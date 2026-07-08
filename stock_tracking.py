from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from functools import lru_cache
from html import unescape
from html.parser import HTMLParser
from io import StringIO
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlsplit, urlunsplit

import pandas as pd
import requests
import yfinance as yf

from config import DATA_DIR, MIN_MARKET_CAP
from database import load_eligible_ticker_symbols


def _load_local_env() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_local_env()

TRACKING_DATA_DIR = DATA_DIR / "stock_tracking"
TRACKING_DB_PATH = TRACKING_DATA_DIR / "stocks.db"
FMP_TRANSCRIPT_RESTRICTED_MESSAGE = "FMP earnings transcript endpoint is restricted on the current API plan."
ALPHAVANTAGE_TRANSCRIPT_MESSAGE = "Alpha Vantage API key is missing. Add ALPHAVANTAGE_API_KEY to .env."
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
GEMMA_MODEL_NAME = os.getenv("GEMMA_MODEL") or os.getenv("OLLAMA_MODEL") or "gemma4:e4b"
NVIDIA_API_URL = os.getenv("NVIDIA_API_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
NVIDIA_SUMMARY_MODEL = os.getenv("NVIDIA_SUMMARY_MODEL", "meta/llama-3.3-70b-instruct")
NVIDIA_TRANSCRIPT_MAX_CHARS = int(os.getenv("NVIDIA_TRANSCRIPT_MAX_CHARS", "120000"))
NVIDIA_TIMEOUT_SECONDS = int(os.getenv("NVIDIA_TIMEOUT_SECONDS", "45"))
IMPORTANT_8K_ITEMS = {
    "1.01",
    "1.02",
    "1.03",
    "1.05",
    "2.01",
    "2.02",
    "2.03",
    "2.05",
    "2.06",
    "3.01",
    "4.02",
    "5.02",
    "5.07",
    "8.01",
}

_FMP_LIMITED_KEYS: set[str] = set()


def _fmp_api_keys() -> list[str]:
    keys: list[str] = []
    for env_key in ["FMP_API_KEY", "FMP_API_KEY_2", "FMP_API_KEY_3"]:
        value = os.getenv(env_key)
        if value and value not in keys:
            keys.append(value)
    return keys


def _fmp_get(url: str, params: dict[str, Any] | None = None, timeout: int = 20) -> requests.Response | None:
    keys = _fmp_api_keys()
    if not keys:
        return None
    base_params = dict(params or {})
    query = dict(parse_qsl(urlsplit(url).query, keep_blank_values=True))
    url_key = str(base_params.get("apikey") or query.get("apikey") or "")
    ordered_keys = ([url_key] if url_key else []) + [key for key in keys if key != url_key]
    clean_url = urlunsplit((urlsplit(url).scheme, urlsplit(url).netloc, urlsplit(url).path, "", urlsplit(url).fragment))
    for api_key in ordered_keys:
        if api_key in _FMP_LIMITED_KEYS:
            continue
        request_params = {**query, **base_params, "apikey": api_key}
        try:
            response = requests.get(clean_url, params=request_params, timeout=timeout)
        except Exception:
            continue
        if response.status_code == 429:
            _FMP_LIMITED_KEYS.add(api_key)
            continue
        return response
    return None
SEC_8K_DEFAULT_PROMPT = (
    "You are a professional equity research analyst. Analyze this SEC 8-K filing for swing traders. "
    "Return a short headline, Sentiment (Bullish/Bearish/Neutral), Risk Level, key facts, likely market impact, "
    "and a concise investor summary under 250 words. Ignore boilerplate legal language."
)


def _connect(db_path: Path = TRACKING_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def tracking_database_exists(db_path: Path = TRACKING_DB_PATH) -> bool:
    return db_path.exists()


def ensure_tracking_schema(db_path: Path = TRACKING_DB_PATH) -> None:
    TRACKING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS stocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL UNIQUE,
                company_name TEXT,
                market_cap TEXT,
                description TEXT,
                fifty_two_week_high TEXT,
                beta REAL,
                latest_news TEXT,
                summary TEXT,
                cik TEXT NOT NULL UNIQUE,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS sec_companies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                company_name TEXT NOT NULL,
                cik TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS stock_news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cik TEXT NOT NULL,
                headline TEXT NOT NULL,
                datetime TEXT
            );

            CREATE TABLE IF NOT EXISTS earnings_growth (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cik TEXT NOT NULL,
                metric TEXT NOT NULL,
                report_type TEXT NOT NULL,
                period_label TEXT NOT NULL,
                reported TEXT,
                estimate TEXT,
                surprise TEXT
            );

            CREATE TABLE IF NOT EXISTS revenue_growth (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cik TEXT NOT NULL,
                report_type TEXT NOT NULL,
                period_label TEXT NOT NULL,
                reported TEXT,
                estimate TEXT,
                surprise TEXT
            );

            CREATE TABLE IF NOT EXISTS quarterly_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cik TEXT NOT NULL,
                quarter TEXT NOT NULL,
                transcript TEXT,
                summary TEXT
            );

            CREATE TABLE IF NOT EXISTS tracker_earnings_growth (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cik TEXT NOT NULL,
                metric TEXT NOT NULL,
                report_type TEXT NOT NULL,
                period_label TEXT NOT NULL,
                reported TEXT
            );

            CREATE TABLE IF NOT EXISTS tracker_revenue_growth (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cik TEXT NOT NULL,
                metric TEXT NOT NULL,
                report_type TEXT NOT NULL,
                period_label TEXT NOT NULL,
                reported TEXT
            );

            CREATE TABLE IF NOT EXISTS tracker_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS earnings_calendar_cache (
                report_date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                company TEXT,
                eps_estimate TEXT,
                revenue_estimate TEXT,
                source TEXT NOT NULL,
                fetched_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (report_date, ticker, source)
            );

            CREATE TABLE IF NOT EXISTS earnings_calendar_detail_cache (
                report_date TEXT NOT NULL,
                include_reported_details INTEGER NOT NULL,
                rows_json TEXT NOT NULL,
                fetched_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (report_date, include_reported_details)
            );
            """
        )
        stock_columns = {row[1] for row in conn.execute("PRAGMA table_info(stocks)").fetchall()}
        if "next_earnings_date" not in stock_columns:
            conn.execute("ALTER TABLE stocks ADD COLUMN next_earnings_date TEXT")
        stock_news_columns = {row[1] for row in conn.execute("PRAGMA table_info(stock_news)").fetchall()}
        for column_name, column_type in {
            "filing_url": "TEXT",
            "accession": "TEXT",
            "form_type": "TEXT",
            "sentiment": "TEXT",
            "source": "TEXT",
            "link": "TEXT",
        }.items():
            if column_name not in stock_news_columns:
                conn.execute(f"ALTER TABLE stock_news ADD COLUMN {column_name} {column_type}")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_stock_news_accession ON stock_news(accession)")
        conn.execute(
            """
            DELETE FROM tracker_earnings_growth
            WHERE id NOT IN (
                SELECT MAX(id)
                FROM tracker_earnings_growth
                GROUP BY cik, metric, report_type, period_label
            )
            """
        )
        conn.execute(
            """
            DELETE FROM tracker_revenue_growth
            WHERE id NOT IN (
                SELECT MAX(id)
                FROM tracker_revenue_growth
                GROUP BY cik, metric, report_type, period_label
            )
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_tracker_earnings_growth_unique
            ON tracker_earnings_growth(cik, metric, report_type, period_label)
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_tracker_revenue_growth_unique
            ON tracker_revenue_growth(cik, metric, report_type, period_label)
            """
        )
        conn.commit()


def _rows(query: str, params: tuple[Any, ...] = (), db_path: Path = TRACKING_DB_PATH) -> list[dict[str, Any]]:
    if not tracking_database_exists(db_path):
        return []
    with _connect(db_path) as conn:
        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def _metadata_get(key: str, db_path: Path = TRACKING_DB_PATH) -> str:
    ensure_tracking_schema(db_path)
    rows = _rows("SELECT value FROM tracker_metadata WHERE key = ?", (key,), db_path)
    return str(rows[0]["value"]) if rows else ""


def _metadata_set(key: str, value: str, db_path: Path = TRACKING_DB_PATH) -> None:
    ensure_tracking_schema(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO tracker_metadata (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
        conn.commit()


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_compact_number(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return ""
    absolute = abs(number)
    if absolute >= 1_000_000_000:
        return f"${number / 1_000_000_000:.2f}B"
    if absolute >= 1_000_000:
        return f"${number / 1_000_000:.2f}M"
    return f"${number:,.0f}"


def _format_plain_number(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return ""
    return f"{number:.2f}"


def _format_percent(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return ""
    return f"{number:.1%}"


def _parse_percent_text(value: Any) -> float | None:
    text = str(value or "").strip().replace("%", "").replace(",", "")
    if not text or text == "-":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _quarter_label(value: Any) -> str:
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return str(value or "")
    quarter = ((date.month - 1) // 3) + 1
    return f"Q{quarter} {date.year}"


def _year_label(value: Any) -> str:
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return str(value or "")
    return str(date.year)


def _news_datetime(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        date = pd.to_datetime(value, unit="s", errors="coerce")
    else:
        date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return str(value)
    return date.strftime("%Y-%m-%d")


def _news_field(item: dict[str, Any], *keys: str) -> Any:
    current: Any = item
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _live_news(stock: yf.Ticker) -> list[dict[str, Any]]:
    try:
        news_items = stock.news or []
    except Exception:
        news_items = []

    rows: list[dict[str, Any]] = []
    for item in news_items[:8]:
        if not isinstance(item, dict):
            continue
        title = (
            item.get("title")
            or _news_field(item, "content", "title")
            or item.get("headline")
            or ""
        )
        publisher = (
            item.get("publisher")
            or _news_field(item, "content", "provider", "displayName")
            or ""
        )
        published_at = (
            item.get("providerPublishTime")
            or _news_field(item, "content", "pubDate")
            or item.get("datetime")
        )
        link = (
            item.get("link")
            or _news_field(item, "content", "canonicalUrl", "url")
            or _news_field(item, "content", "clickThroughUrl", "url")
            or ""
        )
        if title:
            rows.append(
                {
                    "date": _news_datetime(published_at),
                    "source": publisher,
                    "headline": title,
                    "link": link,
                }
            )
    return rows


def _live_eps_quarter_table(stock: yf.Ticker) -> list[dict[str, Any]]:
    income = _quarterly_income(stock)
    try:
        earnings = stock.get_earnings_dates(limit=8)
    except Exception:
        earnings = pd.DataFrame()

    earnings_frame = pd.DataFrame()
    if isinstance(earnings, pd.DataFrame) and not earnings.empty:
        earnings_frame = earnings.copy()
        earnings_frame.index = pd.to_datetime(earnings_frame.index, errors="coerce")
        earnings_frame = earnings_frame[~earnings_frame.index.isna()]
        if "Reported EPS" in earnings_frame.columns:
            earnings_frame = earnings_frame.dropna(subset=["Reported EPS"])
        earnings_frame = earnings_frame.sort_index(ascending=False).head(5)

    if earnings_frame.empty:
        return _live_actual_eps_quarter_table(stock)

    rows = [
        {"Metric": "Expected EPS", "Quarter Info": "Analyst expected EPS before report"},
        {"Metric": "Reported EPS", "Quarter Info": "Yahoo earnings-calendar reported EPS"},
        {"Metric": "Surprise", "Quarter Info": "Reported EPS vs expected EPS"},
        {"Metric": "QoQ EPS Growth", "Quarter Info": "Reported EPS growth vs previous quarter"},
    ]

    fiscal_dates = list(earnings_frame.index[:5])
    reported_values = [_safe_float(row.get("Reported EPS")) for _, row in earnings_frame.iterrows()]
    earnings_records = list(earnings_frame.iterrows())

    for index, date in enumerate(fiscal_dates[:5]):
        label = _quarter_label(date)
        earnings_row = earnings_records[index][1] if index < len(earnings_records) else {}
        estimate = earnings_row.get("EPS Estimate") if hasattr(earnings_row, "get") else None
        reported = reported_values[index] if index < len(reported_values) else None
        surprise = earnings_row.get("Surprise(%)") if hasattr(earnings_row, "get") else None
        if surprise is not None and not pd.isna(surprise):
            surprise = float(surprise)
            if abs(surprise) > 1:
                surprise = surprise / 100
        prior_reported = reported_values[index + 1] if index + 1 < len(reported_values) else None
        qoq_growth = None
        if reported is not None and prior_reported not in (None, 0):
            qoq_growth = (reported - prior_reported) / abs(prior_reported)
        rows[0][label] = _format_plain_number(estimate)
        rows[1][label] = _format_plain_number(reported)
        rows[2][label] = _format_percent(surprise)
        rows[3][label] = _format_percent(qoq_growth)
    return rows


def _live_actual_eps_quarter_table(stock: yf.Ticker) -> list[dict[str, Any]]:
    return []


def _latest_earnings_calendar_record(stock: yf.Ticker) -> dict[str, Any]:
    try:
        earnings = stock.get_earnings_dates(limit=8)
    except Exception:
        return {}
    if not isinstance(earnings, pd.DataFrame) or earnings.empty:
        return {}
    frame = earnings.copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame[~frame.index.isna()]
    if frame.empty:
        return {}
    if "Reported EPS" in frame.columns:
        reported = pd.to_numeric(frame["Reported EPS"], errors="coerce")
        frame = frame.loc[~reported.isna()].copy()
    if frame.empty:
        return {}
    frame = frame.sort_index(ascending=False)
    event_date = frame.index[0]
    row = frame.iloc[0].to_dict()
    row["event_date"] = event_date
    return row


def _fmp_reported_earnings_for_date(ticker: str, report_date: date) -> dict[str, Any]:
    for item in _fmp_earnings_calendar(report_date):
        if _calendar_symbol(item) == ticker.strip().upper():
            return item
    return {}


def _fmp_quarterly_income_records(ticker: str, limit: int = 6) -> list[dict[str, Any]]:
    normalized = ticker.strip().upper()
    if not _fmp_api_keys() or not normalized:
        return []
    urls = (
        (
            "https://financialmodelingprep.com/stable/income-statement",
            {"symbol": normalized, "period": "quarter", "limit": limit},
        ),
        (
            f"https://financialmodelingprep.com/api/v3/income-statement/{normalized}",
            {"period": "quarter", "limit": limit},
        ),
    )
    for url, params in urls:
        response = _fmp_get(url, params=params, timeout=20)
        if response is None:
            continue
        if response.status_code in {401, 402, 403, 429} or response.status_code >= 500:
            continue
        try:
            payload = response.json()
        except ValueError:
            continue
        if isinstance(payload, list) and payload:
            return [item for item in payload if isinstance(item, dict)]
    return []


def _quarterly_income(stock: yf.Ticker) -> pd.DataFrame:
    try:
        income = stock.get_income_stmt(freq="quarterly")
    except Exception:
        income = pd.DataFrame()
    if not isinstance(income, pd.DataFrame):
        return pd.DataFrame()
    return income


def _yearly_income(stock: yf.Ticker) -> pd.DataFrame:
    try:
        income = stock.get_income_stmt(freq="yearly")
    except Exception:
        income = pd.DataFrame()
    if not isinstance(income, pd.DataFrame):
        return pd.DataFrame()
    return income


def _income_series(income: pd.DataFrame, *names: str) -> pd.Series | None:
    for name in names:
        if name in income.index:
            return income.loc[name]
    return None


def _live_revenue_quarter_table(stock: yf.Ticker) -> list[dict[str, Any]]:
    ticker = getattr(stock, "ticker", "") or ""
    income = _quarterly_income(stock)
    rows = [
        {"Metric": "Actual Revenue"},
        {"Metric": "Revenue Growth QoQ"},
    ]

    fmp_records = _fmp_quarterly_income_records(ticker, limit=6) if ticker else []
    if len(fmp_records) >= 2:
        for index, record in enumerate(fmp_records[:5]):
            label = _quarter_label(record.get("date") or record.get("fillingDate") or record.get("acceptedDate"))
            if not label:
                label = str(record.get("period") or f"Quarter {index + 1}")
            current_revenue = _safe_float(record.get("revenue"))
            previous_revenue = _safe_float(fmp_records[index + 1].get("revenue")) if index + 1 < len(fmp_records) else None
            growth = None
            if current_revenue is not None and previous_revenue not in (None, 0):
                growth = (current_revenue - previous_revenue) / abs(previous_revenue)
            rows[0][label] = _format_compact_number(current_revenue)
            rows[1][label] = _format_percent(growth)
        return rows

    revenue = _income_series(income, "Total Revenue", "TotalRevenue") if not income.empty else None
    revenue_values: list[float | None] = []
    if revenue is not None:
        revenue_values = [_safe_float(revenue.iloc[index]) for index in range(min(6, len(revenue)))]

    labels: list[str] = []
    latest_event = _latest_earnings_calendar_record(stock)
    event_date = pd.to_datetime(latest_event.get("event_date"), errors="coerce")
    if not pd.isna(event_date) and getattr(event_date, "tzinfo", None) is not None:
        event_date = event_date.tz_convert(None)
    latest_income_date = pd.to_datetime(income.columns[0], errors="coerce") if not income.empty else pd.NaT
    if ticker and not pd.isna(event_date) and (pd.isna(latest_income_date) or event_date > latest_income_date):
        fmp_event = _fmp_reported_earnings_for_date(ticker, event_date.date())
        fmp_revenue = _safe_float(
            _calendar_value(fmp_event, "revenueActual", "revenue", "actualRevenue", "revenueReported")
        )
        if fmp_revenue is not None:
            labels.append(_quarter_label(event_date))
            previous_revenue = revenue_values[0] if revenue_values else None
            growth = None
            if previous_revenue not in (None, 0):
                growth = (fmp_revenue - previous_revenue) / abs(previous_revenue)
            rows[0][labels[-1]] = _format_compact_number(fmp_revenue)
            rows[1][labels[-1]] = _format_percent(growth)

    if revenue is None:
        return rows if len(rows[0]) > 1 else []

    dates = list(income.columns[:6])
    for index, date in enumerate(dates):
        if len(labels) >= 5:
            break
        label = _quarter_label(date)
        if label in labels:
            continue
        labels.append(label)
        current_revenue = _safe_float(revenue.iloc[index])
        previous_revenue = _safe_float(revenue.iloc[index + 1]) if index + 1 < len(revenue) else None
        growth = None
        if current_revenue is not None and previous_revenue not in (None, 0):
            growth = (current_revenue - previous_revenue) / abs(previous_revenue)
        rows[0][label] = _format_compact_number(current_revenue)
        rows[1][label] = _format_percent(growth)
    return rows


def _live_eps_annual_table(stock: yf.Ticker) -> list[dict[str, Any]]:
    income = _yearly_income(stock)
    if income.empty or income.shape[1] < 1:
        return []
    eps = _income_series(income, "Diluted EPS", "DilutedEPS", "Basic EPS", "BasicEPS")
    if eps is None:
        return []

    rows = [
        {"Metric": "Reported EPS"},
        {"Metric": "EPS Growth YoY"},
    ]
    eps_values = [_safe_float(eps.iloc[index]) for index in range(min(6, len(eps)))]
    for index, date in enumerate(list(income.columns[:5])):
        label = _year_label(date)
        reported = eps_values[index] if index < len(eps_values) else None
        previous = eps_values[index + 1] if index + 1 < len(eps_values) else None
        growth = None
        if reported is not None and previous not in (None, 0):
            growth = (reported - previous) / abs(previous)
        rows[0][label] = _format_plain_number(reported)
        rows[1][label] = _format_percent(growth)
    return rows


def _live_revenue_annual_table(stock: yf.Ticker) -> list[dict[str, Any]]:
    income = _yearly_income(stock)
    if income.empty or income.shape[1] < 1:
        return []
    revenue = _income_series(income, "Total Revenue", "TotalRevenue")
    if revenue is None:
        return []

    rows = [
        {"Metric": "Actual Revenue"},
        {"Metric": "Revenue Growth YoY"},
    ]
    revenue_values = [_safe_float(revenue.iloc[index]) for index in range(min(6, len(revenue)))]
    for index, date in enumerate(list(income.columns[:5])):
        label = _year_label(date)
        current_revenue = revenue_values[index] if index < len(revenue_values) else None
        previous_revenue = revenue_values[index + 1] if index + 1 < len(revenue_values) else None
        growth = None
        if current_revenue is not None and previous_revenue not in (None, 0):
            growth = (current_revenue - previous_revenue) / abs(previous_revenue)
        rows[0][label] = _format_compact_number(current_revenue)
        rows[1][label] = _format_percent(growth)
    return rows


def _table_to_growth_records(table: list[dict[str, Any]], report_type: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for row in table:
        if not isinstance(row, dict):
            continue
        metric = str(row.get("Metric") or "").strip()
        if not metric:
            continue
        for period_label, value in row.items():
            if period_label in {"Metric", "Quarter Info"}:
                continue
            value_text = str(value or "").strip()
            if not value_text:
                continue
            records.append(
                {
                    "metric": metric,
                    "report_type": report_type,
                    "period_label": str(period_label),
                    "reported": value_text,
                    "estimate": "",
                    "surprise": "",
                }
            )
    return records


def _period_sort_key(period_label: str, report_type: str) -> tuple[int, int, str]:
    text = str(period_label or "").strip()
    if report_type == "annual":
        match = re.search(r"(20\d{2}|19\d{2})", text)
        if match:
            return (int(match.group(1)), 0, text)
        return (0, 0, text)
    match = re.search(r"Q([1-4])\s*['-]?\s*(20\d{2}|\d{2})", text, flags=re.IGNORECASE)
    if match:
        quarter = int(match.group(1))
        year_text = match.group(2)
        year = int(year_text)
        if year < 100:
            year += 2000
        return (year, quarter, text)
    date_value = pd.to_datetime(text, errors="coerce")
    if not pd.isna(date_value):
        quarter = ((date_value.month - 1) // 3) + 1
        return (int(date_value.year), quarter, text)
    return (0, 0, text)


def _growth_records_to_table(records: list[dict[str, Any]], report_type: str) -> list[dict[str, Any]]:
    filtered = [row for row in records if str(row.get("report_type") or "") == report_type]
    if not filtered:
        return []

    metric_order: list[str] = []
    period_order: list[str] = []
    for row in filtered:
        metric = str(row.get("metric") or "").strip()
        period = str(row.get("period_label") or "").strip()
        if metric and metric not in metric_order:
            metric_order.append(metric)
        if period and period not in period_order:
            period_order.append(period)
    period_order.sort(key=lambda period: _period_sort_key(period, report_type), reverse=True)

    table: list[dict[str, Any]] = []
    for metric in metric_order:
        out: dict[str, Any] = {"Metric": metric}
        for period in period_order:
            match = next(
                (
                    row
                    for row in filtered
                    if str(row.get("metric") or "") == metric
                    and str(row.get("period_label") or "") == period
                ),
                None,
            )
            out[period] = str(match.get("reported") or "") if match else ""
        table.append(out)
    return table


def _display_number_to_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text or text == "-":
        return None
    multiplier = 1.0
    upper = text.upper()
    if upper.endswith("B"):
        multiplier = 1_000_000_000.0
        text = text[:-1]
    elif upper.endswith("M"):
        multiplier = 1_000_000.0
        text = text[:-1]
    elif upper.endswith("K"):
        multiplier = 1_000.0
        text = text[:-1]
    text = text.replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        return float(text) * multiplier
    except ValueError:
        return None


def _merge_growth_table_rows(
    existing_table: list[dict[str, Any]],
    incoming_table: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_metric: dict[str, dict[str, Any]] = {}
    metric_order: list[str] = []

    def merge_row(row: dict[str, Any]) -> None:
        metric = str(row.get("Metric") or "").strip()
        if not metric:
            return
        if metric not in rows_by_metric:
            rows_by_metric[metric] = {"Metric": metric}
            metric_order.append(metric)
        for key, value in row.items():
            if key in {"Metric", "Quarter Info"}:
                continue
            if str(value or "").strip():
                rows_by_metric[metric][str(key)] = value

    for source_row in existing_table or []:
        if isinstance(source_row, dict):
            merge_row(source_row)
    for source_row in incoming_table or []:
        if isinstance(source_row, dict):
            merge_row(source_row)

    return [rows_by_metric[metric] for metric in metric_order if metric in rows_by_metric]


def _upsert_growth_row_from_actuals(
    table: list[dict[str, Any]],
    actual_metric: str,
    growth_metric: str,
) -> list[dict[str, Any]]:
    actual_row = next((row for row in table if row.get("Metric") == actual_metric), None)
    if not actual_row:
        return table

    periods = [key for key in actual_row.keys() if key not in {"Metric", "Quarter Info"}]
    periods.sort(key=lambda period: _period_sort_key(period, "quarterly"), reverse=True)
    if not periods:
        return table

    growth_row = next((row for row in table if row.get("Metric") == growth_metric), None)
    if growth_row is None:
        growth_row = {"Metric": growth_metric}
        insert_after = next(
            (index + 1 for index, row in enumerate(table) if row.get("Metric") == actual_metric),
            len(table),
        )
        table.insert(insert_after, growth_row)

    for index, period in enumerate(periods):
        current_value = _display_number_to_float(actual_row.get(period))
        previous_period = periods[index + 1] if index + 1 < len(periods) else ""
        previous_value = _display_number_to_float(actual_row.get(previous_period)) if previous_period else None
        if current_value is not None and previous_value not in (None, 0):
            growth_row[period] = _format_percent((current_value - previous_value) / abs(previous_value))
    return table


def _merge_calendar_growth_with_history(
    ticker: str,
    eps_rows: list[dict[str, Any]],
    revenue_rows: list[dict[str, Any]],
    db_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    existing = get_stock_details(ticker, db_path) or {}
    merged_eps = _merge_growth_table_rows(existing.get("eps_quarter_table", []), eps_rows)
    merged_revenue = _merge_growth_table_rows(existing.get("revenue_quarter_table", []), revenue_rows)
    merged_eps = _upsert_growth_row_from_actuals(merged_eps, "Reported EPS", "QoQ EPS Growth")
    merged_revenue = _upsert_growth_row_from_actuals(merged_revenue, "Actual Revenue", "Revenue Growth QoQ")
    return merged_eps, merged_revenue


def _revenue_db_rows_to_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        label = str(row.get("period_label") or "")
        if "|" in label:
            metric, period_label = label.split("|", 1)
        else:
            metric, period_label = "Actual Revenue", label
        out.append(
            {
                **row,
                "metric": metric,
                "period_label": period_label,
            }
        )
    return out


def save_growth_tables(
    ticker: str,
    eps_quarter_table: list[dict[str, Any]],
    revenue_quarter_table: list[dict[str, Any]],
    eps_annual_table: list[dict[str, Any]] | None = None,
    revenue_annual_table: list[dict[str, Any]] | None = None,
    db_path: Path = TRACKING_DB_PATH,
) -> dict[str, int]:
    normalized = ticker.strip().upper()
    ensure_tracking_schema(db_path)
    stock_rows = _rows("SELECT cik FROM stocks WHERE ticker = ?", (normalized,), db_path)
    if not stock_rows:
        return {"eps_rows": 0, "revenue_rows": 0}
    cik = stock_rows[0]["cik"]

    eps_records = _table_to_growth_records(eps_quarter_table, "quarterly")
    eps_records.extend(_table_to_growth_records(eps_annual_table or [], "annual"))
    revenue_records = _table_to_growth_records(revenue_quarter_table, "quarterly")
    revenue_records.extend(_table_to_growth_records(revenue_annual_table or [], "annual"))

    with _connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO tracker_earnings_growth (cik, metric, report_type, period_label, reported)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(cik, metric, report_type, period_label) DO UPDATE SET
                reported = excluded.reported
            """,
            [
                (
                    cik,
                    row["metric"],
                    row["report_type"],
                    row["period_label"],
                    row["reported"],
                )
                for row in eps_records
            ],
        )
        conn.executemany(
            """
            INSERT INTO tracker_revenue_growth (cik, metric, report_type, period_label, reported)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(cik, metric, report_type, period_label) DO UPDATE SET
                reported = excluded.reported
            """,
            [
                (
                    cik,
                    row["metric"],
                    row["report_type"],
                    row["period_label"],
                    row["reported"],
                )
                for row in revenue_records
            ],
        )
        conn.commit()
    return {"eps_rows": len(eps_records), "revenue_rows": len(revenue_records)}


def save_live_news_for_ticker(
    ticker: str,
    cik: str,
    news_rows: list[dict[str, Any]],
    db_path: Path = TRACKING_DB_PATH,
) -> int:
    normalized = ticker.strip().upper()
    ensure_tracking_schema(db_path)
    clean_rows: list[tuple[str, str, str, str]] = []
    for item in news_rows:
        if not isinstance(item, dict):
            continue
        headline = str(item.get("headline") or "").strip()
        if not headline:
            continue
        clean_rows.append(
            (
                str(item.get("date") or item.get("datetime") or ""),
                str(item.get("source") or "Yahoo Finance"),
                headline,
                str(item.get("link") or ""),
            )
        )

    with _connect(db_path) as conn:
        conn.execute("DELETE FROM stock_news WHERE cik = ? AND form_type = 'NEWS'", (cik,))
        conn.executemany(
            """
            INSERT INTO stock_news (cik, headline, datetime, form_type, source, link)
            VALUES (?, ?, ?, 'NEWS', ?, ?)
            """,
            [(cik, headline, datetime_value, source, link) for datetime_value, source, headline, link in clean_rows],
        )
        conn.commit()
    return len(clean_rows)


def save_live_growth_tables_for_ticker(ticker: str, db_path: Path = TRACKING_DB_PATH) -> dict[str, int]:
    normalized = ticker.strip().upper()
    stock = yf.Ticker(normalized)
    return save_growth_tables(
        normalized,
        _live_eps_quarter_table(stock),
        _live_revenue_quarter_table(stock),
        _live_eps_annual_table(stock),
        _live_revenue_annual_table(stock),
        db_path,
    )


def _growth_text(current: Any, prior: Any) -> str:
    current_number = _safe_float(current)
    prior_number = _safe_float(prior)
    if current_number is None or prior_number in (None, 0):
        return ""
    return _format_percent((current_number - prior_number) / abs(prior_number))


def _number_from_text(value: str) -> float | None:
    text = str(value or "").strip().replace(",", "")
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()$ ")
    try:
        number = float(text)
    except ValueError:
        return None
    return -number if negative else number


def _money_from_text(value: str, unit: str = "") -> float | None:
    number = _number_from_text(value)
    if number is None:
        return None
    lowered = str(unit or "").lower()
    if lowered.startswith("b"):
        return number * 1_000_000_000
    if lowered.startswith("m"):
        return number * 1_000_000
    return number


def _quarter_label_from_text(text: str) -> str:
    patterns = (
        r"\b(Q[1-4])\s*['-]?\s*(20\d{2}|\d{2})\b",
        r"\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter(?:\s+of)?\s+(20\d{2})\b",
    )
    quarter_map = {
        "first": "Q1",
        "1st": "Q1",
        "second": "Q2",
        "2nd": "Q2",
        "third": "Q3",
        "3rd": "Q3",
        "fourth": "Q4",
        "4th": "Q4",
    }
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        quarter = match.group(1).upper()
        quarter = quarter_map.get(quarter.lower(), quarter)
        year = int(match.group(2))
        if year < 100:
            year += 2000
        return f"{quarter} {year}"
    return "Transcript quarter"


def _extract_reported_eps_from_text(text: str) -> float | None:
    cleaned = re.sub(r"\s+", " ", str(text or ""))
    patterns = (
        r"(?P<context>(?:non-gaap\s+)?(?:diluted\s+)?(?:net\s+)?(?:loss|income|earnings)?\s*(?:per share|eps|earnings per share))[^.]{0,100}?\$?\s*(?P<number>\(?-?\d+(?:\.\d+)?\)?)",
        r"(?P<number>\(?-?\d+(?:\.\d+)?\)?)\s+(?P<context>per diluted share|per share)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, cleaned, flags=re.IGNORECASE):
            context = match.group("context").lower()
            value = _number_from_text(match.group("number"))
            if value is None:
                continue
            if "loss" in context and value > 0:
                value = -value
            return value
    return None


def _extract_revenue_from_text(text: str) -> float | None:
    cleaned = re.sub(r"\s+", " ", str(text or ""))
    patterns = (
        r"(?:total\s+)?revenue[^.]{0,100}?\$\s*(?P<number>[0-9][0-9,]*(?:\.\d+)?)\s*(?P<unit>billion|million|bn|mm|m|b)?",
        r"\$\s*(?P<number>[0-9][0-9,]*(?:\.\d+)?)\s*(?P<unit>billion|million|bn|mm|m|b)\s+(?:in\s+)?(?:total\s+)?revenue",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if not match:
            continue
        revenue = _money_from_text(match.group("number"), match.groupdict().get("unit") or "")
        if revenue is not None:
            return revenue
    return None


def _extract_sec_press_release_eps_from_text(text: str) -> float | None:
    cleaned = re.sub(r"[ \t]+", " ", str(text or ""))
    patterns = (
        r"Earnings\s+per\s+share\s+Basic\s+\$?\s*\(?-?\d+(?:\.\d+)?\)?\s+\$?\s*\(?-?\d+(?:\.\d+)?\)?\s+\$?\s*\(?-?\d+(?:\.\d+)?\)?\s+Diluted\s+(?P<number>\(?-?\d+(?:\.\d+)?\)?)",
        r"Diluted\s+(?P<number>\(?-?\d+(?:\.\d+)?\)?)\s+\(?-?\d+(?:\.\d+)?\)?\s+\(?-?\d+(?:\.\d+)?\)?",
        r"Adjusted\s+(?:diluted\s+)?earnings\s+per\s+share[^$0-9-]{0,80}\$?\s*(?P<number>\(?-?\d+(?:\.\d{1,2})?\)?)",
        r"Diluted\s+earnings\s+per\s+(?:voting\s+common\s+)?share[^$0-9-]{0,80}\$?\s*(?P<number>\(?-?\d+(?:\.\d{1,2})?\)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.I | re.S)
        if not match:
            continue
        value = _number_from_text(match.group("number"))
        if value is not None and abs(value) < 100:
            return value
    fallback = _extract_reported_eps_from_text(text)
    return fallback if fallback is not None and abs(fallback) < 100 else None


def _extract_sec_press_release_revenue_from_text(text: str) -> float | None:
    cleaned = re.sub(r"[ \t]+", " ", str(text or ""))
    scale = 1.0
    lowered_head = cleaned[:1200].lower()
    if "in thousands" in lowered_head or "in 000s" in lowered_head:
        scale = 1_000.0
    elif "in millions" in lowered_head or "in 000,000s" in lowered_head:
        scale = 1_000_000.0
    patterns = (
        r"(?:Total\s+)?Revenue\s+\$?\s*(?P<number>[0-9][0-9,]*(?:\.\d+)?)\b",
        r"(?:Net\s+sales|Net\s+revenues|Total\s+net\s+revenues)\s+\$?\s*(?P<number>[0-9][0-9,]*(?:\.\d+)?)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.I)
        if not match:
            continue
        value = _number_from_text(match.group("number"))
        if value is None:
            continue
        if scale != 1:
            return value * scale
        if value < 1_000_000:
            return value * 1_000_000
        return value
    return _extract_revenue_from_text(text)


def extract_transcript_metric_tables(transcript_text: str) -> dict[str, list[dict[str, Any]]]:
    text = str(transcript_text or "").strip()
    if not text:
        return {"eps_quarter_table": [], "revenue_quarter_table": []}
    quarter = _quarter_label_from_text(text)
    eps = _extract_reported_eps_from_text(text)
    revenue = _extract_revenue_from_text(text)
    eps_rows = [{"Metric": "Reported EPS", quarter: _format_plain_number(eps)}] if eps is not None else []
    revenue_rows = [{"Metric": "Actual Revenue", quarter: _format_compact_number(revenue)}] if revenue is not None else []
    return {"eps_quarter_table": eps_rows, "revenue_quarter_table": revenue_rows}


def _news_matches(news_rows: list[dict[str, Any]], keywords: tuple[str, ...]) -> str:
    matches: list[str] = []
    for item in news_rows:
        headline = str(item.get("headline") or "").strip()
        if not headline:
            continue
        lowered = headline.lower()
        if any(keyword in lowered for keyword in keywords):
            date = item.get("date") or item.get("datetime") or ""
            source = item.get("source") or ""
            prefix = " - ".join(part for part in [str(date), str(source)] if part)
            matches.append(f"{prefix}: {headline}" if prefix else headline)
    return " | ".join(matches[:3])


def _sentence_matches(text: str, keywords: tuple[str, ...], limit: int = 3) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    matches: list[str] = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in keywords):
            matches.append(sentence.strip())
        if len(matches) >= limit:
            break
    return " ".join(matches)[:700]


def _transcript_payload_text(payload: Any) -> str:
    if isinstance(payload, dict) and isinstance(payload.get("transcript"), list):
        parts: list[str] = []
        for turn in payload["transcript"]:
            if not isinstance(turn, dict):
                continue
            speaker = str(turn.get("speaker") or "").strip()
            title = str(turn.get("title") or "").strip()
            content = str(turn.get("content") or "").strip()
            if not content:
                continue
            label = " - ".join(part for part in [speaker, title] if part)
            parts.append(f"{label}: {content}" if label else content)
        return "\n".join(parts)
    if isinstance(payload, list) and payload:
        payload = payload[0]
    if not isinstance(payload, dict):
        return ""
    for key in ("content", "transcript", "text", "finalTranscript"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        if tag in {"h1", "h2", "h3", "p", "li", "br", "div"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self.parts.append(text)


def _article_text_from_html(html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html)
    text = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", "\n".join(parser.parts))).strip()
    start_markers = ("Full Conference Call Transcript", "TAKEAWAYS", "DATE", "Story Continues")
    starts = [text.find(marker) for marker in start_markers if text.find(marker) >= 0]
    if starts:
        text = text[min(starts) :]
    end_markers = (
        "The Motley Fool has positions",
        "Join Stock Advisor",
        "Related Articles",
        "Invest better with The Motley Fool",
        "Sign in to access",
        "Recommended Stories",
        "Most Read from Bloomberg",
    )
    ends = [text.find(marker) for marker in end_markers if text.find(marker) > 0]
    if ends:
        text = text[: min(ends)]
    return text.strip()


def _fetch_transcript_url_html(url: str) -> str:
    cleaned_url = str(url or "").strip()
    if not cleaned_url.startswith(("https://", "http://")):
        raise ValueError("Transcript URL must start with http:// or https://.")
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = requests.get(
                cleaned_url,
                headers=headers,
                timeout=(8, 45),
            )
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 1:
                time.sleep(2 + attempt)
    raise RuntimeError(f"Could not fetch URL after retries: {last_error}")


def _fetch_transcript_url_text(url: str) -> str:
    html = _fetch_transcript_url_html(url)
    text = _article_text_from_html(html)
    if len(text) < 500:
        raise ValueError("Could not extract enough transcript text from that URL.")
    return text


def _yahoo_article_links_for_ticker(ticker: str, limit: int = 5) -> list[dict[str, str]]:
    normalized = ticker.strip().upper()
    if not normalized:
        return []
    stock = yf.Ticker(normalized)
    links: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in _live_news(stock):
        link = str(item.get("link") or "").strip()
        headline = str(item.get("headline") or "").strip()
        source = str(item.get("source") or "").strip()
        if not link or link in seen:
            continue
        if "finance.yahoo.com" not in link and "yahoo.com" not in link:
            continue
        seen.add(link)
        links.append({"url": link, "headline": headline, "source": source})
        if len(links) >= limit:
            break
    return links


def _fallback_article_rows_for_ticker(
    ticker: str,
    source_url: str,
    article_text: str,
    headline: str = "",
) -> list[dict[str, Any]]:
    status = "Yahoo Finance article fallback used because Motley Fool Takeaways were unavailable."
    if headline:
        status = f"{status} Article: {headline}"
    return _nvidia_fallback_rows_for_ticker(ticker, status, article_text)


def _clean_html_fragment(fragment: str) -> str:
    text = re.sub(r"(?i)<br\s*/?>", "\n", fragment)
    text = re.sub(r"(?i)</(?:p|div|li|h[1-6])>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = unescape(text)
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{2,}", "\n", text)).strip()


def _motley_fool_takeaway_rows_from_html(html: str) -> list[dict[str, str]]:
    if not html:
        return []
    match = re.search(
        r'<h2\b[^>]*id=["\']takeaways["\'][^>]*>.*?</h2>\s*<ul\b[^>]*>(?P<body>.*?)</ul>',
        html,
        flags=re.I | re.S,
    )
    if not match:
        return []

    rows: list[dict[str, str]] = []
    for item in re.findall(r"<li\b[^>]*>(.*?)</li>", match.group("body"), flags=re.I | re.S):
        text = _clean_html_fragment(item)
        if not text:
            continue
        if " -- " in text:
            heading, summary = text.split(" -- ", 1)
        elif "--" in text:
            heading, summary = text.split("--", 1)
        else:
            heading, summary = "Takeaway", text
        heading = heading.strip()
        summary = summary.strip()
        if heading and summary:
            rows.append(
                {
                    "section": heading,
                    "quarter": "Motley Fool Takeaways",
                    "summary": summary,
                }
            )
    return rows


def _motley_fool_transcript_links_for_ticker(ticker: str) -> list[str]:
    normalized = ticker.strip().lower()
    if not normalized:
        return []

    links: list[str] = []
    seen: set[str] = set()
    for exchange in ("nasdaq", "nyse", "amex"):
        quote_url = f"https://www.fool.com/quote/{exchange}/{normalized}/"
        try:
            response = requests.get(quote_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        except Exception:
            continue
        if response.status_code >= 400:
            continue
        raw_links = re.findall(
            r"(?:https?://www\.fool\.com)?/earnings/call-transcripts/[^\"'<>\s]+",
            response.text,
            flags=re.I,
        )
        for raw_link in raw_links:
            cleaned = unescape(raw_link).strip().rstrip("\\,")
            cleaned = cleaned.split("\\", 1)[0].split("?", 1)[0]
            url = urljoin("https://www.fool.com", cleaned)
            if ticker.lower() not in url.lower():
                continue
            if url not in seen:
                links.append(url)
                seen.add(url)
        if links:
            break
    return links


def _motley_fool_takeaway_rows(text: str) -> list[dict[str, str]]:
    if not text:
        return []
    match = re.search(
        r"\bTAKEAWAYS\b(?P<body>.*?)(?:\n\s*(?:CATALYST|FULL CONFERENCE CALL TRANSCRIPT|TRANSCRIPT|DATE|CALL PARTICIPANTS|PREPARED REMARKS)\b|$)",
        text,
        flags=re.I | re.S,
    )
    if not match:
        return []
    body = match.group("body").strip()
    if not body:
        return []

    rows: list[dict[str, str]] = []
    current_heading = ""
    current_summary: list[str] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if " -- " in line:
            heading, summary = line.split(" -- ", 1)
            heading = heading.strip()
            summary = summary.strip()
            if heading and summary:
                if current_heading and current_summary:
                    rows.append(
                        {
                            "section": current_heading,
                            "quarter": "Motley Fool Takeaways",
                            "summary": " ".join(current_summary),
                        }
                    )
                rows.append(
                    {
                        "section": heading,
                        "quarter": "Motley Fool Takeaways",
                        "summary": summary,
                    }
                )
                current_heading = ""
                current_summary = []
                continue
        if line.startswith("--"):
            summary = line.lstrip("-").strip()
            if summary:
                current_summary.append(summary)
            continue

        if current_heading and current_summary:
            rows.append(
                {
                    "section": current_heading,
                    "quarter": "Motley Fool Takeaways",
                    "summary": " ".join(current_summary),
                }
            )
        current_heading = line
        current_summary = []

    if current_heading and current_summary:
        rows.append(
            {
                "section": current_heading,
                "quarter": "Motley Fool Takeaways",
                "summary": " ".join(current_summary),
            }
        )
    return rows


def _latest_quarter_year(income: pd.DataFrame) -> tuple[int | None, int | None]:
    if income.empty:
        return None, None
    date = pd.to_datetime(income.columns[0], errors="coerce")
    if pd.isna(date):
        return None, None
    return ((date.month - 1) // 3) + 1, int(date.year)


def _fetch_alphavantage_transcript(ticker: str, quarter: int, year: int) -> dict[str, str]:
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return {"status": ALPHAVANTAGE_TRANSCRIPT_MESSAGE, "text": "", "source": "Alpha Vantage"}

    quarter_code = f"{year}Q{quarter}"
    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "EARNINGS_CALL_TRANSCRIPT",
                "symbol": ticker,
                "quarter": quarter_code,
                "apikey": api_key,
            },
            timeout=20,
        )
    except Exception as exc:
        return {"status": f"Alpha Vantage transcript request failed: {exc}", "text": "", "source": "Alpha Vantage"}

    if response.status_code >= 400:
        return {
            "status": f"Alpha Vantage transcript request failed with HTTP {response.status_code}.",
            "text": "",
            "source": "Alpha Vantage",
        }
    try:
        payload = response.json()
    except ValueError:
        return {"status": "Alpha Vantage transcript response was not JSON.", "text": "", "source": "Alpha Vantage"}

    if isinstance(payload, dict):
        message = payload.get("Information") or payload.get("Note") or payload.get("Error Message")
        if message:
            normalized = str(message)
            if "rate limit" in normalized.lower() or "premium" in normalized.lower():
                normalized = "Alpha Vantage daily API rate limit was reached. Transcript lookup can run again after the limit resets."
            return {"status": normalized, "text": "", "source": "Alpha Vantage"}

    text = _transcript_payload_text(payload)
    if text:
        return {
            "status": f"Alpha Vantage transcript loaded for {quarter_code}.",
            "text": text,
            "source": "Alpha Vantage",
        }
    return {"status": f"No Alpha Vantage transcript found for {quarter_code}.", "text": "", "source": "Alpha Vantage"}


def _fetch_fmp_transcript(ticker: str, quarter: int, year: int) -> dict[str, str]:
    if not _fmp_api_keys():
        return {"status": "FMP API key is missing.", "text": "", "source": "FMP"}

    urls = [
        (
            "FMP stable transcript",
            "https://financialmodelingprep.com/stable/earning-call-transcript",
            {"symbol": ticker, "quarter": quarter, "year": year},
        ),
        (
            "FMP latest transcript",
            "https://financialmodelingprep.com/stable/earning-call-transcript-latest",
            {"symbol": ticker},
        ),
    ]
    restricted = False
    for source, url, params in urls:
        response = _fmp_get(url, params=params, timeout=15)
        if response is None:
            continue
        if response.status_code == 402:
            restricted = True
            continue
        if response.status_code in {403, 404}:
            continue
        if response.status_code >= 400:
            continue
        try:
            payload = response.json()
        except ValueError:
            continue
        text = _transcript_payload_text(payload)
        if text:
            return {
                "status": f"FMP transcript loaded for Q{quarter} {year}.",
                "text": text,
                "source": source,
            }

    if restricted:
        return {"status": FMP_TRANSCRIPT_RESTRICTED_MESSAGE, "text": "", "source": "FMP"}
    return {"status": f"No FMP transcript found for Q{quarter} {year}.", "text": "", "source": "FMP"}


def _fetch_latest_earnings_transcript(ticker: str, income: pd.DataFrame) -> dict[str, str]:
    quarter, year = _latest_quarter_year(income)
    if quarter is None or year is None:
        return {"status": "Latest fiscal quarter was unavailable.", "text": "", "source": "FMP"}

    alpha_vantage = _fetch_alphavantage_transcript(ticker, quarter, year)
    if alpha_vantage.get("text"):
        return alpha_vantage

    fmp = _fetch_fmp_transcript(ticker, quarter, year)
    if fmp.get("text"):
        return fmp

    return {
        "status": f"{alpha_vantage.get('status', '')} Fallback: {fmp.get('status', '')}".strip(),
        "text": "",
        "source": "Alpha Vantage/FMP",
    }


def _extract_json_object(text: str) -> dict[str, str]:
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): str(value).strip() for key, value in payload.items() if value not in (None, "")}


def _gemma4_transcript_analysis(
    ticker: str,
    quarter_label: str,
    transcript_text: str,
    eps_summary: str,
    revenue_summary: str,
) -> dict[str, str]:
    if not transcript_text.strip():
        return {}

    prompt = f"""
You are an equity research analyst helping build a quarterly summary table for {ticker}.

Use the full earnings call transcript below to summarize the whole earnings call, not just one topic.

Do not calculate EPS or revenue from the transcript. Those structured values are already available:
EPS row: {eps_summary}
Revenue row: {revenue_summary}

Return ONLY valid compact JSON with exactly these keys:
{{
  "Transcript Summary": "3 to 5 concise sentences covering the whole call: business performance, demand, margins, management tone, and major quarter themes.",
  "Key Positives": "1 to 3 concise points from the transcript about what improved or sounded strong.",
  "Risks / Concerns": "1 to 3 concise points about risks, weak areas, supply issues, margin pressure, demand concerns, or uncertainty.",
  "Backlog": "one short sentence, or Not discussed in transcript.",
  "New Product Highlight": "highlight any new products, launches, platforms, approvals, or partnerships. If none, say Not discussed in transcript.",
  "Forward Guidance": "one or two concise sentences about guidance, outlook, forecast, or management expectations.",
  "Trading Takeaway": "one concise swing-trading watchlist takeaway based on the call tone and business momentum, without giving financial advice."
}}

Quarter: {quarter_label}
Full transcript:
{transcript_text}
""".strip()

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": GEMMA_MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1, "num_predict": 1200},
            },
            timeout=NVIDIA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return {}

    content = payload.get("response") if isinstance(payload, dict) else ""
    result = _extract_json_object(str(content or ""))
    allowed = {
        "Transcript Summary",
        "Key Positives",
        "Risks / Concerns",
        "Backlog",
        "New Product Highlight",
        "Forward Guidance",
        "Trading Takeaway",
    }
    return {key: value for key, value in result.items() if key in allowed}


def _transcript_summary_prompt(
    ticker: str,
    quarter_label: str,
    transcript_text: str,
    eps_summary: str,
    revenue_summary: str,
) -> str:
    clipped_text = transcript_text[:NVIDIA_TRANSCRIPT_MAX_CHARS]
    if len(transcript_text) > len(clipped_text):
        clipped_text = f"{clipped_text}\n\n[Transcript clipped at {len(clipped_text):,} characters.]"
    return f"""
You are an equity research analyst helping build a quarterly summary table for {ticker}.

Use the earnings call transcript/article below to summarize the whole call.

Do not calculate EPS or revenue from the transcript. These structured values are already available:
EPS row: {eps_summary}
Revenue row: {revenue_summary}

Return ONLY valid compact JSON with exactly these keys:
{{
  "Transcript Summary": "3 to 5 concise sentences covering the whole call: business performance, demand, margins, management tone, and major quarter themes.",
  "Key Positives": "1 to 3 concise points from the transcript about what improved or sounded strong.",
  "Risks / Concerns": "1 to 3 concise points about risks, weak areas, supply issues, margin pressure, demand concerns, or uncertainty.",
  "Backlog": "one short sentence, or Not discussed in transcript.",
  "New Product Highlight": "highlight any new products, launches, platforms, approvals, or partnerships. If none, say Not discussed in transcript.",
  "Forward Guidance": "one or two concise sentences about guidance, outlook, forecast, or management expectations.",
  "Trading Takeaway": "one concise swing-trading watchlist takeaway based on the call tone and business momentum, without giving financial advice."
}}

Quarter: {quarter_label}
Transcript/article text:
{clipped_text}
""".strip()


def _nvidia_transcript_analysis(
    ticker: str,
    quarter_label: str,
    transcript_text: str,
    eps_summary: str,
    revenue_summary: str,
) -> dict[str, str]:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key or not transcript_text.strip():
        return {}

    prompt = _transcript_summary_prompt(ticker, quarter_label, transcript_text, eps_summary, revenue_summary)
    try:
        response = requests.post(
            NVIDIA_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": NVIDIA_SUMMARY_MODEL,
                "messages": [
                    {"role": "system", "content": "You return strict JSON for earnings transcript analysis."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 1400,
                "stream": False,
            },
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return {}

    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not choices:
        return {}
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    result = _extract_json_object(str(content or ""))
    allowed = {
        "Transcript Summary",
        "Key Positives",
        "Risks / Concerns",
        "Backlog",
        "New Product Highlight",
        "Forward Guidance",
        "Trading Takeaway",
    }
    return {key: value for key, value in result.items() if key in allowed}


def _live_quarterly_summary(
    stock: yf.Ticker,
    transcript_override: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    income = _quarterly_income(stock)
    eps_rows = _live_eps_quarter_table(stock)
    revenue_rows = _live_revenue_quarter_table(stock)
    news_rows = _live_news(stock)
    override_text = str((transcript_override or {}).get("text") or "").strip()
    if income.empty and not eps_rows and not revenue_rows and not news_rows and not override_text:
        return []

    summaries = []
    eps_actual = next((row for row in eps_rows if row.get("Metric") == "Reported EPS"), {})
    eps_qoq = next((row for row in eps_rows if row.get("Metric") == "QoQ EPS Growth"), {})
    revenue_actual = next((row for row in revenue_rows if row.get("Metric") == "Actual Revenue"), {})
    revenue_qoq = next((row for row in revenue_rows if row.get("Metric") == "Revenue Growth QoQ"), {})

    labels: list[str] = []
    for rows in (eps_rows, revenue_rows):
        for row in rows:
            labels.extend([key for key in row if key not in {"Metric", "Quarter Info"}])
    labels = list(dict.fromkeys(labels))[:5]
    latest_label = labels[0] if labels else "Latest quarter"
    ticker = str(getattr(stock, "ticker", "") or "").upper()
    transcript = transcript_override or (_fetch_latest_earnings_transcript(ticker, income) if ticker else {
        "status": "Ticker unavailable for transcript lookup.",
        "text": "",
        "source": "FMP",
    })
    transcript_text = transcript.get("text", "")

    eps_yoy = ""
    revenue_yoy = ""
    if not income.empty and income.shape[1] >= 5:
        eps = _income_series(income, "Diluted EPS", "DilutedEPS", "Basic EPS", "BasicEPS")
        revenue = _income_series(income, "Total Revenue", "TotalRevenue")
        if eps is not None:
            eps_yoy = _growth_text(eps.iloc[0], eps.iloc[4])
        if revenue is not None:
            revenue_yoy = _growth_text(revenue.iloc[0], revenue.iloc[4])

    eps_summary = (
        f"Reported EPS {eps_actual.get(latest_label, '-')}; "
        f"q/q growth {eps_qoq.get(latest_label, '-')}; "
        f"y/y growth {eps_yoy or '-'}. "
        f"Source: Yahoo financials/earnings calendar."
    )
    revenue_summary = (
        f"Revenue {revenue_actual.get(latest_label, '-')}; "
        f"q/q growth {revenue_qoq.get(latest_label, '-')}; "
        f"y/y growth {revenue_yoy or '-'}. "
        f"Source: Yahoo financials."
    )

    summaries.append(
        {
            "section": "EPS",
            "quarter": latest_label,
            "summary": eps_summary,
        }
    )
    summaries.append(
        {
            "section": "Revenue",
            "quarter": latest_label,
            "summary": revenue_summary,
        }
    )

    nvidia_analysis = _nvidia_transcript_analysis(
        ticker,
        latest_label,
        transcript_text,
        eps_summary,
        revenue_summary,
    )
    transcript_status = transcript.get("status", "Transcript unavailable.")
    if transcript_text and nvidia_analysis:
        transcript_status = f"{transcript_status} NVIDIA Build summary used."
    elif transcript_text and os.getenv("NVIDIA_API_KEY"):
        transcript_status = f"{transcript_status} NVIDIA Build failed, so rule-based transcript/news extraction was used."
    elif transcript_text:
        transcript_status = f"{transcript_status} Rule-based transcript/news extraction used."

    summaries.append(
        {
            "section": "Earnings Transcript",
            "quarter": latest_label,
            "summary": transcript_status,
        }
    )

    backlog_keywords = ("backlog", "bookings", "order book", "remaining performance", "rpo", "deferred revenue")
    product_keywords = (
        "launch",
        "launched",
        "unveil",
        "unveiled",
        "new product",
        "platform",
        "partnership",
        "approval",
        "fda",
    )
    guidance_keywords = (
        "guidance",
        "outlook",
        "forecast",
        "expects",
        "expect",
        "raises",
        "raised",
        "lowers",
        "lowered",
        "guides",
    )

    if nvidia_analysis:
        transcript_summary = nvidia_analysis.get("Transcript Summary", "")
        key_positives = nvidia_analysis.get("Key Positives", "")
        risks = nvidia_analysis.get("Risks / Concerns", "")
        backlog = nvidia_analysis.get("Backlog", "")
        new_product = nvidia_analysis.get("New Product Highlight", "")
        guidance = nvidia_analysis.get("Forward Guidance", "")
        trading_takeaway = nvidia_analysis.get("Trading Takeaway", "")
    else:
        transcript_summary = _sentence_matches(transcript_text, ("revenue", "growth", "margin", "demand", "quarter"), limit=5)
        key_positives = _sentence_matches(transcript_text, ("record", "strong", "growth", "improved", "demand"), limit=3)
        risks = _sentence_matches(transcript_text, ("risk", "pressure", "decline", "constraint", "uncertain", "weak"), limit=3)
        backlog = _sentence_matches(transcript_text, backlog_keywords) or _news_matches(
            news_rows,
            backlog_keywords,
        )
        new_product = _sentence_matches(transcript_text, product_keywords) or _news_matches(
            news_rows,
            product_keywords,
        )
        guidance = _sentence_matches(transcript_text, guidance_keywords) or _news_matches(
            news_rows,
            guidance_keywords,
        )
        trading_takeaway = ""
    summaries.extend(
        [
            {
                "section": "Transcript Summary",
                "quarter": latest_label,
                "summary": transcript_summary or "No full transcript summary available.",
            },
            {
                "section": "Key Positives",
                "quarter": latest_label,
                "summary": key_positives or "No clear positives found in the transcript.",
            },
            {
                "section": "Risks / Concerns",
                "quarter": latest_label,
                "summary": risks or "No clear risks found in the transcript.",
            },
            {
                "section": "Backlog",
                "quarter": latest_label,
                "summary": backlog or "No backlog item found in the transcript or latest Yahoo news.",
            },
            {
                "section": "New Product Highlight",
                "quarter": latest_label,
                "summary": new_product or "No new product item found in the transcript or latest Yahoo news.",
            },
            {
                "section": "Forward Guidance",
                "quarter": latest_label,
                "summary": guidance or "No forward guidance item found in the transcript or latest Yahoo news.",
            },
            {
                "section": "Trading Takeaway",
                "quarter": latest_label,
                "summary": trading_takeaway or "Review the technical setup before using this transcript summary for a trade watchlist.",
            },
        ]
    )
    return summaries


@lru_cache(maxsize=256)
def get_live_stock_tracker_details(ticker: str) -> dict[str, Any]:
    stock = yf.Ticker(ticker.strip().upper())
    return {
        "news": _live_news(stock),
        "eps_quarter_table": _live_eps_quarter_table(stock),
        "revenue_quarter_table": _live_revenue_quarter_table(stock),
        "eps_annual_table": _live_eps_annual_table(stock),
        "revenue_annual_table": _live_revenue_annual_table(stock),
        "next_earnings": get_next_earnings_label(ticker),
        "quarterly_summary": _live_quarterly_summary(stock),
    }


def _first_metric_growth(rows: list[dict[str, Any]], metric_name: str) -> tuple[float | None, str]:
    for row in rows:
        if not isinstance(row, dict) or row.get("Metric") != metric_name:
            continue
        for key, value in row.items():
            if key in {"Metric", "Quarter Info"}:
                continue
            parsed = _parse_percent_text(value)
            if parsed is not None:
                return parsed, str(value)
    return None, ""


def _fmp_earnings_calendar(report_date: date) -> list[dict[str, Any]]:
    if not _fmp_api_keys():
        return []
    urls = (
        "https://financialmodelingprep.com/stable/earnings-calendar",
        "https://financialmodelingprep.com/api/v3/earning_calendar",
    )
    params = {"from": report_date.isoformat(), "to": report_date.isoformat()}
    for url in urls:
        response = _fmp_get(url, params=params, timeout=30)
        if response is None:
            continue
        if response.status_code >= 400:
            continue
        try:
            payload = response.json()
        except ValueError:
            continue
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
    return []


def _cached_earnings_calendar_rows_for_source(report_date: date, source: str) -> list[dict[str, Any]]:
    ensure_tracking_schema()
    rows = _rows(
        """
        SELECT ticker, company, eps_estimate, revenue_estimate, source
        FROM earnings_calendar_cache
        WHERE report_date = ? AND source = ?
        ORDER BY ticker
        """,
        (report_date.isoformat(), source),
        TRACKING_DB_PATH,
    )
    return [
        {
            "symbol": row.get("ticker"),
            "company": row.get("company") or "",
            "time": "",
            "epsEstimated": row.get("eps_estimate"),
            "epsActual": None,
            "epsSurprise": None,
            "revenueEstimated": row.get("revenue_estimate"),
            "revenueActual": None,
            "source": row.get("source") or source,
        }
        for row in rows
    ]


def _save_earnings_calendar_rows(report_date: date, rows: list[dict[str, Any]], source: str) -> None:
    if not rows:
        return
    ensure_tracking_schema()
    with _connect(TRACKING_DB_PATH) as conn:
        conn.executemany(
            """
            INSERT INTO earnings_calendar_cache (
                report_date, ticker, company, eps_estimate, revenue_estimate, source, fetched_at
            )
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(report_date, ticker, source) DO UPDATE SET
                company = excluded.company,
                eps_estimate = excluded.eps_estimate,
                revenue_estimate = excluded.revenue_estimate,
                fetched_at = excluded.fetched_at
            """,
            [
                (
                    report_date.isoformat(),
                    str(row.get("symbol") or row.get("ticker") or "").upper(),
                    str(row.get("company") or ""),
                    "" if row.get("epsEstimated") in (None, "") else str(row.get("epsEstimated")),
                    "" if row.get("revenueEstimated") in (None, "") else str(row.get("revenueEstimated")),
                    source,
                )
                for row in rows
                if str(row.get("symbol") or row.get("ticker") or "").strip()
            ],
        )
        conn.commit()


def _earnings_calendar_detail_cache_key(report_date: date, include_reported_details: bool) -> tuple[str, int]:
    return report_date.isoformat(), 1 if include_reported_details else 0


def _load_earnings_calendar_detail_cache(
    report_date: date,
    include_reported_details: bool,
    max_age_minutes: int = 720,
) -> list[dict[str, Any]]:
    ensure_tracking_schema()
    report_date_key, details_key = _earnings_calendar_detail_cache_key(report_date, include_reported_details)
    rows = _rows(
        """
        SELECT rows_json, fetched_at
        FROM earnings_calendar_detail_cache
        WHERE report_date = ? AND include_reported_details = ?
        LIMIT 1
        """,
        (report_date_key, details_key),
        TRACKING_DB_PATH,
    )
    if not rows:
        return []
    fetched_at = pd.to_datetime(rows[0].get("fetched_at"), errors="coerce")
    if pd.isna(fetched_at):
        return []
    age = pd.Timestamp.now(tz=None) - fetched_at
    if age.total_seconds() > max_age_minutes * 60:
        return []
    try:
        payload = json.loads(str(rows[0].get("rows_json") or "[]"))
    except json.JSONDecodeError:
        return []
    return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []


def _save_earnings_calendar_detail_cache(
    report_date: date,
    include_reported_details: bool,
    rows: list[dict[str, Any]],
) -> None:
    ensure_tracking_schema()
    report_date_key, details_key = _earnings_calendar_detail_cache_key(report_date, include_reported_details)
    with _connect(TRACKING_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO earnings_calendar_detail_cache (
                report_date, include_reported_details, rows_json, fetched_at
            )
            VALUES (?, ?, ?, datetime('now'))
            ON CONFLICT(report_date, include_reported_details) DO UPDATE SET
                rows_json = excluded.rows_json,
                fetched_at = excluded.fetched_at
            """,
            (report_date_key, details_key, json.dumps(rows, default=str)),
        )
        conn.commit()


def clear_earnings_calendar_detail_cache(report_date: date | None = None) -> None:
    ensure_tracking_schema()
    with _connect(TRACKING_DB_PATH) as conn:
        if report_date is None:
            conn.execute("DELETE FROM earnings_calendar_detail_cache")
        else:
            conn.execute("DELETE FROM earnings_calendar_detail_cache WHERE report_date = ?", (report_date.isoformat(),))
        conn.commit()


def _alphavantage_earnings_calendar(report_date: date) -> list[dict[str, Any]]:
    cached_rows = _cached_earnings_calendar_rows_for_source(report_date, "Alpha Vantage")
    if cached_rows:
        return cached_rows

    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for horizon in ("3month", "6month", "12month"):
        try:
            response = requests.get(
                "https://www.alphavantage.co/query",
                params={"function": "EARNINGS_CALENDAR", "horizon": horizon, "apikey": api_key},
                timeout=30,
            )
        except Exception:
            continue
        if response.status_code >= 400:
            continue
        text = response.text.strip()
        lowered = text[:300].lower()
        if not text or "symbol" not in lowered:
            continue
        try:
            frame = pd.read_csv(StringIO(text))
        except Exception:
            continue
        if frame.empty or "symbol" not in frame.columns or "reportDate" not in frame.columns:
            continue
        frame["reportDate"] = pd.to_datetime(frame["reportDate"], errors="coerce").dt.date
        frame = frame[frame["reportDate"] == report_date]
        for record in frame.to_dict("records"):
            symbol = str(record.get("symbol") or "").strip().upper().replace(".", "-")
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            rows.append(
                {
                    "symbol": symbol,
                    "company": record.get("name") or "",
                    "time": "",
                    "epsEstimated": record.get("estimate"),
                    "epsActual": None,
                    "epsSurprise": None,
                    "revenueEstimated": record.get("revenueEstimate")
                    or record.get("revenueEstimated")
                    or record.get("revenue_estimate"),
                    "revenueActual": None,
                    "source": "Alpha Vantage",
                }
            )
        if rows:
            break
    _save_earnings_calendar_rows(report_date, rows, "Alpha Vantage")
    return rows


def _yahoo_earnings_calendar(report_date: date) -> list[dict[str, Any]]:
    try:
        calendar = yf.Calendars(
            start=report_date.isoformat(),
            end=(report_date + timedelta(days=1)).isoformat(),
        )
        frame = calendar.get_earnings_calendar(
            market_cap=MIN_MARKET_CAP,
            filter_most_active=False,
            start=report_date.isoformat(),
            end=(report_date + timedelta(days=1)).isoformat(),
            limit=100,
            offset=0,
            force=True,
        )
    except Exception:
        return []
    if frame.empty:
        return []
    if "Symbol" not in frame.columns:
        frame = frame.reset_index()

    rows: list[dict[str, Any]] = []
    for record in frame.to_dict("records"):
        event_date = pd.to_datetime(record.get("Event Start Date"), errors="coerce")
        if pd.isna(event_date) or event_date.date() != report_date:
            continue
        rows.append(
            {
                "symbol": record.get("Symbol"),
                "company": record.get("Company") or record.get("Company Name"),
                "time": record.get("Timing"),
                "epsEstimated": record.get("EPS Estimate"),
                "epsActual": record.get("Reported EPS"),
                "epsSurprise": record.get("Surprise(%)") or record.get("Surprise (%)"),
                "revenueEstimated": None,
                "revenueActual": None,
                "source": "Yahoo Finance",
            }
        )
    return rows


def _calendar_symbol(item: dict[str, Any]) -> str:
    return str(item.get("symbol") or item.get("ticker") or "").strip().upper().replace(".", "-")


def _calendar_value(item: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return value
    return None


def _merge_earnings_calendar_items(*item_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    by_symbol: dict[str, dict[str, Any]] = {}
    for group in item_groups:
        for item in group:
            if not isinstance(item, dict):
                continue
            ticker = _calendar_symbol(item)
            if not ticker:
                continue
            existing = by_symbol.get(ticker)
            if existing is None:
                existing = dict(item)
                existing["symbol"] = ticker
                merged.append(existing)
                by_symbol[ticker] = existing
                continue

            existing_source = str(existing.get("source") or "").strip()
            item_source = str(item.get("source") or "").strip()
            if item_source and item_source not in existing_source.split(" + "):
                existing["source"] = f"{existing_source} + {item_source}" if existing_source else item_source
            for key, value in item.items():
                if value in (None, "") or key == "source":
                    continue
                if existing.get(key) in (None, ""):
                    existing[key] = value
    return merged


def _beat_text(actual: Any, estimate: Any, money: bool = False) -> str:
    actual_number = _safe_float(actual)
    estimate_number = _safe_float(estimate)
    if actual_number is None or estimate_number is None:
        return ""
    difference = actual_number - estimate_number
    pct = ""
    if estimate_number != 0:
        pct = f" ({difference / abs(estimate_number):.1%})"
    if money:
        return f"{_format_compact_number(difference)}{pct}"
    return f"{difference:.2f}{pct}"


@lru_cache(maxsize=1024)
def _sec_cik_for_ticker(ticker: str) -> tuple[str, str]:
    normalized = ticker.strip().upper().replace(".", "-")
    if not normalized:
        return "", ""
    try:
        local_rows = _rows(
            """
            SELECT cik, company_name
            FROM sec_companies
            WHERE ticker = ?
            LIMIT 1
            """,
            (normalized,),
            TRACKING_DB_PATH,
        )
        if local_rows:
            return str(local_rows[0].get("cik") or "").zfill(10), str(local_rows[0].get("company_name") or "")
        stock_rows = _rows(
            """
            SELECT cik, company_name
            FROM stocks
            WHERE ticker = ?
            LIMIT 1
            """,
            (normalized,),
            TRACKING_DB_PATH,
        )
        if stock_rows and stock_rows[0].get("cik"):
            return str(stock_rows[0].get("cik") or "").zfill(10), str(stock_rows[0].get("company_name") or "")
    except Exception:
        pass
    try:
        response = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": _sec_user_agent(), "Accept-Encoding": "gzip, deflate"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return "", ""
    for item in payload.values() if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("ticker") or "").upper().replace(".", "-")
        if symbol == normalized:
            return str(item.get("cik_str") or "").zfill(10), str(item.get("title") or "")
    return "", ""


def _motley_fool_takeaways_preview_for_ticker(ticker: str, limit: int = 3) -> str:
    links = _motley_fool_transcript_links_for_ticker(ticker)
    for url in links[:2]:
        try:
            html = _fetch_transcript_url_html(url)
        except Exception:
            continue
        rows = _motley_fool_takeaway_rows_from_html(html)
        if not rows:
            continue
        snippets = [
            f"{row.get('section')}: {row.get('summary')}"
            for row in rows[:limit]
            if row.get("section") and row.get("summary")
        ]
        if snippets:
            return " | ".join(snippets)
    return ""


@lru_cache(maxsize=512)
def _calendar_summary_preview_for_ticker(ticker: str, limit: int = 3) -> str:
    motley_summary = _motley_fool_takeaways_preview_for_ticker(ticker, limit=limit)
    if motley_summary:
        return motley_summary

    for article in _yahoo_article_links_for_ticker(ticker, limit=3):
        try:
            html = _fetch_transcript_url_html(str(article.get("url") or ""))
        except Exception:
            continue
        text = _article_text_from_html(html)
        summary = _sentence_matches(
            text,
            (
                "revenue",
                "earnings",
                "eps",
                "growth",
                "guidance",
                "outlook",
                "margin",
                "demand",
                "backlog",
                "product",
            ),
            limit=limit,
        )
        if summary:
            return summary
    return ""


def _looks_like_earnings_release_8k(text: str) -> bool:
    lowered = str(text or "").lower()
    return (
        "item 2.02" in lowered
        or "results of operations" in lowered
        or "financial results" in lowered
        or "earnings release" in lowered
        or ("revenue" in lowered and "earnings per share" in lowered)
        or "press release" in lowered and any(word in lowered for word in ("revenue", "eps", "earnings"))
    )


def _clean_earnings_release_summary_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return ""
    start_markers = (
        "Revenue",
        "Net sales",
        "Net revenues",
        "Total net revenues",
        "Financial Results",
        "Quarterly Financial Highlights",
        "Results for",
        "Highlights",
    )
    starts = [cleaned.lower().find(marker.lower()) for marker in start_markers if cleaned.lower().find(marker.lower()) >= 0]
    if starts:
        cleaned = cleaned[min(starts) :]
    boilerplate_markers = (
        "check the appropriate box",
        "securities registered pursuant",
        "written communications pursuant",
        "pre-commencement communications",
        "pursuant to the requirements",
    )
    lowered = cleaned[:800].lower()
    if any(marker in lowered for marker in boilerplate_markers):
        return ""
    return cleaned


def _sec_filing_index_url(cik: str, accession: str) -> str:
    normalized_cik = str(cik or "").lstrip("0")
    normalized_accession = str(accession or "").replace("-", "")
    if not normalized_cik or not normalized_accession:
        return ""
    return f"https://www.sec.gov/Archives/edgar/data/{normalized_cik}/{normalized_accession}/{accession}-index.htm"


def _sec_archive_base_url(index_url: str) -> str:
    cleaned = str(index_url or "").strip()
    if not cleaned:
        return ""
    if cleaned.endswith("/"):
        return cleaned
    return cleaned.rsplit("/", 1)[0] + "/"


def _sec_index_json_url(index_url: str) -> str:
    base_url = _sec_archive_base_url(index_url)
    return urljoin(base_url, "index.json") if base_url else ""


def _sec_earnings_exhibit_from_primary_8k(index_url: str, primary_document: str = "") -> dict[str, str]:
    try:
        primary_url = urljoin(_sec_archive_base_url(index_url), primary_document) if primary_document else _sec_primary_document_url(index_url)
        if not primary_url:
            return {}
        response = requests.get(
            primary_url,
            headers={"User-Agent": _sec_user_agent(), "Accept-Encoding": "gzip, deflate"},
            timeout=30,
        )
        response.raise_for_status()
        html = response.text
    except Exception:
        return {}

    table_start = html.lower().find("exhibit no")
    search_html = html[table_start : table_start + 8000] if table_start >= 0 else html
    links_by_href: dict[str, list[str]] = {}
    for match in re.finditer(r"<a\b[^>]*href=[\"'](?P<href>[^\"']+)[\"'][^>]*>(?P<text>.*?)</a>", search_html, flags=re.I | re.S):
        href = match.group("href")
        links_by_href.setdefault(href, []).append(_clean_html_fragment(match.group("text")))
    for href, pieces in links_by_href.items():
        description = re.sub(r"\s+", " ", " ".join(piece for piece in pieces if piece).strip())
        lower_href = href.lower()
        if "99" in lower_href:
            return {
                "url": urljoin(primary_url, href),
                "description": description,
                "source": "8-K exhibit table",
            }

    for row_html in re.findall(r"<tr\b[^>]*>(.*?)</tr>", html, flags=re.I | re.S):
        row_text = _clean_html_fragment(row_html)
        lowered = row_text.lower()
        if not re.search(r"\b99\.?1\b", lowered):
            continue
        link_text = " ".join(
            _clean_html_fragment(match.group(1))
            for match in re.finditer(r"<a\b[^>]*href=[\"'][^\"']+[\"'][^>]*>(.*?)</a>", row_html, flags=re.I | re.S)
        )
        description = re.sub(r"\s+", " ", link_text or row_text).strip()
        href_match = re.search(r'href=["\']([^"\']+)["\']', row_html, flags=re.I)
        if not href_match:
            continue
        return {
            "url": urljoin(primary_url, href_match.group(1)),
            "description": description,
            "source": "8-K exhibit table",
        }
    return {}


def _sec_earnings_exhibit_document(index_url: str, primary_document: str = "") -> dict[str, str]:
    exhibit = _sec_earnings_exhibit_from_primary_8k(index_url, primary_document=primary_document)
    if exhibit:
        return exhibit

    index_json_url = _sec_index_json_url(index_url)
    if not index_json_url:
        return {}
    try:
        response = requests.get(
            index_json_url,
            headers={"User-Agent": _sec_user_agent(), "Accept-Encoding": "gzip, deflate"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return {}
    items = payload.get("directory", {}).get("item", []) if isinstance(payload, dict) else []
    candidates: list[tuple[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        lower = name.lower()
        if not (lower.endswith(".htm") or lower.endswith(".html")):
            continue
        if "index" in lower:
            continue
        if re.search(r"(?:ex(?:hibit)?[-_ ]?99\.?1|ex991|press[-_ ]?release|earnings)", lower):
            candidates.append((name, str(item.get("description") or name)))
    if not candidates:
        return {}
    candidates.sort(key=lambda item: (0 if "press" in item[0].lower() or "ex99" in item[0].lower() or "ex991" in item[0].lower() else 1, item[0]))
    return {
        "url": urljoin(_sec_archive_base_url(index_url), candidates[0][0]),
        "description": candidates[0][1],
        "source": "SEC filing directory",
    }


def _sec_earnings_exhibit_document_url(index_url: str) -> str:
    return _sec_earnings_exhibit_document(index_url).get("url", "")


def _sec_filing_preferred_earnings_text(index_url: str, max_chars: int = 80000, primary_document: str = "") -> tuple[str, str, str]:
    exhibit = _sec_earnings_exhibit_document(index_url, primary_document=primary_document)
    exhibit_url = exhibit.get("url", "")
    if exhibit_url:
        try:
            response = requests.get(
                exhibit_url,
                headers={"User-Agent": _sec_user_agent(), "Accept-Encoding": "gzip, deflate"},
                timeout=30,
            )
            response.raise_for_status()
            return _article_text_from_html(response.text)[:max_chars], exhibit_url, exhibit.get("description", "")
        except Exception:
            pass
    text, document_url = _sec_filing_document_text(index_url, max_chars=max_chars)
    return text, document_url, ""


@lru_cache(maxsize=128)
def _nvidia_earnings_release_summary_from_url(
    ticker: str,
    document_url: str,
    eps_summary: str,
    revenue_summary: str,
) -> str:
    if not document_url or not os.getenv("NVIDIA_API_KEY"):
        return ""
    try:
        response = requests.get(
            document_url,
            headers={"User-Agent": _sec_user_agent(), "Accept-Encoding": "gzip, deflate"},
            timeout=30,
        )
        response.raise_for_status()
        release_text = _article_text_from_html(response.text)
    except Exception:
        return ""
    analysis = _nvidia_transcript_analysis(
        ticker,
        "Latest earnings release",
        release_text,
        eps_summary,
        revenue_summary,
    )
    if not analysis:
        return ""
    parts = [
        ("Summary", analysis.get("Transcript Summary", "")),
        ("Positives", analysis.get("Key Positives", "")),
        ("Risks", analysis.get("Risks / Concerns", "")),
        ("Guidance", analysis.get("Forward Guidance", "")),
        ("Trading Takeaway", analysis.get("Trading Takeaway", "")),
    ]
    return " | ".join(f"{label}: {value}" for label, value in parts if value)


@lru_cache(maxsize=512)
def _sec_submission_8k_entries_for_ticker(ticker: str, report_date_iso: str) -> list[dict[str, str]]:
    report_day = pd.to_datetime(report_date_iso, errors="coerce")
    if pd.isna(report_day):
        return []
    cik, company_name = _sec_cik_for_ticker(ticker)
    normalized_cik = cik.lstrip("0")
    if not normalized_cik:
        return []
    start_day = report_day.date() - timedelta(days=1)
    end_day = report_day.date() + timedelta(days=3)
    try:
        response = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers={"User-Agent": _sec_user_agent(), "Accept-Encoding": "gzip, deflate"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    recent = payload.get("filings", {}).get("recent", {}) if isinstance(payload, dict) else {}
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    primary_documents = recent.get("primaryDocument", [])
    entries: list[dict[str, str]] = []
    for form, accession, filed_at, report_at, primary_document in zip(
        forms,
        accessions,
        filing_dates,
        report_dates,
        primary_documents,
    ):
        form_type = str(form or "").upper()
        if form_type not in {"8-K", "8-K/A"}:
            continue
        filed = pd.to_datetime(filed_at, errors="coerce")
        if pd.isna(filed):
            continue
        if not (start_day <= filed.date() <= end_day):
            continue
        index_url = _sec_filing_index_url(normalized_cik, str(accession))
        entries.append(
            {
                "title": f"{form_type} - {company_name or ticker.upper()}",
                "summary": f"Filed: {filed_at}; Period: {report_at}; Primary document: {primary_document}",
                "updated": str(filed_at),
                "filing_url": index_url,
                "form_type": form_type,
                "cik": normalized_cik,
                "accession": str(accession),
                "filed": str(filed_at),
                "primary_document": str(primary_document or ""),
            }
        )
    return entries


def _sec_entry_from_yahoo_filing_url(ticker: str, url: str, filed_at: str = "") -> dict[str, str]:
    cleaned_url = str(url or "").strip()
    if not cleaned_url:
        return {}
    path = urlparse(cleaned_url).path
    match = re.search(r"/sec-filing/[^/]+/(?P<accession>[0-9-]+)_(?P<cik>\d+)/?", path, flags=re.I)
    if match:
        accession = match.group("accession")
        cik = match.group("cik")
    else:
        accession_match = re.search(r"([0-9]{10}-[0-9]{2}-[0-9]{6})", cleaned_url)
        cik_match = re.search(r"/data/(\d+)/", cleaned_url)
        if not accession_match or not cik_match:
            return {}
        accession = accession_match.group(1)
        cik = cik_match.group(1)
    _, company_name = _sec_cik_for_ticker(ticker)
    return {
        "title": f"8-K - {company_name or ticker.upper()}",
        "summary": f"Yahoo Finance SEC filing fallback: {cleaned_url}",
        "updated": filed_at,
        "filing_url": _sec_filing_index_url(cik, accession),
        "form_type": "8-K",
        "cik": str(cik).lstrip("0"),
        "accession": accession,
        "filed": filed_at,
        "primary_document": "",
        "source": "Yahoo Finance SEC filing fallback",
    }


@lru_cache(maxsize=512)
def _yahoo_sec_8k_entries_for_ticker(ticker: str, report_date_iso: str) -> list[dict[str, str]]:
    report_day = pd.to_datetime(report_date_iso, errors="coerce")
    if pd.isna(report_day):
        return []
    start_day = report_day.date() - timedelta(days=1)
    end_day = report_day.date() + timedelta(days=3)
    try:
        filings = yf.Ticker(ticker.strip().upper()).get_sec_filings()
    except Exception:
        return []

    records: list[dict[str, Any]]
    if isinstance(filings, pd.DataFrame):
        records = filings.reset_index().to_dict("records")
    elif isinstance(filings, list):
        records = [item for item in filings if isinstance(item, dict)]
    else:
        return []

    entries: list[dict[str, str]] = []
    for record in records:
        form_type = str(record.get("type") or record.get("form") or record.get("Form") or "").upper()
        if form_type and form_type not in {"8-K", "8-K/A"}:
            continue
        filed_value = (
            record.get("date")
            or record.get("filingDate")
            or record.get("filed")
            or record.get("Date")
            or record.get("epochDate")
            or ""
        )
        if isinstance(filed_value, (int, float)):
            filed = pd.to_datetime(filed_value, unit="s", errors="coerce")
        else:
            filed = pd.to_datetime(filed_value, errors="coerce")
        if not pd.isna(filed) and not (start_day <= filed.date() <= end_day):
            continue
        url = (
            record.get("link")
            or record.get("url")
            or record.get("edgarUrl")
            or record.get("filingUrl")
            or record.get("reportUrl")
            or ""
        )
        entry = _sec_entry_from_yahoo_filing_url(ticker, str(url), "" if pd.isna(filed) else filed.date().isoformat())
        if entry:
            entries.append(entry)
    return entries


def _previous_market_day(day: date) -> date:
    previous = day - timedelta(days=1)
    while previous.weekday() >= 5:
        previous -= timedelta(days=1)
    return previous


def _filing_discovery_entries_for_ticker(ticker: str, report_date_iso: str) -> list[dict[str, str]]:
    entries = _sec_submission_8k_entries_for_ticker(ticker, report_date_iso)
    if entries:
        return entries

    report_day = pd.to_datetime(report_date_iso, errors="coerce")
    if not pd.isna(report_day):
        cik, company_name = _sec_cik_for_ticker(ticker)
        normalized_cik = cik.lstrip("0")
        start_day = report_day.date() - timedelta(days=1)
        end_day = report_day.date() + timedelta(days=3)
        try:
            feed_entries = _sec_feed_entries(form_type="8-K", count=500)
        except Exception:
            feed_entries = []
        for entry in feed_entries:
            if str(entry.get("cik") or "").lstrip("0") != normalized_cik:
                continue
            filed = pd.to_datetime(entry.get("filed"), errors="coerce")
            if pd.isna(filed) or not (start_day <= filed.date() <= end_day):
                continue
            entry = dict(entry)
            entry.setdefault("primary_document", "")
            entry["title"] = entry.get("title") or f"8-K - {company_name or ticker.upper()}"
            entries.append(entry)
        if entries:
            return entries

    if pd.isna(report_day) or report_day.date() != _previous_market_day(date.today()):
        return []
    return _yahoo_sec_8k_entries_for_ticker(ticker, report_date_iso)


@lru_cache(maxsize=512)
def _earnings_release_8k_for_ticker(ticker: str, report_date_iso: str) -> dict[str, Any]:
    for entry in _filing_discovery_entries_for_ticker(ticker, report_date_iso):
        feed_text = f"{entry.get('title', '')}\n{entry.get('summary', '')}"
        try:
            filing_text, document_url, exhibit_description = _sec_filing_preferred_earnings_text(
                str(entry.get("filing_url") or ""),
                max_chars=80000,
                primary_document=str(entry.get("primary_document") or ""),
            )
        except Exception:
            filing_text, document_url, exhibit_description = "", "", ""
        combined_text = f"{feed_text}\n\n{filing_text}".strip()
        preferred_exhibit = bool(document_url) and document_url != entry.get("filing_url")
        if not preferred_exhibit and not _looks_like_earnings_release_8k(combined_text):
            continue

        is_earnings_release = _looks_like_earnings_release_8k(combined_text)
        eps_actual = _extract_sec_press_release_eps_from_text(filing_text or combined_text) if is_earnings_release else None
        revenue_actual = _extract_sec_press_release_revenue_from_text(filing_text or combined_text) if is_earnings_release else None
        eps_summary = _format_plain_number(eps_actual)
        revenue_summary = _format_compact_number(revenue_actual)
        nvidia_summary = _nvidia_earnings_release_summary_from_url(
            ticker,
            document_url,
            eps_summary,
            revenue_summary,
        )
        release_summary_text = _clean_earnings_release_summary_text(filing_text)
        summary = nvidia_summary or _sentence_matches(
            release_summary_text,
            (
                "revenue",
                "earnings",
                "eps",
                "guidance",
                "outlook",
                "margin",
                "demand",
                "backlog",
                "product",
                "growth",
            ),
            limit=4,
        )
        return {
            "epsActual": eps_actual,
            "revenueActual": revenue_actual,
            "summary": summary,
            "source": "SEC 8-K / earnings release",
            "filing_url": entry.get("filing_url") or "",
            "document_url": document_url,
            "exhibit_description": exhibit_description,
            "primary_document": entry.get("primary_document") or "",
        }
    return {}


def _remove_unverified_metric_snippets(summary: str, metrics: tuple[str, ...]) -> str:
    if not summary:
        return ""
    parts = [part.strip() for part in re.split(r"\s+\|\s+|(?<=[.!?])\s+", summary) if part.strip()]
    kept: list[str] = []
    for part in parts:
        lowered = part.lower()
        has_metric = any(metric in lowered for metric in metrics)
        has_number = bool(re.search(r"\$|\b\d+(?:\.\d+)?\s*(?:billion|million|bn|mm|%)\b", lowered))
        if has_metric and has_number:
            continue
        kept.append(part)
    return " | ".join(kept[:3])


def _calendar_growth_tables_from_row(row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    report_date = pd.to_datetime(row.get("date"), errors="coerce")
    period_label = _quarter_label(report_date) if not pd.isna(report_date) else str(row.get("date") or "Latest Quarter")
    eps_actual = row.get("eps actual") or row.get("epsActual")
    eps_estimate = row.get("eps estimate") or row.get("epsEstimated")
    eps_beat = row.get("eps beat") or ""
    revenue_actual = row.get("revenue actual") or row.get("revenueActual")
    revenue_estimate = row.get("revenue estimate") or row.get("revenueEstimated")
    revenue_beat = row.get("revenue beat") or ""

    eps_rows: list[dict[str, Any]] = []
    if eps_estimate not in (None, ""):
        eps_rows.append({"Metric": "Expected EPS", period_label: eps_estimate})
    if eps_actual not in (None, ""):
        eps_rows.append({"Metric": "Reported EPS", period_label: eps_actual})
    if eps_beat not in (None, ""):
        eps_rows.append({"Metric": "Surprise", period_label: eps_beat})

    revenue_rows: list[dict[str, Any]] = []
    if revenue_estimate not in (None, ""):
        revenue_rows.append({"Metric": "Expected Revenue", period_label: revenue_estimate})
    if revenue_actual not in (None, ""):
        revenue_rows.append({"Metric": "Actual Revenue", period_label: revenue_actual})
    if revenue_beat not in (None, ""):
        revenue_rows.append({"Metric": "Revenue Beat", period_label: revenue_beat})
    return eps_rows, revenue_rows


def save_yesterday_calendar_metrics_for_ticker(
    ticker: str,
    report_date: date | None = None,
    db_path: Path = TRACKING_DB_PATH,
) -> dict[str, Any]:
    normalized = ticker.strip().upper()
    target_date = report_date or _previous_market_day(date.today())
    rows = earnings_calendar_rows_for_tracker(target_date, include_reported_details=True)
    calendar_row = next((row for row in rows if str(row.get("ticker") or "").upper() == normalized), None)
    if not calendar_row:
        return {
            "ticker": normalized,
            "status": f"No yesterday earnings calendar row found for {normalized}.",
            "eps_rows": 0,
            "revenue_rows": 0,
        }
    eps_rows, revenue_rows = _calendar_growth_tables_from_row(calendar_row)
    if not eps_rows and not revenue_rows:
        return {
            "ticker": normalized,
            "status": "Yesterday earnings calendar row has no EPS/revenue values to save.",
            "eps_rows": 0,
            "revenue_rows": 0,
        }
    eps_rows, revenue_rows = _merge_calendar_growth_with_history(normalized, eps_rows, revenue_rows, db_path)
    saved = save_growth_tables(normalized, eps_rows, revenue_rows, db_path=db_path)
    return {
        "ticker": normalized,
        "status": "Saved EPS/revenue from yesterday earnings calendar and recalculated growth.",
        "eps_rows": saved.get("eps_rows", 0),
        "revenue_rows": saved.get("revenue_rows", 0),
    }


def earnings_calendar_rows_for_tracker(report_date: date, include_reported_details: bool = False) -> list[dict[str, Any]]:
    if include_reported_details:
        cached_rows = _load_earnings_calendar_detail_cache(report_date, include_reported_details=True)
        if cached_rows:
            return cached_rows

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    eligible_symbols = set(load_eligible_ticker_symbols())
    calendar_items = _alphavantage_earnings_calendar(report_date)
    if not calendar_items:
        calendar_items = _merge_earnings_calendar_items(
            _fmp_earnings_calendar(report_date),
            _yahoo_earnings_calendar(report_date),
        )
    for item in calendar_items:
        ticker = _calendar_symbol(item)
        if not ticker:
            continue
        if eligible_symbols and ticker not in eligible_symbols:
            continue
        company = str(_calendar_value(item, "company", "companyName", "name") or "")
        time_label = str(_calendar_value(item, "time", "timeOfDay", "when") or "")
        row_key = (ticker, report_date.isoformat(), time_label.lower())
        if row_key in seen:
            continue
        seen.add(row_key)
        eps_actual = _calendar_value(item, "epsActual", "eps", "actualEps", "epsReported")
        eps_estimate = _calendar_value(item, "epsEstimated", "epsEstimate", "estimatedEps", "epsConsensus")
        revenue_actual = _calendar_value(item, "revenueActual", "revenue", "actualRevenue", "revenueReported")
        revenue_estimate = _calendar_value(item, "revenueEstimated", "revenueEstimate", "estimatedRevenue", "revenueConsensus")
        actual_source = str(_calendar_value(item, "source") or "FMP")
        earnings_release: dict[str, Any] = {}
        if include_reported_details and (eps_actual in (None, "") or revenue_actual in (None, "")):
            earnings_release = _earnings_release_8k_for_ticker(ticker, report_date.isoformat())
            if eps_actual in (None, "") and earnings_release.get("epsActual") not in (None, ""):
                eps_actual = earnings_release["epsActual"]
                actual_source = str(earnings_release.get("source") or actual_source)
            if revenue_actual in (None, "") and earnings_release.get("revenueActual") not in (None, ""):
                revenue_actual = earnings_release["revenueActual"]
                actual_source = str(earnings_release.get("source") or actual_source)
        out = {
            "ticker": ticker,
            "company": company,
            "date": report_date.isoformat(),
            "time": time_label,
            "source": str(_calendar_value(item, "source") or "FMP"),
        }
        if include_reported_details:
            quarterly_summary = str(earnings_release.get("summary") or "").strip()
            exhibit_url = str(earnings_release.get("document_url") or earnings_release.get("filing_url") or "").strip()
            if exhibit_url:
                quarterly_summary = f"99.1 Exhibit: {exhibit_url}" + (f" | {quarterly_summary}" if quarterly_summary else "")
            if not quarterly_summary:
                quarterly_summary = _calendar_summary_preview_for_ticker(ticker)
            unverified_metrics: list[str] = []
            if revenue_actual in (None, ""):
                unverified_metrics.append("revenue")
            if eps_actual in (None, ""):
                unverified_metrics.extend(["eps", "earnings per share"])
            if unverified_metrics:
                quarterly_summary = _remove_unverified_metric_snippets(quarterly_summary, tuple(unverified_metrics))
            out.update(
                {
                    "eps actual": _format_plain_number(eps_actual),
                    "eps estimate": _format_plain_number(eps_estimate),
                    "eps beat": _beat_text(eps_actual, eps_estimate),
                    "revenue actual": _format_compact_number(revenue_actual),
                    "revenue estimate": _format_compact_number(revenue_estimate),
                    "revenue beat": _beat_text(revenue_actual, revenue_estimate, money=True),
                    "actual source": actual_source,
                    "earnings filing": exhibit_url,
                    "exhibit description": earnings_release.get("exhibit_description") or "",
                    "quarterly summary": quarterly_summary,
                }
            )
        rows.append(out)
    if include_reported_details:
        _save_earnings_calendar_detail_cache(report_date, include_reported_details=True, rows=rows)
    return rows


@lru_cache(maxsize=512)
def get_next_earnings_label(ticker: str) -> str:
    normalized = ticker.strip().upper()
    if not normalized:
        return ""
    stock = yf.Ticker(normalized)
    today = pd.Timestamp(date.today())

    try:
        calendar = stock.get_calendar()
    except Exception:
        calendar = None
    candidates: list[pd.Timestamp] = []
    if isinstance(calendar, dict):
        for key, value in calendar.items():
            if "earnings" not in str(key).lower():
                continue
            values = value if isinstance(value, (list, tuple, pd.Series)) else [value]
            for item in values:
                ts = pd.to_datetime(item, errors="coerce")
                if not pd.isna(ts):
                    candidates.append(ts.tz_localize(None) if getattr(ts, "tzinfo", None) else ts)

    try:
        earnings_dates = stock.get_earnings_dates(limit=12)
    except Exception:
        earnings_dates = pd.DataFrame()
    if isinstance(earnings_dates, pd.DataFrame) and not earnings_dates.empty:
        for idx in earnings_dates.index:
            ts = pd.to_datetime(idx, errors="coerce")
            if not pd.isna(ts):
                candidates.append(ts.tz_localize(None) if getattr(ts, "tzinfo", None) else ts)

    future_dates = sorted({candidate.normalize() for candidate in candidates if candidate.normalize() > today})
    if not future_dates:
        return ""
    return future_dates[0].strftime("%Y-%m-%d")


def refresh_next_earnings_dates_for_tracked_stocks(db_path: Path = TRACKING_DB_PATH) -> list[dict[str, Any]]:
    ensure_tracking_schema(db_path)
    get_next_earnings_label.cache_clear()
    stocks = _rows("SELECT ticker FROM stocks ORDER BY ticker", db_path=db_path)
    results: list[dict[str, Any]] = []
    with _connect(db_path) as conn:
        for row in stocks:
            ticker = str(row.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            try:
                next_date = get_next_earnings_label(ticker)
                conn.execute(
                    "UPDATE stocks SET next_earnings_date = ?, updated_at = datetime('now') WHERE ticker = ?",
                    (next_date, ticker),
                )
                results.append({"ticker": ticker, "next_earnings_date": next_date, "status": "updated"})
            except Exception as exc:
                results.append({"ticker": ticker, "next_earnings_date": "", "status": f"failed: {exc}"})
        conn.commit()
    return results


def refresh_due_earnings_data_for_tracked_stocks(db_path: Path = TRACKING_DB_PATH) -> list[dict[str, Any]]:
    ensure_tracking_schema(db_path)
    today = date.today()
    rows = _rows(
        """
        SELECT ticker, next_earnings_date
        FROM stocks
        WHERE next_earnings_date IS NOT NULL
          AND next_earnings_date != ''
          AND next_earnings_date <= ?
        ORDER BY ticker
        """,
        (today.isoformat(),),
        db_path,
    )
    results: list[dict[str, Any]] = []
    for row in rows:
        ticker = str(row.get("ticker") or "").upper()
        earnings_date = str(row.get("next_earnings_date") or "")
        if not ticker:
            continue
        try:
            get_live_stock_tracker_details.cache_clear()
            fetch_and_save_motley_fool_takeaways_for_ticker(ticker, db_path)
            save_live_growth_tables_for_ticker(ticker, db_path)
            next_date = get_next_earnings_label(ticker)
            with _connect(db_path) as conn:
                conn.execute(
                    "UPDATE stocks SET next_earnings_date = ?, updated_at = datetime('now') WHERE ticker = ?",
                    (next_date, ticker),
                )
                conn.commit()
            results.append(
                {
                    "ticker": ticker,
                    "earnings_date": earnings_date,
                    "next_earnings_date": next_date,
                    "status": "refreshed after earnings date",
                }
            )
        except Exception as exc:
            results.append(
                {
                    "ticker": ticker,
                    "earnings_date": earnings_date,
                    "next_earnings_date": "",
                    "status": f"refresh failed: {exc}",
                }
            )
    return results


def auto_refresh_tracker_panel_if_due(db_path: Path = TRACKING_DB_PATH) -> dict[str, Any]:
    today_key = date.today().isoformat()
    last_run = _metadata_get("auto_tracker_panel_refresh_date", db_path)
    if last_run == today_key:
        return {
            "ran": False,
            "status": "Already refreshed today.",
            "run_date": today_key,
            "updated": _metadata_get("auto_tracker_panel_refresh_updated", db_path),
            "post_earnings_refreshed": _metadata_get("auto_tracker_post_earnings_refreshed", db_path),
        }

    results = refresh_due_earnings_data_for_tracked_stocks(db_path)
    refreshed_count = sum(1 for item in results if str(item.get("status", "")).startswith("refreshed"))
    _metadata_set("auto_tracker_panel_refresh_date", today_key, db_path)
    _metadata_set("auto_tracker_panel_refresh_updated", "0", db_path)
    _metadata_set("auto_tracker_post_earnings_refreshed", str(refreshed_count), db_path)
    return {
        "ran": True,
        "status": "Post-earnings refresh checked.",
        "run_date": today_key,
        "updated": 0,
        "post_earnings_refreshed": refreshed_count,
        "results": results,
    }


def scan_recent_earnings_growth_and_add_to_tracker(
    report_date: date | None = None,
    min_eps_qoq_growth: float = 40.0,
    min_revenue_qoq_growth: float = 30.0,
    db_path: Path = TRACKING_DB_PATH,
) -> list[dict[str, Any]]:
    target_date = report_date or (date.today() - timedelta(days=1))
    events = _fmp_earnings_calendar(target_date)
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in events:
        symbol = _calendar_symbol(event)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        stock = yf.Ticker(symbol)
        try:
            eps_rows = _live_eps_quarter_table(stock)
            revenue_rows = _live_revenue_quarter_table(stock)
        except Exception as exc:
            results.append(
                {
                    "ticker": symbol,
                    "reported_date": target_date.isoformat(),
                    "eps_qoq_growth": "",
                    "revenue_qoq_growth": "",
                    "added": False,
                    "status": f"Growth check failed: {exc}",
                }
            )
            continue

        eps_growth, eps_text = _first_metric_growth(eps_rows, "QoQ EPS Growth")
        revenue_growth, revenue_text = _first_metric_growth(revenue_rows, "Revenue Growth QoQ")
        passes = (
            eps_growth is not None
            and revenue_growth is not None
            and eps_growth > min_eps_qoq_growth
            and revenue_growth > min_revenue_qoq_growth
        )
        status = "Did not pass growth filters."
        added = False
        if passes:
            try:
                add_tracked_stock(symbol, db_path)
                save_growth_tables(symbol, eps_rows, revenue_rows, db_path=db_path)
                summary_result = fetch_and_save_motley_fool_takeaways_for_ticker(symbol, db_path)
                status = f"Added to tracker. {summary_result.get('status', '')}".strip()
                added = True
            except Exception as exc:
                status = f"Passed filters but add failed: {exc}"
        results.append(
            {
                "ticker": symbol,
                "reported_date": target_date.isoformat(),
                "eps_qoq_growth": eps_text,
                "revenue_qoq_growth": revenue_text,
                "added": added,
                "status": status,
            }
        )
    return results


def _previous_weekday(today: date | None = None) -> date:
    current = today or date.today()
    candidate = current - timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


def auto_scan_reported_earnings_if_due(
    min_eps_qoq_growth: float = 40.0,
    min_revenue_qoq_growth: float = 30.0,
    db_path: Path = TRACKING_DB_PATH,
) -> dict[str, Any]:
    today_key = date.today().isoformat()
    last_run = _metadata_get("auto_reported_earnings_scan_run_date", db_path)
    if last_run == today_key:
        return {
            "ran": False,
            "status": "Already ran today.",
            "run_date": today_key,
            "reported_date": _metadata_get("auto_reported_earnings_scan_reported_date", db_path),
            "checked": _metadata_get("auto_reported_earnings_scan_checked", db_path),
            "added": _metadata_get("auto_reported_earnings_scan_added", db_path),
            "results": [],
        }

    reported_date = _previous_weekday()
    results = scan_recent_earnings_growth_and_add_to_tracker(
        report_date=reported_date,
        min_eps_qoq_growth=min_eps_qoq_growth,
        min_revenue_qoq_growth=min_revenue_qoq_growth,
        db_path=db_path,
    )
    added_count = sum(1 for item in results if item.get("added"))
    _metadata_set("auto_reported_earnings_scan_run_date", today_key, db_path)
    _metadata_set("auto_reported_earnings_scan_reported_date", reported_date.isoformat(), db_path)
    _metadata_set("auto_reported_earnings_scan_checked", str(len(results)), db_path)
    _metadata_set("auto_reported_earnings_scan_added", str(added_count), db_path)
    return {
        "ran": True,
        "status": "Auto scan completed.",
        "run_date": today_key,
        "reported_date": reported_date.isoformat(),
        "checked": len(results),
        "added": added_count,
        "results": results,
    }


def analyze_pasted_transcript_with_metrics(ticker: str, transcript_text: str) -> dict[str, Any]:
    normalized = ticker.strip().upper()
    cleaned = str(transcript_text or "").strip()
    if not normalized:
        raise ValueError("Ticker is required.")
    if not cleaned:
        raise ValueError("Transcript text is empty.")

    stock = yf.Ticker(normalized)
    summary = _live_quarterly_summary(
        stock,
        {
            "status": "Pasted transcript text loaded.",
            "text": cleaned,
            "source": "Pasted transcript",
        },
    )
    metrics = extract_transcript_metric_tables(cleaned)
    return {"summary": summary, **metrics}


def analyze_uploaded_transcript(ticker: str, transcript_text: str) -> list[dict[str, Any]]:
    return analyze_pasted_transcript_with_metrics(ticker, transcript_text)["summary"]


def _transcript_text_from_url(transcript_url: str) -> tuple[str, list[dict[str, Any]], bool]:
    is_motley_fool = "fool.com" in transcript_url.lower()
    takeaway_rows: list[dict[str, Any]] = []
    if is_motley_fool:
        html = _fetch_transcript_url_html(transcript_url)
        takeaway_rows = _motley_fool_takeaway_rows_from_html(html)
        transcript_text = _article_text_from_html(html)
        if not takeaway_rows:
            takeaway_rows = _motley_fool_takeaway_rows(transcript_text)
        return transcript_text, takeaway_rows, True
    return _fetch_transcript_url_text(transcript_url), [], False


def analyze_transcript_url_with_metrics(ticker: str, transcript_url: str) -> dict[str, Any]:
    normalized = ticker.strip().upper()
    if not normalized:
        raise ValueError("Ticker is required.")
    transcript_text, takeaway_rows, is_motley_fool = _transcript_text_from_url(transcript_url)
    if takeaway_rows:
        summary = takeaway_rows
    else:
        stock = yf.Ticker(normalized)
        status = "Transcript URL loaded."
        if is_motley_fool:
            status = "Motley Fool URL loaded, but no TAKEAWAYS section was found on this page. Full transcript/article summary fallback used."
        summary = _live_quarterly_summary(
            stock,
            {
                "status": status,
                "text": transcript_text,
                "source": "Transcript URL",
            },
        )
    metrics = extract_transcript_metric_tables(transcript_text)
    return {"summary": summary, **metrics}


def analyze_transcript_url(ticker: str, transcript_url: str) -> list[dict[str, Any]]:
    return analyze_transcript_url_with_metrics(ticker, transcript_url)["summary"]


def save_quarterly_summary_rows(
    ticker: str,
    rows: list[dict[str, Any]],
    source_url: str = "",
    db_path: Path = TRACKING_DB_PATH,
    replace_existing: bool = True,
    replace_weak_existing: bool = False,
) -> int:
    normalized = ticker.strip().upper()
    if not normalized or not rows:
        return 0
    ensure_tracking_schema(db_path)
    stock_rows = _rows("SELECT cik FROM stocks WHERE ticker = ?", (normalized,), db_path)
    if not stock_rows:
        return 0
    cik = stock_rows[0]["cik"]
    quarters = {
        str(row.get("quarter") or "Motley Fool Takeaways").strip()
        for row in rows
        if str(row.get("summary") or "").strip()
    }
    quarters = {quarter for quarter in quarters if quarter}
    if not quarters:
        return 0
    with _connect(db_path) as conn:
        if replace_existing:
            conn.executemany(
                "DELETE FROM quarterly_summary WHERE cik = ? AND quarter = ?",
                [(cik, quarter) for quarter in quarters],
            )
        count = 0
        for row in rows:
            section = str(row.get("section") or row.get("transcript") or "Takeaway").strip()
            summary = str(row.get("summary") or "").strip()
            quarter = str(row.get("quarter") or "Motley Fool Takeaways").strip()
            if not section or not summary:
                continue
            if not replace_existing:
                existing = conn.execute(
                    """
                    SELECT summary
                    FROM quarterly_summary
                    WHERE cik = ? AND quarter = ? AND transcript = ?
                    LIMIT 1
                    """,
                    (cik, quarter or "Motley Fool Takeaways", section),
                ).fetchone()
                if existing:
                    existing_summary = str(existing[0] or "")
                    if replace_weak_existing and _is_weak_summary(existing_summary):
                        conn.execute(
                            """
                            DELETE FROM quarterly_summary
                            WHERE cik = ? AND quarter = ? AND transcript = ?
                            """,
                            (cik, quarter or "Motley Fool Takeaways", section),
                        )
                    else:
                        continue
            conn.execute(
                """
                INSERT INTO quarterly_summary (cik, quarter, transcript, summary)
                VALUES (?, ?, ?, ?)
                """,
                (cik, quarter or "Motley Fool Takeaways", section, summary),
            )
            count += 1
        conn.commit()
    return count


def delete_quarterly_summary_sections(
    ticker: str,
    sections: Sequence[str],
    db_path: Path = TRACKING_DB_PATH,
) -> int:
    normalized = ticker.strip().upper()
    clean_sections = [str(section).strip() for section in sections if str(section).strip()]
    if not normalized or not clean_sections:
        return 0
    ensure_tracking_schema(db_path)
    stock_rows = _rows("SELECT cik FROM stocks WHERE ticker = ?", (normalized,), db_path)
    if not stock_rows:
        return 0
    cik = stock_rows[0]["cik"]
    placeholders = ",".join("?" for _ in clean_sections)
    with _connect(db_path) as conn:
        cursor = conn.execute(
            f"DELETE FROM quarterly_summary WHERE cik = ? AND transcript IN ({placeholders})",
            (cik, *clean_sections),
        )
        conn.commit()
        return cursor.rowcount


def _nvidia_fallback_rows_for_ticker(
    ticker: str,
    status: str,
    transcript_text: str = "",
) -> list[dict[str, Any]]:
    stock = yf.Ticker(ticker.strip().upper())
    override = None
    if transcript_text.strip():
        override = {"status": status, "text": transcript_text, "source": "NVIDIA fallback"}
    return _live_quarterly_summary(stock, override)


def _is_weak_summary(summary: str) -> bool:
    text = re.sub(r"\s+", " ", str(summary or "").strip().lower())
    if not text:
        return True
    weak_phrases = (
        "no full transcript summary available",
        "no clear positives found",
        "no clear risks found",
        "no backlog item found",
        "no new product item found",
        "no forward guidance item found",
        "review the technical setup before using this transcript summary",
        "transcript unavailable",
        "no alpha vantage transcript found",
        "fmp earnings transcript endpoint is restricted",
    )
    return any(phrase in text for phrase in weak_phrases)


def fetch_and_save_motley_fool_takeaways_for_ticker(
    ticker: str,
    db_path: Path = TRACKING_DB_PATH,
) -> dict[str, Any]:
    normalized = ticker.strip().upper()
    links = _motley_fool_transcript_links_for_ticker(normalized)
    if not links:
        for article in _yahoo_article_links_for_ticker(normalized):
            url = article.get("url", "")
            try:
                html = _fetch_transcript_url_html(url)
                article_text = _article_text_from_html(html)
            except Exception:
                continue
            if len(article_text) < 500:
                continue
            rows = _fallback_article_rows_for_ticker(
                normalized,
                url,
                article_text,
                article.get("headline", ""),
            )
            rows_saved = save_quarterly_summary_rows(
                normalized,
                rows,
                url,
                db_path,
                replace_existing=False,
                replace_weak_existing=True,
            )
            return {
                "ticker": normalized,
                "status": "No Motley Fool transcript URL found; saved Yahoo article summary fallback." if rows_saved else "No Motley Fool transcript URL found; Yahoo article summary found but existing summary was preserved.",
                "url": url,
                "rows_saved": rows_saved,
            }
        rows = _nvidia_fallback_rows_for_ticker(
            normalized,
            "No Motley Fool transcript URL found. NVIDIA Build/latest transcript fallback used.",
        )
        rows_saved = save_quarterly_summary_rows(normalized, rows, "", db_path, replace_existing=False)
        return {
            "ticker": normalized,
            "status": "No Motley Fool transcript URL found; added missing fallback rows without replacing existing summary." if rows_saved else "No Motley Fool transcript URL found; existing summary preserved.",
            "url": "",
            "rows_saved": rows_saved,
        }

    fallback_url = ""
    fallback_text = ""
    for url in links:
        try:
            html = _fetch_transcript_url_html(url)
            rows = _motley_fool_takeaway_rows_from_html(html)
        except Exception as exc:
            return {"ticker": normalized, "status": f"Fetch failed: {exc}", "url": url, "rows_saved": 0}
        if not fallback_url:
            fallback_url = url
            fallback_text = _article_text_from_html(html)
        if not rows:
            continue
        rows_saved = save_quarterly_summary_rows(normalized, rows, url, db_path)
        return {
            "ticker": normalized,
            "status": "Saved Motley Fool Takeaways.",
            "url": url,
            "rows_saved": rows_saved,
        }
    if fallback_text:
        yahoo_article_used = False
        for article in _yahoo_article_links_for_ticker(normalized):
            url = article.get("url", "")
            try:
                html = _fetch_transcript_url_html(url)
                article_text = _article_text_from_html(html)
            except Exception:
                continue
            if len(article_text) < 500:
                continue
            fallback_url = url
            fallback_text = article_text
            yahoo_article_used = True
            break
        rows = _nvidia_fallback_rows_for_ticker(
            normalized,
            "Yahoo Finance article fallback used because Motley Fool Takeaways were unavailable."
            if yahoo_article_used
            else "Motley Fool transcript URL found, but no Takeaways section was found. NVIDIA Build summary fallback used.",
            fallback_text,
        )
        rows_saved = save_quarterly_summary_rows(
            normalized,
            rows,
            fallback_url,
            db_path,
            replace_existing=False,
            replace_weak_existing=yahoo_article_used,
        )
        return {
            "ticker": normalized,
            "status": (
                "No Takeaways found; saved Yahoo article summary fallback."
                if yahoo_article_used and rows_saved
                else "No Takeaways found; added missing fallback rows without replacing existing summary."
                if rows_saved
                else "No Takeaways found; existing summary preserved."
            ),
            "url": fallback_url,
            "rows_saved": rows_saved,
        }
    return {
        "ticker": normalized,
        "status": "Transcript URL found, but no Takeaways section was found.",
        "url": links[0],
        "rows_saved": 0,
    }


def fetch_and_save_sec_earnings_release_summary_for_ticker(
    ticker: str,
    report_date: date | None = None,
    db_path: Path = TRACKING_DB_PATH,
) -> dict[str, Any]:
    normalized = ticker.strip().upper()
    if not normalized:
        return {"ticker": normalized, "status": "No ticker provided.", "url": "", "rows_saved": 0}

    candidate_dates = [report_date] if report_date else []
    if not candidate_dates:
        today = date.today()
        candidate_dates = [today - timedelta(days=offset) for offset in range(0, 8)]

    release: dict[str, Any] = {}
    used_date: date | None = None
    for candidate_date in candidate_dates:
        if candidate_date is None:
            continue
        release = _earnings_release_8k_for_ticker(normalized, candidate_date.isoformat())
        if release:
            used_date = candidate_date
            break

    exhibit_url = str(release.get("document_url") or release.get("filing_url") or "").strip()
    if not release or not exhibit_url:
        return {
            "ticker": normalized,
            "status": "No SEC 8-K Exhibit 99.1 earnings release found.",
            "url": "",
            "rows_saved": 0,
        }

    quarter_label = used_date.isoformat() if used_date else "SEC earnings release"
    rows = [
        {
            "section": "99.1 Exhibit Link",
            "quarter": quarter_label,
            "summary": exhibit_url,
        }
    ]
    exhibit_description = str(release.get("exhibit_description") or "").strip()
    if exhibit_description:
        rows.append(
            {
                "section": "99.1 Exhibit Description",
                "quarter": quarter_label,
                "summary": exhibit_description,
            }
        )
    summary = str(release.get("summary") or "").strip()
    if summary:
        rows.append(
            {
                "section": "SEC Earnings Release Summary",
                "quarter": quarter_label,
                "summary": summary,
            }
        )

    delete_quarterly_summary_sections(
        normalized,
        ("99.1 Exhibit Link", "99.1 Exhibit Description", "SEC Earnings Release Summary"),
        db_path,
    )
    rows_saved = save_quarterly_summary_rows(
        normalized,
        rows,
        exhibit_url,
        db_path,
        replace_existing=False,
        replace_weak_existing=True,
    )
    return {
        "ticker": normalized,
        "status": "Saved SEC 8-K Exhibit 99.1 earnings release summary." if rows_saved else "SEC 8-K Exhibit 99.1 found; existing summary rows preserved.",
        "url": exhibit_url,
        "rows_saved": rows_saved,
    }


def fetch_and_save_motley_fool_takeaways_for_tracked_stocks(
    db_path: Path = TRACKING_DB_PATH,
) -> list[dict[str, Any]]:
    stocks = list_tracked_stocks(db_path)
    if stocks.empty:
        return []
    results: list[dict[str, Any]] = []
    for ticker in stocks["ticker"].dropna().astype(str).str.upper().tolist():
        results.append(fetch_and_save_motley_fool_takeaways_for_ticker(ticker, db_path))
    return results


def list_tracked_stocks(db_path: Path = TRACKING_DB_PATH) -> pd.DataFrame:
    rows = _rows(
        """
        SELECT
            s.id,
            s.ticker,
            s.company_name,
            s.market_cap,
            s.description,
            s.fifty_two_week_high,
            s.beta,
            s.next_earnings_date,
            s.summary,
            s.cik,
            COUNT(sn.id) AS news_count,
            MAX(sn.datetime) AS latest_news_at
        FROM stocks s
        LEFT JOIN stock_news sn ON sn.cik = s.cik
        GROUP BY s.id
        ORDER BY s.ticker
        """,
        db_path=db_path,
    )
    return pd.DataFrame(rows)


def get_stock_details(ticker: str, db_path: Path = TRACKING_DB_PATH) -> dict[str, Any] | None:
    normalized = ticker.strip().upper()
    stock_rows = _rows("SELECT * FROM stocks WHERE ticker = ?", (normalized,), db_path)
    if not stock_rows:
        return None

    stock = stock_rows[0]
    cik = stock["cik"]
    stock["news"] = _rows(
        """
        SELECT
            headline,
            datetime,
            COALESCE(source, form_type, '') AS source,
            COALESCE(link, filing_url, '') AS link
        FROM stock_news
        WHERE cik = ?
        ORDER BY datetime DESC
        LIMIT 25
        """,
        (cik,),
        db_path,
    )
    stock["earnings_growth"] = _rows(
        """
        SELECT metric, report_type, period_label, reported
        FROM tracker_earnings_growth
        WHERE cik = ?
        ORDER BY report_type, id
        """,
        (cik,),
        db_path,
    )
    stock["revenue_growth"] = _rows(
        """
        SELECT metric, report_type, period_label, reported
        FROM tracker_revenue_growth
        WHERE cik = ?
        ORDER BY report_type, id
        """,
        (cik,),
        db_path,
    )
    stock["quarterly_summary"] = _rows(
        """
        SELECT quarter, transcript, summary
        FROM quarterly_summary
        WHERE cik = ?
        ORDER BY quarter
        """,
        (cik,),
        db_path,
    )
    stock["eps_quarter_table"] = _growth_records_to_table(stock["earnings_growth"], "quarterly")
    stock["eps_annual_table"] = _growth_records_to_table(stock["earnings_growth"], "annual")
    stock["revenue_quarter_table"] = _growth_records_to_table(stock["revenue_growth"], "quarterly")
    stock["revenue_annual_table"] = _growth_records_to_table(stock["revenue_growth"], "annual")
    stock["next_earnings"] = str(stock.get("next_earnings_date") or "")
    return stock


def refresh_live_data_for_tracked_stock(ticker: str, db_path: Path = TRACKING_DB_PATH) -> dict[str, Any]:
    normalized = ticker.strip().upper()
    if not normalized:
        raise ValueError("Ticker is required.")
    ensure_tracking_schema(db_path)
    existing_rows = _rows("SELECT cik FROM stocks WHERE ticker = ?", (normalized,), db_path)
    if not existing_rows:
        return add_tracked_stock(normalized, db_path)

    get_live_stock_tracker_details.cache_clear()
    get_next_earnings_label.cache_clear()
    stock = yf.Ticker(normalized)
    try:
        info = stock.get_info()
    except Exception:
        info = {}
    if not isinstance(info, dict):
        info = {}

    live = get_live_stock_tracker_details(normalized)
    existing_cik = str(existing_rows[0].get("cik") or f"TICKER-{normalized}")
    company_name = info.get("longName") or info.get("shortName") or normalized
    cik = str(info.get("cik") or info.get("secCIK") or existing_cik).zfill(10)
    market_cap = info.get("marketCap")
    fifty_two_week_high = info.get("fiftyTwoWeekHigh")
    beta = info.get("beta")
    description = info.get("longBusinessSummary")
    next_date = str(live.get("next_earnings") or get_next_earnings_label(normalized) or "")

    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE stocks
            SET
                cik = ?,
                company_name = ?,
                market_cap = ?,
                description = ?,
                fifty_two_week_high = ?,
                beta = ?,
                next_earnings_date = ?,
                updated_at = datetime('now')
            WHERE ticker = ?
            """,
            (
                cik,
                company_name,
                None if market_cap is None else str(market_cap),
                description,
                None if fifty_two_week_high is None else str(fifty_two_week_high),
                beta,
                next_date,
                normalized,
            ),
        )
        conn.execute(
            """
            INSERT INTO sec_companies (ticker, company_name, cik)
            VALUES (?, ?, ?)
            ON CONFLICT(cik) DO UPDATE SET
                ticker = excluded.ticker,
                company_name = excluded.company_name
            """,
            (normalized, company_name, cik),
        )
        conn.commit()

    save_growth_tables(
        normalized,
        live.get("eps_quarter_table", []),
        live.get("revenue_quarter_table", []),
        live.get("eps_annual_table", []),
        live.get("revenue_annual_table", []),
        db_path,
    )
    save_live_news_for_ticker(normalized, cik, live.get("news", []), db_path)
    get_live_stock_tracker_details.cache_clear()
    refreshed = get_stock_details(normalized, db_path) or live
    return refreshed


def delete_tracked_stock(ticker: str, db_path: Path = TRACKING_DB_PATH) -> bool:
    normalized = ticker.strip().upper()
    ensure_tracking_schema(db_path)
    with _connect(db_path) as conn:
        cursor = conn.execute("DELETE FROM stocks WHERE ticker = ?", (normalized,))
        conn.commit()
        return cursor.rowcount > 0


def add_tracked_stock(ticker: str, db_path: Path = TRACKING_DB_PATH) -> dict[str, Any]:
    normalized = ticker.strip().upper()
    if not normalized:
        raise ValueError("Ticker is required.")

    ensure_tracking_schema(db_path)
    stock = yf.Ticker(normalized)
    info = stock.get_info()
    if not isinstance(info, dict) or not info.get("symbol"):
        raise RuntimeError(f"No overview data returned for {normalized}.")

    company_name = info.get("longName") or info.get("shortName") or normalized
    cik = str(info.get("cik") or info.get("secCIK") or f"TICKER-{normalized}").zfill(10)
    market_cap = info.get("marketCap")
    fifty_two_week_high = info.get("fiftyTwoWeekHigh")
    beta = info.get("beta")
    description = info.get("longBusinessSummary")
    next_earnings_date = get_next_earnings_label(normalized)

    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO stocks (
                ticker,
                cik,
                company_name,
                market_cap,
                description,
                fifty_two_week_high,
                beta,
                next_earnings_date
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                cik = excluded.cik,
                company_name = excluded.company_name,
                market_cap = excluded.market_cap,
                description = excluded.description,
                fifty_two_week_high = excluded.fifty_two_week_high,
                beta = excluded.beta,
                next_earnings_date = excluded.next_earnings_date,
                updated_at = datetime('now')
            """,
            (
                normalized,
                cik,
                company_name,
                None if market_cap is None else str(market_cap),
                description,
                None if fifty_two_week_high is None else str(fifty_two_week_high),
                beta,
                next_earnings_date,
            ),
        )
        conn.execute(
            """
            INSERT INTO sec_companies (ticker, company_name, cik)
            VALUES (?, ?, ?)
            ON CONFLICT(cik) DO UPDATE SET
                ticker = excluded.ticker,
                company_name = excluded.company_name
            """,
            (normalized, company_name, cik),
        )
        conn.commit()

    save_live_growth_tables_for_ticker(normalized, db_path)
    details = get_stock_details(normalized, db_path)
    if details is None:
        raise RuntimeError(f"Could not save {normalized}.")
    return details


def list_sec_feed(search: str = "", limit: int = 250, db_path: Path = TRACKING_DB_PATH) -> pd.DataFrame:
    if not tracking_database_exists(db_path):
        return pd.DataFrame()

    params: list[Any] = []
    where = ""
    if search.strip():
        term = f"%{search.strip()}%"
        where = """
        WHERE s.ticker LIKE ?
           OR s.company_name LIKE ?
           OR sn.headline LIKE ?
        """
        params.extend([term, term, term])

    params.append(limit * 3)
    rows = _rows(
        f"""
        SELECT
            sn.datetime,
            COALESCE(s.ticker, '') AS ticker,
            COALESCE(s.company_name, '') AS company_name,
            sn.headline,
            COALESCE(sn.filing_url, '') AS filing_url,
            COALESCE(sn.accession, '') AS accession,
            COALESCE(sn.form_type, '') AS form_type,
            COALESCE(sn.sentiment, '') AS sentiment
        FROM stock_news sn
        LEFT JOIN sec_companies s ON sn.cik = s.cik
        {where}
        ORDER BY
            sn.datetime DESC,
            CASE WHEN sn.headline LIKE '%Investor Summary:%' THEN 0 ELSE 1 END,
            sn.id DESC
        LIMIT ?
        """,
        tuple(params),
        db_path,
    )
    if not rows:
        return pd.DataFrame()
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        accession = str(row.get("accession") or "")
        if accession.startswith("http"):
            accession = re.sub(r".*/([^/]+)-index\.htm$", r"\1", accession)
        key = (
            str(row.get("datetime") or ""),
            str(row.get("ticker") or ""),
            accession,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
        if len(deduped) >= limit:
            break
    return pd.DataFrame(deduped)


def extract_feed_headline(text: str | None) -> str:
    if not text:
        return "No headline"
    match = re.search(r"\*\*Headline:\*\*(.*)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()[:160]


def _normalize_sentiment_label(value: str) -> str:
    lowered = value.lower()
    if any(term in lowered for term in ("very very bullish", "extremely bullish", "strongly bullish", "highly bullish", "very bullish")):
        return "Very Very Bullish"
    if any(term in lowered for term in ("very very bearish", "extremely bearish", "strongly bearish", "highly bearish", "very bearish")):
        return "Very Very Bearish"
    if any(term in lowered for term in ("little bearish", "slightly bearish", "mildly bearish", "somewhat bearish")):
        return "Little Bearish"
    if "bearish" in lowered or "negative" in lowered:
        return "Very Very Bearish"
    if "bullish" in lowered or "positive" in lowered:
        return "Bullish"
    if "neutral" in lowered or "mixed" in lowered:
        return "Neutral"
    return ""


def infer_sentiment(text: str | None) -> str:
    if not text:
        return "Neutral"
    normalized = _normalize_sentiment_label(text)
    return normalized or "Neutral"


def classify_sec_filing_sentiment(text: str | None) -> str:
    lowered = str(text or "").lower()
    if not lowered:
        return "Neutral"
    explicit_match = re.search(r"sentiment:\s*([^\n\r]+)", str(text or ""), flags=re.I)
    if explicit_match:
        explicit = _normalize_sentiment_label(explicit_match.group(1).strip())
        if explicit:
            return explicit
    bearish_terms = [
        "bankruptcy",
        "chapter 11",
        "going concern",
        "default",
        "delisting",
        "termination",
        "resignation",
        "impairment",
        "restatement",
        "investigation",
        "subpoena",
        "material weakness",
        "layoff",
        "restructuring",
        "offering",
        "dilution",
        "adverse",
    ]
    bullish_terms = [
        "acquisition",
        "merger",
        "strategic partnership",
        "agreement",
        "contract",
        "award",
        "approval",
        "fda approval",
        "guidance raised",
        "raises guidance",
        "record revenue",
        "share repurchase",
        "buyback",
        "dividend increase",
        "new product",
        "launch",
        "positive",
    ]
    bearish_score = sum(1 for term in bearish_terms if term in lowered)
    bullish_score = sum(1 for term in bullish_terms if term in lowered)
    if bearish_score >= bullish_score + 2:
        return "Very Very Bearish"
    if bearish_score > bullish_score:
        return "Little Bearish"
    if bullish_score >= bearish_score + 2:
        return "Very Very Bullish"
    if bullish_score > bearish_score:
        return "Bullish"
    return "Neutral"


def _sec_user_agent() -> str:
    return os.getenv("SEC_USER_AGENT", "Kalyani Setup Scanner kalyani@example.com")


def _sec_form_type_from_title(title: str) -> str:
    match = re.match(r"\s*([A-Z0-9/-]+)\s+-\s+", str(title or ""), flags=re.I)
    return match.group(1).upper() if match else ""


def _sec_form_type_matches(requested_form_type: str, actual_form_type: str) -> bool:
    requested = str(requested_form_type or "").upper()
    actual = str(actual_form_type or "").upper()
    if requested == "4":
        return actual in {"4", "4/A"}
    if requested == "8-K":
        return actual in {"8-K", "8-K/A"}
    return requested == actual


def _eligible_cik_map(db_path: Path = TRACKING_DB_PATH) -> dict[str, dict[str, str]]:
    eligible_symbols = set(load_eligible_ticker_symbols())
    if not eligible_symbols:
        return {}
    sync_sec_company_map_for_eligible_tickers(db_path)
    rows = _rows(
        """
        SELECT cik, ticker, company_name
        FROM sec_companies
        WHERE ticker IS NOT NULL AND ticker != ''
        """,
        db_path=db_path,
    )
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        ticker = str(row.get("ticker") or "").upper()
        cik = str(row.get("cik") or "").lstrip("0")
        if ticker in eligible_symbols and cik:
            out[cik] = {
                "ticker": ticker,
                "company_name": str(row.get("company_name") or ticker),
            }
    return out


def sync_sec_company_map_for_eligible_tickers(db_path: Path = TRACKING_DB_PATH) -> int:
    eligible_symbols = set(load_eligible_ticker_symbols())
    if not eligible_symbols:
        return 0
    last_sync = _metadata_get("sec_company_tickers_last_sync", db_path)
    if last_sync[:10] == date.today().isoformat():
        return 0
    response = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"User-Agent": _sec_user_agent(), "Accept-Encoding": "gzip, deflate"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    rows: list[tuple[str, str, str]] = []
    for item in payload.values() if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker") or "").upper().replace(".", "-")
        if ticker not in eligible_symbols:
            continue
        cik = str(item.get("cik_str") or "").zfill(10)
        company_name = str(item.get("title") or ticker)
        if cik and ticker:
            rows.append((ticker, company_name, cik))
    ensure_tracking_schema(db_path)
    with _connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO sec_companies (ticker, company_name, cik)
            VALUES (?, ?, ?)
            ON CONFLICT(cik) DO UPDATE SET
                ticker = excluded.ticker,
                company_name = excluded.company_name
            """,
            rows,
        )
        conn.commit()
    _metadata_set("sec_company_tickers_last_sync", date.today().isoformat(), db_path)
    return len(rows)


def _sec_feed_entries(form_type: str = "8-K", count: int = 100) -> list[dict[str, str]]:
    url = "https://www.sec.gov/cgi-bin/browse-edgar"
    response = requests.get(
        url,
        params={"action": "getcurrent", "type": form_type, "count": count, "output": "atom"},
        headers={"User-Agent": _sec_user_agent(), "Accept-Encoding": "gzip, deflate"},
        timeout=30,
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries: list[dict[str, str]] = []
    for entry in root.findall("atom:entry", ns):
        title = entry.findtext("atom:title", default="", namespaces=ns)
        actual_form_type = _sec_form_type_from_title(title)
        if actual_form_type and not _sec_form_type_matches(form_type, actual_form_type):
            continue
        summary = unescape(entry.findtext("atom:summary", default="", namespaces=ns))
        updated = entry.findtext("atom:updated", default="", namespaces=ns)
        link_el = entry.find("atom:link", ns)
        filing_url = link_el.attrib.get("href", "") if link_el is not None else ""
        cik_match = re.search(r"\((\d{6,10})\)\s+\(Filer\)", title)
        accession_match = re.search(r"AccNo:\s*([0-9-]+)", summary)
        filed_match = re.search(r"Filed:\s*([0-9-]+)", summary)
        item_text = re.sub(r"<[^>]+>", " ", summary)
        item_text = re.sub(r"\s+", " ", item_text).strip()
        accession_match = accession_match or re.search(r"AccNo:\s*([0-9-]+)", item_text)
        filed_match = filed_match or re.search(r"Filed:\s*([0-9-]+)", item_text)
        entries.append(
            {
                "title": title.strip(),
                "summary": item_text,
                "updated": updated,
                "filing_url": filing_url,
                "form_type": actual_form_type or form_type,
                "cik": str(int(cik_match.group(1))) if cik_match else "",
                "accession": accession_match.group(1) if accession_match else filing_url,
                "filed": filed_match.group(1) if filed_match else updated[:10],
            }
        )
    return entries


def _sec_primary_document_url(index_url: str, prefer_xml: bool = False) -> str:
    if not index_url:
        return ""
    response = requests.get(
        index_url,
        headers={"User-Agent": _sec_user_agent(), "Accept-Encoding": "gzip, deflate"},
        timeout=30,
    )
    response.raise_for_status()
    html = response.text
    candidates = re.findall(r'href=["\']([^"\']+\.(?:htm|html|xml))["\']', html, flags=re.I)
    if prefer_xml:
        for candidate in candidates:
            lower = candidate.lower()
            if not lower.endswith(".xml"):
                continue
            if any(skip in lower for skip in ("filingsummary", "metadata", "xbrl")):
                continue
            candidate = re.sub(r"/xslf345x\d+/", "/", candidate, flags=re.I)
            return urljoin("https://www.sec.gov", candidate) if candidate.startswith("/") else urljoin(index_url, candidate)
    for candidate in candidates:
        if "/ix?doc=" in candidate:
            doc_path = candidate.split("/ix?doc=", 1)[1]
            return urljoin("https://www.sec.gov", doc_path)
    for candidate in candidates:
        lower = candidate.lower()
        if not (lower.endswith(".htm") or lower.endswith(".html")):
            continue
        if "index.htm" in lower:
            continue
        return urljoin(index_url, candidate)
    return ""


def _sec_filing_document_text(index_url: str, max_chars: int = 60000, prefer_xml: bool = False) -> tuple[str, str]:
    document_url = _sec_primary_document_url(index_url, prefer_xml=prefer_xml)
    if not document_url:
        return "", ""
    response = requests.get(
        document_url,
        headers={"User-Agent": _sec_user_agent(), "Accept-Encoding": "gzip, deflate"},
        timeout=30,
    )
    response.raise_for_status()
    text = response.text if document_url.lower().endswith(".xml") else _article_text_from_html(response.text)
    return text[:max_chars], document_url


def _legacy_8k_prompts() -> dict[str, str]:
    try:
        from imported_stock_tracking_app.prompts import prompts
    except Exception:
        return {}
    return {str(key): str(value).strip() for key, value in prompts.items()}


def _extract_8k_items(text: str) -> dict[str, str]:
    if not text:
        return {}
    cleaned = re.sub(r"[ \t]+", " ", text)
    item_pattern = re.compile(r"Item\s*\n?\s*(\d+\.\d+)", re.I)
    matches = list(item_pattern.finditer(cleaned))
    items: dict[str, str] = {}
    for index, match in enumerate(matches):
        item_number = match.group(1)
        if item_number not in IMPORTANT_8K_ITEMS:
            continue
        start = match.end()
        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            signature_match = re.search(r"SIGNATURES?", cleaned[start:], re.I)
            end = start + signature_match.start() if signature_match else len(cleaned)
        section = cleaned[start:end].strip()
        if section:
            items[item_number] = section[:25000]
    return items


def _nvidia_sec_8k_summary(
    ticker: str,
    company_name: str,
    accession: str,
    item_number: str,
    item_text: str,
    prompt_instruction: str,
) -> str:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key or not item_text.strip():
        return ""
    prompt = f"""
{prompt_instruction}

Company: {company_name}
Ticker: {ticker}
SEC accession: {accession}
8-K item: {item_number}

Use only the filing text below. Do not invent missing facts. If a detail is not disclosed, say not disclosed.
Follow the return format requested in the prompt above.

--- CONTENT TO SUMMARIZE ---
{item_text[:30000]}
""".strip()
    try:
        response = requests.post(
            NVIDIA_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": NVIDIA_SUMMARY_MODEL,
                "messages": [
                    {"role": "system", "content": "You summarize SEC 8-K filings for equity research. Be factual and concise."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 900,
                "stream": False,
            },
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return ""
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not choices:
        return ""
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    return str(message.get("content") or "").strip()


def _summarize_sec_8k_with_nvidia(
    ticker: str,
    company_name: str,
    accession: str,
    filing_text: str,
    feed_text: str,
) -> str:
    prompts = _legacy_8k_prompts()
    items = _extract_8k_items(filing_text)
    summaries: list[str] = []
    if not items and filing_text:
        items = {"default": filing_text[:25000]}
    for item_number, item_text in items.items():
        prompt_instruction = prompts.get(item_number) or prompts.get("default") or SEC_8K_DEFAULT_PROMPT
        summary = _nvidia_sec_8k_summary(
            ticker,
            company_name,
            accession,
            item_number,
            item_text,
            prompt_instruction,
        )
        if summary:
            summaries.append(f"Item {item_number}\n{summary}")
    if summaries:
        return "\n\n".join(summaries)
    return f"{feed_text}\n\nFull 8-K text excerpt:\n{filing_text[:5000]}" if filing_text else feed_text


def _xml_local_name(tag: str) -> str:
    return str(tag).rsplit("}", 1)[-1]


def _xml_children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(element) if _xml_local_name(child.tag) == name]


def _xml_descendants(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element.iter() if _xml_local_name(child.tag) == name]


def _xml_child_text(element: ET.Element, path: tuple[str, ...]) -> str:
    current = element
    for name in path:
        matches = _xml_children(current, name)
        if not matches:
            return ""
        current = matches[0]
    return str(current.text or "").strip()


def _xml_first_text(element: ET.Element, name: str) -> str:
    for child in element.iter():
        if _xml_local_name(child.tag) == name:
            return str(child.text or "").strip()
    return ""


def _to_float(value: str) -> float | None:
    cleaned = str(value or "").replace(",", "").replace("$", "").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _format_share_count(value: float | None) -> str:
    if value is None:
        return "not disclosed"
    return f"{value:,.0f}" if value == int(value) else f"{value:,.2f}"


def _format_money(value: float | None) -> str:
    if value is None:
        return "not disclosed"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.1f}K"
    return f"${value:,.2f}"


FORM4_TRANSACTION_CODES = {
    "P": "Open-market/direct purchase",
    "S": "Open-market/direct sale",
    "A": "Grant or award",
    "D": "Sale or disposition back to issuer",
    "F": "Tax withholding/payment",
    "M": "Option/warrant exercise",
    "C": "Conversion",
    "G": "Gift",
    "J": "Other transaction",
    "V": "Voluntary reported transaction",
}


def _form4_owner_role(owner: ET.Element) -> str:
    relationship = _xml_children(owner, "reportingOwnerRelationship")
    if not relationship:
        return "role not disclosed"
    rel = relationship[0]
    role_parts: list[str] = []
    if _xml_child_text(rel, ("isDirector",)).lower() == "true":
        role_parts.append("Director")
    if _xml_child_text(rel, ("isOfficer",)).lower() == "true":
        title = _xml_child_text(rel, ("officerTitle",))
        role_parts.append(title or "Officer")
    if _xml_child_text(rel, ("isTenPercentOwner",)).lower() == "true":
        role_parts.append("10% owner")
    if _xml_child_text(rel, ("isOther",)).lower() == "true":
        other = _xml_child_text(rel, ("otherText",))
        role_parts.append(other or "Other")
    return ", ".join(dict.fromkeys(part for part in role_parts if part)) or "role not disclosed"


def _form4_transaction_row(transaction: ET.Element, transaction_kind: str) -> dict[str, Any]:
    code = _xml_child_text(transaction, ("transactionCoding", "transactionCode")).upper()
    shares = _to_float(_xml_child_text(transaction, ("transactionAmounts", "transactionShares", "value")))
    price = _to_float(_xml_child_text(transaction, ("transactionAmounts", "transactionPricePerShare", "value")))
    acquired_disposed = _xml_child_text(transaction, ("transactionAmounts", "transactionAcquiredDisposedCode", "value")).upper()
    value = shares * price if shares is not None and price is not None else None
    return {
        "date": _xml_child_text(transaction, ("transactionDate", "value")),
        "security": _xml_child_text(transaction, ("securityTitle", "value")) or transaction_kind,
        "code": code or "N/A",
        "code_meaning": FORM4_TRANSACTION_CODES.get(code, "Other/undisclosed transaction"),
        "action": "Acquired" if acquired_disposed == "A" else "Disposed" if acquired_disposed == "D" else "Not disclosed",
        "shares": shares,
        "price": price,
        "value": value,
        "owned_after": _to_float(_xml_child_text(transaction, ("postTransactionAmounts", "sharesOwnedFollowingTransaction", "value"))),
        "ownership": _xml_child_text(transaction, ("ownershipNature", "directOrIndirectOwnership", "value")) or "not disclosed",
    }


def _extract_form4_transaction_summary(text: str) -> str:
    if not text:
        return ""
    try:
        root = ET.fromstring(text.encode("utf-8"))
    except Exception:
        cleaned = re.sub(r"\s+", " ", text).strip()
        return (
            "Headline: Form 4 insider ownership transaction\n"
            "Sentiment: Neutral\n"
            "Risk Level: Low\n"
            "Key Facts: Could not parse structured Form 4 XML.\n"
            f"Investor Summary: {cleaned[:500] or 'No readable filing text found.'}\n"
            "Likely Market Impact: Review the SEC filing directly."
        )

    issuer_name = _xml_first_text(root, "issuerName")
    issuer_symbol = _xml_first_text(root, "issuerTradingSymbol")
    period = _xml_first_text(root, "periodOfReport")
    owners = _xml_descendants(root, "reportingOwner")
    owner_details = []
    for owner in owners:
        name = _xml_child_text(owner, ("reportingOwnerId", "rptOwnerName"))
        owner_details.append(f"{name or 'Owner not disclosed'} ({_form4_owner_role(owner)})")

    transactions = [
        _form4_transaction_row(transaction, "Common/Non-derivative")
        for transaction in _xml_descendants(root, "nonDerivativeTransaction")
    ]
    transactions.extend(
        _form4_transaction_row(transaction, "Derivative security")
        for transaction in _xml_descendants(root, "derivativeTransaction")
    )
    transactions = [transaction for transaction in transactions if transaction["code"] != "N/A" or transaction["shares"] is not None]
    open_market_purchases = [item for item in transactions if item["code"] == "P" and item["action"] == "Acquired"]
    open_market_sales = [item for item in transactions if item["code"] == "S" and item["action"] == "Disposed"]
    awards_or_exercises = [item for item in transactions if item["code"] in {"A", "M"}]
    purchase_value = sum(float(item["value"] or 0) for item in open_market_purchases)
    sale_value = sum(float(item["value"] or 0) for item in open_market_sales)

    if purchase_value > sale_value and purchase_value:
        sentiment = "Very Very Bullish" if purchase_value >= 1_000_000 else "Bullish"
        headline = f"Form 4 insider purchase: {_format_money(purchase_value)} bought"
    elif sale_value > purchase_value and sale_value:
        sentiment = "Little Bearish"
        headline = f"Form 4 insider sale: {_format_money(sale_value)} sold"
    elif awards_or_exercises:
        sentiment = "Neutral"
        headline = "Form 4 equity award or option exercise"
    else:
        sentiment = "Neutral"
        headline = "Form 4 insider ownership update"

    transaction_lines = []
    for item in transactions[:6]:
        transaction_lines.append(
            (
                f"{item['date'] or period or 'date not disclosed'}: {item['action']} "
                f"{_format_share_count(item['shares'])} shares of {item['security']} "
                f"at {_format_money(item['price'])}/share; value {_format_money(item['value'])}; "
                f"code {item['code']} ({item['code_meaning']}); owned after {_format_share_count(item['owned_after'])}."
            )
        )
    if len(transactions) > 6:
        transaction_lines.append(f"{len(transactions) - 6} additional transactions not shown.")

    key_facts = [
        f"Issuer: {issuer_name or 'not disclosed'} ({issuer_symbol or 'ticker not disclosed'})",
        f"Reporting owner: {'; '.join(owner_details) if owner_details else 'not disclosed'}",
        f"Period of report: {period or 'not disclosed'}",
        f"Open-market purchase value: {_format_money(purchase_value) if purchase_value else '$0.00'}",
        f"Open-market sale value: {_format_money(sale_value) if sale_value else '$0.00'}",
    ]
    return (
        f"Headline: {headline}\n"
        f"Sentiment: {sentiment}\n"
        "Risk Level: Low\n"
        f"Key Facts: {' | '.join(key_facts)}\n"
        f"Transactions: {' | '.join(transaction_lines) if transaction_lines else 'No reportable transaction rows found.'}\n"
        "Investor Summary: This is a structured Form 4 insider ownership filing. Open-market purchases are usually more useful bullish signals than grants, option exercises, gifts, or tax-withholding sales. Open-market sales can be bearish, but should be judged against owner role, size, and remaining ownership.\n"
        "Likely Market Impact: Usually limited unless the transaction is large, open-market, repeated, or from a key executive."
    )


def classify_form4_sentiment(text: str | None) -> str:
    lowered = str(text or "").lower()
    explicit_match = re.search(r"sentiment:\s*([^\n\r]+)", str(text or ""), flags=re.I)
    if explicit_match:
        explicit = _normalize_sentiment_label(explicit_match.group(1).strip())
        if explicit:
            return explicit
    if "open-market purchase value: $" in lowered and not "open-market purchase value: $0.00" in lowered:
        return "Bullish"
    if "open-market sale value: $" in lowered and not "open-market sale value: $0.00" in lowered:
        return "Little Bearish"
    return "Neutral"


def _accession_has_nvidia_summary(accession: str, db_path: Path = TRACKING_DB_PATH) -> bool:
    if not accession:
        return False
    rows = _rows(
        "SELECT 1 FROM stock_news WHERE accession = ? AND headline LIKE ? LIMIT 1",
        (accession, "%Investor Summary:%"),
        db_path,
    )
    return bool(rows)


def _accession_existing_summary(accession: str, db_path: Path = TRACKING_DB_PATH) -> dict[str, Any] | None:
    if not accession:
        return None
    rows = _rows(
        """
        SELECT headline, filing_url, sentiment
        FROM stock_news
        WHERE accession = ? AND headline LIKE ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (accession, "%Sentiment%"),
        db_path,
    )
    return rows[0] if rows else None


def cleanup_duplicate_sec_feed_rows(db_path: Path = TRACKING_DB_PATH) -> int:
    ensure_tracking_schema(db_path)
    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            DELETE FROM stock_news
            WHERE accession LIKE 'https://www.sec.gov/%-index.htm'
              AND EXISTS (
                  SELECT 1
                  FROM stock_news newer
                  WHERE newer.cik = stock_news.cik
                    AND newer.datetime = stock_news.datetime
                    AND newer.headline LIKE '%Investor Summary:%'
              )
            """
        )
        conn.commit()
        return cursor.rowcount


def prune_sec_feed_rows(max_rows: int = 2000, db_path: Path = TRACKING_DB_PATH) -> int:
    ensure_tracking_schema(db_path)
    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            DELETE FROM stock_news
            WHERE id NOT IN (
                SELECT id
                FROM stock_news
                ORDER BY datetime DESC, id DESC
                LIMIT ?
            )
            """,
            (max_rows,),
        )
        conn.commit()
        return cursor.rowcount


def poll_sec_8k_feed_once(
    count: int = 100,
    db_path: Path = TRACKING_DB_PATH,
    progress: bool = False,
    force_resummarize: bool = False,
    form_types: tuple[str, ...] = ("8-K", "4"),
) -> dict[str, Any]:
    ensure_tracking_schema(db_path)
    eligible_by_cik = _eligible_cik_map(db_path)
    if not eligible_by_cik:
        return {"checked": 0, "matched": 0, "inserted": 0, "status": "No saved $500M CIK map found."}

    entries: list[dict[str, str]] = []
    for form_type in form_types:
        entries.extend(_sec_feed_entries(form_type=form_type, count=count))
    matched = 0
    inserted = 0
    for entry in entries:
        cik = entry.get("cik", "")
        if cik not in eligible_by_cik:
            continue
        company = eligible_by_cik[cik]
        ticker = company.get("ticker", "")
        company_name = company.get("company_name", ticker)
        accession = entry.get("accession") or entry.get("filing_url")
        form_type = str(entry.get("form_type") or "8-K")
        matched += 1
        existing_summary = _accession_existing_summary(accession, db_path)
        if existing_summary and not force_resummarize:
            if progress:
                print(f"Skipping already summarized {ticker} {accession}", flush=True)
            headline = str(existing_summary.get("headline") or "")
            document_url = str(existing_summary.get("filing_url") or "")
            sentiment = str(existing_summary.get("sentiment") or classify_sec_filing_sentiment(headline))
            row = {
                "cik": cik.zfill(10),
                "headline": headline,
                "datetime": entry.get("filed") or entry.get("updated"),
                "filing_url": document_url or entry.get("filing_url"),
                "accession": accession,
                "form_type": form_type,
                "sentiment": sentiment,
            }
            with _connect(db_path) as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO stock_news (
                        cik, headline, datetime, filing_url, accession, form_type, sentiment
                    )
                    VALUES (
                        :cik, :headline, :datetime, :filing_url, :accession, :form_type, :sentiment
                    )
                    ON CONFLICT(accession) DO UPDATE SET
                        headline = excluded.headline,
                        datetime = excluded.datetime,
                        filing_url = excluded.filing_url,
                        form_type = excluded.form_type,
                        sentiment = excluded.sentiment
                    """,
                    row,
                )
                inserted += cursor.rowcount
                conn.commit()
            continue
        if progress:
            print(f"Processing {form_type} {ticker} {accession}", flush=True)
        feed_text = f"{entry.get('title', '')} {entry.get('summary', '')}"
        filing_text = ""
        document_url = ""
        headline = f"{entry.get('title', '')} - {entry.get('summary', '')}"
        try:
            filing_text, document_url = _sec_filing_document_text(
                entry.get("filing_url", ""),
                prefer_xml=form_type in {"4", "4/A"},
            )
        except Exception:
            filing_text = ""
            document_url = ""
        if form_type in {"4", "4/A"}:
            headline = _extract_form4_transaction_summary(filing_text or feed_text)
        elif filing_text:
            headline = _summarize_sec_8k_with_nvidia(
                ticker,
                company_name,
                accession,
                filing_text,
                headline,
            )
        text = headline or filing_text or feed_text
        sentiment = classify_form4_sentiment(text) if form_type in {"4", "4/A"} else classify_sec_filing_sentiment(text)
        row = {
            "cik": cik.zfill(10),
            "headline": headline,
            "datetime": entry.get("filed") or entry.get("updated"),
            "filing_url": document_url or entry.get("filing_url"),
            "accession": accession,
            "form_type": form_type,
            "sentiment": sentiment,
        }
        with _connect(db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO stock_news (
                    cik, headline, datetime, filing_url, accession, form_type, sentiment
                )
                VALUES (
                    :cik, :headline, :datetime, :filing_url, :accession, :form_type, :sentiment
                )
                ON CONFLICT(accession) DO UPDATE SET
                    headline = excluded.headline,
                    datetime = excluded.datetime,
                    filing_url = excluded.filing_url,
                    form_type = excluded.form_type,
                    sentiment = excluded.sentiment
                """,
                row,
            )
            inserted += cursor.rowcount
            conn.commit()
        if progress:
            print(f"Saved {form_type} {ticker} {accession} sentiment={sentiment}", flush=True)
    _metadata_set("sec_8k_last_poll", pd.Timestamp.utcnow().isoformat(), db_path)
    cleanup_duplicate_sec_feed_rows(db_path)
    pruned = prune_sec_feed_rows(2000, db_path)
    return {"checked": len(entries), "matched": matched, "inserted": inserted, "status": "ok"}


def run_sec_8k_feed_poller(
    interval_seconds: int = 600,
    count: int = 100,
    db_path: Path = TRACKING_DB_PATH,
    force_resummarize: bool = False,
) -> None:
    while True:
        try:
            result = poll_sec_8k_feed_once(
                count=count,
                db_path=db_path,
                progress=True,
                force_resummarize=force_resummarize,
            )
            print(
                f"{pd.Timestamp.utcnow().isoformat()} checked={result['checked']} "
                f"matched={result['matched']} inserted={result['inserted']} status={result['status']}",
                flush=True,
            )
        except Exception as exc:
            print(f"{pd.Timestamp.utcnow().isoformat()} SEC 8-K poll failed: {exc}", flush=True)
        time.sleep(interval_seconds)


def format_market_cap(value: Any) -> str:
    try:
        number = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return str(value or "")

    abs_number = abs(number)
    if abs_number >= 1_000_000_000_000:
        return f"${number / 1_000_000_000_000:.2f}T"
    if abs_number >= 1_000_000_000:
        return f"${number / 1_000_000_000:.2f}B"
    if abs_number >= 1_000_000:
        return f"${number / 1_000_000:.2f}M"
    return f"${number:,.0f}"
