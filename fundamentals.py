from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
import requests
import yfinance as yf

from config import DATA_DIR, MIN_AVG_VOLUME, MIN_EPS_GROWTH, MIN_MARKET_CAP, MIN_REVENUE_GROWTH


_FMP_RATE_LIMITED = False
_FMP_RATE_LIMITED_KEYS: set[str] = set()
_FMP_TOP50_CACHE_PATH = DATA_DIR / "fmp_top50_scorecard_cache.json"
_FMP_TOP50_CACHE_VERSION = "fmp_top50_v2_growth_star"


@dataclass(frozen=True)
class Fundamentals:
    ticker: str
    revenue_growth: float | None
    eps_growth: float | None
    market_cap: float | None
    average_volume: float | None
    source: str
    error: str | None = None

    @property
    def passes(self) -> bool:
        return (
            self.revenue_growth is not None
            and self.eps_growth is not None
            and self.market_cap is not None
            and self.average_volume is not None
            and self.revenue_growth > MIN_REVENUE_GROWTH
            and self.eps_growth > MIN_EPS_GROWTH
            and self.market_cap > MIN_MARKET_CAP
            and self.average_volume > MIN_AVG_VOLUME
        )


@dataclass(frozen=True)
class EPSGrowthDetail:
    period_type: str
    reported_period: str | None
    reported_date: str | None
    reported_eps: float | None
    previous_period: str | None
    previous_date: str | None
    previous_eps: float | None
    eps_growth: float | None


def _growth(newest: float | None, prior: float | None) -> float | None:
    if newest is None or prior in (None, 0) or pd.isna(newest) or pd.isna(prior):
        return None
    return (float(newest) - float(prior)) / abs(float(prior))


def _safe_info_value(info: dict, *keys: str) -> float | None:
    for key in keys:
        value = info.get(key)
        if value is not None and not pd.isna(value):
            return float(value)
    return None


def _period_label(date_value: object, period_type: str) -> str | None:
    date = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(date):
        return None
    if period_type == "annual":
        return f"FY {date.year}"
    quarter = ((date.month - 1) // 3) + 1
    return f"Q{quarter} {date.year}"


def _safe_numeric(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _eps_series(income: pd.DataFrame) -> pd.Series | None:
    for row_name in ["Diluted EPS", "DilutedEPS", "Basic EPS", "BasicEPS", "Normalized Diluted EPS"]:
        if row_name in income.index:
            return income.loc[row_name]
    if {"Net Income", "Diluted Average Shares"}.issubset(income.index):
        shares = income.loc["Diluted Average Shares"].replace(0, pd.NA)
        return income.loc["Net Income"] / shares
    if {"NetIncome", "DilutedAverageShares"}.issubset(income.index):
        shares = income.loc["DilutedAverageShares"].replace(0, pd.NA)
        return income.loc["NetIncome"] / shares
    if {"Net Income", "Basic Average Shares"}.issubset(income.index):
        shares = income.loc["Basic Average Shares"].replace(0, pd.NA)
        return income.loc["Net Income"] / shares
    if {"NetIncome", "BasicAverageShares"}.issubset(income.index):
        shares = income.loc["BasicAverageShares"].replace(0, pd.NA)
        return income.loc["NetIncome"] / shares
    return None


def _revenue_series(income: pd.DataFrame) -> pd.Series | None:
    for row_name in ["Total Revenue", "TotalRevenue"]:
        if row_name in income.index:
            return income.loc[row_name]
    return None


def _eps_growth_detail(stock: yf.Ticker, freq: str, period_type: str) -> EPSGrowthDetail:
    try:
        income = stock.get_income_stmt(freq=freq)
    except Exception:
        income = pd.DataFrame()

    if not isinstance(income, pd.DataFrame) or income.shape[1] < 2:
        return EPSGrowthDetail(period_type, None, None, None, None, None, None, None)

    eps = _eps_series(income)
    if eps is None or len(eps.dropna()) < 2:
        return EPSGrowthDetail(period_type, None, None, None, None, None, None, None)

    newest_date = income.columns[0]
    previous_date = income.columns[1]
    newest_eps = _safe_numeric(eps.iloc[0])
    previous_eps = _safe_numeric(eps.iloc[1])
    newest_ts = pd.to_datetime(newest_date, errors="coerce")
    previous_ts = pd.to_datetime(previous_date, errors="coerce")
    return EPSGrowthDetail(
        period_type=period_type,
        reported_period=_period_label(newest_date, period_type),
        reported_date=None if pd.isna(newest_ts) else newest_ts.date().isoformat(),
        reported_eps=newest_eps,
        previous_period=_period_label(previous_date, period_type),
        previous_date=None if pd.isna(previous_ts) else previous_ts.date().isoformat(),
        previous_eps=previous_eps,
        eps_growth=_growth(newest_eps, previous_eps),
    )


def _request_fmp_json(url: str) -> list | dict:
    global _FMP_RATE_LIMITED
    if _FMP_RATE_LIMITED:
        return {}
    candidate_keys = _fmp_api_keys_for_url(url)
    if not candidate_keys:
        return {}
    for api_key in candidate_keys:
        if api_key in _FMP_RATE_LIMITED_KEYS:
            continue
        request_url = _replace_fmp_api_key(url, api_key)
        response = requests.get(request_url, timeout=15)
        if response.status_code == 429:
            _FMP_RATE_LIMITED_KEYS.add(api_key)
            _FMP_RATE_LIMITED = len(_FMP_RATE_LIMITED_KEYS) >= len(_configured_fmp_api_keys())
            continue
        if response.status_code in {402, 403}:
            continue
        response.raise_for_status()
        try:
            return response.json()
        except ValueError:
            return {}
    _FMP_RATE_LIMITED = len(_FMP_RATE_LIMITED_KEYS) >= len(_configured_fmp_api_keys())
    return {}


def _configured_fmp_api_keys() -> list[str]:
    keys: list[str] = []
    for env_key in ["FMP_API_KEY", "FMP_API_KEY_2", "FMP_API_KEY_3"]:
        value = os.getenv(env_key)
        if value and value not in keys:
            keys.append(value)
    return keys


def _replace_fmp_api_key(url: str, api_key: str) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["apikey"] = api_key
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _fmp_api_keys_for_url(url: str) -> list[str]:
    configured = _configured_fmp_api_keys()
    if not configured:
        return []
    query = dict(parse_qsl(urlsplit(url).query, keep_blank_values=True))
    url_key = query.get("apikey")
    ordered: list[str] = []
    if url_key:
        ordered.append(url_key)
    for key in configured:
        if key not in ordered:
            ordered.append(key)
    return ordered


def get_fmp_rate_limit_status() -> dict[str, int]:
    configured = _configured_fmp_api_keys()
    limited = [key for key in configured if key in _FMP_RATE_LIMITED_KEYS]
    return {
        "configured_keys": len(configured),
        "rate_limited_keys": len(limited),
        "available_keys": max(len(configured) - len(limited), 0),
    }


def _fmp_url(path: str, api_key: str, **params: object) -> str:
    query = "&".join(
        f"{key}={value}"
        for key, value in {**params, "apikey": api_key}.items()
        if value is not None
    )
    return f"https://financialmodelingprep.com/{path}?{query}"


def _first_numeric(record: dict, *keys: str) -> float | None:
    for key in keys:
        value = _safe_numeric(record.get(key))
        if value is not None:
            return value
    return None


def _qoq_growth_trend_from_records(records: list, value_getter) -> list[float]:
    values: list[float] = []
    for record in reversed(records[:5]):
        if not isinstance(record, dict):
            continue
        value = value_getter(record)
        if value is None:
            return []
        values.append(float(value))
    if len(values) < 5:
        return []
    growth_rates: list[float] = []
    for prior, current in zip(values, values[1:], strict=False):
        growth = _growth(current, prior)
        if growth is None:
            return []
        growth_rates.append(round(growth * 100, 2))
    return growth_rates


def _growth_trending_up(growth_rates: list[float]) -> bool:
    if len(growth_rates) < 4:
        return False
    return all(current > prior for prior, current in zip(growth_rates, growth_rates[1:], strict=False))


def _quarter_values_from_records(records: list, value_getter, limit: int = 4) -> list[float]:
    values: list[float] = []
    for record in reversed(records[:limit]):
        if not isinstance(record, dict):
            continue
        value = value_getter(record)
        if value is None:
            return []
        values.append(round(float(value), 4))
    return values if len(values) == limit else []


def _values_trending_up(values: list[float]) -> bool:
    if len(values) < 4:
        return False
    return all(current > prior for prior, current in zip(values, values[1:], strict=False))


def _fmp_latest_eps_surprise_pct(ticker: str, api_key: str) -> float | None:
    rows = _request_fmp_json(
        f"https://financialmodelingprep.com/api/v3/historical/earning_calendar/{ticker}?limit=8&apikey={api_key}"
    )
    if not isinstance(rows, list) or not rows:
        return None
    dated_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_date = pd.to_datetime(
            row.get("date") or row.get("reportDate") or row.get("fiscalDateEnding"),
            errors="coerce",
        )
        dated_rows.append((row_date, row))
    dated_rows.sort(key=lambda item: pd.Timestamp.min if pd.isna(item[0]) else item[0], reverse=True)
    for _, row in dated_rows:
        direct = _first_numeric(row, "epsSurprise", "surprise", "surprisePercentage", "surprisePercent")
        if direct is not None:
            return direct * 100 if abs(direct) <= 1 else direct
        actual = _first_numeric(row, "epsActual", "actualEarningResult", "actual", "eps")
        estimate = _first_numeric(row, "epsEstimated", "estimatedEarning", "estimate", "estimatedEPS")
        if actual is not None and estimate not in (None, 0):
            return ((actual - estimate) / abs(estimate)) * 100
    return None


def _load_fmp_top50_cache() -> dict[str, dict[str, object]]:
    if not _FMP_TOP50_CACHE_PATH.exists():
        return {}
    try:
        payload = json.loads(_FMP_TOP50_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_fmp_top50_cache(cache: dict[str, dict[str, object]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _FMP_TOP50_CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def get_fmp_top50_scorecard(ticker: str) -> dict[str, object]:
    """FMP-only EPS/revenue fields used by the Top 50 Strength Score scanner."""
    api_key = os.getenv("FMP_API_KEY")
    if not api_key or _FMP_RATE_LIMITED:
        return {}
    ticker = str(ticker).strip().upper()
    today = date.today().isoformat()
    cache = _load_fmp_top50_cache()
    cached = cache.get(ticker)
    if (
        isinstance(cached, dict)
        and cached.get("fetched_date") == today
        and cached.get("cache_version") == _FMP_TOP50_CACHE_VERSION
    ):
        return {
            key: value
            for key, value in cached.items()
            if key not in {"fetched_date", "cache_version"}
        }

    result: dict[str, object] = {
        "revenue_yoy_growth_pct": None,
        "revenue_qoq_growth_pct": None,
        "eps_yoy_growth_pct": None,
        "eps_qoq_growth_pct": None,
        "eps_surprise_pct": None,
        "revenue_growth_trend_up": False,
        "eps_growth_trend_up": False,
        "revenue_qoq_growth_trend": [],
        "eps_qoq_growth_trend": [],
        "revenue_quarter_values": [],
        "eps_quarter_values": [],
        "fundamental_source": "FMP",
        "fundamental_error": "",
    }

    try:
        income = _request_fmp_json(
            _fmp_url("stable/income-statement", api_key, symbol=ticker, period="quarter", limit=5)
        )
    except Exception as exc:
        result["fundamental_error"] = str(exc)
        return result

    if isinstance(income, list) and len(income) >= 2:
        latest = income[0] if isinstance(income[0], dict) else {}
        prior_q = income[1] if isinstance(income[1], dict) else {}
        prior_y = income[4] if len(income) >= 5 and isinstance(income[4], dict) else {}
        latest_revenue = _safe_numeric(latest.get("revenue"))
        prior_q_revenue = _safe_numeric(prior_q.get("revenue"))
        prior_y_revenue = _safe_numeric(prior_y.get("revenue"))
        latest_eps = _first_numeric(latest, "eps", "epsdiluted")
        prior_q_eps = _first_numeric(prior_q, "eps", "epsdiluted")
        prior_y_eps = _first_numeric(prior_y, "eps", "epsdiluted")
        revenue_yoy = _growth(latest_revenue, prior_y_revenue)
        revenue_qoq = _growth(latest_revenue, prior_q_revenue)
        eps_yoy = _growth(latest_eps, prior_y_eps)
        eps_qoq = _growth(latest_eps, prior_q_eps)
        result["revenue_yoy_growth_pct"] = round(revenue_yoy * 100, 2) if revenue_yoy is not None else None
        result["revenue_qoq_growth_pct"] = round(revenue_qoq * 100, 2) if revenue_qoq is not None else None
        result["eps_yoy_growth_pct"] = round(eps_yoy * 100, 2) if eps_yoy is not None else None
        result["eps_qoq_growth_pct"] = round(eps_qoq * 100, 2) if eps_qoq is not None else None
        revenue_trend = _qoq_growth_trend_from_records(income, lambda record: _safe_numeric(record.get("revenue")))
        eps_trend = _qoq_growth_trend_from_records(income, lambda record: _first_numeric(record, "eps", "epsdiluted"))
        revenue_values = _quarter_values_from_records(income, lambda record: _safe_numeric(record.get("revenue")))
        eps_values = _quarter_values_from_records(income, lambda record: _first_numeric(record, "eps", "epsdiluted"))
        result["revenue_qoq_growth_trend"] = revenue_trend
        result["eps_qoq_growth_trend"] = eps_trend
        result["revenue_quarter_values"] = revenue_values
        result["eps_quarter_values"] = eps_values
        result["revenue_growth_trend_up"] = _growth_trending_up(revenue_trend) or _values_trending_up(revenue_values)
        result["eps_growth_trend_up"] = _growth_trending_up(eps_trend) or _values_trending_up(eps_values)

    surprise_pct = _fmp_latest_eps_surprise_pct(ticker, api_key)
    result["eps_surprise_pct"] = round(surprise_pct, 2) if surprise_pct is not None else None

    cache[ticker] = {**result, "fetched_date": today, "cache_version": _FMP_TOP50_CACHE_VERSION}
    _save_fmp_top50_cache(cache)
    return result


def _fmp_eps_growth_detail(ticker: str, period: str, api_key: str) -> EPSGrowthDetail:
    period_type = "annual" if period == "annual" else "quarterly"
    url = (
        f"https://financialmodelingprep.com/stable/income-statement"
        f"?symbol={ticker}&period={period}&limit=8&apikey={api_key}"
    )
    records = _request_fmp_json(url)
    if not isinstance(records, list) or len(records) < 2:
        return EPSGrowthDetail(period_type, None, None, None, None, None, None, None)

    newest = records[0]
    previous = records[1]
    newest_eps = _safe_numeric(newest.get("eps") or newest.get("epsdiluted"))
    previous_eps = _safe_numeric(previous.get("eps") or previous.get("epsdiluted"))
    newest_date = newest.get("date")
    previous_date = previous.get("date")
    return EPSGrowthDetail(
        period_type=period_type,
        reported_period=str(newest.get("period") or _period_label(newest_date, period_type) or ""),
        reported_date=str(newest_date) if newest_date else None,
        reported_eps=newest_eps,
        previous_period=str(previous.get("period") or _period_label(previous_date, period_type) or ""),
        previous_date=str(previous_date) if previous_date else None,
        previous_eps=previous_eps,
        eps_growth=_growth(newest_eps, previous_eps),
    )


def _fmp_quarterly_growth_details(ticker: str, api_key: str) -> dict[str, float | str | None]:
    url = (
        f"https://financialmodelingprep.com/stable/income-statement"
        f"?symbol={ticker}&period=quarter&limit=4&apikey={api_key}"
    )
    records = _request_fmp_json(url)
    if not isinstance(records, list) or len(records) < 2:
        return {}

    newest = records[0]
    previous = records[1]
    newest_date = newest.get("date")
    previous_date = previous.get("date")
    newest_revenue = _safe_numeric(newest.get("revenue"))
    previous_revenue = _safe_numeric(previous.get("revenue"))
    newest_eps = _safe_numeric(newest.get("eps") or newest.get("epsdiluted"))
    previous_eps = _safe_numeric(previous.get("eps") or previous.get("epsdiluted"))
    return {
        "quarterly_revenue_period": str(newest.get("period") or _period_label(newest_date, "quarterly") or ""),
        "quarterly_revenue_date": str(newest_date) if newest_date else None,
        "quarterly_revenue": newest_revenue,
        "quarterly_prior_revenue_period": str(previous.get("period") or _period_label(previous_date, "quarterly") or ""),
        "quarterly_prior_revenue_date": str(previous_date) if previous_date else None,
        "quarterly_prior_revenue": previous_revenue,
        "quarterly_revenue_growth": _growth(newest_revenue, previous_revenue),
        "quarterly_eps_period": str(newest.get("period") or _period_label(newest_date, "quarterly") or ""),
        "quarterly_eps_date": str(newest_date) if newest_date else None,
        "quarterly_reported_eps": newest_eps,
        "quarterly_prior_eps_period": str(previous.get("period") or _period_label(previous_date, "quarterly") or ""),
        "quarterly_prior_eps_date": str(previous_date) if previous_date else None,
        "quarterly_prior_eps": previous_eps,
        "quarterly_eps_growth": _growth(newest_eps, previous_eps),
    }


def _earnings_dates_frame(stock: yf.Ticker) -> pd.DataFrame:
    try:
        earnings = stock.get_earnings_dates(limit=16)
    except Exception:
        return pd.DataFrame()
    if not isinstance(earnings, pd.DataFrame) or earnings.empty or "Reported EPS" not in earnings.columns:
        return pd.DataFrame()

    out = earnings.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()]
    out = out.sort_index(ascending=False)
    out["Reported EPS"] = pd.to_numeric(out["Reported EPS"], errors="coerce")
    out = out.dropna(subset=["Reported EPS"])
    return out


def _quarterly_eps_from_earnings_dates(earnings: pd.DataFrame) -> EPSGrowthDetail:
    if len(earnings) < 2:
        return EPSGrowthDetail("quarterly", None, None, None, None, None, None, None)

    newest_date = earnings.index[0]
    comparison_index = 4 if len(earnings) >= 5 else 1
    previous_date = earnings.index[comparison_index]
    newest_eps = _safe_numeric(earnings.iloc[0]["Reported EPS"])
    previous_eps = _safe_numeric(earnings.iloc[comparison_index]["Reported EPS"])
    return EPSGrowthDetail(
        period_type="quarterly",
        reported_period=_period_label(newest_date, "quarterly"),
        reported_date=newest_date.date().isoformat(),
        reported_eps=newest_eps,
        previous_period=_period_label(previous_date, "quarterly"),
        previous_date=previous_date.date().isoformat(),
        previous_eps=previous_eps,
        eps_growth=_growth(newest_eps, previous_eps),
    )


def _qoq_eps_from_earnings_dates(earnings: pd.DataFrame) -> EPSGrowthDetail:
    if len(earnings) < 2:
        return EPSGrowthDetail("quarterly", None, None, None, None, None, None, None)

    newest_date = earnings.index[0]
    previous_date = earnings.index[1]
    newest_eps = _safe_numeric(earnings.iloc[0]["Reported EPS"])
    previous_eps = _safe_numeric(earnings.iloc[1]["Reported EPS"])
    return EPSGrowthDetail(
        period_type="quarterly",
        reported_period=_period_label(newest_date, "quarterly"),
        reported_date=newest_date.date().isoformat(),
        reported_eps=newest_eps,
        previous_period=_period_label(previous_date, "quarterly"),
        previous_date=previous_date.date().isoformat(),
        previous_eps=previous_eps,
        eps_growth=_growth(newest_eps, previous_eps),
    )


def _annual_eps_from_earnings_dates(earnings: pd.DataFrame) -> EPSGrowthDetail:
    if len(earnings) < 8:
        return EPSGrowthDetail("annual", None, None, None, None, None, None, None)

    latest_four = earnings.iloc[:4]["Reported EPS"]
    previous_four = earnings.iloc[4:8]["Reported EPS"]
    newest_eps = _safe_numeric(latest_four.sum())
    previous_eps = _safe_numeric(previous_four.sum())
    newest_date = earnings.index[0]
    previous_date = earnings.index[4]
    return EPSGrowthDetail(
        period_type="annual",
        reported_period=f"TTM through {_period_label(newest_date, 'quarterly')}",
        reported_date=newest_date.date().isoformat(),
        reported_eps=newest_eps,
        previous_period=f"TTM through {_period_label(previous_date, 'quarterly')}",
        previous_date=previous_date.date().isoformat(),
        previous_eps=previous_eps,
        eps_growth=_growth(newest_eps, previous_eps),
    )


def _date_from_unix(value: object) -> pd.Timestamp | None:
    numeric = _safe_numeric(value)
    if numeric is None:
        return None
    date = pd.to_datetime(numeric, unit="s", errors="coerce")
    if pd.isna(date):
        return None
    return date


def _eps_details_from_quote_info(stock: yf.Ticker) -> tuple[EPSGrowthDetail, EPSGrowthDetail]:
    try:
        info = stock.get_info()
    except Exception:
        info = {}

    quarter_date = _date_from_unix(info.get("mostRecentQuarter"))
    quarter_period = _period_label(quarter_date, "quarterly") if quarter_date is not None else None
    annual_period = f"TTM through {quarter_period}" if quarter_period else None
    annual_date = quarter_date.date().isoformat() if quarter_date is not None else None
    trailing_eps = _safe_info_value(info, "trailingEps")
    quarterly_growth = _safe_info_value(info, "earningsQuarterlyGrowth", "earningsGrowth")
    annual_growth = _safe_info_value(info, "earningsGrowth", "earningsQuarterlyGrowth")

    annual = EPSGrowthDetail(
        period_type="annual",
        reported_period=annual_period,
        reported_date=annual_date,
        reported_eps=trailing_eps,
        previous_period=None,
        previous_date=None,
        previous_eps=None,
        eps_growth=annual_growth,
    )
    quarterly = EPSGrowthDetail(
        period_type="quarterly",
        reported_period=quarter_period,
        reported_date=annual_date,
        reported_eps=None,
        previous_period=None,
        previous_date=None,
        previous_eps=None,
        eps_growth=quarterly_growth,
    )
    return annual, quarterly


def get_eps_growth_details(ticker: str) -> dict[str, float | str | None]:
    api_key = os.getenv("FMP_API_KEY")
    if api_key and not _FMP_RATE_LIMITED:
        try:
            annual = _fmp_eps_growth_detail(ticker, "annual", api_key)
            quarterly = _fmp_eps_growth_detail(ticker, "quarter", api_key)
            if annual.reported_eps is not None or quarterly.reported_eps is not None:
                return {
                    "annual_eps_period": annual.reported_period,
                    "annual_eps_date": annual.reported_date,
                    "annual_reported_eps": annual.reported_eps,
                    "annual_prior_eps_period": annual.previous_period,
                    "annual_prior_eps_date": annual.previous_date,
                    "annual_prior_eps": annual.previous_eps,
                    "annual_eps_growth": annual.eps_growth,
                    "quarterly_eps_period": quarterly.reported_period,
                    "quarterly_eps_date": quarterly.reported_date,
                    "quarterly_reported_eps": quarterly.reported_eps,
                    "quarterly_prior_eps_period": quarterly.previous_period,
                    "quarterly_prior_eps_date": quarterly.previous_date,
                    "quarterly_prior_eps": quarterly.previous_eps,
                    "quarterly_eps_growth": quarterly.eps_growth,
                }
        except Exception:
            pass

    stock = yf.Ticker(ticker)
    annual = _eps_growth_detail(stock, freq="yearly", period_type="annual")
    quarterly = _eps_growth_detail(stock, freq="quarterly", period_type="quarterly")

    if annual.reported_eps is None or quarterly.reported_eps is None:
        earnings = _earnings_dates_frame(stock)
        if annual.reported_eps is None:
            annual = _annual_eps_from_earnings_dates(earnings)
        if quarterly.reported_eps is None:
            quarterly = _quarterly_eps_from_earnings_dates(earnings)

    if annual.reported_period is None or quarterly.reported_period is None:
        info_annual, info_quarterly = _eps_details_from_quote_info(stock)
        if annual.reported_period is None:
            annual = info_annual
        if quarterly.reported_period is None:
            quarterly = info_quarterly

    return {
        "annual_eps_period": annual.reported_period,
        "annual_eps_date": annual.reported_date,
        "annual_reported_eps": annual.reported_eps,
        "annual_prior_eps_period": annual.previous_period,
        "annual_prior_eps_date": annual.previous_date,
        "annual_prior_eps": annual.previous_eps,
        "annual_eps_growth": annual.eps_growth,
        "quarterly_eps_period": quarterly.reported_period,
        "quarterly_eps_date": quarterly.reported_date,
        "quarterly_reported_eps": quarterly.reported_eps,
        "quarterly_prior_eps_period": quarterly.previous_period,
        "quarterly_prior_eps_date": quarterly.previous_date,
        "quarterly_prior_eps": quarterly.previous_eps,
        "quarterly_eps_growth": quarterly.eps_growth,
    }


def get_quarterly_growth_details(ticker: str) -> dict[str, float | str | None]:
    """Return latest quarter vs immediately previous quarter revenue and EPS growth."""
    api_key = os.getenv("FMP_API_KEY")
    if api_key and not _FMP_RATE_LIMITED:
        try:
            details = _fmp_quarterly_growth_details(ticker, api_key)
            if details:
                return details
        except Exception:
            pass

    stock = yf.Ticker(ticker)
    try:
        income = stock.get_income_stmt(freq="quarterly")
    except Exception:
        income = pd.DataFrame()

    details: dict[str, float | str | None] = {
        "quarterly_revenue_period": None,
        "quarterly_revenue_date": None,
        "quarterly_revenue": None,
        "quarterly_prior_revenue_period": None,
        "quarterly_prior_revenue_date": None,
        "quarterly_prior_revenue": None,
        "quarterly_revenue_growth": None,
        "quarterly_eps_period": None,
        "quarterly_eps_date": None,
        "quarterly_reported_eps": None,
        "quarterly_prior_eps_period": None,
        "quarterly_prior_eps_date": None,
        "quarterly_prior_eps": None,
        "quarterly_eps_growth": None,
    }

    if isinstance(income, pd.DataFrame) and income.shape[1] >= 2:
        newest_date = income.columns[0]
        previous_date = income.columns[1]
        newest_ts = pd.to_datetime(newest_date, errors="coerce")
        previous_ts = pd.to_datetime(previous_date, errors="coerce")
        newest_period = _period_label(newest_date, "quarterly")
        previous_period = _period_label(previous_date, "quarterly")

        revenue = _revenue_series(income)
        if revenue is not None:
            newest_revenue = _safe_numeric(revenue.iloc[0])
            previous_revenue = _safe_numeric(revenue.iloc[1])
            details.update(
                {
                    "quarterly_revenue_period": newest_period,
                    "quarterly_revenue_date": None if pd.isna(newest_ts) else newest_ts.date().isoformat(),
                    "quarterly_revenue": newest_revenue,
                    "quarterly_prior_revenue_period": previous_period,
                    "quarterly_prior_revenue_date": None if pd.isna(previous_ts) else previous_ts.date().isoformat(),
                    "quarterly_prior_revenue": previous_revenue,
                    "quarterly_revenue_growth": _growth(newest_revenue, previous_revenue),
                }
            )

        eps = _eps_series(income)
        if eps is not None:
            newest_eps = _safe_numeric(eps.iloc[0])
            previous_eps = _safe_numeric(eps.iloc[1])
            details.update(
                {
                    "quarterly_eps_period": newest_period,
                    "quarterly_eps_date": None if pd.isna(newest_ts) else newest_ts.date().isoformat(),
                    "quarterly_reported_eps": newest_eps,
                    "quarterly_prior_eps_period": previous_period,
                    "quarterly_prior_eps_date": None if pd.isna(previous_ts) else previous_ts.date().isoformat(),
                    "quarterly_prior_eps": previous_eps,
                    "quarterly_eps_growth": _growth(newest_eps, previous_eps),
                }
            )

    if details["quarterly_reported_eps"] is None:
        qoq_eps = _qoq_eps_from_earnings_dates(_earnings_dates_frame(stock))
        details.update(
            {
                "quarterly_eps_period": qoq_eps.reported_period,
                "quarterly_eps_date": qoq_eps.reported_date,
                "quarterly_reported_eps": qoq_eps.reported_eps,
                "quarterly_prior_eps_period": qoq_eps.previous_period,
                "quarterly_prior_eps_date": qoq_eps.previous_date,
                "quarterly_prior_eps": qoq_eps.previous_eps,
                "quarterly_eps_growth": qoq_eps.eps_growth,
            }
        )

    return details


def _from_fmp(ticker: str, api_key: str) -> Fundamentals:
    profile_url = f"https://financialmodelingprep.com/stable/profile?symbol={ticker}&apikey={api_key}"
    income_url = (
        f"https://financialmodelingprep.com/stable/income-statement"
        f"?symbol={ticker}&period=annual&limit=2&apikey={api_key}"
    )
    profile = _request_fmp_json(profile_url)
    income = _request_fmp_json(income_url)
    if not isinstance(profile, list) or not profile:
        raise ValueError("FMP did not return profile data")

    revenue_growth = None
    eps_growth = None
    if isinstance(income, list) and len(income) >= 2:
        newest, prior = income[0], income[1]
        revenue_growth = _growth(newest.get("revenue"), prior.get("revenue"))
        eps_growth = _growth(newest.get("eps") or newest.get("epsdiluted"), prior.get("eps") or prior.get("epsdiluted"))

    fallback = None
    if revenue_growth is None or eps_growth is None:
        fallback = _from_yfinance(ticker)
        revenue_growth = revenue_growth if revenue_growth is not None else fallback.revenue_growth
        eps_growth = eps_growth if eps_growth is not None else fallback.eps_growth

    profile_row = profile[0]
    return Fundamentals(
        ticker=ticker,
        revenue_growth=revenue_growth,
        eps_growth=eps_growth,
        market_cap=_safe_info_value(profile_row, "marketCap", "mktCap"),
        average_volume=_safe_info_value(profile_row, "averageVolume", "volAvg")
        or (fallback.average_volume if fallback else None),
        source="fmp" if isinstance(income, list) and len(income) >= 2 else "fmp_profile+yfinance",
    )


def _from_yfinance(ticker: str) -> Fundamentals:
    stock = yf.Ticker(ticker)
    info = stock.get_info()
    income = stock.get_income_stmt(freq="yearly")

    revenue_growth = None
    eps_growth = None

    if isinstance(income, pd.DataFrame) and income.shape[1] >= 2:
        newest = income.iloc[:, 0]
        prior = income.iloc[:, 1]
        if "Total Revenue" in income.index:
            revenue_growth = _growth(newest.get("Total Revenue"), prior.get("Total Revenue"))
        if revenue_growth is None and "TotalRevenue" in income.index:
            revenue_growth = _growth(newest.get("TotalRevenue"), prior.get("TotalRevenue"))
        eps_candidates = ["Diluted EPS", "DilutedEPS", "Basic EPS", "BasicEPS", "Normalized Diluted EPS"]
        for row_name in eps_candidates:
            if row_name in income.index:
                eps_growth = _growth(newest.get(row_name), prior.get(row_name))
                if eps_growth is not None:
                    break
        if eps_growth is None and {"Net Income", "Diluted Average Shares"}.issubset(income.index):
            newest_eps = newest.get("Net Income") / newest.get("Diluted Average Shares")
            prior_eps = prior.get("Net Income") / prior.get("Diluted Average Shares")
            eps_growth = _growth(newest_eps, prior_eps)
        if eps_growth is None and {"NetIncome", "DilutedAverageShares"}.issubset(income.index):
            newest_eps = newest.get("NetIncome") / newest.get("DilutedAverageShares")
            prior_eps = prior.get("NetIncome") / prior.get("DilutedAverageShares")
            eps_growth = _growth(newest_eps, prior_eps)

    revenue_growth = revenue_growth if revenue_growth is not None else _safe_info_value(info, "revenueGrowth")
    eps_growth = eps_growth if eps_growth is not None else _safe_info_value(
        info, "earningsGrowth", "earningsQuarterlyGrowth"
    )

    return Fundamentals(
        ticker=ticker,
        revenue_growth=revenue_growth,
        eps_growth=eps_growth,
        market_cap=_safe_info_value(info, "marketCap"),
        average_volume=_safe_info_value(info, "averageVolume", "averageDailyVolume10Day"),
        source="yfinance",
    )


def get_fundamentals(ticker: str) -> Fundamentals:
    api_key = os.getenv("FMP_API_KEY")
    try:
        if api_key and not _FMP_RATE_LIMITED:
            return _from_fmp(ticker, api_key)
        return _from_yfinance(ticker)
    except Exception as exc:
        if api_key:
            try:
                fallback = _from_yfinance(ticker)
                return Fundamentals(
                    ticker=ticker,
                    revenue_growth=fallback.revenue_growth,
                    eps_growth=fallback.eps_growth,
                    market_cap=fallback.market_cap,
                    average_volume=fallback.average_volume,
                    source="yfinance",
                    error=None,
                )
            except Exception:
                pass
        return Fundamentals(
            ticker=ticker,
            revenue_growth=None,
            eps_growth=None,
            market_cap=None,
            average_volume=None,
            source="none",
            error=str(exc),
        )
