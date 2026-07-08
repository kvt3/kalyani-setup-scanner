from __future__ import annotations

import os
from datetime import datetime, time, timedelta
from dataclasses import dataclass
from io import StringIO
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

from config import HISTORY_PERIOD, NASDAQ_LISTED_URL, OTHER_LISTED_URL, PRICE_CHUNK_SIZE


NASDAQ_HISTORICAL_URL = "https://api.nasdaq.com/api/quote/{ticker}/historical"
ALPHAVANTAGE_LISTING_STATUS_URL = "https://www.alphavantage.co/query"
NASDAQ_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0",
}


FALLBACK_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "AVGO",
    "TSLA",
    "LLY",
    "AMD",
    "NFLX",
    "COST",
    "CRM",
    "NOW",
    "PANW",
    "CRWD",
]


@dataclass(frozen=True)
class TickerUniverse:
    tickers: list[str]
    source: str
    error: str | None = None

    @property
    def is_fallback(self) -> bool:
        return self.source == "fallback"


def _to_yfinance_symbol(symbol: object) -> str | None:
    if not isinstance(symbol, str):
        return None
    cleaned = symbol.strip()
    if not cleaned or cleaned.lower() == "nan":
        return None
    return cleaned.replace(".", "-")


def _normalize_symbols(symbols: pd.Series) -> list[str]:
    normalized = (_to_yfinance_symbol(symbol) for symbol in symbols.dropna())
    return sorted({symbol for symbol in normalized if symbol})


def _read_nasdaq_pipe_file(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    text = "\n".join(
        line for line in response.text.splitlines() if not line.startswith("File Creation Time")
    )
    return pd.read_csv(StringIO(text), sep="|")


def _read_alphavantage_listing_status() -> pd.DataFrame:
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY is not configured.")
    response = requests.get(
        ALPHAVANTAGE_LISTING_STATUS_URL,
        params={"function": "LISTING_STATUS", "state": "active", "apikey": api_key},
        timeout=30,
    )
    response.raise_for_status()
    if "symbol" not in response.text[:200].lower():
        raise RuntimeError(response.text[:200].strip() or "Alpha Vantage listing response was empty.")
    frame = pd.read_csv(StringIO(response.text))
    if frame.empty or "symbol" not in frame.columns:
        raise RuntimeError("Alpha Vantage listing status returned no symbols.")
    exchange = frame.get("exchange", pd.Series(dtype=str)).fillna("").astype(str).str.upper()
    asset_type = frame.get("assetType", pd.Series(dtype=str)).fillna("").astype(str).str.upper()
    frame = frame[
        asset_type.eq("STOCK")
        & exchange.isin(["NASDAQ", "NYSE", "NYSE AMERICAN", "NYSE MKT", "AMEX"])
    ]
    return frame.rename(columns={"symbol": "Symbol"})[["Symbol"]]


def load_us_ticker_universe(limit: int | None = None) -> TickerUniverse:
    """Load ordinary common-stock tickers from NASDAQ and NYSE."""
    try:
        nasdaq = _read_nasdaq_pipe_file(NASDAQ_LISTED_URL)
        nasdaq = nasdaq[
            (nasdaq["Test Issue"] == "N")
            & (nasdaq["Financial Status"].fillna("N") != "D")
            & (nasdaq["ETF"] == "N")
        ][["Symbol"]]

        other = _read_nasdaq_pipe_file(OTHER_LISTED_URL)
        other = other[
            (other["Exchange"].isin(["N", "A"]))
            & (other["Test Issue"] == "N")
            & (other["ETF"] == "N")
        ][["ACT Symbol"]].rename(columns={"ACT Symbol": "Symbol"})

        tickers = _normalize_symbols(pd.concat([nasdaq, other])["Symbol"])
        source = "nasdaqtrader"
        error = None
    except Exception as exc:
        nasdaqtrader_error = str(exc)
        try:
            tickers = _normalize_symbols(_read_alphavantage_listing_status()["Symbol"])
            source = "alphavantage_listing_status"
            error = f"nasdaqtrader unavailable: {nasdaqtrader_error}"
        except Exception as alpha_exc:
            tickers = FALLBACK_TICKERS.copy()
            source = "fallback"
            error = f"nasdaqtrader unavailable: {nasdaqtrader_error}; alphavantage unavailable: {alpha_exc}"

    if limit:
        tickers = tickers[:limit]
    return TickerUniverse(tickers=tickers, source=source, error=error)


def load_us_tickers(limit: int | None = None) -> list[str]:
    return load_us_ticker_universe(limit=limit).tickers


def chunked(items: Iterable[str], size: int = PRICE_CHUNK_SIZE) -> Iterable[list[str]]:
    batch: list[str] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def latest_completed_us_session(now: datetime | None = None) -> pd.Timestamp:
    """Return the latest U.S. market date whose regular session is complete.

    This intentionally ignores intraday/premarket updates. If the current New York
    session has not closed, the scanner uses the previous weekday and then lets
    downloaded data choose the latest available row on or before that date.
    """
    ny_tz = ZoneInfo("America/New_York")
    ny_now = (now or datetime.now(tz=ny_tz)).astimezone(ny_tz)
    candidate = ny_now.date()
    market_closed = ny_now.weekday() < 5 and ny_now.time() >= time(16, 10)
    if not market_closed:
        candidate -= timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return pd.Timestamp(candidate)


def _normalize_download(downloaded: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    if downloaded.empty:
        return {}

    data: dict[str, pd.DataFrame] = {}
    if isinstance(downloaded.columns, pd.MultiIndex):
        level_names = list(downloaded.columns.names)
        first_level = set(downloaded.columns.get_level_values(0))
        second_level = set(downloaded.columns.get_level_values(1))
        for ticker in tickers:
            frame = pd.DataFrame()
            if ticker in first_level:
                frame = downloaded[ticker].dropna(how="all").copy()
            elif ticker in second_level:
                frame = downloaded.xs(ticker, axis=1, level=1).dropna(how="all").copy()
            elif len(tickers) == 1 and "Ticker" not in level_names and "Price" in level_names:
                frame = downloaded.droplevel(1, axis=1).dropna(how="all").copy()
            if not frame.empty:
                data[ticker] = frame
    elif len(tickers) == 1:
        data[tickers[0]] = downloaded.dropna(how="all").copy()
    return data


def _parse_nasdaq_number(value: object) -> float | None:
    text = str(value or "").replace("$", "").replace(",", "").strip()
    if not text or text.upper() == "N/A":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def download_ohlcv_from_nasdaq(ticker: str, completed_date: pd.Timestamp) -> pd.DataFrame | None:
    from_date = (completed_date - pd.Timedelta(days=420)).date().isoformat()
    to_date = completed_date.date().isoformat()
    try:
        response = requests.get(
            NASDAQ_HISTORICAL_URL.format(ticker=ticker),
            params={
                "assetclass": "stocks",
                "fromdate": from_date,
                "todate": to_date,
                "limit": 9999,
            },
            headers=NASDAQ_HEADERS,
            timeout=20,
        )
        response.raise_for_status()
        rows = response.json().get("data", {}).get("tradesTable", {}).get("rows", [])
    except Exception:
        return None

    parsed_rows: list[dict[str, float | pd.Timestamp]] = []
    for row in rows:
        date = pd.to_datetime(row.get("date"), errors="coerce")
        open_ = _parse_nasdaq_number(row.get("open"))
        high = _parse_nasdaq_number(row.get("high"))
        low = _parse_nasdaq_number(row.get("low"))
        close = _parse_nasdaq_number(row.get("close"))
        volume = _parse_nasdaq_number(row.get("volume"))
        if pd.isna(date) or None in (open_, high, low, close, volume):
            continue
        parsed_rows.append(
            {
                "Date": date.normalize(),
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Adj Close": close,
                "Volume": volume,
            }
        )

    if not parsed_rows:
        return None

    frame = pd.DataFrame(parsed_rows).drop_duplicates(subset=["Date"]).set_index("Date").sort_index()
    frame = frame[frame.index <= completed_date].dropna(subset=["Open", "High", "Low", "Close"])
    return frame if len(frame) >= 210 else None


def download_ohlcv(
    tickers: list[str],
    period: str = HISTORY_PERIOD,
    completed_date: pd.Timestamp | None = None,
    use_nasdaq_fallback: bool = True,
) -> dict[str, pd.DataFrame]:
    completed_date = completed_date or latest_completed_us_session()
    all_data: dict[str, pd.DataFrame] = {}
    for batch in chunked(tickers, size=PRICE_CHUNK_SIZE):
        try:
            downloaded = yf.download(
                tickers=batch,
                period=period,
                interval="1d",
                auto_adjust=False,
                prepost=False,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception:
            continue

        for ticker, frame in _normalize_download(downloaded, batch).items():
            frame = frame.rename(columns=str.title)
            frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
            frame = frame[frame.index <= completed_date].dropna(subset=["Open", "High", "Low", "Close"])
            if len(frame) >= 210:
                all_data[ticker] = frame

    if use_nasdaq_fallback:
        missing_tickers = [ticker for ticker in tickers if ticker not in all_data]
        for ticker in missing_tickers:
            frame = download_ohlcv_from_nasdaq(ticker, completed_date)
            if frame is not None:
                all_data[ticker] = frame

    return all_data
