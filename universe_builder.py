from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import requests
import yfinance as yf

from config import MIN_MARKET_CAP
from data_loader import load_us_ticker_universe


ProgressCallback = Callable[[int, int, int, str], None]


@dataclass(frozen=True)
class EligibleUniverseResult:
    rows: list[dict[str, float | str]]
    total_tickers: int
    source: str
    errors: list[str]


NASDAQ_SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks"
NASDAQ_SCREENER_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0",
}


def normalize_symbol(symbol: object) -> str | None:
    if not isinstance(symbol, str):
        return None
    cleaned = symbol.strip().upper()
    if not cleaned or cleaned == "NA":
        return None
    return cleaned.replace(".", "-").replace("/", "-")


def parse_market_cap(value: object) -> float | None:
    text = str(value or "").replace(",", "").replace("$", "").strip()
    if not text or text.upper() == "NA":
        return None
    try:
        market_cap = float(text)
    except ValueError:
        return None
    return market_cap if market_cap > 0 else None


def fetch_nasdaq_screener_rows(exchange: str) -> list[dict[str, Any]]:
    response = requests.get(
        NASDAQ_SCREENER_URL,
        params={"tableonly": "true", "limit": 10000, "exchange": exchange},
        headers=NASDAQ_SCREENER_HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("data", {}).get("table", {}).get("rows", [])


def build_from_nasdaq_screener(min_market_cap: int) -> EligibleUniverseResult:
    rows_by_symbol: dict[str, dict[str, float | str]] = {}
    errors: list[str] = []
    total = 0

    for exchange in ["nasdaq", "nyse"]:
        try:
            screener_rows = fetch_nasdaq_screener_rows(exchange)
        except Exception as exc:
            errors.append(f"{exchange}: screener unavailable ({exc})")
            continue

        total += len(screener_rows)
        for row in screener_rows:
            ticker = normalize_symbol(row.get("symbol"))
            market_cap = parse_market_cap(row.get("marketCap"))
            if not ticker:
                continue
            if market_cap is None:
                errors.append(f"{ticker}: market cap unavailable")
                continue
            if market_cap > min_market_cap:
                rows_by_symbol[ticker] = {
                    "ticker": ticker,
                    "market_cap": market_cap,
                    "source": f"nasdaq_screener:{exchange}",
                }

    if not rows_by_symbol:
        raise ValueError("Nasdaq screener returned no eligible market-cap rows")

    return EligibleUniverseResult(
        rows=sorted(rows_by_symbol.values(), key=lambda item: float(item["market_cap"]), reverse=True),
        total_tickers=total,
        source="nasdaq_screener",
        errors=errors,
    )


def fetch_market_cap(ticker: str) -> float | None:
    stock = yf.Ticker(ticker)

    try:
        fast_info = stock.fast_info
        market_cap = fast_info.get("market_cap") if hasattr(fast_info, "get") else None
        if market_cap and market_cap > 0:
            return float(market_cap)
    except Exception:
        pass

    try:
        info = stock.get_info()
        market_cap = info.get("marketCap")
        if market_cap and market_cap > 0:
            return float(market_cap)
    except Exception:
        pass

    return None


def build_eligible_market_cap_universe(
    min_market_cap: int = MIN_MARKET_CAP,
    progress_callback: ProgressCallback | None = None,
) -> EligibleUniverseResult:
    try:
        result = build_from_nasdaq_screener(min_market_cap)
        if progress_callback:
            progress_callback(
                result.total_tickers,
                result.total_tickers,
                len(result.rows),
                "nasdaq_screener",
            )
        return result
    except Exception as exc:
        screener_error = str(exc)

    universe = load_us_ticker_universe()
    rows: list[dict[str, float | str]] = []
    errors: list[str] = [f"nasdaq_screener: fallback to yfinance ({screener_error})"]
    total = len(universe.tickers)

    for index, ticker in enumerate(universe.tickers, start=1):
        market_cap = fetch_market_cap(ticker)
        if market_cap is None:
            errors.append(f"{ticker}: market cap unavailable")
        elif market_cap > min_market_cap:
            rows.append(
                {
                    "ticker": ticker,
                    "market_cap": market_cap,
                    "source": "yfinance",
                }
            )

        if progress_callback and (index == 1 or index % 25 == 0 or index == total):
            progress_callback(index, total, len(rows), ticker)

    return EligibleUniverseResult(
        rows=rows,
        total_tickers=total,
        source=universe.source,
        errors=errors,
    )
