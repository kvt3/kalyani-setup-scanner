#!/usr/bin/env python3
"""
Weekly V1 RS5 STRICT-HIGH Sector + Market Breakout Backtest

Strategy:
1. Use weekly candles.
2. QQQ must be in weekly uptrend.
3. Stock must be in weekly uptrend:
   Close > 9 EMA > 20 EMA > 50 SMA > 200 SMA.
4. Sector ETF must be in weekly uptrend and stronger than SPY.
5. Market ETF/SPY must be in weekly uptrend.
6. Stock must be stronger than SPY:
   RS 5-week vs SPY > 5%.
   RS Ratio ROC 5-week > 5%.
5. Stock makes weekly breakout:
   Weekly close breaks above the prior 52-week HIGH or prior all-time HIGH.
6. Breakout week volume must confirm:
   Weekly volume >= 1.2 × 20-week average volume.
7. Breakout week becomes signal candle.
   Next week buy only if price breaks above breakout candle high.
8. Stop below breakout weekly candle low.
9. Hold until weekly 9 EMA crosses below weekly 20 EMA, or price hits stop.

Install:
pip install yfinance pandas numpy

Run:
python weekly_v1_rs5_breakout_STRICT_HIGH_SECTOR_MARKET_backtest.py --universe market_cap_over_10b.csv
"""

from __future__ import annotations

import argparse
import math
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


# =========================
# STRATEGY SETTINGS
# =========================

OUTPUT_PREFIX = "weekly_v6_rs5_52w_ath_cup_handle"

# Date settings
START_DATE = "2010-01-01"
END_DATE = None  # None means today

# Market ETFs
SPY_TICKER = "SPY"
QQQ_TICKER = "QQQ"
MARKET_ETF_TICKER = SPY_TICKER

# Strength confirmation rules
# stock_strong is the stock weekly uptrend rule:
# Close > 9 EMA > 20 EMA > 50 SMA > 200 SMA
REQUIRE_STOCK_STRONG = True
REQUIRE_SECTOR_ETF_STRONG = True
REQUIRE_MARKET_ETF_STRONG = True
REQUIRE_SECTOR_RS_VS_SPY = True

# Sector vs SPY relative strength.
# 0.00 means sector ETF must outperform SPY over 5 weeks.
# Use 0.05 if you want sector ETF to beat SPY by more than 5%.
MIN_SECTOR_RS_VS_SPY = 0.00
MIN_SECTOR_RS_RATIO_ROC = 0.00

# Optional: require the stock to outperform its sector ETF.
# Keep False first; output columns are still created for analysis.
REQUIRE_STOCK_RS_VS_SECTOR = True
MIN_STOCK_RS_VS_SECTOR = 0.05
MIN_STOCK_SECTOR_RATIO_ROC = 0.00

# If ticker sector cannot be identified:
# "skip" = do not trade that ticker
# "spy" = use SPY as fallback sector ETF
UNKNOWN_SECTOR_ACTION = "skip"
SECTOR_CACHE_FILE = "sector_etf_cache.csv"

SECTOR_TO_ETF = {
    "Technology": "XLK",
    "Information Technology": "XLK",
    "Healthcare": "XLV",
    "Health Care": "XLV",
    "Financial Services": "XLF",
    "Financial": "XLF",
    "Financials": "XLF",
    "Consumer Cyclical": "XLY",
    "Consumer Discretionary": "XLY",
    "Consumer Defensive": "XLP",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Industrial Goods": "XLI",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
    "Materials": "XLB",
    "Communication Services": "XLC",
    "Telecom Services": "XLC",
    "Telecommunication Services": "XLC",
}

TICKER_SECTOR_ETF_OVERRIDES = {
    "GOOG": ("Communication Services", "XLC"),
    "GOOGL": ("Communication Services", "XLC"),
    "META": ("Communication Services", "XLC"),
    "AMZN": ("Consumer Cyclical", "XLY"),
    "TSLA": ("Consumer Cyclical", "XLY"),
    "BRK-B": ("Financial Services", "XLF"),
    "BRK-A": ("Financial Services", "XLF"),
}

# Weekly moving averages
EMA_FAST = 9
EMA_SLOW = 20
SMA_MID = 50
SMA_LONG = 200

# Relative strength
RS_LOOKBACK_WEEKS = 5
MIN_RS_VS_SPY = 0.05
MIN_RS_RATIO_ROC = 0.05

# Breakout
BREAKOUT_LOOKBACK_WEEKS = 52
MIN_BREAKOUT_VOLUME_RATIO = 1.20

# V5 quality filters requested:
# Keep both 52-week breakout and ATH breakout.
# 20 SMA must be above 50 SMA, but not more than 15% above it.
MAX_SMA20_ABOVE_SMA50_PCT = 15.0

# Avoid either:
# 1. Very big green breakout candle:
#    signal candle green AND range >= 20% AND body >= 10%
# OR
# 2. Very high volume breakout:
#    breakout volume ratio >= 4.0x
AVOID_BIG_GREEN_OR_VERY_HIGH_VOLUME = True
BIG_GREEN_MIN_RANGE_PCT = 20.0
BIG_GREEN_MIN_BODY_PCT = 10.0
VERY_HIGH_VOLUME_RATIO = 4.0

# Entry/stop
ENTRY_TRIGGER_BUFFER_PCT = 0.0
STOP_BUFFER_PCT = 0.005

# Gap-aware entry:
# If next week opens above the trigger, entry is next week open.
# Otherwise entry is the trigger price.
USE_GAP_AWARE_ENTRY = True

# Optional liquidity filter.
# Keep None to follow the exact weekly V1 rules the user specified.
MIN_20W_AVG_VOLUME = None  # Example: 1_000_000

# Cup-and-handle pattern filter.
# This is a practical weekly-chart approximation, not a perfect visual pattern detector.
REQUIRE_CUP_HANDLE_PATTERN = False
CUP_LOOKBACK_MAX_WEEKS = 104
CUP_MIN_TOTAL_WEEKS = 12
CUP_MIN_DEPTH_PCT = 12.0
CUP_MAX_DEPTH_PCT = 50.0
CUP_LEFT_RIM_TOLERANCE_PCT = 8.0
CUP_RIGHT_RIM_TOLERANCE_PCT = 10.0
HANDLE_MIN_WEEKS = 2
HANDLE_MAX_WEEKS = 8
HANDLE_MAX_DEPTH_PCT = 15.0
HANDLE_MAX_CUP_DEPTH_RATIO = 0.50
REQUIRE_HANDLE_LOW_ABOVE_CUP_MIDPOINT = True


@dataclass
class TradeResult:
    trade: dict[str, Any]
    exit_index: int


def normalize_ticker(ticker: str) -> str:
    """Convert ticker format for yfinance."""
    ticker = str(ticker).strip().upper()
    ticker = ticker.replace(".", "-")
    return ticker


def load_universe(path: Path, max_tickers: int | None = None) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Universe file is empty: {path}")

    if "Ticker" in df.columns:
        raw_tickers = df["Ticker"].dropna().tolist()
    elif "Symbol" in df.columns:
        raw_tickers = df["Symbol"].dropna().tolist()
    else:
        raw_tickers = df.iloc[:, 0].dropna().tolist()

    tickers = []
    seen = set()
    for t in raw_tickers:
        ticker = normalize_ticker(t)
        if ticker and ticker not in seen:
            tickers.append(ticker)
            seen.add(ticker)

    if max_tickers:
        tickers = tickers[:max_tickers]

    return tickers


def load_sector_cache(cache_path: Path) -> dict[str, dict[str, str]]:
    if not cache_path.exists():
        return {}
    try:
        df = pd.read_csv(cache_path)
    except Exception:
        return {}
    required = {"Ticker", "Sector", "Sector ETF"}
    if not required.issubset(df.columns):
        return {}

    cache: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        ticker = normalize_ticker(row.get("Ticker", ""))
        sector = str(row.get("Sector", "")).strip()
        sector_etf = str(row.get("Sector ETF", "")).strip().upper()
        if ticker and sector_etf:
            cache[ticker] = {"Sector": sector, "Sector ETF": sector_etf}
    return cache


def save_sector_cache(cache_path: Path, cache: dict[str, dict[str, str]]) -> None:
    rows = []
    for ticker, data in sorted(cache.items()):
        rows.append(
            {
                "Ticker": ticker,
                "Sector": data.get("Sector", ""),
                "Sector ETF": data.get("Sector ETF", ""),
            }
        )
    pd.DataFrame(rows).to_csv(cache_path, index=False)


def sector_to_etf(sector: str) -> str | None:
    sector = str(sector).strip()
    if not sector:
        return None
    return SECTOR_TO_ETF.get(sector)


def infer_sector_from_universe(path: Path, tickers: list[str]) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}

    ticker_col = None
    for col in ["Ticker", "Symbol", "ticker", "symbol"]:
        if col in df.columns:
            ticker_col = col
            break
    if ticker_col is None:
        ticker_col = df.columns[0]

    sector_col = None
    for col in ["Sector", "sector", "GICS Sector", "GICS_Sector", "Industry Sector"]:
        if col in df.columns:
            sector_col = col
            break

    sector_etf_col = None
    for col in ["Sector ETF", "Sector_ETF", "sector_etf", "ETF", "SectorETF"]:
        if col in df.columns:
            sector_etf_col = col
            break

    result: dict[str, dict[str, str]] = {}
    tickers_set = set(tickers)
    for _, row in df.iterrows():
        ticker = normalize_ticker(row.get(ticker_col, ""))
        if not ticker or ticker not in tickers_set:
            continue
        sector = str(row.get(sector_col, "")).strip() if sector_col else ""
        sector_etf = str(row.get(sector_etf_col, "")).strip().upper() if sector_etf_col else ""
        if not sector_etf and sector:
            sector_etf = sector_to_etf(sector) or ""
        if ticker in TICKER_SECTOR_ETF_OVERRIDES:
            override_sector, override_etf = TICKER_SECTOR_ETF_OVERRIDES[ticker]
            sector = sector or override_sector
            sector_etf = override_etf
        if sector_etf:
            result[ticker] = {"Sector": sector, "Sector ETF": sector_etf}
    return result


def fetch_sector_from_yfinance(ticker: str) -> dict[str, str] | None:
    if ticker in TICKER_SECTOR_ETF_OVERRIDES:
        sector, sector_etf = TICKER_SECTOR_ETF_OVERRIDES[ticker]
        return {"Sector": sector, "Sector ETF": sector_etf}
    try:
        info = yf.Ticker(ticker).info
        sector = str(info.get("sector", "")).strip()
    except Exception:
        return None
    sector_etf = sector_to_etf(sector)
    if sector_etf:
        return {"Sector": sector, "Sector ETF": sector_etf}
    return None


def build_sector_etf_map(universe_path: Path, tickers: list[str], cache_path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}

    for ticker in tickers:
        if ticker in TICKER_SECTOR_ETF_OVERRIDES:
            sector, sector_etf = TICKER_SECTOR_ETF_OVERRIDES[ticker]
            result[ticker] = {"Sector": sector, "Sector ETF": sector_etf}

    result.update(infer_sector_from_universe(universe_path, tickers))

    cache = load_sector_cache(cache_path)
    for ticker in tickers:
        if ticker not in result and ticker in cache:
            result[ticker] = cache[ticker]

    missing = [ticker for ticker in tickers if ticker not in result]
    if missing:
        print(f"Sector ETF missing for {len(missing)} tickers. Fetching sector info with yfinance and caching...")

    for idx, ticker in enumerate(missing, start=1):
        data = fetch_sector_from_yfinance(ticker)
        if data:
            result[ticker] = data
            cache[ticker] = data
        elif UNKNOWN_SECTOR_ACTION == "spy":
            result[ticker] = {"Sector": "Unknown", "Sector ETF": SPY_TICKER}
            cache[ticker] = result[ticker]

        if idx % 25 == 0:
            save_sector_cache(cache_path, cache)

    save_sector_cache(cache_path, cache)
    return result


def flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        # Common yfinance shape for a single ticker: ('Close', 'AAPL')
        new_cols = []
        for col in df.columns:
            first = str(col[0])
            last = str(col[-1])
            if first in {"Open", "High", "Low", "Close", "Adj Close", "Volume"}:
                new_cols.append(first)
            elif last in {"Open", "High", "Low", "Close", "Adj Close", "Volume"}:
                new_cols.append(last)
            else:
                new_cols.append(first)
        df.columns = new_cols

    return df


def download_weekly(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1wk",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    df = flatten_yfinance_columns(df.copy())

    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return pd.DataFrame()

    df = df[needed].copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=needed)

    # Remove zero-volume rows, usually bad data.
    df = df[df["Volume"] > 0]

    return df


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()



def empty_cup_handle_result() -> dict[str, Any]:
    return {
        "Cup Handle Pattern": False,
        "Cup Left Rim Date": "",
        "Cup Bottom Date": "",
        "Cup Handle Start Date": "",
        "Cup Left Rim High": np.nan,
        "Cup Bottom Low": np.nan,
        "Cup Depth %": np.nan,
        "Cup Length Weeks": np.nan,
        "Handle Length Weeks": np.nan,
        "Handle Depth %": np.nan,
        "Handle Low Above Midpoint": False,
        "Right Side Recovery %": np.nan,
    }


def detect_cup_handle_at(df: pd.DataFrame, i: int) -> dict[str, Any]:
    result = empty_cup_handle_result()

    if i <= CUP_MIN_TOTAL_WEEKS + HANDLE_MIN_WEEKS:
        return result

    row = df.iloc[i]
    if not (bool(row.get("Breakout 52W", False)) or bool(row.get("Breakout ATH", False))):
        return result

    breakout_level = row.get("Breakout Level", np.nan)
    if pd.isna(breakout_level) or breakout_level <= 0:
        return result

    lookback_start = max(0, i - CUP_LOOKBACK_MAX_WEEKS)

    for handle_len in range(HANDLE_MIN_WEEKS, HANDLE_MAX_WEEKS + 1):
        handle_start = i - handle_len
        if handle_start <= lookback_start + CUP_MIN_TOTAL_WEEKS:
            continue

        pre_handle = df.iloc[lookback_start:handle_start]
        if pre_handle.empty:
            continue

        highs = pre_handle["High"].to_numpy(dtype=float)
        if np.all(np.isnan(highs)):
            continue

        left_rel_pos = int(np.nanargmax(highs))
        left_pos = lookback_start + left_rel_pos
        left_rim_high = float(df.iloc[left_pos]["High"])

        if not np.isfinite(left_rim_high) or left_rim_high <= 0:
            continue

        if left_rim_high < float(breakout_level) * (1 - CUP_LEFT_RIM_TOLERANCE_PCT / 100):
            continue

        cup_len = i - left_pos
        if cup_len < CUP_MIN_TOTAL_WEEKS or cup_len > CUP_LOOKBACK_MAX_WEEKS:
            continue

        bottom_start = left_pos + 1
        bottom_end = handle_start
        if bottom_end - bottom_start < 4:
            continue

        bottom_segment = df.iloc[bottom_start:bottom_end]
        lows = bottom_segment["Low"].to_numpy(dtype=float)
        if np.all(np.isnan(lows)):
            continue

        bottom_rel_pos = int(np.nanargmin(lows))
        bottom_pos = bottom_start + bottom_rel_pos
        cup_bottom_low = float(df.iloc[bottom_pos]["Low"])

        if not np.isfinite(cup_bottom_low) or cup_bottom_low <= 0:
            continue

        if bottom_pos - left_pos < 2:
            continue
        if handle_start - bottom_pos < 4:
            continue

        cup_depth_pct = (left_rim_high - cup_bottom_low) / left_rim_high * 100
        if cup_depth_pct < CUP_MIN_DEPTH_PCT or cup_depth_pct > CUP_MAX_DEPTH_PCT:
            continue

        right_side = df.iloc[bottom_pos + 1:handle_start]
        if right_side.empty:
            continue

        right_recovery_high = float(right_side["High"].max())
        right_side_recovery_pct = right_recovery_high / left_rim_high * 100
        if right_recovery_high < left_rim_high * (1 - CUP_RIGHT_RIM_TOLERANCE_PCT / 100):
            continue

        handle = df.iloc[handle_start:i]
        if handle.empty:
            continue

        handle_high = float(handle["High"].max())
        handle_low = float(handle["Low"].min())
        if handle_high <= 0 or not np.isfinite(handle_high) or not np.isfinite(handle_low):
            continue

        handle_depth_pct = (handle_high - handle_low) / handle_high * 100

        if handle_depth_pct > HANDLE_MAX_DEPTH_PCT:
            continue

        if handle_depth_pct > cup_depth_pct * HANDLE_MAX_CUP_DEPTH_RATIO:
            continue

        cup_midpoint = cup_bottom_low + (left_rim_high - cup_bottom_low) * 0.50
        handle_low_above_midpoint = handle_low > cup_midpoint
        if REQUIRE_HANDLE_LOW_ABOVE_CUP_MIDPOINT and not handle_low_above_midpoint:
            continue

        return {
            "Cup Handle Pattern": True,
            "Cup Left Rim Date": str(pd.Timestamp(df.index[left_pos]).date()),
            "Cup Bottom Date": str(pd.Timestamp(df.index[bottom_pos]).date()),
            "Cup Handle Start Date": str(pd.Timestamp(df.index[handle_start]).date()),
            "Cup Left Rim High": left_rim_high,
            "Cup Bottom Low": cup_bottom_low,
            "Cup Depth %": cup_depth_pct,
            "Cup Length Weeks": cup_len,
            "Handle Length Weeks": handle_len,
            "Handle Depth %": handle_depth_pct,
            "Handle Low Above Midpoint": bool(handle_low_above_midpoint),
            "Right Side Recovery %": right_side_recovery_pct,
        }

    return result


def add_cup_handle_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    records = [detect_cup_handle_at(out, i) for i in range(len(out))]
    ch = pd.DataFrame(records, index=out.index)
    for col in ch.columns:
        out[col] = ch[col]
    return out


def add_base_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["EMA9"] = ema(out["Close"], EMA_FAST)
    out["EMA20"] = ema(out["Close"], EMA_SLOW)
    out["SMA20"] = out["Close"].rolling(20).mean()
    out["SMA50"] = out["Close"].rolling(SMA_MID).mean()
    out["SMA200"] = out["Close"].rolling(SMA_LONG).mean()
    out["SMA20 vs SMA50 %"] = (out["SMA20"] / out["SMA50"] - 1) * 100
    out["Volume SMA20"] = out["Volume"].rolling(20).mean()
    out["RSI14"] = rsi(out["Close"], 14)
    out["ADX14"] = adx(out, 14)

    # STRICT breakout definition:
    # Previous version used highest weekly CLOSE. That can create false breakouts.
    # This version requires the weekly CLOSE to break above the previous weekly HIGH level.
    out["Prior 52W High"] = out["High"].shift(1).rolling(BREAKOUT_LOOKBACK_WEEKS).max()
    out["Prior ATH High"] = out["High"].shift(1).cummax()

    out["Breakout 52W"] = out["Close"] > out["Prior 52W High"]
    out["Breakout ATH"] = out["Close"] > out["Prior ATH High"]
    out["Breakout Type"] = np.where(
        out["Breakout ATH"],
        "ATH",
        np.where(out["Breakout 52W"], "52W", ""),
    )

    out["Breakout Level"] = np.where(
        out["Breakout ATH"],
        out["Prior ATH High"],
        np.where(out["Breakout 52W"], out["Prior 52W High"], np.nan),
    )
    out["Close Above Breakout Level %"] = (out["Close"] / out["Breakout Level"] - 1) * 100
    out["High Above Breakout Level %"] = (out["High"] / out["Breakout Level"] - 1) * 100

    out["Breakout Volume Ratio"] = out["Volume"] / out["Volume SMA20"]

    out["Signal Candle Range %"] = (out["High"] - out["Low"]) / out["Close"]
    out["Signal Candle Body %"] = (out["Close"] - out["Open"]).abs() / out["Close"]
    out["Signal Candle Green"] = out["Close"] > out["Open"]
    out["Signal Candle Close Position"] = (out["Close"] - out["Low"]) / (out["High"] - out["Low"]).replace(0, np.nan)

    out["Big Green Candle"] = (
        out["Signal Candle Green"]
        & ((out["Signal Candle Range %"] * 100) >= BIG_GREEN_MIN_RANGE_PCT)
        & ((out["Signal Candle Body %"] * 100) >= BIG_GREEN_MIN_BODY_PCT)
    )
    out["Very High Volume Breakout"] = out["Breakout Volume Ratio"] >= VERY_HIGH_VOLUME_RATIO
    out["Avoid Big Green Or Very High Volume"] = (
        out["Big Green Candle"] | out["Very High Volume Breakout"]
    )

    # Backward-compatible debug column name.
    out["Big Green High Volume Candle"] = out["Avoid Big Green Or Very High Volume"]

    out = add_cup_handle_columns(out)

    return out


def add_weekly_uptrend_flag(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = add_base_indicators(df)
    out[f"{label} Weekly Uptrend"] = (
        (out["Close"] > out["EMA9"])
        & (out["EMA9"] > out["EMA20"])
        & (out["EMA20"] > out["SMA50"])
    )
    return out


def prepare_etf_data(spy: pd.DataFrame, qqq: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    spy = add_weekly_uptrend_flag(spy, "SPY")
    qqq = add_weekly_uptrend_flag(qqq, "QQQ")
    return spy, qqq


def prepare_sector_etf_data(etf_raw: pd.DataFrame, sector_etf: str) -> pd.DataFrame:
    return add_weekly_uptrend_flag(etf_raw, sector_etf)


def add_relative_strength(
    stock: pd.DataFrame,
    spy: pd.DataFrame,
    qqq: pd.DataFrame,
    sector_df: pd.DataFrame,
    sector_name: str,
    sector_etf: str,
) -> pd.DataFrame:
    out = add_base_indicators(stock)

    # Broad market / SPY confirmation
    out["SPY Close"] = spy["Close"].reindex(out.index).ffill()
    out["SPY EMA9"] = spy["EMA9"].reindex(out.index).ffill()
    out["SPY EMA20"] = spy["EMA20"].reindex(out.index).ffill()
    out["SPY SMA50"] = spy["SMA50"].reindex(out.index).ffill()
    out["SPY Weekly Uptrend"] = spy["SPY Weekly Uptrend"].reindex(out.index).ffill().astype(bool)

    # QQQ confirmation from original weekly strategy
    out["QQQ Close"] = qqq["Close"].reindex(out.index).ffill()
    out["QQQ EMA9"] = qqq["EMA9"].reindex(out.index).ffill()
    out["QQQ EMA20"] = qqq["EMA20"].reindex(out.index).ffill()
    out["QQQ SMA50"] = qqq["SMA50"].reindex(out.index).ffill()
    out["QQQ Weekly Uptrend"] = qqq["QQQ Weekly Uptrend"].reindex(out.index).ffill().astype(bool)

    # Sector ETF confirmation
    sector_uptrend_col = f"{sector_etf} Weekly Uptrend"
    out["Sector"] = sector_name
    out["Sector ETF"] = sector_etf
    out["Sector ETF Close"] = sector_df["Close"].reindex(out.index).ffill()
    out["Sector ETF EMA9"] = sector_df["EMA9"].reindex(out.index).ffill()
    out["Sector ETF EMA20"] = sector_df["EMA20"].reindex(out.index).ffill()
    out["Sector ETF SMA50"] = sector_df["SMA50"].reindex(out.index).ffill()
    out["Sector ETF Weekly Uptrend"] = sector_df[sector_uptrend_col].reindex(out.index).ffill().astype(bool)

    # Stock vs SPY relative strength from original weekly strategy
    out["Stock 5W Return"] = out["Close"].pct_change(RS_LOOKBACK_WEEKS)
    out["SPY 5W Return"] = out["SPY Close"].pct_change(RS_LOOKBACK_WEEKS)
    out["RS 5W vs SPY"] = out["Stock 5W Return"] - out["SPY 5W Return"]
    out["RS Ratio"] = out["Close"] / out["SPY Close"]
    out["RS Ratio ROC 5W"] = out["RS Ratio"].pct_change(RS_LOOKBACK_WEEKS)

    # Sector vs SPY relative strength: this is the new sector-SPY check.
    out["Sector 5W Return"] = out["Sector ETF Close"].pct_change(RS_LOOKBACK_WEEKS)
    out["Sector RS 5W vs SPY"] = out["Sector 5W Return"] - out["SPY 5W Return"]
    out["Sector RS Ratio"] = out["Sector ETF Close"] / out["SPY Close"]
    out["Sector RS Ratio ROC 5W"] = out["Sector RS Ratio"].pct_change(RS_LOOKBACK_WEEKS)

    # Extra debug: stock vs its sector ETF. Not required by default.
    out["Stock RS 5W vs Sector"] = out["Stock 5W Return"] - out["Sector 5W Return"]
    out["Stock/Sector RS Ratio"] = out["Close"] / out["Sector ETF Close"]
    out["Stock/Sector RS Ratio ROC 5W"] = out["Stock/Sector RS Ratio"].pct_change(RS_LOOKBACK_WEEKS)

    return out


def is_finite_values(row: pd.Series, columns: list[str]) -> bool:
    for col in columns:
        val = row.get(col, np.nan)
        if pd.isna(val) or not np.isfinite(float(val)):
            return False
    return True


def setup_passes(row: pd.Series) -> bool:
    required = [
        "Close", "High", "Low", "EMA9", "EMA20", "SMA20", "SMA50", "SMA200", "SMA20 vs SMA50 %",
        "Volume SMA20", "Breakout Volume Ratio",
        "RS 5W vs SPY", "RS Ratio ROC 5W",
        "QQQ Close", "QQQ EMA9", "QQQ EMA20", "QQQ SMA50",
        "SPY Close", "SPY EMA9", "SPY EMA20", "SPY SMA50",
        "Sector ETF Close", "Sector ETF EMA9", "Sector ETF EMA20", "Sector ETF SMA50",
        "Sector RS 5W vs SPY", "Sector RS Ratio ROC 5W",
        "Stock RS 5W vs Sector", "Stock/Sector RS Ratio ROC 5W",
    ]
    if not is_finite_values(row, required):
        return False

    # 1. Original QQQ weekly uptrend filter
    qqq_uptrend = (
        row["QQQ Close"] > row["QQQ EMA9"]
        and row["QQQ EMA9"] > row["QQQ EMA20"]
        and row["QQQ EMA20"] > row["QQQ SMA50"]
    )
    if not qqq_uptrend:
        return False

    # 2. market_etf_strong = True, using SPY
    market_etf_strong = (
        row["SPY Close"] > row["SPY EMA9"]
        and row["SPY EMA9"] > row["SPY EMA20"]
        and row["SPY EMA20"] > row["SPY SMA50"]
    )
    if REQUIRE_MARKET_ETF_STRONG and not market_etf_strong:
        return False

    # 3. stock_strong = True
    # Keep the existing weekly trend rule, and add requested SMA20 structure:
    # SMA20 must be above SMA50, but not more than 15% above SMA50.
    stock_strong = (
        row["Close"] > row["EMA9"]
        and row["EMA9"] > row["EMA20"]
        and row["EMA20"] > row["SMA50"]
        and row["SMA50"] > row["SMA200"]
        and row["SMA20"] > row["SMA50"]
        and row["SMA20 vs SMA50 %"] < MAX_SMA20_ABOVE_SMA50_PCT
    )
    if REQUIRE_STOCK_STRONG and not stock_strong:
        return False

    # 4. sector_etf_strong = True
    sector_etf_strong = (
        row["Sector ETF Close"] > row["Sector ETF EMA9"]
        and row["Sector ETF EMA9"] > row["Sector ETF EMA20"]
        and row["Sector ETF EMA20"] > row["Sector ETF SMA50"]
    )
    if REQUIRE_SECTOR_ETF_STRONG and not sector_etf_strong:
        return False

    # 5. Original stock vs SPY relative strength
    if row["RS 5W vs SPY"] <= MIN_RS_VS_SPY:
        return False
    if row["RS Ratio ROC 5W"] <= MIN_RS_RATIO_ROC:
        return False

    # 6. New sector vs SPY relative strength
    if REQUIRE_SECTOR_RS_VS_SPY:
        if row["Sector RS 5W vs SPY"] <= MIN_SECTOR_RS_VS_SPY:
            return False
        if row["Sector RS Ratio ROC 5W"] <= MIN_SECTOR_RS_RATIO_ROC:
            return False

    # 7. Optional stock vs sector relative strength
    if REQUIRE_STOCK_RS_VS_SECTOR:
        if row["Stock RS 5W vs Sector"] <= MIN_STOCK_RS_VS_SECTOR:
            return False
        if row["Stock/Sector RS Ratio ROC 5W"] <= MIN_STOCK_SECTOR_RATIO_ROC:
            return False

    # 8. Weekly breakout
    if not (bool(row["Breakout 52W"]) or bool(row["Breakout ATH"])):
        return False

    # 8b. Cup-and-handle pattern before breakout.
    if REQUIRE_CUP_HANDLE_PATTERN and not bool(row.get("Cup Handle Pattern", False)):
        return False

    # 9. Breakout volume
    if row["Breakout Volume Ratio"] < MIN_BREAKOUT_VOLUME_RATIO:
        return False

    # 10. Avoid either:
    # - very big green breakout candle, OR
    # - very high breakout volume >= 4.0x.
    if AVOID_BIG_GREEN_OR_VERY_HIGH_VOLUME and bool(row.get("Avoid Big Green Or Very High Volume", False)):
        return False

    # Optional liquidity filter
    if MIN_20W_AVG_VOLUME is not None and row["Volume SMA20"] < MIN_20W_AVG_VOLUME:
        return False

    return True


def trend_cross_down(df: pd.DataFrame, j: int) -> bool:
    if j <= 0:
        return False

    prev = df.iloc[j - 1]
    row = df.iloc[j]

    if pd.isna(prev["EMA9"]) or pd.isna(prev["EMA20"]) or pd.isna(row["EMA9"]) or pd.isna(row["EMA20"]):
        return False

    return prev["EMA9"] >= prev["EMA20"] and row["EMA9"] < row["EMA20"]


def simulate_trade(ticker: str, df: pd.DataFrame, signal_i: int) -> TradeResult | None:
    signal = df.iloc[signal_i]
    if signal_i + 1 >= len(df):
        return None

    next_week = df.iloc[signal_i + 1]

    trigger_price = signal["High"] * (1 + ENTRY_TRIGGER_BUFFER_PCT)
    stop_price = signal["Low"] * (1 - STOP_BUFFER_PCT)

    # Next week must break above breakout candle high.
    if next_week["High"] < trigger_price:
        return None

    if USE_GAP_AWARE_ENTRY:
        entry_price = max(float(trigger_price), float(next_week["Open"]))
    else:
        entry_price = float(trigger_price)

    initial_risk_pct = (entry_price - stop_price) / entry_price
    if not np.isfinite(initial_risk_pct) or initial_risk_pct <= 0:
        return None

    entry_i = signal_i + 1
    exit_price = np.nan
    exit_date = None
    exit_reason = ""
    exit_i = len(df) - 1

    for j in range(entry_i, len(df)):
        row = df.iloc[j]

        # Conservative rule:
        # If entry and stop both happen inside the same weekly candle, count stop loss.
        if row["Low"] <= stop_price:
            exit_price = float(stop_price)
            exit_date = row.name
            exit_reason = "Stop Loss"
            exit_i = j
            break

        if j > entry_i and trend_cross_down(df, j):
            exit_price = float(row["Close"])
            exit_date = row.name
            exit_reason = "Weekly 9 EMA crossed below 20 EMA"
            exit_i = j
            break

    if exit_reason == "":
        final = df.iloc[-1]
        exit_price = float(final["Close"])
        exit_date = final.name
        exit_reason = "End of Data"
        exit_i = len(df) - 1

    pl_pct = (exit_price - entry_price) / entry_price
    r_multiple = pl_pct / initial_risk_pct if initial_risk_pct > 0 else np.nan

    signal_range_pct = (signal["High"] - signal["Low"]) / signal["Close"] if signal["Close"] else np.nan
    entry_gap_pct = (entry_price - trigger_price) / trigger_price if trigger_price else np.nan

    trade = {
        "Ticker": ticker,
        "Signal Date": signal.name.date(),
        "Entry Date": df.iloc[entry_i].name.date(),
        "Exit Date": pd.Timestamp(exit_date).date(),
        "Breakout Type": signal["Breakout Type"],
        "Prior 52W High": float(signal["Prior 52W High"]) if pd.notna(signal["Prior 52W High"]) else np.nan,
        "Prior ATH High": float(signal["Prior ATH High"]) if pd.notna(signal["Prior ATH High"]) else np.nan,
        "Breakout Level": float(signal["Breakout Level"]) if pd.notna(signal["Breakout Level"]) else np.nan,
        "Close Above Breakout Level %": float(signal["Close Above Breakout Level %"]) if pd.notna(signal["Close Above Breakout Level %"]) else np.nan,
        "High Above Breakout Level %": float(signal["High Above Breakout Level %"]) if pd.notna(signal["High Above Breakout Level %"]) else np.nan,
        "Signal Open": float(signal["Open"]),
        "Signal Close": float(signal["Close"]),
        "Signal High": float(signal["High"]),
        "Signal Low": float(signal["Low"]),
        "SMA20": float(signal["SMA20"]) if pd.notna(signal["SMA20"]) else np.nan,
        "SMA50": float(signal["SMA50"]) if pd.notna(signal["SMA50"]) else np.nan,
        "SMA20 vs SMA50 %": float(signal["SMA20 vs SMA50 %"]) if pd.notna(signal["SMA20 vs SMA50 %"]) else np.nan,
        "Cup Handle Pattern": bool(signal.get("Cup Handle Pattern", False)),
        "Cup Left Rim Date": signal.get("Cup Left Rim Date", ""),
        "Cup Bottom Date": signal.get("Cup Bottom Date", ""),
        "Cup Handle Start Date": signal.get("Cup Handle Start Date", ""),
        "Cup Left Rim High": float(signal.get("Cup Left Rim High", np.nan)) if pd.notna(signal.get("Cup Left Rim High", np.nan)) else np.nan,
        "Cup Bottom Low": float(signal.get("Cup Bottom Low", np.nan)) if pd.notna(signal.get("Cup Bottom Low", np.nan)) else np.nan,
        "Cup Depth %": float(signal.get("Cup Depth %", np.nan)) if pd.notna(signal.get("Cup Depth %", np.nan)) else np.nan,
        "Cup Length Weeks": float(signal.get("Cup Length Weeks", np.nan)) if pd.notna(signal.get("Cup Length Weeks", np.nan)) else np.nan,
        "Handle Length Weeks": float(signal.get("Handle Length Weeks", np.nan)) if pd.notna(signal.get("Handle Length Weeks", np.nan)) else np.nan,
        "Handle Depth %": float(signal.get("Handle Depth %", np.nan)) if pd.notna(signal.get("Handle Depth %", np.nan)) else np.nan,
        "Handle Low Above Midpoint": bool(signal.get("Handle Low Above Midpoint", False)),
        "Right Side Recovery %": float(signal.get("Right Side Recovery %", np.nan)) if pd.notna(signal.get("Right Side Recovery %", np.nan)) else np.nan,
        "Entry Trigger Price": float(trigger_price),
        "Entry Price": float(entry_price),
        "Stop Price": float(stop_price),
        "Exit Price": float(exit_price),
        "Exit Reason": exit_reason,
        "P/L %": pl_pct * 100,
        "Initial Risk %": initial_risk_pct * 100,
        "R Multiple": r_multiple,
        "Holding Weeks": int(exit_i - entry_i + 1),
        "QQQ Close": float(signal["QQQ Close"]),
        "SPY Weekly Uptrend": bool(signal["SPY Weekly Uptrend"]),
        "Sector": signal["Sector"],
        "Sector ETF": signal["Sector ETF"],
        "Sector ETF Close": float(signal["Sector ETF Close"]),
        "Sector ETF Weekly Uptrend": bool(signal["Sector ETF Weekly Uptrend"]),
        "Sector 5W Return": float(signal["Sector 5W Return"]),
        "Sector RS 5W vs SPY": float(signal["Sector RS 5W vs SPY"]),
        "Sector RS Ratio ROC 5W": float(signal["Sector RS Ratio ROC 5W"]),
        "Stock RS 5W vs Sector": float(signal["Stock RS 5W vs Sector"]),
        "Stock/Sector RS Ratio ROC 5W": float(signal["Stock/Sector RS Ratio ROC 5W"]),
        "RS 5W vs SPY": float(signal["RS 5W vs SPY"]),
        "RS Ratio ROC 5W": float(signal["RS Ratio ROC 5W"]),
        "Stock 5W Return": float(signal["Stock 5W Return"]),
        "SPY 5W Return": float(signal["SPY 5W Return"]),
        "Breakout Volume Ratio": float(signal["Breakout Volume Ratio"]),
        "Weekly Volume": float(signal["Volume"]),
        "20W Avg Volume": float(signal["Volume SMA20"]),
        "Weekly RSI14": float(signal["RSI14"]) if pd.notna(signal["RSI14"]) else np.nan,
        "Weekly ADX14": float(signal["ADX14"]) if pd.notna(signal["ADX14"]) else np.nan,
        "Signal Candle Range %": signal_range_pct * 100,
        "Signal Candle Body %": float(signal["Signal Candle Body %"] * 100) if pd.notna(signal["Signal Candle Body %"]) else np.nan,
        "Signal Candle Green": bool(signal["Signal Candle Green"]) if pd.notna(signal["Signal Candle Green"]) else False,
        "Big Green Candle": bool(signal["Big Green Candle"]) if pd.notna(signal["Big Green Candle"]) else False,
        "Very High Volume Breakout": bool(signal["Very High Volume Breakout"]) if pd.notna(signal["Very High Volume Breakout"]) else False,
        "Avoid Big Green Or Very High Volume": bool(signal["Avoid Big Green Or Very High Volume"]) if pd.notna(signal["Avoid Big Green Or Very High Volume"]) else False,
        "Big Green High Volume Candle": bool(signal["Big Green High Volume Candle"]) if pd.notna(signal["Big Green High Volume Candle"]) else False,
        "Signal Candle Close Position": float(signal["Signal Candle Close Position"]) if pd.notna(signal["Signal Candle Close Position"]) else np.nan,
        "Entry Gap %": entry_gap_pct * 100,
    }

    return TradeResult(trade=trade, exit_index=exit_i)


def backtest_ticker(ticker: str, df: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trades: list[dict[str, Any]] = []
    setups: list[dict[str, Any]] = []

    # Need enough history for SMA200 and prior ATH/52W high.
    i = max(SMA_LONG, BREAKOUT_LOOKBACK_WEEKS, 20, RS_LOOKBACK_WEEKS) + 1

    while i < len(df) - 1:
        row = df.iloc[i]

        if not setup_passes(row):
            i += 1
            continue

        trigger_price = row["High"] * (1 + ENTRY_TRIGGER_BUFFER_PCT)
        stop_price = row["Low"] * (1 - STOP_BUFFER_PCT)
        next_week = df.iloc[i + 1]
        entry_triggered = bool(next_week["High"] >= trigger_price)

        setup_record = {
            "Ticker": ticker,
            "Signal Date": row.name.date(),
            "Breakout Type": row["Breakout Type"],
            "Prior 52W High": float(row["Prior 52W High"]) if pd.notna(row["Prior 52W High"]) else np.nan,
            "Prior ATH High": float(row["Prior ATH High"]) if pd.notna(row["Prior ATH High"]) else np.nan,
            "Breakout Level": float(row["Breakout Level"]) if pd.notna(row["Breakout Level"]) else np.nan,
            "Close Above Breakout Level %": float(row["Close Above Breakout Level %"]) if pd.notna(row["Close Above Breakout Level %"]) else np.nan,
            "High Above Breakout Level %": float(row["High Above Breakout Level %"]) if pd.notna(row["High Above Breakout Level %"]) else np.nan,
            "Signal Open": float(row["Open"]),
            "Signal Close": float(row["Close"]),
            "Signal High": float(row["High"]),
            "Signal Low": float(row["Low"]),
            "SMA20": float(row["SMA20"]) if pd.notna(row["SMA20"]) else np.nan,
            "SMA50": float(row["SMA50"]) if pd.notna(row["SMA50"]) else np.nan,
            "SMA20 vs SMA50 %": float(row["SMA20 vs SMA50 %"]) if pd.notna(row["SMA20 vs SMA50 %"]) else np.nan,
            "Cup Handle Pattern": bool(row.get("Cup Handle Pattern", False)),
            "Cup Left Rim Date": row.get("Cup Left Rim Date", ""),
            "Cup Bottom Date": row.get("Cup Bottom Date", ""),
            "Cup Handle Start Date": row.get("Cup Handle Start Date", ""),
            "Cup Left Rim High": float(row.get("Cup Left Rim High", np.nan)) if pd.notna(row.get("Cup Left Rim High", np.nan)) else np.nan,
            "Cup Bottom Low": float(row.get("Cup Bottom Low", np.nan)) if pd.notna(row.get("Cup Bottom Low", np.nan)) else np.nan,
            "Cup Depth %": float(row.get("Cup Depth %", np.nan)) if pd.notna(row.get("Cup Depth %", np.nan)) else np.nan,
            "Cup Length Weeks": float(row.get("Cup Length Weeks", np.nan)) if pd.notna(row.get("Cup Length Weeks", np.nan)) else np.nan,
            "Handle Length Weeks": float(row.get("Handle Length Weeks", np.nan)) if pd.notna(row.get("Handle Length Weeks", np.nan)) else np.nan,
            "Handle Depth %": float(row.get("Handle Depth %", np.nan)) if pd.notna(row.get("Handle Depth %", np.nan)) else np.nan,
            "Handle Low Above Midpoint": bool(row.get("Handle Low Above Midpoint", False)),
            "Right Side Recovery %": float(row.get("Right Side Recovery %", np.nan)) if pd.notna(row.get("Right Side Recovery %", np.nan)) else np.nan,
            "Entry Trigger Price": float(trigger_price),
            "Stop Price": float(stop_price),
            "Entry Triggered": entry_triggered,
            "Next Week Date": next_week.name.date(),
            "Next Week Open": float(next_week["Open"]),
            "Next Week High": float(next_week["High"]),
            "Next Week Low": float(next_week["Low"]),
            "Next Week Close": float(next_week["Close"]),
            "SPY Weekly Uptrend": bool(row["SPY Weekly Uptrend"]),
            "Sector": row["Sector"],
            "Sector ETF": row["Sector ETF"],
            "Sector ETF Close": float(row["Sector ETF Close"]),
            "Sector ETF Weekly Uptrend": bool(row["Sector ETF Weekly Uptrend"]),
            "Sector 5W Return": float(row["Sector 5W Return"]),
            "Sector RS 5W vs SPY": float(row["Sector RS 5W vs SPY"]),
            "Sector RS Ratio ROC 5W": float(row["Sector RS Ratio ROC 5W"]),
            "Stock RS 5W vs Sector": float(row["Stock RS 5W vs Sector"]),
            "Stock/Sector RS Ratio ROC 5W": float(row["Stock/Sector RS Ratio ROC 5W"]),
            "RS 5W vs SPY": float(row["RS 5W vs SPY"]),
            "RS Ratio ROC 5W": float(row["RS Ratio ROC 5W"]),
            "Stock 5W Return": float(row["Stock 5W Return"]),
            "SPY 5W Return": float(row["SPY 5W Return"]),
            "Breakout Volume Ratio": float(row["Breakout Volume Ratio"]),
            "Weekly Volume": float(row["Volume"]),
            "20W Avg Volume": float(row["Volume SMA20"]),
            "Weekly RSI14": float(row["RSI14"]) if pd.notna(row["RSI14"]) else np.nan,
            "Weekly ADX14": float(row["ADX14"]) if pd.notna(row["ADX14"]) else np.nan,
            "Signal Candle Range %": float(row["Signal Candle Range %"] * 100) if pd.notna(row["Signal Candle Range %"]) else np.nan,
            "Signal Candle Body %": float(row["Signal Candle Body %"] * 100) if pd.notna(row["Signal Candle Body %"]) else np.nan,
            "Signal Candle Green": bool(row["Signal Candle Green"]) if pd.notna(row["Signal Candle Green"]) else False,
            "Big Green Candle": bool(row["Big Green Candle"]) if pd.notna(row["Big Green Candle"]) else False,
            "Very High Volume Breakout": bool(row["Very High Volume Breakout"]) if pd.notna(row["Very High Volume Breakout"]) else False,
            "Avoid Big Green Or Very High Volume": bool(row["Avoid Big Green Or Very High Volume"]) if pd.notna(row["Avoid Big Green Or Very High Volume"]) else False,
            "Big Green High Volume Candle": bool(row["Big Green High Volume Candle"]) if pd.notna(row["Big Green High Volume Candle"]) else False,
            "Signal Candle Close Position": float(row["Signal Candle Close Position"]) if pd.notna(row["Signal Candle Close Position"]) else np.nan,
        }

        result = simulate_trade(ticker, df, i)

        if result is not None:
            trade = result.trade
            trades.append(trade)
            setup_record.update(
                {
                    "Entry Date": trade["Entry Date"],
                    "Exit Date": trade["Exit Date"],
                    "Exit Reason": trade["Exit Reason"],
                    "P/L %": trade["P/L %"],
                    "Initial Risk %": trade["Initial Risk %"],
                    "R Multiple": trade["R Multiple"],
                    "Holding Weeks": trade["Holding Weeks"],
                }
            )
            setups.append(setup_record)

            # No overlapping trades on the same ticker.
            i = result.exit_index + 1
            continue

        setups.append(setup_record)
        i += 1

    return trades, setups


def profit_factor(pl_series: pd.Series) -> float:
    gains = pl_series[pl_series > 0].sum()
    losses = pl_series[pl_series < 0].sum()
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return gains / abs(losses)


def summarize_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(
            [
                {
                    "Strategy": OUTPUT_PREFIX,
                    "Total Trades": 0,
                    "Win Rate %": np.nan,
                    "Average P/L %": np.nan,
                    "Median P/L %": np.nan,
                    "Profit Factor": np.nan,
                    "Average R": np.nan,
                    "Median R": np.nan,
                    "Average Initial Risk %": np.nan,
                    "Average Holding Weeks": np.nan,
                    "Biggest Winner %": np.nan,
                    "Biggest Loser %": np.nan,
                }
            ]
        )

    wins = trades_df["P/L %"] > 0
    summary = {
        "Strategy": OUTPUT_PREFIX,
        "Total Trades": int(len(trades_df)),
        "Winners": int(wins.sum()),
        "Losers": int((~wins).sum()),
        "Win Rate %": float(wins.mean() * 100),
        "Average P/L %": float(trades_df["P/L %"].mean()),
        "Median P/L %": float(trades_df["P/L %"].median()),
        "Profit Factor": float(profit_factor(trades_df["P/L %"])),
        "Average R": float(trades_df["R Multiple"].mean()),
        "Median R": float(trades_df["R Multiple"].median()),
        "Average Initial Risk %": float(trades_df["Initial Risk %"].mean()),
        "Median Initial Risk %": float(trades_df["Initial Risk %"].median()),
        "Average Holding Weeks": float(trades_df["Holding Weeks"].mean()),
        "Median Holding Weeks": float(trades_df["Holding Weeks"].median()),
        "Biggest Winner %": float(trades_df["P/L %"].max()),
        "Biggest Loser %": float(trades_df["P/L %"].min()),
        "First Entry Date": trades_df["Entry Date"].min(),
        "Last Entry Date": trades_df["Entry Date"].max(),
    }

    return pd.DataFrame([summary])


def grouped_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    if df.empty or group_col not in df.columns:
        return pd.DataFrame()

    for name, group in df.groupby(group_col, dropna=False):
        if group.empty:
            continue

        wins = group["P/L %"] > 0
        rows.append(
            {
                "Bucket Type": group_col,
                "Bucket": str(name),
                "Trades": int(len(group)),
                "Win Rate %": float(wins.mean() * 100),
                "Average P/L %": float(group["P/L %"].mean()),
                "Median P/L %": float(group["P/L %"].median()),
                "Profit Factor": float(profit_factor(group["P/L %"])),
                "Average R": float(group["R Multiple"].mean()),
                "Average Initial Risk %": float(group["Initial Risk %"].mean()),
                "Average Holding Weeks": float(group["Holding Weeks"].mean()),
            }
        )

    return pd.DataFrame(rows)


def add_trade_buckets(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df

    out = trades_df.copy()

    out["Initial Risk Bucket"] = pd.cut(
        out["Initial Risk %"],
        bins=[-np.inf, 5, 10, 15, 20, np.inf],
        labels=["0-5%", "5-10%", "10-15%", "15-20%", ">20%"],
    )

    out["Breakout Volume Bucket"] = pd.cut(
        out["Breakout Volume Ratio"],
        bins=[-np.inf, 1.2, 1.5, 2.0, 3.0, np.inf],
        labels=["<1.2x", "1.2-1.5x", "1.5-2.0x", "2.0-3.0x", ">3.0x"],
    )

    out["RS 5W vs SPY Bucket"] = pd.cut(
        out["RS 5W vs SPY"] * 100,
        bins=[-np.inf, 5, 10, 15, 20, 30, np.inf],
        labels=["<=5%", "5-10%", "10-15%", "15-20%", "20-30%", ">30%"],
    )

    out["RS Ratio ROC 5W Bucket"] = pd.cut(
        out["RS Ratio ROC 5W"] * 100,
        bins=[-np.inf, 5, 10, 15, 20, 30, np.inf],
        labels=["<=5%", "5-10%", "10-15%", "15-20%", "20-30%", ">30%"],
    )

    out["Sector RS 5W vs SPY Bucket"] = pd.cut(
        out["Sector RS 5W vs SPY"] * 100,
        bins=[-np.inf, 0, 2, 5, 10, np.inf],
        labels=["<=0%", "0-2%", "2-5%", "5-10%", ">10%"],
    )

    out["Sector RS Ratio ROC 5W Bucket"] = pd.cut(
        out["Sector RS Ratio ROC 5W"] * 100,
        bins=[-np.inf, 0, 2, 5, 10, np.inf],
        labels=["<=0%", "0-2%", "2-5%", "5-10%", ">10%"],
    )

    out["Stock RS 5W vs Sector Bucket"] = pd.cut(
        out["Stock RS 5W vs Sector"] * 100,
        bins=[-np.inf, 0, 2, 5, 10, 20, np.inf],
        labels=["<=0%", "0-2%", "2-5%", "5-10%", "10-20%", ">20%"],
    )

    out["Weekly ADX Bucket"] = pd.cut(
        out["Weekly ADX14"],
        bins=[-np.inf, 20, 25, 30, 35, 40, 50, np.inf],
        labels=["<20", "20-25", "25-30", "30-35", "35-40", "40-50", ">50"],
    )

    out["SMA20 vs SMA50 Bucket"] = pd.cut(
        out["SMA20 vs SMA50 %"],
        bins=[-np.inf, 0, 5, 10, 15, 20, np.inf],
        labels=["<=0%", "0-5%", "5-10%", "10-15%", "15-20%", ">20%"],
    )

    out["Cup Depth Bucket"] = pd.cut(
        out["Cup Depth %"],
        bins=[-np.inf, 12, 20, 30, 40, 50, np.inf],
        labels=["<12%", "12-20%", "20-30%", "30-40%", "40-50%", ">50%"],
    )

    out["Cup Length Bucket"] = pd.cut(
        out["Cup Length Weeks"],
        bins=[-np.inf, 12, 20, 35, 52, 78, 104, np.inf],
        labels=["<12w", "12-20w", "20-35w", "35-52w", "52-78w", "78-104w", ">104w"],
    )

    out["Handle Depth Bucket"] = pd.cut(
        out["Handle Depth %"],
        bins=[-np.inf, 3, 5, 8, 10, 15, np.inf],
        labels=["0-3%", "3-5%", "5-8%", "8-10%", "10-15%", ">15%"],
    )

    out["Signal Candle Range Bucket"] = pd.cut(
        out["Signal Candle Range %"],
        bins=[-np.inf, 5, 10, 15, 20, np.inf],
        labels=["0-5%", "5-10%", "10-15%", "15-20%", ">20%"],
    )

    out["Signal Candle Body Bucket"] = pd.cut(
        out["Signal Candle Body %"],
        bins=[-np.inf, 5, 10, 15, 20, np.inf],
        labels=["0-5%", "5-10%", "10-15%", "15-20%", ">20%"],
    )

    out["Big Green Or Very High Volume Bucket"] = np.where(
        out["Avoid Big Green Or Very High Volume"],
        "Rejected candle/volume pattern",
        "Normal",
    )

    out["Signal Close Position Bucket"] = pd.cut(
        out["Signal Candle Close Position"],
        bins=[-np.inf, 0.5, 0.7, 0.85, np.inf],
        labels=["<50%", "50-70%", "70-85%", "85-100%"],
    )

    out["Entry Gap Bucket"] = pd.cut(
        out["Entry Gap %"],
        bins=[-np.inf, 0, 1, 3, 5, np.inf],
        labels=["<=0%", "0-1%", "1-3%", "3-5%", ">5%"],
    )

    return out


def create_bucket_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()

    bucket_cols = [
        "Initial Risk Bucket",
        "Breakout Volume Bucket",
        "RS 5W vs SPY Bucket",
        "RS Ratio ROC 5W Bucket",
        "Sector ETF",
        "Sector RS 5W vs SPY Bucket",
        "Sector RS Ratio ROC 5W Bucket",
        "Stock RS 5W vs Sector Bucket",
        "Weekly ADX Bucket",
        "SMA20 vs SMA50 Bucket",
        "Cup Depth Bucket",
        "Cup Length Bucket",
        "Handle Depth Bucket",
        "Signal Candle Range Bucket",
        "Signal Candle Body Bucket",
        "Big Green Or Very High Volume Bucket",
        "Signal Close Position Bucket",
        "Entry Gap Bucket",
        "Exit Reason",
        "Breakout Type",
    ]

    parts = []
    for col in bucket_cols:
        summary = grouped_summary(trades_df, col)
        if not summary.empty:
            parts.append(summary)

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True)


def summary_by_year(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()

    out = trades_df.copy()
    out["Entry Year"] = pd.to_datetime(out["Entry Date"]).dt.year

    rows = []
    for year, group in out.groupby("Entry Year"):
        wins = group["P/L %"] > 0
        rows.append(
            {
                "Entry Year": int(year),
                "Trades": int(len(group)),
                "Win Rate %": float(wins.mean() * 100),
                "Average P/L %": float(group["P/L %"].mean()),
                "Median P/L %": float(group["P/L %"].median()),
                "Profit Factor": float(profit_factor(group["P/L %"])),
                "Average R": float(group["R Multiple"].mean()),
                "Average Holding Weeks": float(group["Holding Weeks"].mean()),
            }
        )

    return pd.DataFrame(rows)


def save_outputs(
    output_dir: Path,
    trades_df: pd.DataFrame,
    setups_df: pd.DataFrame,
    errors_df: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    trades_with_buckets = add_trade_buckets(trades_df)

    summary = summarize_trades(trades_with_buckets)
    by_year = summary_by_year(trades_with_buckets)
    by_exit = grouped_summary(trades_with_buckets, "Exit Reason") if not trades_with_buckets.empty else pd.DataFrame()
    bucket_summary = create_bucket_summary(trades_with_buckets)

    summary.to_csv(output_dir / f"{OUTPUT_PREFIX}_summary.csv", index=False)
    trades_with_buckets.to_csv(output_dir / f"{OUTPUT_PREFIX}_trades.csv", index=False)
    setups_df.to_csv(output_dir / f"{OUTPUT_PREFIX}_entry_debug_all_setups.csv", index=False)

    by_year.to_csv(output_dir / f"{OUTPUT_PREFIX}_summary_by_year.csv", index=False)
    by_exit.to_csv(output_dir / f"{OUTPUT_PREFIX}_summary_by_exit.csv", index=False)
    bucket_summary.to_csv(output_dir / f"{OUTPUT_PREFIX}_entry_debug_bucket_summary.csv", index=False)

    if not trades_with_buckets.empty:
        trades_with_buckets.sort_values("P/L %", ascending=False).head(30).to_csv(
            output_dir / f"{OUTPUT_PREFIX}_entry_debug_top_winners.csv",
            index=False,
        )
        trades_with_buckets.sort_values("P/L %", ascending=True).head(30).to_csv(
            output_dir / f"{OUTPUT_PREFIX}_entry_debug_top_losers.csv",
            index=False,
        )
    else:
        pd.DataFrame().to_csv(output_dir / f"{OUTPUT_PREFIX}_entry_debug_top_winners.csv", index=False)
        pd.DataFrame().to_csv(output_dir / f"{OUTPUT_PREFIX}_entry_debug_top_losers.csv", index=False)

    errors_df.to_csv(output_dir / f"{OUTPUT_PREFIX}_backtest_errors.csv", index=False)

    print("\nSaved files:")
    for name in [
        f"{OUTPUT_PREFIX}_summary.csv",
        f"{OUTPUT_PREFIX}_trades.csv",
        f"{OUTPUT_PREFIX}_entry_debug_all_setups.csv",
        f"{OUTPUT_PREFIX}_entry_debug_bucket_summary.csv",
        f"{OUTPUT_PREFIX}_entry_debug_top_winners.csv",
        f"{OUTPUT_PREFIX}_entry_debug_top_losers.csv",
        f"{OUTPUT_PREFIX}_summary_by_year.csv",
        f"{OUTPUT_PREFIX}_summary_by_exit.csv",
        f"{OUTPUT_PREFIX}_backtest_errors.csv",
    ]:
        print(f"  {output_dir / name}")


def print_settings() -> None:
    print("=" * 80)
    print("WEEKLY V6 RS5 52W/ATH + CUP-AND-HANDLE BACKTEST")
    print("=" * 80)
    print("Entry setup:")
    print("- Weekly candles")
    print("- QQQ weekly uptrend: QQQ Close > 9 EMA > 20 EMA > 50 SMA")
    print("- market_etf_strong: SPY Close > 9 EMA > 20 EMA > 50 SMA")
    print("- stock_strong: Close > 9 EMA > 20 EMA > 50 SMA > 200 SMA")
    print("- SMA structure: 20 SMA > 50 SMA, but 20 SMA must be less than 15% above 50 SMA")
    print("- sector_etf_strong: Sector ETF Close > 9 EMA > 20 EMA > 50 SMA")
    print("- Stock RS 5-week vs SPY > 5%")
    print("- Stock RS Ratio ROC 5-week > 5%")
    print("- Sector RS 5-week vs SPY > 0%")
    print("- Sector RS Ratio ROC 5-week > 0%")
    print("- Weekly close breaks above prior 52-week HIGH or ATH HIGH")
    print("- Cup-and-handle pattern required before breakout")
    print("- Breakout week volume >= 1.2 x 20-week average volume")
    print("- Avoid very big green breakout candles OR breakout volume >= 4.0x")
    print("- Breakout week becomes signal candle")
    print("- Next week buy only if price breaks above breakout candle high")
    print("- Debug output includes cup depth, cup length, handle depth, prior 52W high, prior ATH high, and breakout level")
    print("Exit:")
    print("- Stop below breakout weekly candle low")
    print("- Or weekly 9 EMA crosses below weekly 20 EMA")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--universe",
        type=str,
        default="market_cap_over_10b.csv",
        help="CSV file with Ticker column. Default: market_cap_over_10b.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Folder to save output CSV files. Default: current folder.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=START_DATE,
        help=f"Backtest start date. Default: {START_DATE}",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=END_DATE,
        help="Backtest end date. Default: today.",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        help="Optional limit for quick testing.",
    )
    args = parser.parse_args()

    print_settings()

    universe_path = Path(args.universe)
    output_dir = Path(args.output_dir)

    tickers = load_universe(universe_path, args.max_tickers)
    print(f"Loaded tickers: {len(tickers)}")

    print("\nDownloading SPY and QQQ weekly data...")
    spy = download_weekly(SPY_TICKER, args.start, args.end)
    qqq = download_weekly(QQQ_TICKER, args.start, args.end)

    if spy.empty:
        raise RuntimeError("Could not download SPY data.")
    if qqq.empty:
        raise RuntimeError("Could not download QQQ data.")

    spy, qqq = prepare_etf_data(spy, qqq)

    sector_cache_path = output_dir / SECTOR_CACHE_FILE
    sector_map = build_sector_etf_map(universe_path, tickers, sector_cache_path)
    mapped = len([t for t in tickers if t in sector_map])
    print(f"Sector ETF mapped tickers: {mapped}/{len(tickers)}")

    unique_sector_etfs = sorted({data["Sector ETF"] for data in sector_map.values() if data.get("Sector ETF")})
    print(f"Downloading sector ETF weekly data: {', '.join(unique_sector_etfs)}")
    sector_data: dict[str, pd.DataFrame] = {}
    for sector_etf in unique_sector_etfs:
        raw_sector = download_weekly(sector_etf, args.start, args.end)
        if raw_sector.empty:
            print(f"Warning: could not download sector ETF {sector_etf}; affected tickers will be skipped.")
            continue
        sector_data[sector_etf] = prepare_sector_etf_data(raw_sector, sector_etf)

    all_trades: list[dict[str, Any]] = []
    all_setups: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for idx, ticker in enumerate(tickers, start=1):
        print(f"[{idx}/{len(tickers)}] {ticker}")

        try:
            sector_info = sector_map.get(ticker)
            if not sector_info:
                errors.append(
                    {
                        "Ticker": ticker,
                        "Error": "Missing sector ETF mapping; skipped",
                    }
                )
                continue

            sector_name = sector_info.get("Sector", "")
            sector_etf = sector_info.get("Sector ETF", "")
            sector_df = sector_data.get(sector_etf)
            if sector_df is None or sector_df.empty:
                errors.append(
                    {
                        "Ticker": ticker,
                        "Sector": sector_name,
                        "Sector ETF": sector_etf,
                        "Error": "Sector ETF data missing; skipped",
                    }
                )
                continue

            raw = download_weekly(ticker, args.start, args.end)
            if raw.empty or len(raw) < SMA_LONG + 20:
                errors.append(
                    {
                        "Ticker": ticker,
                        "Sector": sector_name,
                        "Sector ETF": sector_etf,
                        "Error": "Not enough weekly data or download failed",
                    }
                )
                continue

            df = add_relative_strength(raw, spy, qqq, sector_df, sector_name, sector_etf)
            trades, setups = backtest_ticker(ticker, df)

            all_trades.extend(trades)
            all_setups.extend(setups)

        except Exception as exc:
            errors.append(
                {
                    "Ticker": ticker,
                    "Error": str(exc),
                    "Traceback": traceback.format_exc(),
                }
            )

    trades_df = pd.DataFrame(all_trades)
    setups_df = pd.DataFrame(all_setups)
    errors_df = pd.DataFrame(errors)

    save_outputs(output_dir, trades_df, setups_df, errors_df)

    print("\nDone.")
    if not trades_df.empty:
        summary = summarize_trades(add_trade_buckets(trades_df))
        print("\nSummary:")
        print(summary.to_string(index=False))
    else:
        print("No trades found.")


if __name__ == "__main__":
    main()
