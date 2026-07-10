from __future__ import annotations

import json
import os
from datetime import time
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config import DATA_DIR
from data_loader import latest_completed_us_session
from database import load_eligible_tickers
from universe_builder import fetch_nasdaq_screener_rows, normalize_symbol, parse_market_cap


TICKERS = ["SPY", "QQQ", "DIA", "IWM"]

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLC": "Communication Services",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLE": "Energy",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
}

PERIOD = "3y"
INTERVAL = "1d"

EMA_9_PERIOD = 9
SMA_20_PERIOD = 20
SMA_50_PERIOD = 50
SMA_200_PERIOD = 200

ADX_PERIOD = 14

PIVOT_LEFT = 5
PIVOT_RIGHT = 5

EMA_9_20_CLOSE_THRESHOLD = 0.01
CLOSE_NEAR_20_THRESHOLD = 0.01

VOLUME_SHORT_MA = 5
VOLUME_LONG_MA = 20
VOLUME_DIRECTION_RATIO_THRESHOLD = 1.10
VOLUME_SUPPORT_RATIO = 0.75

ONLY_COMPLETED_DAILY_CANDLE = True
US_MARKET_CLOSE_TIME = time(16, 0)
MARKET_CLOSE_BUFFER_MINUTES = 15
SECTOR_PROFILE_CACHE_PATH = DATA_DIR / "sector_profile_cache.json"
_FMP_PROFILE_RATE_LIMITED = False


def remove_unfinished_daily_candle(df: pd.DataFrame) -> pd.DataFrame:
    if not ONLY_COMPLETED_DAILY_CANDLE or df.empty:
        return df

    now_ny = pd.Timestamp.now(tz="America/New_York")
    today_ny = now_ny.date()
    last_candle_date = df.index[-1].date()
    market_close_with_buffer = (
        pd.Timestamp.combine(today_ny, US_MARKET_CLOSE_TIME)
        .tz_localize("America/New_York")
        + pd.Timedelta(minutes=MARKET_CLOSE_BUFFER_MINUTES)
    )

    if last_candle_date == today_ny and now_ny < market_close_with_buffer:
        return df.iloc[:-1].copy()
    return df


def download_single_ticker_data(ticker: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(period=PERIOD, interval=INTERVAL, auto_adjust=False)
    except Exception as history_exc:
        df = yf.download(
            ticker,
            period=PERIOD,
            interval=INTERVAL,
            auto_adjust=False,
            prepost=False,
            progress=False,
            threads=False,
        )
        if df.empty:
            raise history_exc

    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        if ticker in set(df.columns.get_level_values(0)):
            df = df[ticker]
        elif ticker in set(df.columns.get_level_values(1)):
            df = df.xs(ticker, axis=1, level=1)
        else:
            df.columns = df.columns.get_level_values(0)

    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for {ticker}: {missing}")

    df = df[required_columns].copy()
    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=required_columns)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = remove_unfinished_daily_candle(df)

    if len(df) < SMA_200_PERIOD + 20:
        raise ValueError(f"Not enough data for {ticker}")
    return df


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA 9"] = df["Close"].ewm(span=EMA_9_PERIOD, adjust=False).mean()
    df["SMA 20"] = df["Close"].rolling(SMA_20_PERIOD).mean()
    df["SMA 50"] = df["Close"].rolling(SMA_50_PERIOD).mean()
    df["SMA 200"] = df["Close"].rolling(SMA_200_PERIOD).mean()
    return df


def add_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.DataFrame:
    df = df.copy()

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    previous_close = close.shift(1)

    true_range_array = np.maximum.reduce(
        [
            (high - low).to_numpy(),
            (high - previous_close).abs().to_numpy(),
            (low - previous_close).abs().to_numpy(),
        ]
    )
    true_range = pd.Series(true_range_array, index=df.index)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    atr = true_range.ewm(alpha=1 / period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1 / period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * plus_dm_smooth / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm_smooth / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)

    df["ADX"] = dx.ewm(alpha=1 / period, adjust=False).mean()
    df["+DI"] = plus_di
    df["-DI"] = minus_di
    return df


def to_weekly_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    weekly = (
        df[["Open", "High", "Low", "Close", "Volume"]]
        .resample("W-FRI")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna(subset=["Open", "High", "Low", "Close"])
    )
    return weekly


def latest_weekly_adx_reading(df: pd.DataFrame) -> dict[str, Any]:
    weekly = to_weekly_ohlcv(df)
    if len(weekly) < ADX_PERIOD * 3:
        return {
            "Weekly Date": None,
            "Weekly ADX": None,
            "Weekly +DI": None,
            "Weekly -DI": None,
        }
    weekly = add_adx(weekly, ADX_PERIOD).dropna(subset=["ADX", "+DI", "-DI"])
    if weekly.empty:
        return {
            "Weekly Date": None,
            "Weekly ADX": None,
            "Weekly +DI": None,
            "Weekly -DI": None,
        }
    latest = weekly.iloc[-1]
    return {
        "Weekly Date": weekly.index[-1].date(),
        "Weekly ADX": round(float(latest["ADX"]), 2),
        "Weekly +DI": round(float(latest["+DI"]), 2),
        "Weekly -DI": round(float(latest["-DI"]), 2),
    }


def add_volume_analysis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    volume = df["Volume"].astype(float)
    close = df["Close"].astype(float)
    price_change = close.diff()

    df["Volume SMA 5"] = volume.rolling(VOLUME_SHORT_MA).mean()
    df["Volume SMA 20"] = volume.rolling(VOLUME_LONG_MA).mean()
    df["Up Day Volume"] = np.where(price_change > 0, volume, 0.0)
    df["Down Day Volume"] = np.where(price_change < 0, volume, 0.0)
    df["Up Volume 20"] = df["Up Day Volume"].rolling(VOLUME_LONG_MA).sum()
    df["Down Volume 20"] = df["Down Day Volume"].rolling(VOLUME_LONG_MA).sum()
    df["Up/Down Volume Ratio"] = df["Up Volume 20"] / df["Down Volume 20"].replace(0, np.nan)
    df["Down/Up Volume Ratio"] = df["Down Volume 20"] / df["Up Volume 20"].replace(0, np.nan)
    df["OBV Change"] = np.where(price_change > 0, volume, np.where(price_change < 0, -volume, 0.0))
    df["OBV"] = df["OBV Change"].cumsum()
    df["OBV Change 20"] = df["OBV"].diff(VOLUME_LONG_MA)
    return df


def get_volume_confirmation(latest: pd.Series) -> dict[str, Any]:
    volume_sma_5 = latest["Volume SMA 5"]
    volume_sma_20 = latest["Volume SMA 20"]
    if pd.isna(volume_sma_5) or pd.isna(volume_sma_20) or volume_sma_20 == 0:
        return {
            "Volume Confirmation": "VOLUME UNKNOWN",
            "Volume Reason": "Not enough volume data",
            "Volume SMA 5/20 Ratio": None,
        }

    volume_5_20_ratio = volume_sma_5 / volume_sma_20
    up_down_ratio = latest["Up/Down Volume Ratio"]
    down_up_ratio = latest["Down/Up Volume Ratio"]
    obv_change_20 = latest["OBV Change 20"]

    volume_expanding = volume_sma_5 >= volume_sma_20
    volume_supported = volume_5_20_ratio >= VOLUME_SUPPORT_RATIO
    up_volume_stronger = pd.notna(up_down_ratio) and up_down_ratio >= VOLUME_DIRECTION_RATIO_THRESHOLD
    down_volume_stronger = pd.notna(down_up_ratio) and down_up_ratio >= VOLUME_DIRECTION_RATIO_THRESHOLD
    obv_rising = obv_change_20 > 0
    obv_falling = obv_change_20 < 0

    if volume_expanding and up_volume_stronger and obv_rising:
        return {
            "Volume Confirmation": "VOLUME CONFIRMS UPTREND",
            "Volume Reason": "5-day volume >= 20-day volume, up-volume stronger than down-volume, OBV rising",
            "Volume SMA 5/20 Ratio": volume_5_20_ratio,
        }
    if volume_expanding and down_volume_stronger and obv_falling:
        return {
            "Volume Confirmation": "VOLUME CONFIRMS DOWNTREND",
            "Volume Reason": "5-day volume >= 20-day volume, down-volume stronger than up-volume, OBV falling",
            "Volume SMA 5/20 Ratio": volume_5_20_ratio,
        }
    if volume_supported and up_volume_stronger and obv_rising:
        return {
            "Volume Confirmation": "VOLUME SUPPORTS UPTREND",
            "Volume Reason": f"5-day volume is {volume_5_20_ratio:.0%} of 20-day volume, up-volume stronger than down-volume, OBV rising",
            "Volume SMA 5/20 Ratio": volume_5_20_ratio,
        }
    if volume_supported and down_volume_stronger and obv_falling:
        return {
            "Volume Confirmation": "VOLUME SUPPORTS DOWNTREND",
            "Volume Reason": f"5-day volume is {volume_5_20_ratio:.0%} of 20-day volume, down-volume stronger than up-volume, OBV falling",
            "Volume SMA 5/20 Ratio": volume_5_20_ratio,
        }
    return {
        "Volume Confirmation": "VOLUME MIXED / NOT CONFIRMING",
        "Volume Reason": f"5-day volume is {volume_5_20_ratio:.0%} of 20-day volume, but volume direction is not clean",
        "Volume SMA 5/20 Ratio": volume_5_20_ratio,
    }


def add_pivots(df: pd.DataFrame, left: int = PIVOT_LEFT, right: int = PIVOT_RIGHT) -> pd.DataFrame:
    df = df.copy()
    df["Pivot High"] = False
    df["Pivot Low"] = False

    for index in range(left, len(df) - right):
        current_high = df["High"].iloc[index]
        current_low = df["Low"].iloc[index]
        left_highs = df["High"].iloc[index - left:index]
        right_highs = df["High"].iloc[index + 1:index + right + 1]
        left_lows = df["Low"].iloc[index - left:index]
        right_lows = df["Low"].iloc[index + 1:index + right + 1]

        if current_high > left_highs.max() and current_high > right_highs.max():
            df.loc[df.index[index], "Pivot High"] = True
        if current_low < left_lows.min() and current_low < right_lows.min():
            df.loc[df.index[index], "Pivot Low"] = True
    return df


def get_swing_structure(df: pd.DataFrame) -> dict[str, Any]:
    pivot_highs = df[df["Pivot High"]]
    pivot_lows = df[df["Pivot Low"]]

    higher_high = higher_low = lower_high = lower_low = False
    if len(pivot_highs) >= 2:
        previous_high = pivot_highs["High"].iloc[-2]
        recent_high = pivot_highs["High"].iloc[-1]
        higher_high = recent_high > previous_high
        lower_high = recent_high < previous_high
    if len(pivot_lows) >= 2:
        previous_low = pivot_lows["Low"].iloc[-2]
        recent_low = pivot_lows["Low"].iloc[-1]
        higher_low = recent_low > previous_low
        lower_low = recent_low < previous_low

    if higher_high and higher_low:
        label = "HH + HL"
        reason = "Price structure supports uptrend"
    elif lower_high and lower_low:
        label = "LH + LL"
        reason = "Price structure supports downtrend"
    elif higher_high and lower_low:
        label = "HH + LL - WIDE CHOPPY RANGE"
        reason = "Wide choppy range, not clean trend"
    elif lower_high and higher_low:
        label = "LH + HL - COMPRESSION"
        reason = "Compression, trend unclear"
    else:
        label = "UNCLEAR"
        reason = "Swing structure is unclear"

    return {
        "Higher High": higher_high,
        "Higher Low": higher_low,
        "Lower High": lower_high,
        "Lower Low": lower_low,
        "Swing Structure": label,
        "Structure Reason": reason,
    }


def classify_market_trend(df: pd.DataFrame) -> dict[str, Any]:
    latest = df.iloc[-1]
    close = latest["Close"]
    ema_9 = latest["EMA 9"]
    sma_20 = latest["SMA 20"]
    sma_50 = latest["SMA 50"]
    sma_200 = latest["SMA 200"]
    adx = latest["ADX"]
    plus_di = latest["+DI"]
    minus_di = latest["-DI"]

    swing = get_swing_structure(df)
    volume = get_volume_confirmation(latest)
    higher_high = swing["Higher High"]
    higher_low = swing["Higher Low"]
    lower_high = swing["Lower High"]
    lower_low = swing["Lower Low"]

    volume_up = volume["Volume Confirmation"] in {"VOLUME CONFIRMS UPTREND", "VOLUME SUPPORTS UPTREND"}
    volume_down = volume["Volume Confirmation"] in {"VOLUME CONFIRMS DOWNTREND", "VOLUME SUPPORTS DOWNTREND"}

    ema_close_to_20 = abs(ema_9 - sma_20) / sma_20 <= EMA_9_20_CLOSE_THRESHOLD
    close_near_20 = abs(close - sma_20) / sma_20 <= CLOSE_NEAR_20_THRESHOLD
    strong_bullish_ma = close > ema_9 > sma_20 > sma_50 > sma_200
    weak_bullish_ma = ema_close_to_20 and sma_20 > sma_50 > sma_200 and (close >= sma_20 or close_near_20)
    strong_bearish_ma = close < ema_9 < sma_20 < sma_50 < sma_200
    weak_bearish_ma = close < sma_20 and ema_9 < sma_20 and sma_20 > sma_50 > sma_200
    mixed_structure = not ((higher_high and higher_low) or (lower_high and lower_low))

    reasons: list[str] = []
    if adx < 20:
        trend = "SIDEWAYS"
        reasons.extend(["ADX below 20 means trendless market", swing["Structure Reason"], volume["Volume Reason"]])
    elif strong_bullish_ma and adx > 25 and plus_di > minus_di and higher_high and higher_low and volume_up:
        trend = "STRONG UPTREND"
        reasons.extend(["Close > 9 EMA > 20 SMA > 50 SMA > 200 SMA", "ADX > 25", "+DI > -DI", "Swing structure is HH + HL", volume["Volume Reason"]])
    elif strong_bullish_ma and adx > 25 and plus_di > minus_di and higher_high and higher_low:
        trend = "WEAK UPTREND"
        reasons.extend(["Price, MA stack, ADX, and structure support uptrend", "But volume does not support uptrend", volume["Volume Reason"]])
    elif strong_bearish_ma and adx > 25 and minus_di > plus_di and lower_high and lower_low and volume_down:
        trend = "STRONG DOWNTREND"
        reasons.extend(["Close < 9 EMA < 20 SMA < 50 SMA < 200 SMA", "ADX > 25", "-DI > +DI", "Swing structure is LH + LL", volume["Volume Reason"]])
    elif strong_bearish_ma and adx > 25 and minus_di > plus_di and lower_high and lower_low:
        trend = "WEAK DOWNTREND"
        reasons.extend(["Price, MA stack, ADX, and structure support downtrend", "But volume does not support downtrend", volume["Volume Reason"]])
    elif weak_bullish_ma and plus_di >= minus_di and adx >= 20:
        trend = "WEAK UPTREND"
        reasons.extend(["9 EMA close to 20 SMA", "20 SMA > 50 SMA > 200 SMA", "+DI >= -DI", "Trend is bullish but not strongly separated", volume["Volume Reason"]])
    elif weak_bearish_ma and minus_di > plus_di and adx >= 20:
        trend = "WEAK DOWNTREND"
        reasons.extend(["9 EMA < 20 SMA", "20 SMA > 50 SMA > 200 SMA", "-DI > +DI", "Short-term downtrend inside bigger uptrend", volume["Volume Reason"]])
    elif 20 <= adx <= 25 and mixed_structure:
        trend = "SIDEWAYS"
        reasons.extend(["ADX between 20 and 25 with unclear structure", swing["Structure Reason"], volume["Volume Reason"]])
    else:
        trend = "SIDEWAYS"
        reasons.extend(["No clean trend condition matched", swing["Structure Reason"], volume["Volume Reason"]])

    volume_window = df.tail(VOLUME_LONG_MA).copy()
    comparison_window = df.tail(VOLUME_LONG_MA + 1).copy()
    window_price_change = comparison_window["Close"].diff().tail(VOLUME_LONG_MA)
    up_day_count = int((window_price_change > 0).sum())
    down_day_count = int((window_price_change < 0).sum())
    flat_day_count = int((window_price_change == 0).sum())
    above_average_volume = volume_window["Volume"] > volume_window["Volume SMA 20"]
    above_average_price_change = window_price_change.reindex(volume_window.index)
    above_average_volume_count = int(above_average_volume.sum())
    above_average_bullish_count = int(((above_average_volume) & (above_average_price_change > 0)).sum())
    above_average_bearish_count = int(((above_average_volume) & (above_average_price_change < 0)).sum())
    above_average_flat_count = int(((above_average_volume) & (above_average_price_change == 0)).sum())
    biggest_volume_date = volume_window["Volume"].idxmax()
    biggest_volume_row = volume_window.loc[biggest_volume_date]
    biggest_volume_index = comparison_window.index.get_loc(biggest_volume_date)
    if biggest_volume_index > 0:
        previous_close = comparison_window["Close"].iloc[biggest_volume_index - 1]
        biggest_direction = "Up" if biggest_volume_row["Close"] > previous_close else "Down" if biggest_volume_row["Close"] < previous_close else "Flat"
    else:
        biggest_direction = "Flat"

    return {
        "Date": df.index[-1].date(),
        "Close": round(close, 2),
        "EMA 9": round(ema_9, 2),
        "SMA 20": round(sma_20, 2),
        "SMA 50": round(sma_50, 2),
        "SMA 200": round(sma_200, 2),
        "ADX": round(adx, 2),
        "+DI": round(plus_di, 2),
        "-DI": round(minus_di, 2),
        "Swing Structure": swing["Swing Structure"],
        "Higher High": higher_high,
        "Higher Low": higher_low,
        "Lower High": lower_high,
        "Lower Low": lower_low,
        "Volume": int(latest["Volume"]),
        "Volume SMA 5": round(latest["Volume SMA 5"], 0),
        "Volume SMA 20": round(latest["Volume SMA 20"], 0),
        "Volume SMA 5/20 Ratio": round(volume["Volume SMA 5/20 Ratio"], 2) if volume["Volume SMA 5/20 Ratio"] is not None else None,
        "Up Volume 20": round(latest["Up Volume 20"], 0),
        "Down Volume 20": round(latest["Down Volume 20"], 0),
        "Up Day Count 20": up_day_count,
        "Down Day Count 20": down_day_count,
        "Flat Day Count 20": flat_day_count,
        "Above Avg Volume Count 20": above_average_volume_count,
        "Above Avg Volume Bullish Count 20": above_average_bullish_count,
        "Above Avg Volume Bearish Count 20": above_average_bearish_count,
        "Above Avg Volume Flat Count 20": above_average_flat_count,
        "Biggest Volume Date 20": biggest_volume_date.date(),
        "Biggest Volume 20": int(biggest_volume_row["Volume"]),
        "Biggest Volume Direction 20": biggest_direction,
        "Up/Down Volume Ratio": round(latest["Up/Down Volume Ratio"], 2) if pd.notna(latest["Up/Down Volume Ratio"]) else None,
        "Down/Up Volume Ratio": round(latest["Down/Up Volume Ratio"], 2) if pd.notna(latest["Down/Up Volume Ratio"]) else None,
        "OBV Change 20": round(latest["OBV Change 20"], 0),
        "Volume Confirmation": volume["Volume Confirmation"],
        "Volume Reason": volume["Volume Reason"],
        "Trend": trend,
        "Reason": " | ".join(reasons),
    }


def trend_to_score(trend: str) -> int:
    scores = {
        "STRONG UPTREND": 2,
        "WEAK UPTREND": 1,
        "SIDEWAYS": 0,
        "WEAK DOWNTREND": -1,
        "STRONG DOWNTREND": -2,
    }
    return scores.get(trend, 0)


def get_overall_market_view(result_df: pd.DataFrame) -> tuple[str, str, float]:
    trend_map = dict(zip(result_df["Ticker"], result_df["Trend"]))
    weighted_score = (
        trend_to_score(trend_map.get("SPY", "SIDEWAYS")) * 0.40
        + trend_to_score(trend_map.get("QQQ", "SIDEWAYS")) * 0.25
        + trend_to_score(trend_map.get("DIA", "SIDEWAYS")) * 0.20
        + trend_to_score(trend_map.get("IWM", "SIDEWAYS")) * 0.15
    )

    if weighted_score >= 1.50:
        market = "STRONG BULLISH MARKET"
    elif weighted_score >= 0.50:
        market = "BULLISH / WEAK UPTREND"
    elif weighted_score > -0.50:
        market = "SIDEWAYS / CAUTION"
    elif weighted_score > -1.50:
        market = "BEARISH / WEAK DOWNTREND"
    else:
        market = "STRONG BEARISH MARKET"

    reason = f"Weighted score = {weighted_score:.2f}. SPY 40%, QQQ 25%, DIA 20%, IWM 15%."
    return market, reason, weighted_score


def trading_interpretation(overall_market: str) -> str:
    interpretations = {
        "STRONG BULLISH MARKET": "Long trades allowed. Breakouts and pullbacks are both acceptable.",
        "BULLISH / WEAK UPTREND": "Long trades allowed, but prefer clean pullbacks and strong relative-strength stocks.",
        "SIDEWAYS / CAUTION": "Market is choppy. Reduce trade count. Avoid chasing breakouts.",
        "BEARISH / WEAK DOWNTREND": "Avoid aggressive longs. Wait for recovery or only watch strongest stocks.",
        "STRONG BEARISH MARKET": "Avoid new long trades.",
    }
    return interpretations.get(overall_market, "Market condition is unclear. Keep risk small.")


def _return_pct(close: pd.Series, periods: int) -> float | None:
    clean = pd.to_numeric(close, errors="coerce").dropna()
    if len(clean) <= periods:
        return None
    prior = float(clean.iloc[-1 - periods])
    if prior == 0:
        return None
    return float(clean.iloc[-1]) / prior - 1


def _sector_etf_row(ticker: str, sector: str, df: pd.DataFrame, spy_df: pd.DataFrame) -> dict[str, Any]:
    df = add_moving_averages(df)
    df = add_adx(df, ADX_PERIOD)
    df = add_volume_analysis(df)
    df = add_pivots(df, PIVOT_LEFT, PIVOT_RIGHT)
    df = df.dropna(
        subset=[
            "EMA 9",
            "SMA 20",
            "SMA 50",
            "SMA 200",
            "ADX",
            "+DI",
            "-DI",
            "Volume SMA 5",
            "Volume SMA 20",
            "Up Volume 20",
            "Down Volume 20",
            "OBV Change 20",
        ]
    )
    if df.empty:
        raise ValueError(f"Not enough clean indicator data for {ticker}")

    result = classify_market_trend(df)
    result.update(latest_weekly_adx_reading(df))
    close = pd.to_numeric(df["Close"], errors="coerce")
    spy_close = pd.to_numeric(spy_df["Close"], errors="coerce")
    common_index = close.index.intersection(spy_close.index)
    sector_return_20d = _return_pct(close.loc[common_index], 20)
    spy_return_20d = _return_pct(spy_close.loc[common_index], 20)
    sector_return_5d = _return_pct(close.loc[common_index], 5)
    sector_return_1d = _return_pct(close.loc[common_index], 1)
    latest = df.iloc[-1]
    result.update(
        {
            "Ticker": ticker,
            "Sector": sector,
            "Return 1D %": round(sector_return_1d * 100, 2) if sector_return_1d is not None else None,
            "Return 5D %": round(sector_return_5d * 100, 2) if sector_return_5d is not None else None,
            "Return 20D %": round(sector_return_20d * 100, 2) if sector_return_20d is not None else None,
            "SPY Return 20D %": round(spy_return_20d * 100, 2) if spy_return_20d is not None else None,
            "RS 20D vs SPY %": (
                round((sector_return_20d - spy_return_20d) * 100, 2)
                if sector_return_20d is not None and spy_return_20d is not None
                else None
            ),
            "Above 9 EMA": bool(latest["Close"] > latest["EMA 9"]),
            "MA Stack": (
                "C > 9E > 20 > 50 > 200"
                if latest["Close"] > latest["EMA 9"] > latest["SMA 20"] > latest["SMA 50"] > latest["SMA 200"]
                else "Not fully stacked"
            ),
        }
    )
    return result


def run_sector_etf_scan() -> dict[str, Any]:
    completed_date = latest_completed_us_session()
    tickers = sorted(set(SECTOR_ETFS) | {"SPY"})
    results: list[dict[str, Any]] = []
    errors: list[str] = []

    price_data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            price_data[ticker] = download_single_ticker_data(ticker)
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")

    spy_df = price_data.get("SPY")
    if spy_df is None or spy_df.empty:
        return {
            "completed_date": str(completed_date.date()),
            "results": pd.DataFrame(),
            "leaders": [],
            "laggards": [],
            "errors": errors or ["SPY benchmark data unavailable."],
        }

    for ticker, sector in SECTOR_ETFS.items():
        df = price_data.get(ticker)
        if df is None or df.empty:
            continue
        try:
            results.append(_sector_etf_row(ticker, sector, df, spy_df))
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values(
            by=["RS 20D vs SPY %", "Return 5D %", "Volume SMA 5/20 Ratio"],
            ascending=[False, False, False],
        )

    leaders = []
    laggards = []
    if not result_df.empty and "RS 20D vs SPY %" in result_df.columns:
        leaders = result_df.head(3)[["Ticker", "Sector", "RS 20D vs SPY %"]].to_dict("records")
        laggards = result_df.tail(3)[["Ticker", "Sector", "RS 20D vs SPY %"]].sort_values(
            by="RS 20D vs SPY %", ascending=True
        ).to_dict("records")

    return {
        "completed_date": str(completed_date.date()),
        "results": result_df,
        "leaders": leaders,
        "laggards": laggards,
        "errors": errors,
    }


def _load_nasdaq_sector_map() -> tuple[dict[str, dict[str, str]], list[str]]:
    rows_by_ticker: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    for exchange in ("nasdaq", "nyse"):
        try:
            rows = fetch_nasdaq_screener_rows(exchange)
        except Exception as exc:
            errors.append(f"{exchange}: sector map unavailable ({exc})")
            continue
        for row in rows:
            ticker = normalize_symbol(row.get("symbol"))
            if not ticker:
                continue
            sector = str(row.get("sector") or row.get("Sector") or "Unknown").strip() or "Unknown"
            industry = str(row.get("industry") or row.get("Industry") or "").strip()
            company = str(row.get("name") or row.get("Name") or row.get("companyName") or "").strip()
            rows_by_ticker[ticker] = {
                "sector": sector,
                "industry": industry,
                "company": company,
            }
    return rows_by_ticker, errors


def _load_sector_profile_cache() -> dict[str, dict[str, str]]:
    try:
        if SECTOR_PROFILE_CACHE_PATH.exists():
            payload = json.loads(SECTOR_PROFILE_CACHE_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return {
                    str(key).upper(): value
                    for key, value in payload.items()
                    if isinstance(value, dict) and value.get("sector")
                }
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _save_sector_profile_cache(cache: dict[str, dict[str, str]]) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        SECTOR_PROFILE_CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        pass


def _fetch_sector_profile(ticker: str, cache: dict[str, dict[str, str]] | None = None) -> dict[str, str]:
    global _FMP_PROFILE_RATE_LIMITED
    ticker = ticker.upper()
    if cache is not None and ticker in cache:
        return cache[ticker]

    profile = {"sector": "Unknown", "industry": "", "company": ""}
    api_key = os.getenv("FMP_API_KEY")
    if api_key and not _FMP_PROFILE_RATE_LIMITED:
        try:
            response = requests.get(
                "https://financialmodelingprep.com/stable/profile",
                params={"symbol": ticker, "apikey": api_key},
                timeout=8,
            )
            if response.status_code == 429:
                _FMP_PROFILE_RATE_LIMITED = True
                raise RuntimeError("FMP profile rate limit reached")
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and "limit" in str(payload).lower():
                _FMP_PROFILE_RATE_LIMITED = True
                raise RuntimeError("FMP profile rate limit reached")
            if isinstance(payload, list) and payload:
                row = payload[0]
                sector = str(row.get("sector") or "").strip()
                if sector:
                    profile = {
                        "sector": sector,
                        "industry": str(row.get("industry") or "").strip(),
                        "company": str(row.get("companyName") or "").strip(),
                    }
                    if cache is not None:
                        cache[ticker] = profile
                    return profile
        except Exception:
            pass

    try:
        info = yf.Ticker(ticker).get_info()
        sector = str(info.get("sector") or "").strip()
        if sector:
            profile = {
                "sector": sector,
                "industry": str(info.get("industry") or "").strip(),
                "company": str(info.get("longName") or info.get("shortName") or "").strip(),
            }
            if cache is not None:
                cache[ticker] = profile
            return profile
    except Exception:
        pass
    return profile


def _parse_nasdaq_float(value: object) -> float | None:
    text = str(value or "").replace("$", "").replace(",", "").replace("%", "").strip()
    if not text or text.upper() in {"N/A", "NA"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _looks_like_common_stock_symbol(ticker: str) -> bool:
    suffixes = ("W", "WS", "WT", "WTS", "U", "R", "RT")
    if ticker.endswith((".W", ".WS", ".WT", ".U", ".R")):
        return False
    if len(ticker) >= 5 and any(ticker.endswith(suffix) for suffix in suffixes):
        return False
    return True


def run_sector_top_gainers_scan(max_per_sector: int = 5, sector_lookup_limit: int = 80) -> dict[str, Any]:
    completed_date = latest_completed_us_session()
    eligible = load_eligible_tickers()
    if eligible.empty:
        return {
            "completed_date": str(completed_date.date()),
            "results": pd.DataFrame(),
            "errors": ["Saved $500M+ ticker universe is empty."],
        }

    eligible["ticker"] = eligible["ticker"].astype(str).str.upper()
    tickers = eligible["ticker"].dropna().drop_duplicates().tolist()
    eligible_set = set(tickers)
    market_caps = dict(zip(eligible["ticker"], pd.to_numeric(eligible["market_cap"], errors="coerce"), strict=False))
    errors: list[str] = []

    rows: list[dict[str, Any]] = []
    for exchange in ("nasdaq", "nyse"):
        try:
            screener_rows = fetch_nasdaq_screener_rows(exchange)
        except Exception as exc:
            errors.append(f"{exchange}: Nasdaq mover rows unavailable ({exc})")
            continue
        for screener_row in screener_rows:
            ticker = normalize_symbol(screener_row.get("symbol"))
            if not ticker or ticker not in eligible_set:
                continue
            if not _looks_like_common_stock_symbol(ticker):
                continue
            close = _parse_nasdaq_float(screener_row.get("lastsale"))
            net_change = _parse_nasdaq_float(screener_row.get("netchange"))
            gain_pct = _parse_nasdaq_float(screener_row.get("pctchange"))
            if close is None or gain_pct is None:
                continue
            previous_close = close - net_change if net_change is not None else None
            market_cap = market_caps.get(ticker)
            if market_cap is None or pd.isna(market_cap):
                market_cap = parse_market_cap(screener_row.get("marketCap"))
            rows.append(
                {
                    "ticker": ticker,
                    "company": str(screener_row.get("name") or "").strip(),
                    "sector": "Unknown",
                    "industry": "",
                    "date": str(completed_date.date()),
                    "close": round(close, 2),
                    "previous_close": round(previous_close, 2) if previous_close is not None else None,
                    "gain_pct": round(gain_pct, 2),
                    "market_cap": float(market_cap) if market_cap is not None and pd.notna(market_cap) else None,
                    "exchange": exchange.upper(),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return {
            "completed_date": str(completed_date.date()),
            "results": frame,
            "errors": errors or ["No Nasdaq mover data was available for the saved $500M+ universe."],
            "downloaded_tickers": 0,
            "requested_tickers": len(tickers),
        }

    frame = frame.drop_duplicates(subset=["ticker"], keep="first")
    nasdaq_mover_count = len(frame)

    frame = frame.sort_values(["gain_pct", "market_cap"], ascending=[False, False]).reset_index(drop=True)
    enrichment_errors: list[str] = []
    enriched_rows: list[dict[str, Any]] = []
    sector_counts: dict[str, int] = {}
    profile_cache = _load_sector_profile_cache()
    original_cache_size = len(profile_cache)
    for row in frame.head(sector_lookup_limit).to_dict("records"):
        ticker = str(row.get("ticker") or "")
        sector = str(row.get("sector") or "Unknown")
        if sector == "Unknown":
            profile = _fetch_sector_profile(ticker, profile_cache)
            sector = profile.get("sector") or "Unknown"
            row["sector"] = sector
            row["industry"] = profile.get("industry") or row.get("industry") or ""
            row["company"] = profile.get("company") or row.get("company") or ""
        if sector == "Unknown":
            enrichment_errors.append(f"{ticker}: sector unavailable")
            continue
        enriched_rows.append(row)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    if len(profile_cache) != original_cache_size:
        _save_sector_profile_cache(profile_cache)

    frame = pd.DataFrame(enriched_rows)
    if frame.empty:
        return {
            "completed_date": str(completed_date.date()),
            "results": frame,
            "errors": errors + enrichment_errors or ["Sector mapping returned no usable sectors."],
            "downloaded_tickers": len(price_data),
            "requested_tickers": len(tickers),
            "sector_lookup_limit": sector_lookup_limit,
        }

    frame = frame.sort_values(["sector", "gain_pct", "market_cap"], ascending=[True, False, False])
    frame["sector_rank"] = frame.groupby("sector")["gain_pct"].rank(method="first", ascending=False).astype(int)
    top = frame[frame["sector_rank"] <= max_per_sector].sort_values(["sector", "sector_rank"]).reset_index(drop=True)

    sector_summary = (
        frame.groupby("sector")
        .agg(
            tickers=("ticker", "count"),
            advancers=("gain_pct", lambda values: int((values > 0).sum())),
            decliners=("gain_pct", lambda values: int((values < 0).sum())),
            avg_gain_pct=("gain_pct", "mean"),
            best_gain_pct=("gain_pct", "max"),
        )
        .reset_index()
        .sort_values("avg_gain_pct", ascending=False)
    )
    sector_summary["avg_gain_pct"] = sector_summary["avg_gain_pct"].round(2)
    sector_summary["best_gain_pct"] = sector_summary["best_gain_pct"].round(2)

    return {
        "completed_date": str(completed_date.date()),
        "results": top,
        "sector_summary": sector_summary,
        "errors": errors + enrichment_errors[:50],
        "downloaded_tickers": nasdaq_mover_count,
        "requested_tickers": len(tickers),
        "profile_checked_tickers": min(sector_lookup_limit, nasdaq_mover_count),
        "sector_lookup_limit": sector_lookup_limit,
        "sector_mapped_tickers": len(frame),
    }


def run_market_trend_scanner() -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    errors: list[str] = []

    for ticker in TICKERS:
        try:
            df = download_single_ticker_data(ticker)
            df = add_moving_averages(df)
            df = add_adx(df, ADX_PERIOD)
            df = add_volume_analysis(df)
            df = add_pivots(df, PIVOT_LEFT, PIVOT_RIGHT)

            required_indicators = [
                "EMA 9",
                "SMA 20",
                "SMA 50",
                "SMA 200",
                "ADX",
                "+DI",
                "-DI",
                "Volume SMA 5",
                "Volume SMA 20",
                "Up Volume 20",
                "Down Volume 20",
                "OBV Change 20",
            ]
            df = df.dropna(subset=required_indicators)
            if df.empty:
                errors.append(f"{ticker}: not enough clean indicator data")
                continue

            result = classify_market_trend(df)
            result.update(latest_weekly_adx_reading(df))
            result["Ticker"] = ticker
            results.append(result)
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")

    result_df = pd.DataFrame(results)
    if result_df.empty:
        return {
            "results": result_df,
            "overall_market": "UNKNOWN",
            "overall_reason": "No index data could be analyzed.",
            "weighted_score": 0.0,
            "interpretation": "Market condition is unavailable. Keep risk small until data refreshes.",
            "errors": errors,
        }

    columns = [
        "Ticker",
        "Date",
        "Trend",
        "Reason",
        "Close",
        "EMA 9",
        "SMA 20",
        "SMA 50",
        "SMA 200",
        "ADX",
        "+DI",
        "-DI",
        "Weekly Date",
        "Weekly ADX",
        "Weekly +DI",
        "Weekly -DI",
        "Swing Structure",
        "Higher High",
        "Higher Low",
        "Lower High",
        "Lower Low",
        "Volume",
        "Volume SMA 5",
        "Volume SMA 20",
        "Volume SMA 5/20 Ratio",
        "Up Volume 20",
        "Down Volume 20",
        "Up Day Count 20",
        "Down Day Count 20",
        "Flat Day Count 20",
        "Above Avg Volume Count 20",
        "Above Avg Volume Bullish Count 20",
        "Above Avg Volume Bearish Count 20",
        "Above Avg Volume Flat Count 20",
        "Biggest Volume Date 20",
        "Biggest Volume 20",
        "Biggest Volume Direction 20",
        "Up/Down Volume Ratio",
        "Down/Up Volume Ratio",
        "OBV Change 20",
        "Volume Confirmation",
        "Volume Reason",
    ]
    result_df = result_df[columns]
    overall_market, overall_reason, weighted_score = get_overall_market_view(result_df)

    return {
        "results": result_df,
        "overall_market": overall_market,
        "overall_reason": overall_reason,
        "weighted_score": weighted_score,
        "interpretation": trading_interpretation(overall_market),
        "errors": errors,
    }
