from __future__ import annotations

import pandas as pd


def candle_parts(row: pd.Series) -> dict[str, float]:
    open_ = float(row["Open"])
    high = float(row["High"])
    low = float(row["Low"])
    close = float(row["Close"])
    body = abs(close - open_)
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low
    total_range = max(high - low, 0.0001)
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "body": body,
        "upper_wick": upper_wick,
        "lower_wick": lower_wick,
        "range": total_range,
    }


def has_lower_wick_twice_body(row: pd.Series) -> bool:
    parts = candle_parts(row)
    body = max(parts["body"], parts["range"] * 0.03)
    return parts["lower_wick"] >= 2 * body


def is_hammer(row: pd.Series) -> bool:
    parts = candle_parts(row)
    body = max(parts["body"], parts["range"] * 0.03)
    body_top = max(parts["open"], parts["close"])
    candle_position = (body_top - parts["low"]) / parts["range"]
    return (
        parts["lower_wick"] >= 2 * body
        and parts["upper_wick"] <= body * 1.2
        and candle_position >= 0.6
    )


def is_bullish_engulfing(prev: pd.Series, curr: pd.Series) -> bool:
    prev_bearish = prev["Close"] < prev["Open"]
    curr_bullish = curr["Close"] > curr["Open"]
    body_engulfed = curr["Open"] <= prev["Close"] and curr["Close"] >= prev["Open"]
    return bool(prev_bearish and curr_bullish and body_engulfed)


def is_morning_star(
    first: pd.Series,
    second: pd.Series,
    third: pd.Series,
    first_long_body_pct: float = 0.60,
    small_body_pct: float = 0.35,
    recovery_pct: float = 0.50,
    third_long_body_pct: float = 0.60,
    gap_tolerance_pct: float = 0.005,
) -> bool:
    first_parts = candle_parts(first)
    second_parts = candle_parts(second)
    third_parts = candle_parts(third)
    first_body = max(first_parts["body"], first_parts["range"] * 0.03)
    second_body = second_parts["body"]
    third_body = third_parts["body"]

    first_bearish = first["Close"] < first["Open"]
    third_bullish = third["Close"] > third["Open"]
    first_long_body = first_body / first_parts["range"] >= first_long_body_pct
    third_long_body = third_body / third_parts["range"] >= third_long_body_pct
    first_body_midpoint = float(first["Close"]) + (first_body * recovery_pct)
    second_is_small = second_body <= first_body * small_body_pct and second_body <= third_body
    gap_down = float(second["Open"]) <= float(first["Close"]) * (1 + gap_tolerance_pct)
    gap_up = float(third["Open"]) >= float(second["Close"]) * (1 - gap_tolerance_pct)
    recovery = float(third["Close"]) >= first_body_midpoint
    return bool(
        first_bearish
        and first_long_body
        and second_is_small
        and gap_down
        and third_bullish
        and third_long_body
        and gap_up
        and recovery
    )


def is_bullish_rejection(row: pd.Series) -> bool:
    parts = candle_parts(row)
    return bool(row["Close"] >= row["Open"] and parts["lower_wick"] >= 2 * max(parts["body"], parts["range"] * 0.03))


def detect_signal(df: pd.DataFrame) -> dict[str, bool | str]:
    if len(df) < 2:
        return {
            "hammer": False,
            "bullish_engulfing": False,
            "bullish_rejection": False,
            "lower_wick_2x_body": False,
            "pattern": "",
        }

    prev = df.iloc[-2]
    curr = df.iloc[-1]
    hammer = is_hammer(curr)
    engulfing = is_bullish_engulfing(prev, curr)
    rejection = is_bullish_rejection(curr)
    lower_wick = has_lower_wick_twice_body(curr)

    if hammer:
        pattern = "Hammer"
    elif engulfing:
        pattern = "Bullish engulfing"
    elif rejection:
        pattern = "Bullish rejection"
    else:
        pattern = ""

    return {
        "hammer": hammer,
        "bullish_engulfing": engulfing,
        "bullish_rejection": rejection,
        "lower_wick_2x_body": lower_wick,
        "pattern": pattern,
    }


def detect_morning_star(
    df: pd.DataFrame,
    first_long_body_pct: float = 0.60,
    small_body_pct: float = 0.35,
    recovery_pct: float = 0.50,
    third_long_body_pct: float = 0.60,
    gap_tolerance_pct: float = 0.005,
) -> bool:
    if len(df) < 3:
        return False
    return is_morning_star(
        df.iloc[-3],
        df.iloc[-2],
        df.iloc[-1],
        first_long_body_pct,
        small_body_pct,
        recovery_pct,
        third_long_body_pct,
        gap_tolerance_pct,
    )
