#!/usr/bin/env python3
"""
Weekly V9 Scanner: V8 breakout setup + cup-and-handle information

This scanner is built on top of:
weekly_v8_rs5_52w_ath_sma20_no_cup_breakout_backtest.py

It keeps the same V8 breakout conditions:
- 52-week breakout OR ATH breakout
- No cup filter in the base V8 setup
- 20 SMA > 50 SMA
- SMA20 less than 20% above SMA50
- Stock/sector/market relative strength filters
- Avoid huge green breakout candle
- Avoid breakout volume >= 4x

Then it adds a cup-and-handle detector on top and outputs:
- Cup depth %
- Handle depth %
- Cup length
- Handle length
- Pivot/handle high
- Big-volume green candles inside the cup
- Big-volume green candles on the right side of the cup

Run:
python weekly_v9_v8_breakout_plus_cup_handle_scanner.py --universe market_cap_over_10b.csv

Default output:
weekly_v9_v8_breakout_plus_cup_handle_scan_candidates.csv

Optional all V8 candidates with cup info:
weekly_v9_v8_breakout_plus_cup_handle_all_v8_candidates_with_cup_info.csv
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import weekly_v8_rs5_52w_ath_sma20_no_cup_breakout_backtest as strat


SCAN_OUTPUT = "weekly_v9_v8_breakout_plus_cup_handle_scan_candidates.csv"
ALL_V8_OUTPUT = "weekly_v9_v8_breakout_plus_cup_handle_all_v8_candidates_with_cup_info.csv"
ERROR_OUTPUT = "weekly_v9_v8_breakout_plus_cup_handle_scanner_errors.csv"


# Broad cup-and-handle defaults.
# These are intentionally broad so the scanner can show cup statistics.
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

# Green candle with big volume inside the cup.
# A cup accumulation candle is counted when:
# Close > Open and Volume / 20W Avg Volume >= this threshold.
CUP_BIG_GREEN_VOLUME_RATIO = 1.50


def drop_incomplete_latest_week(df: pd.DataFrame, keep_latest: bool = False) -> pd.DataFrame:
    """Avoid using the current unfinished weekly candle.

    yfinance weekly bars are usually timestamped at the week start.
    If the latest bar is less than 5 calendar days old, treat it as incomplete.
    """
    if keep_latest or df.empty:
        return df

    latest_date = pd.Timestamp(df.index[-1]).tz_localize(None)
    now = pd.Timestamp.utcnow().tz_localize(None)
    if (now - latest_date).days < 5:
        return df.iloc[:-1].copy()
    return df


def empty_cup_result() -> dict[str, Any]:
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
        "Handle High": np.nan,
        "Handle Low": np.nan,
        "Handle Depth %": np.nan,
        "Handle Low Above Midpoint": False,
        "Right Side Recovery %": np.nan,
        "Pivot Price": np.nan,

        "Cup Big Green Volume Count": 0,
        "Cup Big Green Volume Dates": "",
        "Cup Big Green Max Volume Ratio": np.nan,
        "Cup Big Green Avg Volume Ratio": np.nan,

        "Right Side Big Green Volume Count": 0,
        "Right Side Big Green Volume Dates": "",
        "Right Side Big Green Max Volume Ratio": np.nan,
        "Right Side Big Green Avg Volume Ratio": np.nan,
    }


def summarize_big_green_volume_candles(
    df: pd.DataFrame,
    start_pos: int,
    end_pos_exclusive: int,
    min_volume_ratio: float,
) -> dict[str, Any]:
    """Summarize green weekly candles with large volume.

    Green big-volume candle:
    - Close > Open
    - Volume / Volume SMA20 >= min_volume_ratio
    """
    if start_pos < 0 or end_pos_exclusive <= start_pos:
        return {
            "Count": 0,
            "Dates": "",
            "Max Ratio": np.nan,
            "Avg Ratio": np.nan,
        }

    segment = df.iloc[start_pos:end_pos_exclusive].copy()
    if segment.empty:
        return {
            "Count": 0,
            "Dates": "",
            "Max Ratio": np.nan,
            "Avg Ratio": np.nan,
        }

    if "Volume SMA20" not in segment.columns:
        return {
            "Count": 0,
            "Dates": "",
            "Max Ratio": np.nan,
            "Avg Ratio": np.nan,
        }

    segment["Volume Ratio"] = segment["Volume"] / segment["Volume SMA20"]
    mask = (
        (segment["Close"] > segment["Open"])
        & segment["Volume Ratio"].replace([np.inf, -np.inf], np.nan).notna()
        & (segment["Volume Ratio"] >= min_volume_ratio)
    )

    hits = segment.loc[mask].copy()
    if hits.empty:
        return {
            "Count": 0,
            "Dates": "",
            "Max Ratio": np.nan,
            "Avg Ratio": np.nan,
        }

    details = []
    for idx, row in hits.iterrows():
        details.append(f"{pd.Timestamp(idx).date()}:{row['Volume Ratio']:.2f}x")

    return {
        "Count": int(len(hits)),
        "Dates": "; ".join(details),
        "Max Ratio": float(hits["Volume Ratio"].max()),
        "Avg Ratio": float(hits["Volume Ratio"].mean()),
    }


def detect_cup_handle_at(
    df: pd.DataFrame,
    i: int,
    min_cup_depth_pct: float,
    max_cup_depth_pct: float,
    big_green_volume_ratio: float,
) -> dict[str, Any]:
    """Detect a practical weekly cup-with-handle ending at row i.

    The breakout row i is expected to already pass the V8 breakout setup.
    This detector is used by the scanner to add cup/handle columns.
    """
    result = empty_cup_result()

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

        # Left rim should be close to the eventual breakout zone.
        if left_rim_high < float(breakout_level) * (1 - CUP_LEFT_RIM_TOLERANCE_PCT / 100):
            continue

        cup_len = i - left_pos
        if cup_len < CUP_MIN_TOTAL_WEEKS or cup_len > CUP_LOOKBACK_MAX_WEEKS:
            continue

        bottom_start = left_pos + 1
        bottom_end = handle_start
        if bottom_end - bottom_start < 3:
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

        # Need a meaningful left side and right side.
        if bottom_pos - left_pos < 1:
            continue
        if handle_start - bottom_pos < 2:
            continue

        cup_depth_pct = (left_rim_high - cup_bottom_low) / left_rim_high * 100
        if cup_depth_pct < min_cup_depth_pct or cup_depth_pct > max_cup_depth_pct:
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

        pivot_price = handle_high

        # Current breakout candle should close above handle pivot.
        if float(row["Close"]) <= pivot_price:
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

        cup_big_green = summarize_big_green_volume_candles(
            df=df,
            start_pos=left_pos,
            end_pos_exclusive=handle_start,
            min_volume_ratio=big_green_volume_ratio,
        )
        right_side_big_green = summarize_big_green_volume_candles(
            df=df,
            start_pos=bottom_pos + 1,
            end_pos_exclusive=handle_start,
            min_volume_ratio=big_green_volume_ratio,
        )

        return {
            "Cup Handle Pattern": True,
            "Cup Left Rim Date": str(pd.Timestamp(df.index[left_pos]).date()),
            "Cup Bottom Date": str(pd.Timestamp(df.index[bottom_pos]).date()),
            "Cup Handle Start Date": str(pd.Timestamp(df.index[handle_start]).date()),
            "Cup Left Rim High": left_rim_high,
            "Cup Bottom Low": cup_bottom_low,
            "Cup Depth %": cup_depth_pct,
            "Cup Length Weeks": float(cup_len),
            "Handle Length Weeks": float(handle_len),
            "Handle High": handle_high,
            "Handle Low": handle_low,
            "Handle Depth %": handle_depth_pct,
            "Handle Low Above Midpoint": bool(handle_low_above_midpoint),
            "Right Side Recovery %": right_side_recovery_pct,
            "Pivot Price": pivot_price,

            "Cup Big Green Volume Count": cup_big_green["Count"],
            "Cup Big Green Volume Dates": cup_big_green["Dates"],
            "Cup Big Green Max Volume Ratio": cup_big_green["Max Ratio"],
            "Cup Big Green Avg Volume Ratio": cup_big_green["Avg Ratio"],

            "Right Side Big Green Volume Count": right_side_big_green["Count"],
            "Right Side Big Green Volume Dates": right_side_big_green["Dates"],
            "Right Side Big Green Max Volume Ratio": right_side_big_green["Max Ratio"],
            "Right Side Big Green Avg Volume Ratio": right_side_big_green["Avg Ratio"],
        }

    return result


def build_candidate_record(ticker: str, row: pd.Series, cup: dict[str, Any]) -> dict[str, Any]:
    trigger_price = float(row["High"] * (1 + strat.ENTRY_TRIGGER_BUFFER_PCT))
    stop_price = float(row["Low"] * (1 - strat.STOP_BUFFER_PCT))
    risk_pct = (trigger_price - stop_price) / trigger_price * 100 if trigger_price > 0 else np.nan

    record = {
        "Ticker": ticker,
        "Signal Date": pd.Timestamp(row.name).date(),
        "Breakout Type": row["Breakout Type"],
        "Sector": row["Sector"],
        "Sector ETF": row["Sector ETF"],

        "Signal Open": float(row["Open"]),
        "Signal Close": float(row["Close"]),
        "Signal High": float(row["High"]),
        "Signal Low": float(row["Low"]),

        "Entry Trigger Price": trigger_price,
        "Stop Price": stop_price,
        "Initial Risk %": risk_pct,

        "Prior 52W High": float(row["Prior 52W High"]) if pd.notna(row["Prior 52W High"]) else np.nan,
        "Prior ATH High": float(row["Prior ATH High"]) if pd.notna(row["Prior ATH High"]) else np.nan,
        "Breakout Level": float(row["Breakout Level"]) if pd.notna(row["Breakout Level"]) else np.nan,
        "Close Above Breakout Level %": float(row["Close Above Breakout Level %"]) if pd.notna(row["Close Above Breakout Level %"]) else np.nan,

        "SMA20": float(row["SMA20"]) if pd.notna(row["SMA20"]) else np.nan,
        "SMA50": float(row["SMA50"]) if pd.notna(row["SMA50"]) else np.nan,
        "SMA20 vs SMA50 %": float(row["SMA20 vs SMA50 %"]) if pd.notna(row["SMA20 vs SMA50 %"]) else np.nan,

        "RS 5W vs SPY": float(row["RS 5W vs SPY"]),
        "RS Ratio ROC 5W": float(row["RS Ratio ROC 5W"]),
        "Sector RS 5W vs SPY": float(row["Sector RS 5W vs SPY"]),
        "Sector RS Ratio ROC 5W": float(row["Sector RS Ratio ROC 5W"]),
        "Stock RS 5W vs Sector": float(row["Stock RS 5W vs Sector"]),
        "Stock/Sector RS Ratio ROC 5W": float(row["Stock/Sector RS Ratio ROC 5W"]),

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

        "SPY Weekly Uptrend": bool(row["SPY Weekly Uptrend"]),
        "QQQ Weekly Uptrend": bool(row["QQQ Weekly Uptrend"]),
        "Sector ETF Weekly Uptrend": bool(row["Sector ETF Weekly Uptrend"]),
    }

    record.update(cup)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", type=str, default="market_cap_over_10b.csv")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--start", type=str, default=strat.START_DATE)
    parser.add_argument("--end", type=str, default=strat.END_DATE)
    parser.add_argument("--max-tickers", type=int, default=None)
    parser.add_argument("--min-cup-depth", type=float, default=CUP_MIN_DEPTH_PCT)
    parser.add_argument("--max-cup-depth", type=float, default=CUP_MAX_DEPTH_PCT)
    parser.add_argument("--big-green-volume-ratio", type=float, default=CUP_BIG_GREEN_VOLUME_RATIO)
    parser.add_argument(
        "--include-non-cup-v8-candidates",
        action="store_true",
        help="Also include V8 breakout candidates without a valid cup handle in the main output. Default: only cup-handle candidates.",
    )
    parser.add_argument(
        "--include-current-week",
        action="store_true",
        help="Use the latest weekly bar even if it may be incomplete. Default: false.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    universe_path = Path(args.universe)

    print("=" * 80)
    print("WEEKLY V9: V8 BREAKOUT + CUP-HANDLE INFO SCANNER")
    print("=" * 80)
    print("Base setup: Weekly V8 52W/ATH no-cup breakout.")
    print("Added: cup depth, handle depth, big-volume green candles inside cup.")
    print(f"Cup depth scan range: {args.min_cup_depth:.1f}% to {args.max_cup_depth:.1f}%")
    print(f"Big-volume green candle threshold: Volume / 20W avg volume >= {args.big_green_volume_ratio:.2f}x")
    print("=" * 80)

    tickers = strat.load_universe(universe_path, args.max_tickers)
    print(f"Loaded tickers: {len(tickers)}")

    print("Downloading SPY and QQQ...")
    spy_raw = strat.download_weekly(strat.SPY_TICKER, args.start, args.end)
    qqq_raw = strat.download_weekly(strat.QQQ_TICKER, args.start, args.end)
    spy_raw = drop_incomplete_latest_week(spy_raw, args.include_current_week)
    qqq_raw = drop_incomplete_latest_week(qqq_raw, args.include_current_week)

    if spy_raw.empty or qqq_raw.empty:
        raise RuntimeError("Could not download enough SPY/QQQ weekly data.")

    spy, qqq = strat.prepare_etf_data(spy_raw, qqq_raw)

    sector_cache_path = output_dir / strat.SECTOR_CACHE_FILE
    sector_map = strat.build_sector_etf_map(universe_path, tickers, sector_cache_path)
    unique_sector_etfs = sorted({data["Sector ETF"] for data in sector_map.values() if data.get("Sector ETF")})

    print(f"Downloading sector ETFs: {', '.join(unique_sector_etfs)}")
    sector_data: dict[str, pd.DataFrame] = {}
    for etf in unique_sector_etfs:
        raw = strat.download_weekly(etf, args.start, args.end)
        raw = drop_incomplete_latest_week(raw, args.include_current_week)
        if raw.empty:
            print(f"Warning: missing sector ETF data for {etf}")
            continue
        sector_data[etf] = strat.prepare_sector_etf_data(raw, etf)

    cup_candidates: list[dict[str, Any]] = []
    all_v8_candidates: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for idx, ticker in enumerate(tickers, start=1):
        print(f"[{idx}/{len(tickers)}] {ticker}")
        try:
            sector_info = sector_map.get(ticker)
            if not sector_info:
                errors.append({"Ticker": ticker, "Error": "Missing sector ETF mapping"})
                continue

            sector_name = sector_info.get("Sector", "")
            sector_etf = sector_info.get("Sector ETF", "")
            sector_df = sector_data.get(sector_etf)
            if sector_df is None or sector_df.empty:
                errors.append({"Ticker": ticker, "Sector": sector_name, "Sector ETF": sector_etf, "Error": "Missing sector ETF data"})
                continue

            raw = strat.download_weekly(ticker, args.start, args.end)
            raw = drop_incomplete_latest_week(raw, args.include_current_week)
            if raw.empty or len(raw) < strat.SMA_LONG + 20:
                errors.append({"Ticker": ticker, "Sector": sector_name, "Sector ETF": sector_etf, "Error": "Not enough weekly data"})
                continue

            df = strat.add_relative_strength(raw, spy, qqq, sector_df, sector_name, sector_etf)
            if df.empty:
                continue

            row = df.iloc[-1]

            # First apply your V8 breakout setup.
            if not strat.setup_passes(row):
                continue

            cup = detect_cup_handle_at(
                df=df,
                i=len(df) - 1,
                min_cup_depth_pct=args.min_cup_depth,
                max_cup_depth_pct=args.max_cup_depth,
                big_green_volume_ratio=args.big_green_volume_ratio,
            )

            record = build_candidate_record(ticker, row, cup)
            all_v8_candidates.append(record)

            if bool(cup["Cup Handle Pattern"]) or args.include_non_cup_v8_candidates:
                cup_candidates.append(record)

        except Exception as exc:
            errors.append({"Ticker": ticker, "Error": str(exc), "Traceback": traceback.format_exc()})

    candidates_df = pd.DataFrame(cup_candidates)
    if not candidates_df.empty:
        candidates_df = candidates_df.sort_values(
            [
                "Cup Handle Pattern",
                "Right Side Big Green Volume Count",
                "Cup Big Green Volume Count",
                "Stock RS 5W vs Sector",
                "RS 5W vs SPY",
                "Breakout Volume Ratio",
            ],
            ascending=[False, False, False, False, False, False],
        )

    all_v8_df = pd.DataFrame(all_v8_candidates)
    if not all_v8_df.empty:
        all_v8_df = all_v8_df.sort_values(
            [
                "Cup Handle Pattern",
                "Right Side Big Green Volume Count",
                "Cup Big Green Volume Count",
                "Stock RS 5W vs Sector",
                "RS 5W vs SPY",
                "Breakout Volume Ratio",
            ],
            ascending=[False, False, False, False, False, False],
        )

    output_path = output_dir / SCAN_OUTPUT
    all_v8_path = output_dir / ALL_V8_OUTPUT
    errors_path = output_dir / ERROR_OUTPUT

    candidates_df.to_csv(output_path, index=False)
    all_v8_df.to_csv(all_v8_path, index=False)
    pd.DataFrame(errors).to_csv(errors_path, index=False)

    print("\nDone.")
    print(f"Cup-handle candidates saved: {len(candidates_df)} -> {output_path}")
    print(f"All V8 candidates with cup info saved: {len(all_v8_df)} -> {all_v8_path}")
    print(f"Errors saved: {errors_path}")


if __name__ == "__main__":
    main()
