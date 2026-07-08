#!/usr/bin/env python3
"""
Weekly V6 Cup-and-Handle Scanner

Scans the latest completed weekly candle for the same entry setup used in:
weekly_v6_rs5_52w_ath_cup_handle_backtest.py

Run:
python weekly_v6_rs5_52w_ath_cup_handle_scanner.py --universe market_cap_over_10b.csv

Output:
weekly_v6_rs5_52w_ath_cup_handle_scan_candidates.csv
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import weekly_v6_rs5_52w_ath_cup_handle_backtest as strat


SCAN_OUTPUT = "weekly_v6_rs5_52w_ath_cup_handle_scan_candidates.csv"


def drop_incomplete_latest_week(df: pd.DataFrame, keep_latest: bool = False) -> pd.DataFrame:
    if keep_latest or df.empty:
        return df

    latest_date = pd.Timestamp(df.index[-1]).tz_localize(None)
    now = pd.Timestamp.utcnow().tz_localize(None)
    if (now - latest_date).days < 5:
        return df.iloc[:-1].copy()
    return df


def build_candidate_record(ticker: str, row: pd.Series) -> dict[str, Any]:
    trigger_price = float(row["High"] * (1 + strat.ENTRY_TRIGGER_BUFFER_PCT))
    stop_price = float(row["Low"] * (1 - strat.STOP_BUFFER_PCT))
    risk_pct = (trigger_price - stop_price) / trigger_price * 100 if trigger_price > 0 else np.nan

    return {
        "Ticker": ticker,
        "Signal Date": pd.Timestamp(row.name).date(),
        "Breakout Type": row["Breakout Type"],
        "Sector": row["Sector"],
        "Sector ETF": row["Sector ETF"],

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
        "Right Side Recovery %": float(row.get("Right Side Recovery %", np.nan)) if pd.notna(row.get("Right Side Recovery %", np.nan)) else np.nan,

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", type=str, default="market_cap_over_10b.csv")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--start", type=str, default=strat.START_DATE)
    parser.add_argument("--end", type=str, default=strat.END_DATE)
    parser.add_argument("--max-tickers", type=int, default=None)
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
    print("WEEKLY V6 CUP-AND-HANDLE SCANNER")
    print("=" * 80)
    print("Scanning latest completed weekly candle.")
    print("Entry trigger is next week above the signal candle high.")
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

    candidates: list[dict[str, Any]] = []
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
            if strat.setup_passes(row):
                candidates.append(build_candidate_record(ticker, row))

        except Exception as exc:
            errors.append({"Ticker": ticker, "Error": str(exc), "Traceback": traceback.format_exc()})

    candidates_df = pd.DataFrame(candidates)
    if not candidates_df.empty:
        candidates_df = candidates_df.sort_values(
            ["Cup Depth %", "Stock RS 5W vs Sector", "RS 5W vs SPY"],
            ascending=[True, False, False],
        )

    output_path = output_dir / SCAN_OUTPUT
    candidates_df.to_csv(output_path, index=False)

    errors_df = pd.DataFrame(errors)
    errors_df.to_csv(output_dir / "weekly_v6_scanner_errors.csv", index=False)

    print("\nDone.")
    print(f"Candidates: {len(candidates_df)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
