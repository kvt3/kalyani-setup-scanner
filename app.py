from __future__ import annotations

import calendar
import json
import importlib
import re
import subprocess
import sys
from datetime import date, timedelta
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import market_breadth
import market_trend
import trade_journal
import fundamentals as fundamentals_module
from config import APP_NAME, DATA_DIR, DB_PATH, MIN_MARKET_CAP
from data_loader import latest_completed_us_session
from database import (
    list_scan_dates,
    list_rule_watchlists,
    load_eligible_ticker_symbols,
    load_eligible_tickers,
    load_rule_watchlist,
    load_universe_metadata,
    load_watchlist,
    save_eligible_tickers,
    save_rule_watchlist,
    save_watchlist,
)
import rule_config as rule_config_module
import scanner as scanner_module
import scan_state as scan_state_module
import schedule_config as schedule_config_module

trade_journal = importlib.reload(trade_journal)
fundamentals_module = importlib.reload(fundamentals_module)
rule_config_module = importlib.reload(rule_config_module)
scanner_module = importlib.reload(scanner_module)
scan_state_module = importlib.reload(scan_state_module)
schedule_config_module = importlib.reload(schedule_config_module)

from cloud_sqlite_store import cloud_sqlite_status
from rule_config import (
    MarubozuRuleConfig,
    MorningStarRuleConfig,
    PullbackRuleConfig,
    RuleConfig,
    TechnicalStrengthRuleConfig,
    WeeklyATHRuleConfig,
    WeeklyMomentumRuleConfig,
    load_rule_config,
    save_rule_config,
)
from scanner import (
    explain_rules,
    run_marubozu_breakout_scan,
    run_morning_star_scan,
    run_monthly_big_volume_scan,
    run_saved_database_scan,
    run_score_above60_setup_scan,
    run_top50_strength_score_scan,
    run_technical_breakout_scan,
    run_technical_pullback_scan,
    run_technical_strength_scan,
    run_weekly_v6_cup_handle_scan,
    run_weekly_ath_breakout_scan,
    run_weekly_momentum_scan,
)
from scan_state import (
    load_latest_marubozu_scan_state,
    load_latest_morning_star_scan_state,
    load_latest_monthly_big_volume_scan_state,
    load_latest_scan_state,
    load_latest_score_above60_setup_scan_state,
    load_latest_technical_breakout_scan_state,
    load_latest_technical_pullback_scan_state,
    load_latest_technical_strength_scan_state,
    load_latest_top50_strength_score_scan_state,
    load_latest_weekly_v6_cup_handle_scan_state,
    load_latest_weekly_ath_scan_state,
    load_latest_weekly_momentum_scan_state,
    save_latest_marubozu_scan_state,
    save_latest_morning_star_scan_state,
    save_latest_monthly_big_volume_scan_state,
    save_latest_scan_state,
    save_latest_score_above60_setup_scan_state,
    save_latest_technical_breakout_scan_state,
    save_latest_technical_pullback_scan_state,
    save_latest_technical_strength_scan_state,
    save_latest_top50_strength_score_scan_state,
    save_latest_weekly_v6_cup_handle_scan_state,
    save_latest_weekly_ath_scan_state,
    save_latest_weekly_momentum_scan_state,
)
from schedule_config import RULE_LABELS, RULE_ORDER, RuleSchedule, ScheduleConfig, WEEKDAY_NAMES, load_schedule_config, save_schedule_config
from stock_tracking import (
    add_tracked_stock,
    analyze_pasted_transcript_with_metrics,
    analyze_transcript_url_with_metrics,
    auto_refresh_tracker_panel_if_due,
    auto_scan_reported_earnings_if_due,
    clear_earnings_calendar_detail_cache,
    delete_tracked_stock,
    extract_feed_headline,
    earnings_calendar_rows_for_tracker,
    fetch_and_save_motley_fool_takeaways_for_ticker,
    fetch_and_save_sec_earnings_release_summary_for_ticker,
    format_market_cap,
    get_live_stock_tracker_details,
    get_stock_details,
    infer_sentiment,
    list_tracked_stocks,
    refresh_live_data_for_tracked_stock,
    save_growth_tables,
    save_yesterday_calendar_metrics_for_ticker,
    scan_recent_earnings_growth_and_add_to_tracker,
    tracking_database_exists,
)
from universe_builder import build_eligible_market_cap_universe


st.set_page_config(page_title=APP_NAME, layout="wide")

MANUAL_SCAN_STATUS_PATH = DATA_DIR / "manual_scan_status.json"
MANUAL_SCAN_LOG_PATH = DATA_DIR / "manual_scan_background.log"


@st.cache_data(ttl=900, show_spinner=False)
def cached_market_trend_scan(cache_version: int = 5) -> dict[str, object]:
    latest_market_trend = importlib.reload(market_trend)
    return latest_market_trend.run_market_trend_scanner()


@st.cache_data(ttl=900, show_spinner=False)
def cached_sector_etf_scan(cache_version: int = 1) -> dict[str, object]:
    latest_market_trend = importlib.reload(market_trend)
    return latest_market_trend.run_sector_etf_scan()


@st.cache_data(ttl=14_400, show_spinner=False)
def cached_sector_top_gainers_scan(universe_signature: str, cache_version: int = 1) -> dict[str, object]:
    latest_market_trend = importlib.reload(market_trend)
    return latest_market_trend.run_sector_top_gainers_scan(max_per_sector=5)


@st.cache_data(ttl=14_400, show_spinner=False)
def cached_market_breadth_scan(tickers: tuple[str, ...], cache_version: int = 7) -> dict[str, object]:
    latest_market_breadth = importlib.reload(market_breadth)
    return latest_market_breadth.run_market_breadth_scan(list(tickers))


def load_manual_scan_status() -> dict[str, object]:
    if not MANUAL_SCAN_STATUS_PATH.exists():
        return {}
    try:
        return json.loads(MANUAL_SCAN_STATUS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def start_background_scan(rule_keys: list[str]) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MANUAL_SCAN_STATUS_PATH.write_text(
        json.dumps(
            {
                "status": "starting",
                "rules": rule_keys,
                "rule_labels": [RULE_LABELS[rule_key] for rule_key in rule_keys],
                "message": "Starting background scan.",
                "summaries": [],
                "updated_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "scheduled_scan.py"),
        "--rules",
        ",".join(rule_keys),
        "--status-file",
        str(MANUAL_SCAN_STATUS_PATH),
    ]
    with MANUAL_SCAN_LOG_PATH.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parent),
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )
    return int(process.pid)


def show_background_scan_status(active_rule_keys: set[str] | None = None) -> bool:
    status = load_manual_scan_status()
    if not status:
        return False
    state = str(status.get("status", ""))
    labels = ", ".join(str(label) for label in status.get("rule_labels", []) if label)
    updated_at = str(status.get("updated_at", ""))
    active_rule_keys = active_rule_keys or set(RULE_LABELS)
    status_rule_keys = {str(rule_key) for rule_key in status.get("rules", []) if str(rule_key)}
    if status_rule_keys and not status_rule_keys.intersection(active_rule_keys):
        return False
    if state in {"starting", "running"}:
        message = str(status.get("message") or "")
        st.info(f"Background scan running: {labels or 'selected rules'}")
        if message:
            st.caption(message)
        progress = status.get("progress")
        if isinstance(progress, dict):
            earnings_checked = progress.get("earnings_checked")
            earnings_total = progress.get("earnings_total")
            if earnings_checked is not None and earnings_total:
                st.progress(
                    min(float(earnings_checked) / max(float(earnings_total), 1.0), 1.0),
                    text=f"Earnings checked: {int(earnings_checked):,} / {int(earnings_total):,}",
                )
        if updated_at:
            st.caption(f"Last update: {updated_at}")
        return True
    if state == "completed":
        summaries = status.get("summaries", [])
        if st.session_state.get("manual_scan_seen_completed_at") != updated_at:
            for key in (
                "latest_stats",
                "latest_results",
                "latest_errors",
                "latest_scanned_count",
                "latest_saved_count",
                "marubozu_results",
                "marubozu_errors",
                "marubozu_stats",
                "morning_star_results",
                "morning_star_errors",
                "morning_star_stats",
                "weekly_ath_results",
                "weekly_ath_errors",
                "weekly_ath_stats",
                "weekly_momentum_results",
                "weekly_momentum_errors",
                "weekly_momentum_stats",
                "technical_strength_results",
                "technical_strength_errors",
                "technical_strength_stats",
                "technical_breakout_results",
                "technical_breakout_errors",
                "technical_breakout_stats",
                "technical_pullback_results",
                "technical_pullback_errors",
                "technical_pullback_stats",
                "monthly_big_volume_results",
                "monthly_big_volume_errors",
                "monthly_big_volume_stats",
                "score_above60_setup_results",
                "score_above60_setup_errors",
                "score_above60_setup_stats",
                "top50_strength_results",
                "top50_strength_errors",
                "top50_strength_stats",
            ):
                st.session_state.pop(key, None)
            st.session_state["manual_scan_seen_completed_at"] = updated_at
        st.success("Background scan complete.")
        if isinstance(summaries, list) and summaries:
            st.caption(" | ".join(str(item) for item in summaries))
        elif updated_at:
            st.caption(f"Completed at: {updated_at}")
        return False
    if state == "failed":
        st.error(f"Background scan failed: {status.get('message', 'Unknown error')}")
        if updated_at:
            st.caption(f"Failed at: {updated_at}")
        return False
    return False


def format_percent_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.map(lambda value: f"{value * 100:.1f}%" if pd.notna(value) else "")


def format_dashboard(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    risk = out["entry"] - out["stop"]
    reward = out["target"] - out["entry"]
    out["risk:reward"] = [
        f"1:{value:.2f}" if pd.notna(value) and value > 0 else "-"
        for value in (reward / risk.replace(0, pd.NA))
    ]
    out["revenue growth"] = (out["revenue_growth"] * 100).round(1)
    out["EPS growth"] = (out["eps_growth"] * 100).round(1)
    out["market cap"] = (out["market_cap"] / 1_000_000).round(0).astype("Int64").astype(str) + "M"
    out["avg volume"] = out["average_volume"].round(0).astype("Int64")
    return out[
        [
            "ticker",
            "close",
            "entry",
            "stop",
            "target",
            "risk:reward",
            "setup_grade",
            "reason",
            "revenue growth",
            "EPS growth",
            "market cap",
            "avg volume",
            "signal_date",
        ]
    ].rename(columns={"setup_grade": "setup grade", "signal_date": "signal date"})


def show_rules() -> None:
    rules = explain_rules()
    with st.expander("Setup rules", expanded=False):
        st.write(f"Technical Breakout: {rules['technical_breakout']}")
        st.write(f"9 EMA Pullback: {rules['technical_pullback_9ema']}")
        st.write("The scan uses completed U.S. market candles only and filters out incomplete daily or weekly sessions.")


def show_scan_summary(scanned_count: int, match_count: int, skipped_count: int, saved_count: int | None = None) -> None:
    columns = st.columns(4)
    columns[0].metric("Tickers scanned", f"{scanned_count:,}")
    columns[1].metric("Matches", f"{match_count:,}")
    columns[2].metric("Skipped/data issues", f"{skipped_count:,}")
    columns[3].metric("Saved", "-" if saved_count is None else f"{saved_count:,}")


def format_price_candidates(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    out = frame.copy()
    if "risk_per_share" not in out.columns:
        out["risk_per_share"] = out["entry"] - out["stop"]
    if "quantity_for_1000_risk" not in out.columns:
        out["quantity_for_1000_risk"] = (1000 // out["risk_per_share"]).where(out["risk_per_share"] > 0, 0)
    reward = out["target"] - out["entry"]
    out["risk:reward"] = [
        f"1:{value:.2f}" if pd.notna(value) and value > 0 else "-"
        for value in (reward / out["risk_per_share"].replace(0, pd.NA))
    ]
    out["risk_per_share"] = out["risk_per_share"].round(2)
    out["quantity_for_1000_risk"] = out["quantity_for_1000_risk"].astype("Int64")
    out["market cap"] = (out["market_cap"] / 1_000_000).round(0).astype("Int64").astype(str) + "M"
    out["volume"] = out["volume"].round(0).astype("Int64")
    out["avg volume 20d"] = out["avg_volume_20d"].round(0).astype("Int64")
    return out[
        [
            "ticker",
            "signal_date",
            "close",
            "entry",
            "stop",
            "target",
            "risk_per_share",
            "quantity_for_1000_risk",
            "risk:reward",
            "pattern",
            "pullback_ma",
            "distance_to_9ma_pct",
            "volume",
            "avg volume 20d",
            "market cap",
        ]
    ].rename(
        columns={
            "signal_date": "signal date",
            "risk_per_share": "risk/share",
            "quantity_for_1000_risk": "quantity for $1,000 risk",
            "pullback_ma": "pullback MA",
            "distance_to_9ma_pct": "distance to 9MA %",
        }
    )


def show_scan_stats(stats: dict[str, object]) -> None:
    if not stats:
        return
    def fmt_stat(value: object) -> str:
        try:
            return f"{int(value):,}"
        except (TypeError, ValueError):
            return str(value)

    st.subheader("Scan Summary")
    summary = pd.DataFrame(
        [
            {"Metric": "Stored tickers scanned", "Value": fmt_stat(stats.get("stored_tickers", 0))},
            {"Metric": "Latest completed U.S. candle used", "Value": stats.get("scan_date", "-")},
            {"Metric": "OHLCV loaded", "Value": fmt_stat(stats.get("ohlcv_loaded", 0))},
            {
                "Metric": "Price/setup candidates before fundamentals",
                "Value": fmt_stat(stats.get("price_candidates", 0)),
            },
            {
                "Metric": "Final matches after active fundamental filters",
                "Value": fmt_stat(stats.get("matches", 0)),
            },
            {"Metric": "Saved to watchlist", "Value": fmt_stat(stats.get("saved_count", 0))},
            {"Metric": "Missing OHLCV/data issues", "Value": fmt_stat(stats.get("error_count", 0))},
        ]
    )
    st.dataframe(summary, width="stretch", hide_index=True)

    candidate_rows = stats.get("price_candidate_rows", [])
    if isinstance(candidate_rows, list) and candidate_rows:
        st.subheader(f"Price/setup candidates before fundamentals: {len(candidate_rows):,}")
        st.dataframe(format_price_candidates(candidate_rows), width="stretch", hide_index=True)


def show_eligible_universe_builder() -> None:
    st.subheader("Stored $500M+ Universe")
    if "universe_refresh_message" in st.session_state:
        st.success(st.session_state.pop("universe_refresh_message"))

    eligible = load_eligible_tickers(DB_PATH)
    metadata = load_universe_metadata(DB_PATH)
    columns = st.columns(3)
    columns[0].metric("Stored tickers", f"{len(eligible):,}")
    columns[1].metric("Minimum market cap", f"${MIN_MARKET_CAP / 1_000_000:.0f}M")
    columns[2].metric("Last refresh", metadata.get("last_refreshed", "-")[:19])
    if metadata:
        st.caption(
            f"Source: {metadata.get('source', '-')}. "
            f"Checked: {metadata.get('total_tickers_checked', '-')}. "
            f"Market-cap gaps: {metadata.get('market_cap_error_count', '-')}."
        )

    if st.button("Download $500M+ NASDAQ/NYSE Tickers"):
        progress_bar = st.progress(0)
        status = st.empty()

        def update_progress(done: int, total: int, kept: int, ticker: str) -> None:
            progress_bar.progress(done / total if total else 0)
            status.write(f"Checked {done:,}/{total:,}: {ticker}. Kept {kept:,} tickers.")

        with st.spinner("Downloading NASDAQ/NYSE tickers and checking market caps..."):
            result = build_eligible_market_cap_universe(
                min_market_cap=MIN_MARKET_CAP,
                progress_callback=update_progress,
            )
            saved = save_eligible_tickers(
                result.rows,
                source=result.source,
                min_market_cap=MIN_MARKET_CAP,
                total_tickers=result.total_tickers,
                error_count=len(result.errors),
                db_path=DB_PATH,
            )

        progress_bar.empty()
        status.empty()
        st.session_state["universe_refresh_message"] = (
            f"Stored {saved:,} tickers over ${MIN_MARKET_CAP / 1_000_000:.0f}M market cap "
            f"from {result.total_tickers:,} NASDAQ/NYSE symbols."
        )
        if result.errors:
            with st.expander(f"Market-cap lookup issues ({len(result.errors):,})"):
                st.text("\n".join(result.errors[:500]))
                if len(result.errors) > 500:
                    st.caption("Showing first 500 issues.")
        st.rerun()

    if not eligible.empty:
        with st.expander("Preview stored eligible tickers"):
            preview = eligible.head(50).copy()
            preview["market_cap"] = (preview["market_cap"] / 1_000_000).round(0).astype("Int64").astype(str) + "M"
            st.dataframe(preview, width="stretch", hide_index=True)


def format_marubozu_results(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["volume"] = out["volume"].round(0).astype("Int64")
    out["avg volume 20d"] = out["avg_volume_20d"].round(0).astype("Int64")
    if "volume_2" in out.columns:
        out["star volume"] = out["volume_2"].round(0).astype("Int64")
    out["market cap"] = (out["market_cap"] / 1_000_000).round(0).astype("Int64").astype(str) + "M"
    return out[
        [
            "ticker",
            "signal_date",
            "open",
            "high",
            "low",
            "close",
            "prior_52w_high",
            "volume_ratio",
            "body_pct",
            "upper_wick_pct",
            "lower_wick_pct",
            "volume",
            "avg volume 20d",
            "market cap",
        ]
    ].rename(
        columns={
            "signal_date": "signal date",
            "prior_52w_high": "prior 52W high",
            "volume_ratio": "volume/avg",
            "body_pct": "body %",
            "upper_wick_pct": "upper wick %",
            "lower_wick_pct": "lower wick %",
        }
    )


def format_morning_star_results(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    optional_columns = {
        "first_body_pct": pd.NA,
        "third_body_pct": pd.NA,
        "gap_tolerance_pct": pd.NA,
        "volume_ratio_2": pd.NA,
        "volume_2": pd.NA,
    }
    for column, default in optional_columns.items():
        if column not in out.columns:
            out[column] = default
    out["volume"] = out["volume"].round(0).astype("Int64")
    out["avg volume 20d"] = out["avg_volume_20d"].round(0).astype("Int64")
    out["star volume"] = pd.to_numeric(out["volume_2"], errors="coerce").round(0).astype("Int64")
    out["market cap"] = (out["market_cap"] / 1_000_000).round(0).astype("Int64").astype(str) + "M"
    out["risk:reward"] = [
        f"1:{value:.2f}" if pd.notna(value) and value > 0 else "-"
        for value in pd.to_numeric(out["risk_reward"], errors="coerce")
    ]
    return out[
        [
            "ticker",
            "signal_date",
            "pattern",
            "close",
            "entry",
            "stop",
            "target",
            "risk:reward",
            "pattern_low",
            "first_body_pct",
            "third_body_pct",
            "gap_tolerance_pct",
            "volume_ratio_2",
            "volume_ratio",
            "star volume",
            "volume",
            "avg volume 20d",
            "ema_21",
            "sma_50",
            "sma_200",
            "market cap",
            "reason",
        ]
    ].rename(
        columns={
            "signal_date": "signal date",
            "pattern_low": "pattern low",
            "first_body_pct": "first body %",
            "third_body_pct": "third body %",
            "gap_tolerance_pct": "gap tolerance %",
            "volume_ratio_2": "star volume/avg",
            "volume_ratio": "volume/avg",
            "ema_21": "21 EMA",
            "sma_50": "50 SMA",
            "sma_200": "200 SMA",
        }
    )


def format_monthly_big_volume_results(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column in ["volume", "previous_month_volume", "avg_volume_12m"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").round(0).astype("Int64")
    if "market_cap" in out.columns:
        out["market cap"] = (pd.to_numeric(out["market_cap"], errors="coerce") / 1_000_000).round(0).astype("Int64").astype(str) + "M"
    columns = [
        "ticker",
        "signal_month",
        "pattern",
        "open",
        "high",
        "low",
        "close",
        "volume_vs_previous",
        "volume_vs_12m_avg",
        "body_pct",
        "lower_wick_body_ratio",
        "upper_wick_pct",
        "close_position_pct",
        "volume",
        "previous_month_volume",
        "avg_volume_12m",
        "market cap",
        "reason",
    ]
    return out[[column for column in columns if column in out.columns]].rename(
        columns={
            "signal_month": "signal month",
            "volume_vs_previous": "volume/prev month",
            "volume_vs_12m_avg": "volume/12M avg",
            "body_pct": "body %",
            "lower_wick_body_ratio": "lower wick/body",
            "upper_wick_pct": "upper wick %",
            "close_position_pct": "close position %",
            "previous_month_volume": "prev month volume",
            "avg_volume_12m": "12M avg volume",
        }
    )


def format_weekly_ath_results(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    optional_columns = [
        "revenue_growth",
        "eps_growth",
        "average_volume",
        "annual_eps_period",
        "annual_eps_date",
        "annual_reported_eps",
        "annual_prior_eps_period",
        "annual_prior_eps_date",
        "annual_prior_eps",
        "annual_eps_growth",
        "quarterly_eps_period",
        "quarterly_eps_date",
        "quarterly_reported_eps",
        "quarterly_prior_eps_period",
        "quarterly_prior_eps_date",
        "quarterly_prior_eps",
        "quarterly_eps_growth",
    ]
    for column in optional_columns:
        if column not in out.columns:
            out[column] = pd.NA
    numeric_optional_columns = [
        "revenue_growth",
        "eps_growth",
        "average_volume",
        "annual_reported_eps",
        "annual_prior_eps",
        "annual_eps_growth",
        "quarterly_reported_eps",
        "quarterly_prior_eps",
        "quarterly_eps_growth",
    ]
    for column in numeric_optional_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["volume"] = out["volume"].round(0).astype("Int64")
    out["avg volume 20w"] = out["avg_volume_20w"].round(0).astype("Int64")
    out["market cap"] = (out["market_cap"] / 1_000_000).round(0).astype("Int64").astype(str) + "M"
    out["revenue growth"] = (out["revenue_growth"] * 100).round(1)
    out["EPS growth"] = (out["eps_growth"] * 100).round(1)
    out["annual EPS growth"] = (out["annual_eps_growth"] * 100).round(1)
    out["quarterly EPS growth"] = (out["quarterly_eps_growth"] * 100).round(1)
    out["avg daily volume"] = out["average_volume"].round(0).astype("Int64")
    return out[
        [
            "ticker",
            "signal_week",
            "open",
            "high",
            "low",
            "close",
            "prior_all_time_high",
            "volume_ratio",
            "volume",
            "avg volume 20w",
            "ema_10w",
            "sma_30w",
            "market cap",
            "revenue growth",
            "EPS growth",
            "avg daily volume",
            "annual_eps_period",
            "annual_eps_date",
            "annual_reported_eps",
            "annual_prior_eps_period",
            "annual_prior_eps_date",
            "annual_prior_eps",
            "annual EPS growth",
            "quarterly_eps_period",
            "quarterly_eps_date",
            "quarterly_reported_eps",
            "quarterly_prior_eps_period",
            "quarterly_prior_eps_date",
            "quarterly_prior_eps",
            "quarterly EPS growth",
            "uptrend",
        ]
    ].rename(
        columns={
            "signal_week": "signal week",
            "prior_all_time_high": "prior ATH",
            "volume_ratio": "volume/avg",
            "ema_10w": "10W EMA",
            "sma_30w": "30W SMA",
            "annual_eps_period": "annual EPS period",
            "annual_eps_date": "annual EPS date",
            "annual_reported_eps": "annual reported EPS",
            "annual_prior_eps_period": "annual compare period",
            "annual_prior_eps_date": "annual compare date",
            "annual_prior_eps": "annual prior EPS",
            "quarterly_eps_period": "quarterly EPS period",
            "quarterly_eps_date": "quarterly EPS date",
            "quarterly_reported_eps": "quarterly reported EPS",
            "quarterly_prior_eps_period": "quarterly compare period",
            "quarterly_prior_eps_date": "quarterly compare date",
            "quarterly_prior_eps": "quarterly prior EPS",
        }
    )


def format_weekly_momentum_results(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    optional_columns = [
        "revenue_growth",
        "eps_growth",
        "average_volume_weekly",
        "annual_eps_period",
        "annual_eps_date",
        "annual_reported_eps",
        "quarterly_revenue_period",
        "quarterly_revenue_date",
        "quarterly_revenue",
        "quarterly_prior_revenue_period",
        "quarterly_prior_revenue_date",
        "quarterly_prior_revenue",
        "quarterly_revenue_growth",
        "quarterly_eps_period",
        "quarterly_eps_date",
        "quarterly_reported_eps",
        "quarterly_prior_eps_period",
        "quarterly_prior_eps_date",
        "quarterly_prior_eps",
        "quarterly_eps_growth",
        "fundamental_status",
    ]
    for column in optional_columns:
        if column not in out.columns:
            out[column] = pd.NA
    numeric_columns = [
        "revenue_growth",
        "eps_growth",
        "average_volume_weekly",
        "annual_reported_eps",
        "quarterly_revenue",
        "quarterly_prior_revenue",
        "quarterly_revenue_growth",
        "quarterly_reported_eps",
        "quarterly_prior_eps",
        "quarterly_eps_growth",
    ]
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["weekly volume"] = out["volume"].round(0).astype("Int64")
    out["avg weekly volume"] = out["avg_volume_20w"].round(0).astype("Int64")
    out["market cap"] = (out["market_cap"] / 1_000_000).round(0).astype("Int64").astype(str) + "M"
    out["revenue growth"] = format_percent_series(out["revenue_growth"])
    out["EPS growth"] = format_percent_series(out["eps_growth"])
    out["QoQ revenue growth"] = format_percent_series(out["quarterly_revenue_growth"])
    out["QoQ EPS growth"] = format_percent_series(out["quarterly_eps_growth"])
    out["quarterly revenue"] = (out["quarterly_revenue"] / 1_000_000).round(1)
    out["quarterly prior revenue"] = (out["quarterly_prior_revenue"] / 1_000_000).round(1)
    out["risk:reward"] = [
        f"1:{value:.2f}" if pd.notna(value) and value > 0 else "-"
        for value in pd.to_numeric(out["risk_reward"], errors="coerce")
    ]
    return out[
        [
            "ticker",
            "signal_week",
            "pattern",
            "close",
            "entry",
            "stop",
            "target",
            "risk:reward",
            "pullback_ma",
            "distance_to_9w_ma_pct",
            "weekly volume",
            "avg weekly volume",
            "volume_ratio",
            "ema_9w",
            "sma_9w",
            "ema_21w",
            "sma_50w",
            "sma_200w",
            "market cap",
            "revenue growth",
            "EPS growth",
            "quarterly_revenue_period",
            "quarterly_revenue_date",
            "QoQ revenue growth",
            "quarterly revenue",
            "quarterly prior revenue",
            "quarterly_eps_period",
            "quarterly_eps_date",
            "QoQ EPS growth",
            "quarterly_reported_eps",
            "quarterly_prior_eps",
            "fundamental_status",
            "annual_eps_period",
            "annual_eps_date",
            "annual_reported_eps",
            "uptrend",
        ]
    ].rename(
        columns={
            "signal_week": "signal week",
            "pullback_ma": "pullback MA",
            "distance_to_9w_ma_pct": "distance to 9W MA %",
            "volume_ratio": "volume/avg",
            "ema_9w": "9W EMA",
            "sma_9w": "9W SMA",
            "ema_21w": "21W EMA",
            "sma_50w": "50W SMA",
            "sma_200w": "200W SMA",
            "quarterly_revenue_period": "QoQ revenue period",
            "quarterly_revenue_date": "QoQ revenue date",
            "quarterly revenue": "quarterly revenue $M",
            "quarterly prior revenue": "prior quarter revenue $M",
            "quarterly_eps_period": "QoQ EPS period",
            "quarterly_eps_date": "QoQ EPS date",
            "quarterly_reported_eps": "quarterly EPS",
            "quarterly_prior_eps": "prior quarter EPS",
            "annual_eps_period": "annual EPS period",
            "annual_eps_date": "annual EPS date",
            "annual_reported_eps": "annual reported EPS",
            "fundamental_status": "fundamental status",
        }
    )


def format_technical_strength_results(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column in ["volume", "avg_volume_20d"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").round(0).astype("Int64")
    if "market_cap" in out.columns:
        out["market cap"] = (pd.to_numeric(out["market_cap"], errors="coerce") / 1_000_000).round(0).astype("Int64").astype(str) + "M"
    columns = [
        "ticker",
        "signal_date",
        "latest_date",
        "setup_type",
        "days_since_breakout",
        "close",
        "entry",
        "stop",
        "target_2r",
        "risk_reward",
        "breakout_level",
        "breakout_high",
        "breakout_close",
        "breakout_volume_ratio",
        "latest_volume_ratio",
        "rsi_14",
        "rs_20d_vs_spy_pct",
        "rs_ratio_roc_20d_pct",
        "stock_return_5d_pct",
        "spy_return_5d_pct",
        "outperformance_5d_pct",
        "weak_market_filter",
        "volume_ratio",
        "volume",
        "avg_volume_20d",
        "prior_52w_high",
        "distance_to_9ema_pct",
        "signal",
        "ma_stack",
        "volume_support",
        "exit_rule",
        "market cap",
        "reason",
    ]
    return out[[column for column in columns if column in out.columns]].rename(
        columns={
            "signal_date": "signal date",
            "latest_date": "latest date",
            "setup_type": "setup type",
            "days_since_breakout": "days since breakout",
            "target_2r": "target 2R",
            "risk_reward": "risk:reward",
            "breakout_level": "breakout level",
            "breakout_high": "breakout high",
            "breakout_close": "breakout close",
            "breakout_volume_ratio": "breakout volume/20D avg",
            "latest_volume_ratio": "latest volume/20D avg",
            "rsi_14": "RSI 14",
            "rs_20d_vs_spy_pct": "RS 20D vs SPY %",
            "rs_ratio_roc_20d_pct": "RS ratio ROC 20D %",
            "stock_return_5d_pct": "stock 5D return %",
            "spy_return_5d_pct": "SPY 5D return %",
            "outperformance_5d_pct": "5D outperformance %",
            "weak_market_filter": "weak market filter",
            "volume_ratio": "volume/20D avg",
            "avg_volume_20d": "avg volume 20D",
            "prior_52w_high": "prior 52W high",
            "distance_to_9ema_pct": "distance to 9 EMA %",
            "ma_stack": "MA stack",
            "volume_support": "volume support",
            "exit_rule": "exit rule",
        }
    )


def format_technical_strength_breakout_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    if "market_cap" in out.columns:
        out["market cap"] = (pd.to_numeric(out["market_cap"], errors="coerce") / 1_000_000).round(0).astype("Int64").astype(str) + "M"
    columns = [
        "ticker",
        "latest_date",
        "breakout_date",
        "days_since_breakout",
        "pullback_status",
        "close",
        "ema_9",
        "distance_to_9ema_pct",
        "breakout_high",
        "breakout_close",
        "breakout_level",
        "breakout_volume_ratio",
        "latest_volume_ratio",
        "rsi_14",
        "rs_20d_vs_spy_pct",
        "rs_ratio_roc_20d_pct",
        "stock_return_5d_pct",
        "outperformance_5d_pct",
        "weak_market_filter",
        "ma_stack",
        "market cap",
        "reason",
    ]
    return out[[column for column in columns if column in out.columns]].rename(
        columns={
            "latest_date": "latest date",
            "breakout_date": "breakout date",
            "days_since_breakout": "days since breakout",
            "pullback_status": "pullback status",
            "ema_9": "9 EMA",
            "distance_to_9ema_pct": "distance to 9 EMA %",
            "breakout_high": "breakout high",
            "breakout_close": "breakout close",
            "breakout_level": "breakout level",
            "breakout_volume_ratio": "breakout volume/20D avg",
            "latest_volume_ratio": "latest volume/20D avg",
            "rsi_14": "RSI 14",
            "rs_20d_vs_spy_pct": "RS 20D vs SPY %",
            "rs_ratio_roc_20d_pct": "RS ratio ROC 20D %",
            "stock_return_5d_pct": "stock 5D return %",
            "outperformance_5d_pct": "5D outperformance %",
            "weak_market_filter": "weak market filter",
            "ma_stack": "MA stack",
        }
    )


def format_weekly_v6_cup_handle_results(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    numeric_columns = [
        "Entry Trigger Price",
        "Stop Price",
        "Initial Risk %",
        "Signal Close",
        "Signal High",
        "Signal Low",
        "Breakout Level",
        "Close Above Breakout Level %",
        "Cup Depth %",
        "Cup Length Weeks",
        "Handle Length Weeks",
        "Handle Depth %",
        "RS 5W vs SPY",
        "RS Ratio ROC 5W",
        "Sector RS 5W vs SPY",
        "Stock RS 5W vs Sector",
        "Breakout Volume Ratio",
        "Weekly RSI14",
        "Weekly ADX14",
    ]
    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").round(2)
    if "Weekly Volume" in out.columns:
        out["Weekly Volume"] = pd.to_numeric(out["Weekly Volume"], errors="coerce").round(0).astype("Int64")
    if "20W Avg Volume" in out.columns:
        out["20W Avg Volume"] = pd.to_numeric(out["20W Avg Volume"], errors="coerce").round(0).astype("Int64")
    columns = [
        "Ticker",
        "Signal Date",
        "Breakout Type",
        "Sector",
        "Sector ETF",
        "Entry Trigger Price",
        "Stop Price",
        "Initial Risk %",
        "Signal Close",
        "Signal High",
        "Signal Low",
        "Breakout Level",
        "Close Above Breakout Level %",
        "RS 5W vs SPY",
        "RS Ratio ROC 5W",
        "Sector RS 5W vs SPY",
        "Stock RS 5W vs Sector",
        "Breakout Volume Ratio",
        "Weekly Volume",
        "20W Avg Volume",
        "Weekly RSI14",
        "Weekly ADX14",
        "SPY Weekly Uptrend",
        "QQQ Weekly Uptrend",
        "Sector ETF Weekly Uptrend",
    ]
    return out[[column for column in columns if column in out.columns]].rename(
        columns={
            "Signal Date": "signal week",
            "Breakout Type": "breakout type",
            "Sector ETF": "sector ETF",
            "Entry Trigger Price": "entry trigger",
            "Stop Price": "stop",
            "Initial Risk %": "risk %",
            "Signal Close": "signal close",
            "Signal High": "signal high",
            "Signal Low": "signal low",
            "Breakout Level": "breakout level",
            "Close Above Breakout Level %": "close above breakout %",
            "RS 5W vs SPY": "RS 5W vs SPY",
            "RS Ratio ROC 5W": "RS ratio ROC 5W",
            "Sector RS 5W vs SPY": "sector RS 5W",
            "Stock RS 5W vs Sector": "stock RS vs sector",
            "Breakout Volume Ratio": "volume/20W avg",
            "Weekly Volume": "weekly volume",
            "20W Avg Volume": "20W avg volume",
            "Weekly RSI14": "weekly RSI14",
            "Weekly ADX14": "weekly ADX14",
            "SPY Weekly Uptrend": "SPY uptrend",
            "QQQ Weekly Uptrend": "QQQ uptrend",
            "Sector ETF Weekly Uptrend": "sector uptrend",
        }
    )


def format_weekly_cup_handle_details(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    numeric_columns = [
        "Entry Trigger Price",
        "Stop Price",
        "Initial Risk %",
        "Signal Close",
        "Breakout Level",
        "Cup Left Rim High",
        "Cup Bottom Low",
        "Cup Depth %",
        "Cup Length Weeks",
        "Handle Length Weeks",
        "Handle High",
        "Handle Low",
        "Handle Depth %",
        "Right Side Recovery %",
        "Pivot Price",
        "Cup Big Green Volume Count",
        "Cup Big Green Max Volume Ratio",
        "Cup Big Green Avg Volume Ratio",
        "Right Side Big Green Volume Count",
        "Right Side Big Green Max Volume Ratio",
        "Right Side Big Green Avg Volume Ratio",
        "RS 5W vs SPY",
        "Stock RS 5W vs Sector",
        "Breakout Volume Ratio",
    ]
    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").round(2)
    columns = [
        "Ticker",
        "Signal Date",
        "Breakout Type",
        "Sector",
        "Sector ETF",
        "Entry Trigger Price",
        "Stop Price",
        "Initial Risk %",
        "Signal Close",
        "Breakout Level",
        "Cup Left Rim Date",
        "Cup Bottom Date",
        "Cup Handle Start Date",
        "Cup Left Rim High",
        "Cup Bottom Low",
        "Cup Depth %",
        "Cup Length Weeks",
        "Handle Length Weeks",
        "Handle High",
        "Handle Low",
        "Handle Depth %",
        "Handle Low Above Midpoint",
        "Right Side Recovery %",
        "Pivot Price",
        "Cup Big Green Volume Count",
        "Cup Big Green Max Volume Ratio",
        "Cup Big Green Avg Volume Ratio",
        "Cup Big Green Volume Dates",
        "Right Side Big Green Volume Count",
        "Right Side Big Green Max Volume Ratio",
        "Right Side Big Green Avg Volume Ratio",
        "Right Side Big Green Volume Dates",
        "RS 5W vs SPY",
        "Stock RS 5W vs Sector",
        "Breakout Volume Ratio",
    ]
    return out[[column for column in columns if column in out.columns]].rename(
        columns={
            "Signal Date": "signal week",
            "Breakout Type": "breakout type",
            "Sector ETF": "sector ETF",
            "Entry Trigger Price": "entry trigger",
            "Stop Price": "stop",
            "Initial Risk %": "risk %",
            "Signal Close": "signal close",
            "Breakout Level": "breakout level",
            "Cup Left Rim Date": "left rim",
            "Cup Bottom Date": "cup bottom",
            "Cup Handle Start Date": "handle start",
            "Cup Left Rim High": "left rim high",
            "Cup Bottom Low": "cup bottom low",
            "Cup Depth %": "cup depth %",
            "Cup Length Weeks": "cup weeks",
            "Handle Length Weeks": "handle weeks",
            "Handle High": "handle high",
            "Handle Low": "handle low",
            "Handle Depth %": "handle depth %",
            "Handle Low Above Midpoint": "handle above midpoint",
            "Right Side Recovery %": "right recovery %",
            "Pivot Price": "pivot price",
            "Cup Big Green Volume Count": "cup green high-vol candles",
            "Cup Big Green Max Volume Ratio": "cup max vol ratio",
            "Cup Big Green Avg Volume Ratio": "cup avg vol ratio",
            "Cup Big Green Volume Dates": "cup green high-vol dates",
            "Right Side Big Green Volume Count": "right-side green high-vol candles",
            "Right Side Big Green Max Volume Ratio": "right-side max vol ratio",
            "Right Side Big Green Avg Volume Ratio": "right-side avg vol ratio",
            "Right Side Big Green Volume Dates": "right-side green high-vol dates",
            "Stock RS 5W vs Sector": "stock RS vs sector",
            "Breakout Volume Ratio": "volume/20W avg",
        }
    )


def show_marubozu_breakout_section() -> None:
    st.subheader("Green Marubozu 52W Breakout")
    st.caption(
        "Rules: green marubozu candle, body >= 80% of candle range, small upper/lower wicks, "
        "volume above 20-day average, candle breaks prior 52-week high, latest completed U.S. candle only."
    )

    if "marubozu_stats" not in st.session_state:
        latest_state = load_latest_marubozu_scan_state()
        if latest_state:
            results, errors, stats = latest_state
            st.session_state["marubozu_results"] = results
            st.session_state["marubozu_errors"] = errors
            st.session_state["marubozu_stats"] = stats

    stats = st.session_state.get("marubozu_stats", {})
    if stats:
        columns = st.columns(5)
        columns[0].metric("Stored tickers", f"{int(stats.get('stored_tickers', 0)):,}")
        columns[1].metric("OHLCV loaded", f"{int(stats.get('ohlcv_loaded', 0)):,}")
        columns[2].metric("Price candidates", f"{int(stats.get('price_candidates', 0)):,}")
        columns[3].metric("Matches", f"{int(stats.get('matches', 0)):,}")
        columns[4].metric("Skipped/data gaps", f"{int(stats.get('error_count', 0)):,}")

    results = st.session_state.get("marubozu_results", pd.DataFrame())
    if isinstance(results, pd.DataFrame) and not results.empty:
        st.dataframe(format_marubozu_results(results), width="stretch", hide_index=True)
    elif stats:
        st.write("No green marubozu 52-week breakout matches found in the latest run.")


def show_morning_star_section() -> None:
    st.subheader("Morning Star")
    st.caption(
        "Rules: first candle long bearish body, second candle small/doji with flexible gap down, "
        "third candle long bullish body with flexible gap up and close above first-body midpoint. "
        "Second and third candle volume must be above 20-day average by default."
    )

    if "morning_star_stats" not in st.session_state:
        latest_state = load_latest_morning_star_scan_state()
        if latest_state:
            results, errors, stats = latest_state
            st.session_state["morning_star_results"] = results
            st.session_state["morning_star_errors"] = errors
            st.session_state["morning_star_stats"] = stats

    stats = st.session_state.get("morning_star_stats", {})
    if stats:
        columns = st.columns(5)
        columns[0].metric("Stored tickers", f"{int(stats.get('stored_tickers', 0)):,}")
        columns[1].metric("OHLCV loaded", f"{int(stats.get('ohlcv_loaded', 0)):,}")
        columns[2].metric("Candidates", f"{int(stats.get('price_candidates', 0)):,}")
        columns[3].metric("Matches", f"{int(stats.get('matches', 0)):,}")
        columns[4].metric("Skipped/data gaps", f"{int(stats.get('error_count', 0)):,}")

    results = st.session_state.get("morning_star_results", pd.DataFrame())
    if isinstance(results, pd.DataFrame) and not results.empty:
        st.dataframe(format_morning_star_results(results), width="stretch", hide_index=True)
    elif stats:
        st.write("No Morning Star matches found in the latest run.")
    else:
        st.info("No Morning Star scan has run yet. Select Morning Star in the sidebar and click Run Selected Scan Rules.")


def show_monthly_big_volume_section() -> None:
    st.subheader("Monthly Big Volume Candle")
    st.caption(
        "Rules: latest completed monthly candle is a big green candle or green hammer; "
        "monthly volume must be greater than the previous monthly candle volume and greater than the 12-month average volume."
    )

    if "monthly_big_volume_stats" not in st.session_state:
        latest_state = load_latest_monthly_big_volume_scan_state()
        if latest_state:
            results, errors, stats = latest_state
            st.session_state["monthly_big_volume_results"] = results
            st.session_state["monthly_big_volume_errors"] = errors
            st.session_state["monthly_big_volume_stats"] = stats

    stats = st.session_state.get("monthly_big_volume_stats", {})
    if stats:
        columns = st.columns(5)
        columns[0].metric("Stored tickers", f"{int(stats.get('stored_tickers', 0)):,}")
        columns[1].metric("OHLCV loaded", f"{int(stats.get('ohlcv_loaded', 0)):,}")
        columns[2].metric("Candidates", f"{int(stats.get('price_candidates', 0)):,}")
        columns[3].metric("Matches", f"{int(stats.get('matches', 0)):,}")
        columns[4].metric("Skipped/data gaps", f"{int(stats.get('error_count', 0)):,}")

    results = st.session_state.get("monthly_big_volume_results", pd.DataFrame())
    if isinstance(results, pd.DataFrame) and not results.empty:
        st.dataframe(format_monthly_big_volume_results(results), width="stretch", hide_index=True)
    elif stats:
        st.write("No monthly big-volume candle matches found in the latest run.")
    else:
        st.info("No monthly big-volume candle scan has run yet. Select it in the sidebar and click Run Selected Scan Rules.")


def show_weekly_ath_breakout_section() -> None:
    st.subheader("Weekly ATH Breakout")
    st.caption(
        "Rules: completed weekly candle breaks prior all-time high, weekly volume above 20-week average, "
        "weekly uptrend is Close > 10W EMA > 30W SMA with 30W SMA rising, "
        "then fundamentals must pass revenue growth, EPS growth, market cap, and average-volume filters."
    )

    if "weekly_ath_stats" not in st.session_state:
        latest_state = load_latest_weekly_ath_scan_state()
        if latest_state:
            results, errors, stats = latest_state
            st.session_state["weekly_ath_results"] = results
            st.session_state["weekly_ath_errors"] = errors
            st.session_state["weekly_ath_stats"] = stats

    stats = st.session_state.get("weekly_ath_stats", {})
    if stats:
        columns = st.columns(5)
        columns[0].metric("Stored tickers", f"{int(stats.get('stored_tickers', 0)):,}")
        columns[1].metric("All-time OHLCV loaded", f"{int(stats.get('ohlcv_loaded', 0)):,}")
        columns[2].metric("Technical candidates", f"{int(stats.get('price_candidates', 0)):,}")
        columns[3].metric("Final matches", f"{int(stats.get('matches', 0)):,}")
        columns[4].metric("Data issues", f"{int(stats.get('error_count', 0)):,}")

    results = st.session_state.get("weekly_ath_results", pd.DataFrame())
    if isinstance(results, pd.DataFrame) and not results.empty:
        st.subheader("Final Matches After Fundamentals")
        st.dataframe(format_weekly_ath_results(results), width="stretch", hide_index=True)
    elif stats:
        st.write("No weekly all-time-high breakout matches found in the latest run.")
    else:
        st.info("No Weekly ATH breakout scan has run yet. Select Weekly ATH breakout in the sidebar and click Run Selected Scan Rules.")

    technical_candidate_rows = stats.get("technical_candidate_rows", []) if isinstance(stats, dict) else []
    if isinstance(technical_candidate_rows, list) and technical_candidate_rows:
        st.subheader(f"Technical Candidates Before Fundamentals: {len(technical_candidate_rows):,}")
        st.dataframe(
            format_weekly_ath_results(pd.DataFrame(technical_candidate_rows)),
            width="stretch",
            hide_index=True,
        )


def show_weekly_momentum_section() -> None:
    st.subheader("Weekly Price Momentum")
    st.caption(
        "Rules: revenue growth > 30%, EPS growth > 30%, market cap > $500M, "
        "20-week average volume > 50,000,000, Close > 21W EMA, "
        "50W/200W SMA checks only when available, and the full weekly candle range "
        "must be within 5% of 9W EMA/SMA with hammer or bullish engulfing candle."
    )

    if "weekly_momentum_stats" not in st.session_state:
        latest_state = load_latest_weekly_momentum_scan_state()
        if latest_state:
            results, errors, stats = latest_state
            st.session_state["weekly_momentum_results"] = results
            st.session_state["weekly_momentum_errors"] = errors
            st.session_state["weekly_momentum_stats"] = stats

    stats = st.session_state.get("weekly_momentum_stats", {})
    if stats:
        columns = st.columns(5)
        columns[0].metric("Stored tickers", f"{int(stats.get('stored_tickers', 0)):,}")
        columns[1].metric("All-time OHLCV loaded", f"{int(stats.get('ohlcv_loaded', 0)):,}")
        columns[2].metric("Technical candidates", f"{int(stats.get('price_candidates', 0)):,}")
        columns[3].metric("Final matches", f"{int(stats.get('matches', 0)):,}")
        columns[4].metric("Skipped/data gaps", f"{int(stats.get('error_count', 0)):,}")

    results = st.session_state.get("weekly_momentum_results", pd.DataFrame())
    if isinstance(results, pd.DataFrame) and not results.empty:
        st.subheader("Final Matches After Fundamentals")
        st.dataframe(format_weekly_momentum_results(results), width="stretch", hide_index=True)
    elif stats:
        st.write("No weekly price momentum matches found in the latest run.")
    else:
        st.info("No Weekly Price Momentum scan has run yet. Select it in the sidebar and click Run Selected Scan Rules.")

    technical_candidate_rows = stats.get("technical_candidate_rows", []) if isinstance(stats, dict) else []
    if isinstance(technical_candidate_rows, list) and technical_candidate_rows:
        st.subheader(f"Technical Candidates Before Fundamentals: {len(technical_candidate_rows):,}")
        st.dataframe(
            format_weekly_momentum_results(pd.DataFrame(technical_candidate_rows)),
            width="stretch",
            hide_index=True,
        )


def _show_technical_scan_section(
    title: str,
    caption: str,
    state_prefix: str,
    load_latest_state,
    match_label: str,
) -> None:
    st.subheader(title)
    st.caption(caption)
    st.info(
        "RSI 14 > 50 means the 14-day Relative Strength Index is above its midpoint. "
        "In simple words: recent upward price momentum is stronger than recent downward momentum."
    )

    stats_key = f"{state_prefix}_stats"
    results_key = f"{state_prefix}_results"
    errors_key = f"{state_prefix}_errors"
    if stats_key not in st.session_state:
        latest_state = load_latest_state()
        if latest_state:
            results, errors, stats = latest_state
            st.session_state[results_key] = results
            st.session_state[errors_key] = errors
            st.session_state[stats_key] = stats

    stats = st.session_state.get(stats_key, {})
    if stats:
        columns = st.columns(5)
        columns[0].metric("Stored tickers", f"{int(stats.get('stored_tickers', 0)):,}")
        columns[1].metric("OHLCV loaded", f"{int(stats.get('ohlcv_loaded', 0)):,}")
        columns[2].metric(match_label, f"{int(stats.get('matches', 0)):,}")
        columns[3].metric("Skipped/data gaps", f"{int(stats.get('error_count', 0)):,}")
        columns[4].metric("Candle date", str(stats.get("scan_date", "-")))
        st.caption("Weak market resilience filter: always applied for this scanner.")

    results = st.session_state.get(results_key, pd.DataFrame())
    if isinstance(results, pd.DataFrame) and not results.empty:
        st.subheader(match_label)
        st.dataframe(format_technical_strength_results(results), width="stretch", hide_index=True)
    elif stats:
        st.write(f"No {match_label.lower()} found in the latest run.")
    else:
        st.info(f"No {title} scan has run yet. Select it in the sidebar and click Run Selected Scan Rules.")


def show_technical_breakout_section() -> None:
    _show_technical_scan_section(
        "Technical Breakout",
        "Rules: Close > 9 EMA > 20 SMA > 50 SMA > 200 SMA, RSI > 50, "
        "20D relative strength vs SPY > 5%, RS ratio ROC 20D > 5%, "
        "5D outperformance vs SPY > 3%, 5D return > -2%, "
        "and recent ATH/52-week high breakout on >= 1.2x 20-day average volume.",
        "technical_breakout",
        load_latest_technical_breakout_scan_state,
        "Breakout matches",
    )


def show_technical_pullback_section() -> None:
    _show_technical_scan_section(
        "9 EMA Pullback",
        "Rules: Close > 9 EMA > 20 SMA > 50 SMA > 200 SMA, RSI > 50, "
        "20D relative strength vs SPY > 5%, RS ratio ROC 20D > 5%, "
        "5D outperformance vs SPY > 3%, 5D return > -2%, "
        "and 9 EMA pullback/reclaim with a bullish signal candle.",
        "technical_pullback",
        load_latest_technical_pullback_scan_state,
        "9 EMA pullback matches",
    )


def show_weekly_v6_cup_handle_section() -> None:
    st.subheader("Weekly Breakout")
    st.caption(
        "Rules: latest completed weekly candle, stock weekly uptrend, SPY/QQQ/sector ETF confirmation, "
        "RS 5W vs SPY > 5%, RS ratio ROC 5W > 5%, stock stronger than sector, "
        "52W/ATH breakout, and breakout volume >= 1.2x 20-week average."
    )

    if "weekly_v6_cup_handle_stats" not in st.session_state:
        latest_state = load_latest_weekly_v6_cup_handle_scan_state()
        if latest_state:
            results, errors, stats = latest_state
            st.session_state["weekly_v6_cup_handle_results"] = results
            st.session_state["weekly_v6_cup_handle_errors"] = errors
            st.session_state["weekly_v6_cup_handle_stats"] = stats

    stats = st.session_state.get("weekly_v6_cup_handle_stats", {})
    if stats:
        columns = st.columns(6)
        columns[0].metric("Stored tickers", f"{int(stats.get('stored_tickers', 0)):,}")
        columns[1].metric("Weekly OHLCV loaded", f"{int(stats.get('ohlcv_loaded', 0)):,}")
        columns[2].metric("Weekly breakouts", f"{int(stats.get('matches', 0)):,}")
        columns[3].metric("Cup/handle", f"{int(stats.get('cup_handle_candidates', 0)):,}")
        columns[4].metric("Data issues", f"{int(stats.get('error_count', 0)):,}")
        columns[5].metric("Signal week", str(stats.get("scan_date", "-")))
        st.caption(
            f"Sector mapped: {int(stats.get('sector_mapped', 0)):,}; "
            f"sector ETFs checked: {int(stats.get('sector_etfs', 0)):,}."
        )

    results = st.session_state.get("weekly_v6_cup_handle_results", pd.DataFrame())
    if isinstance(results, pd.DataFrame) and not results.empty:
        st.subheader("Weekly Breakout Matches")
        st.dataframe(format_weekly_v6_cup_handle_results(results), width="stretch", hide_index=True)
        st.download_button(
            "Export Weekly Breakout CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="weekly_breakout_matches.csv",
            mime="text/csv",
        )

        cup_handle_rows = stats.get("cup_handle_rows", [])
        cup_handle_results = pd.DataFrame(cup_handle_rows) if cup_handle_rows else pd.DataFrame()
        st.subheader("Cup And Handle Candidates")
        st.caption(
            "This is the cup-and-handle subset inside the Weekly Breakout scan. "
            "It keeps cup depth, handle depth, cup length, handle length, and high-volume green candle evidence."
        )
        if not cup_handle_results.empty:
            st.dataframe(format_weekly_cup_handle_details(cup_handle_results), width="stretch", hide_index=True)
            st.download_button(
                "Export Cup And Handle CSV",
                data=cup_handle_results.to_csv(index=False).encode("utf-8"),
                file_name="weekly_cup_handle_candidates.csv",
                mime="text/csv",
            )
        else:
            st.write("No cup-and-handle candidates found inside the latest Weekly Breakout run.")
    elif stats:
        st.write("No Weekly Breakout matches found in the latest run.")
    else:
        st.info("No Weekly Breakout scan has run yet. Select it under Search Rules To Run and click Run Selected Scan Rules.")


def show_pullback_results() -> None:
    results = st.session_state.get("latest_results", pd.DataFrame())
    errors = st.session_state.get("latest_errors", [])
    scanned_count = st.session_state.get("latest_scanned_count")
    saved_count = st.session_state.get("latest_saved_count")

    st.subheader("Pullback Setup")
    if scanned_count is not None:
        show_scan_summary(scanned_count, len(results), len(errors), saved_count)
        show_scan_stats(st.session_state.get("latest_stats", {}))

    if not results.empty:
        st.subheader("Watchlist")
        st.dataframe(format_dashboard(results), width="stretch", hide_index=True)
        st.download_button(
            "Export pullback watchlist to CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name=f"kalyani_watchlist_{results['scan_date'].iloc[0]}.csv",
            mime="text/csv",
        )
    elif scanned_count is not None:
        st.write("No pullback setup matches found in the latest run.")
    else:
        st.write("No pullback setup scan results yet.")


def show_data_issues() -> None:
    if "technical_breakout_stats" not in st.session_state:
        latest_state = load_latest_technical_breakout_scan_state()
        if latest_state:
            results, errors, stats = latest_state
            st.session_state["technical_breakout_results"] = results
            st.session_state["technical_breakout_errors"] = errors
            st.session_state["technical_breakout_stats"] = stats
    if "technical_pullback_stats" not in st.session_state:
        latest_state = load_latest_technical_pullback_scan_state()
        if latest_state:
            results, errors, stats = latest_state
            st.session_state["technical_pullback_results"] = results
            st.session_state["technical_pullback_errors"] = errors
            st.session_state["technical_pullback_stats"] = stats
    if "weekly_v6_cup_handle_stats" not in st.session_state:
        latest_state = load_latest_weekly_v6_cup_handle_scan_state()
        if latest_state:
            results, errors, stats = latest_state
            st.session_state["weekly_v6_cup_handle_results"] = results
            st.session_state["weekly_v6_cup_handle_errors"] = errors
            st.session_state["weekly_v6_cup_handle_stats"] = stats
    if "monthly_big_volume_stats" not in st.session_state:
        latest_state = load_latest_monthly_big_volume_scan_state()
        if latest_state:
            results, errors, stats = latest_state
            st.session_state["monthly_big_volume_results"] = results
            st.session_state["monthly_big_volume_errors"] = errors
            st.session_state["monthly_big_volume_stats"] = stats
    technical_breakout_errors = st.session_state.get("technical_breakout_errors", [])
    technical_pullback_errors = st.session_state.get("technical_pullback_errors", [])
    weekly_v6_errors = st.session_state.get("weekly_v6_cup_handle_errors", [])
    monthly_errors = st.session_state.get("monthly_big_volume_errors", [])

    st.subheader("Data Issues")
    columns = st.columns(4)
    columns[0].metric("Technical Breakout gaps", f"{len(technical_breakout_errors):,}")
    columns[1].metric("9 EMA Pullback gaps", f"{len(technical_pullback_errors):,}")
    columns[2].metric("Weekly Breakout gaps", f"{len(weekly_v6_errors):,}")
    columns[3].metric("Monthly gaps", f"{len(monthly_errors):,}")
    if technical_breakout_errors:
        with st.expander("Technical Breakout skipped tickers and data gaps", expanded=False):
            st.text("\n".join(technical_breakout_errors[:500]))
            if len(technical_breakout_errors) > 500:
                st.caption("Showing first 500 issues.")
    if technical_pullback_errors:
        with st.expander("9 EMA Pullback skipped tickers and data gaps", expanded=False):
            st.text("\n".join(technical_pullback_errors[:500]))
            if len(technical_pullback_errors) > 500:
                st.caption("Showing first 500 issues.")
    if weekly_v6_errors:
        with st.expander("Weekly Breakout skipped tickers and data gaps", expanded=False):
            st.text("\n".join(weekly_v6_errors[:500]))
            if len(weekly_v6_errors) > 500:
                st.caption("Showing first 500 issues.")
    if monthly_errors:
        with st.expander("Monthly Big Volume skipped tickers and data gaps", expanded=False):
            st.text("\n".join(monthly_errors[:500]))
            if len(monthly_errors) > 500:
                st.caption("Showing first 500 issues.")
    if not technical_breakout_errors and not technical_pullback_errors and not weekly_v6_errors and not monthly_errors:
        st.write("No data issues recorded yet.")


def show_scanner_results() -> None:
    st.header("Scanner Results")
    breakout_tab, pullback_tab, weekly_v6_tab, monthly_tab, data_tab = st.tabs(
        ["Technical Breakout", "9 EMA Pullback", "Weekly Breakout", "Monthly Big Volume", "Data Issues"]
    )
    with breakout_tab:
        show_technical_breakout_section()
    with pullback_tab:
        show_technical_pullback_section()
    with weekly_v6_tab:
        show_weekly_v6_cup_handle_section()
    with monthly_tab:
        show_monthly_big_volume_section()
    with data_tab:
        show_data_issues()


def market_condition_style(overall_market: str) -> dict[str, str]:
    market = str(overall_market or "").upper()
    if "STRONG BULLISH" in market:
        return {"accent": "#166534", "bg": "#ecfdf5", "border": "#86efac", "text": "#14532d"}
    if "BULLISH" in market:
        return {"accent": "#16a34a", "bg": "#f0fdf4", "border": "#bbf7d0", "text": "#166534"}
    if "STRONG BEARISH" in market:
        return {"accent": "#7f1d1d", "bg": "#fef2f2", "border": "#fecaca", "text": "#7f1d1d"}
    if "BEARISH" in market:
        return {"accent": "#dc2626", "bg": "#fff1f2", "border": "#fecdd3", "text": "#991b1b"}
    if "UNKNOWN" in market:
        return {"accent": "#6b7280", "bg": "#f9fafb", "border": "#d1d5db", "text": "#374151"}
    return {"accent": "#ca8a04", "bg": "#fffbeb", "border": "#fde68a", "text": "#713f12"}


def trend_style(trend: str) -> dict[str, str]:
    text = str(trend or "").upper()
    if text == "STRONG UPTREND":
        return {"bg": "#dcfce7", "color": "#14532d", "label": "Strong uptrend"}
    if text == "WEAK UPTREND":
        return {"bg": "#f0fdf4", "color": "#166534", "label": "Weak uptrend"}
    if text == "STRONG DOWNTREND":
        return {"bg": "#fee2e2", "color": "#7f1d1d", "label": "Strong downtrend"}
    if text == "WEAK DOWNTREND":
        return {"bg": "#f4e3e3", "color": "#991b1b", "label": "Weak downtrend"}
    return {"bg": "#f1e8cf", "color": "#713f12", "label": "Sideways"}


def _market_float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _market_number(value: object, decimals: int = 2) -> str:
    number = _market_float(value)
    if number is None:
        return "-"
    return f"{number:,.{decimals}f}"


def _market_compact(value: object) -> str:
    number = _market_float(value)
    if number is None:
        return "-"
    absolute = abs(number)
    if absolute >= 1_000_000_000:
        return f"{number / 1_000_000_000:.1f}B"
    if absolute >= 1_000_000:
        return f"{number / 1_000_000:.0f}M"
    if absolute >= 1_000:
        return f"{number / 1_000:.0f}K"
    return f"{number:.0f}"


def _adx_reading(adx: object) -> str:
    value = _market_float(adx)
    if value is None:
        return "unknown"
    if value < 20:
        return "trendless"
    if value <= 25:
        return "forming"
    return "trending"


def _ma_order_text(row: dict[str, object]) -> tuple[str, str]:
    close = _market_float(row.get("Close"))
    ema9 = _market_float(row.get("EMA 9"))
    sma20 = _market_float(row.get("SMA 20"))
    sma50 = _market_float(row.get("SMA 50"))
    sma200 = _market_float(row.get("SMA 200"))
    if None in (close, ema9, sma20, sma50, sma200):
        return "-", "#6b7280"
    if close > ema9 > sma20 > sma50 > sma200:
        return "Close > 9 EMA > 20 SMA > 50 SMA > 200 SMA", "#166534"
    if close < ema9 < sma20 < sma50 < sma200:
        return "Close < 9 EMA < 20 SMA < 50 SMA < 200 SMA", "#991b1b"

    pieces = [
        f"Close {'>' if close > ema9 else '<='} 9 EMA",
        f"9 EMA {'>' if ema9 > sma20 else '<='} 20 SMA",
        f"20 SMA {'>' if sma20 > sma50 else '<='} 50 SMA",
        f"50 SMA {'>' if sma50 > sma200 else '<='} 200 SMA",
    ]
    color = "#111827" if sma50 > sma200 else "#991b1b"
    return "Mixed: " + " | ".join(pieces), color


def _price_momentum_text(row: dict[str, object]) -> tuple[str, str]:
    close = _market_float(row.get("Close"))
    ema9 = _market_float(row.get("EMA 9"))
    if close is None or ema9 is None:
        return "Price momentum unknown", "#6b7280"
    if close >= ema9:
        return f"Above EMA 9 ({ema9:.2f})", "#166534"
    return f"Below EMA 9 ({ema9:.2f})", "#991b1b"


def _swing_short(row: dict[str, object]) -> tuple[str, str]:
    hh = bool(row.get("Higher High"))
    hl = bool(row.get("Higher Low"))
    lh = bool(row.get("Lower High"))
    ll = bool(row.get("Lower Low"))
    if hh and hl:
        return "HH + HL", "#166534"
    if lh and ll:
        return "LH + LL", "#991b1b"
    if hh and ll:
        return "HH + LL", "#713f12"
    if lh and hl:
        return "LH + HL", "#713f12"
    return str(row.get("Swing Structure") or "Unclear"), "#6b7280"


def _volume_pill_style(text: str) -> dict[str, str]:
    lowered = str(text or "").lower()
    if "uptrend" in lowered:
        return {"bg": "#dff1d6", "color": "#166534", "label": "Volume confirms uptrend" if "confirms" in lowered else "Volume supports uptrend"}
    if "downtrend" in lowered:
        return {"bg": "#f4e3e3", "color": "#991b1b", "label": "Volume confirms downtrend" if "confirms" in lowered else "Volume supports downtrend"}
    return {"bg": "#f3f4f6", "color": "#374151", "label": "Volume mixed"}


def _market_takeaways(result_df: pd.DataFrame) -> list[tuple[str, str]]:
    if result_df.empty:
        return [("#6b7280", "No index readings are available yet.")]

    rows = result_df.to_dict("records")
    down = [row["Ticker"] for row in rows if "DOWNTREND" in str(row.get("Trend", "")).upper()]
    up = [row["Ticker"] for row in rows if "UPTREND" in str(row.get("Trend", "")).upper()]
    sideways = [row["Ticker"] for row in rows if str(row.get("Trend", "")).upper() == "SIDEWAYS"]
    macro_up = [row["Ticker"] for row in rows if _market_float(row.get("SMA 50")) and _market_float(row.get("SMA 200")) and _market_float(row.get("SMA 50")) > _market_float(row.get("SMA 200"))]
    di_bearish = [row["Ticker"] for row in rows if (_market_float(row.get("-DI")) or 0) > (_market_float(row.get("+DI")) or 0)]
    di_bullish = [row["Ticker"] for row in rows if (_market_float(row.get("+DI")) or 0) > (_market_float(row.get("-DI")) or 0)]

    takeaways: list[tuple[str, str]] = []
    if down:
        takeaways.append(("#dc2626", f"{' & '.join(down)} are in short-term pullback; sellers control momentum where -DI is dominant."))
    if up:
        takeaways.append(("#65a30d", f"{' & '.join(up)} have bullish trend readings; favor long setups with relative strength."))
    if sideways:
        takeaways.append(("#ca8a04", f"{' & '.join(sideways)} are sideways; ADX is too low or structure is mixed, so avoid chasing."))
    if len(macro_up) == len(rows):
        takeaways.append(("#65a30d", "All four ETFs have SMA 50 above SMA 200, so the broader macro uptrend is intact."))
    elif macro_up:
        takeaways.append(("#65a30d", f"{' & '.join(macro_up)} still have SMA 50 above SMA 200, supporting the longer-term trend."))
    if di_bearish:
        takeaways.append(("#dc2626", f"Bearish momentum is visible in {' & '.join(di_bearish)} because -DI is above +DI."))
    if di_bullish:
        takeaways.append(("#2563eb", f"Buyer momentum is visible in {' & '.join(di_bullish)} because +DI is above -DI."))
    return takeaways[:4]


def _breadth_value(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _breadth_color(value: object) -> str:
    number = _breadth_value(value)
    if number is None:
        return "#6b7280"
    if number >= 60:
        return "#166534"
    if number >= 45:
        return "#ca8a04"
    return "#991b1b"


def _breadth_bar(label: str, value: object, status: object) -> str:
    number = _breadth_value(value)
    width = min(100, max(0, number or 0))
    color = _breadth_color(value)
    text = "-" if number is None else f"{number:.1f}%"
    return (
        '<div class="mb-row">'
        f'<div class="mb-label">{escape(label)}</div>'
        '<div class="mb-track">'
        f'<div class="mb-fill" style="width:{width:.1f}%;background:{color};"></div>'
        "</div>"
        f'<div class="mb-value" style="color:{color};">{escape(text)}</div>'
        f'<div class="mb-status">{escape(str(status or ""))}</div>'
        "</div>"
    )


def show_market_breadth_panel(tickers: list[str]) -> None:
    breadth_ui_version = 8
    if st.session_state.get("market_breadth_ui_version") != breadth_ui_version:
        st.session_state.pop("market_breadth_summary", None)
        st.session_state["market_breadth_ui_version"] = breadth_ui_version

    st.markdown("**Index Breadth**")
    st.caption(
        "Calculates breadth inside each index: S&P 500, QQQ / Nasdaq 100, Dow 30, and Russell 2000/IWM when holdings are available. "
        "Includes advance/decline, new highs/lows, % above 20/50/200 SMA, and SMA crossover counts."
    )

    st.caption("Auto-calculates on the latest completed U.S. session and caches for 4 hours. All four index panels are shown below.")

    if "market_breadth_summary" not in st.session_state:
        with st.spinner("Calculating index breadth from constituents..."):
            st.session_state["market_breadth_summary"] = cached_market_breadth_scan(tuple(tickers))

    summary = st.session_state.get("market_breadth_summary", {})
    if isinstance(summary, dict) and "indexes" not in summary:
        st.session_state.pop("market_breadth_summary", None)
        cached_market_breadth_scan.clear()
        with st.spinner("Updating old breadth cache to index breadth..."):
            st.session_state["market_breadth_summary"] = cached_market_breadth_scan(tuple(tickers))
        summary = st.session_state.get("market_breadth_summary", {})

    if not isinstance(summary, dict) or not summary:
        st.info("No breadth summary available yet.")
        return

    index_rows = summary.get("indexes", [])
    if not isinstance(index_rows, list) or not index_rows:
        st.warning("Index breadth could not be calculated.")
        errors = summary.get("errors", [])
        if isinstance(errors, list) and errors:
            st.text("\n".join(str(error) for error in errors[:20]))
        return

    st.markdown(
        """
        <style>
            .mb-box {border:1px solid #e5e7eb; border-radius:8px; background:#fff; padding:16px; margin:8px 0 16px;}
            .mb-head {display:flex; justify-content:space-between; gap:16px; align-items:flex-start; margin-bottom:12px;}
            .mb-title {font-size:18px; font-weight:850; color:#111827;}
            .mb-sub {font-size:12px; color:#6b7280; margin-top:3px;}
            .mb-pill {border-radius:999px; padding:5px 10px; font-size:12px; font-weight:850; white-space:nowrap;}
            .mb-grid {display:grid; grid-template-columns:1.2fr .8fr; gap:18px;}
            .mb-row {display:grid; grid-template-columns:118px 1fr 58px 68px; gap:9px; align-items:center; margin:9px 0; font-size:12px;}
            .mb-label {color:#374151; font-weight:750;}
            .mb-track {height:8px; background:#e5e7eb; border-radius:999px; overflow:hidden;}
            .mb-fill {height:100%; border-radius:999px;}
            .mb-value {font-weight:850; text-align:right;}
            .mb-status {color:#6b7280; font-size:11px;}
            .mb-counts {display:grid; gap:8px;}
            .mb-count {border:1px solid #eef0f3; border-radius:8px; padding:9px 10px; background:#fafafa;}
            .mb-count span {display:block; font-size:11px; color:#6b7280;}
            .mb-count b {font-size:18px; color:#111827;}
            .mb-note {font-size:13px; color:#374151; line-height:1.45; margin-top:12px; border-top:1px solid #eef0f3; padding-top:12px;}
            .mb-mini-table {width:100%; border-collapse:collapse; font-size:12px; margin-top:10px;}
            .mb-mini-table td {border-top:1px solid #eef0f3; padding:7px 0; vertical-align:top;}
            .mb-mini-table td:first-child {color:#6b7280; width:160px;}
            .mb-mini-table td:last-child {font-weight:750; color:#374151;}
            @media (max-width: 900px) {.mb-grid {grid-template-columns:1fr;} .mb-row {grid-template-columns:110px 1fr 54px;} .mb-status {display:none;}}
        </style>
        """,
        unsafe_allow_html=True,
    )

    for index_summary in index_rows:
        if not isinstance(index_summary, dict):
            continue
        processed = int(index_summary.get("processed_tickers") or 0)
        proxy = str(index_summary.get("proxy") or "Index")
        if processed == 0:
            st.warning(f"{proxy}: {index_summary.get('error') or 'No constituents had enough data.'}")
            continue

        pct200 = _breadth_value(index_summary.get("pct_above_200"))
        breadth_color = _breadth_color(pct200)
        advancers = int(index_summary.get("advancers") or 0)
        decliners = int(index_summary.get("decliners") or 0)
        unchanged = int(index_summary.get("unchanged") or 0)
        ad_ratio = _breadth_value(index_summary.get("advance_decline_ratio"))
        ad_ratio_text = f"{ad_ratio:.2f}" if ad_ratio is not None else "No decliners"
        label = str(index_summary.get("breadth_label") or "Unknown breadth")
        constituents = int(index_summary.get("constituents") or 0)

        st.markdown(
            f"""
            <div class="mb-box">
                <div class="mb-head">
                    <div>
                        <div class="mb-title">{escape(str(index_summary.get('label') or 'Index'))} Breadth</div>
                        <div class="mb-sub">Proxy: {escape(proxy)} • Latest candle: {escape(str(index_summary.get('date') or summary.get('completed_date') or '-'))} • Processed {processed:,} of {constituents:,} constituents</div>
                        <div class="mb-sub">Source: {escape(str(index_summary.get('source') or '-'))}</div>
                    </div>
                    <div class="mb-pill" style="background:{breadth_color}1a;color:{breadth_color};">{escape(label)}</div>
                </div>
                <div class="mb-grid">
                    <div>
                        {_breadth_bar('% above 20 SMA', index_summary.get('pct_above_20'), index_summary.get('pct_above_20_status'))}
                        {_breadth_bar('% above 50 SMA', index_summary.get('pct_above_50'), index_summary.get('pct_above_50_status'))}
                        {_breadth_bar('% above 200 SMA', index_summary.get('pct_above_200'), index_summary.get('pct_above_200_status'))}
                        <table class="mb-mini-table">
                            <tr><td>New highs / new lows</td><td>{int(index_summary.get('new_highs') or 0):,} / {int(index_summary.get('new_lows') or 0):,}</td></tr>
                            <tr><td>Above / below 20 SMA</td><td>{int(index_summary.get('above_20') or 0):,} / {int(index_summary.get('below_20') or 0):,}</td></tr>
                            <tr><td>Above / below 50 SMA</td><td>{int(index_summary.get('above_50') or 0):,} / {int(index_summary.get('below_50') or 0):,}</td></tr>
                            <tr><td>Above / below 200 SMA</td><td>{int(index_summary.get('above_200') or 0):,} / {int(index_summary.get('below_200') or 0):,}</td></tr>
                            <tr><td>SMA 20 > 50</td><td>{int(index_summary.get('sma20_above_50') or 0):,} constituents</td></tr>
                            <tr><td>SMA 50 > 200</td><td>{int(index_summary.get('sma50_above_200') or 0):,} constituents</td></tr>
                            <tr><td>20/50 SMA cross today</td><td>Bullish {int(index_summary.get('bull_cross_20_50') or 0):,} / Bearish {int(index_summary.get('bear_cross_20_50') or 0):,}</td></tr>
                            <tr><td>50/200 SMA cross today</td><td>Bullish {int(index_summary.get('bull_cross_50_200') or 0):,} / Bearish {int(index_summary.get('bear_cross_50_200') or 0):,}</td></tr>
                        </table>
                        <div class="mb-note">
                            <b>New highs:</b> {escape(', '.join(str(t) for t in (index_summary.get('new_high_tickers') or [])[:50]) or 'None')}
                        </div>
                    </div>
                    <div class="mb-counts">
                        <div class="mb-count"><span>Advancers</span><b style="color:#166534;">{advancers:,}</b></div>
                        <div class="mb-count"><span>Decliners</span><b style="color:#991b1b;">{decliners:,}</b></div>
                        <div class="mb-count"><span>Unchanged</span><b>{unchanged:,}</b></div>
                        <div class="mb-count"><span>A/D ratio</span><b>{escape(ad_ratio_text)}</b></div>
                        <div class="mb-count"><span>A/D line 10-day trend</span><b>{escape(str(index_summary.get('ad_line_trend') or '-'))}</b></div>
                    </div>
                </div>
                <div class="mb-note">{escape(str(index_summary.get('interpretation') or ''))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sector_rows = index_summary.get("sector_breadth", [])
        if isinstance(sector_rows, list) and sector_rows:
            with st.expander(f"{proxy} sector advancing / declining", expanded=False):
                st.dataframe(pd.DataFrame(sector_rows), width="stretch", hide_index=True)

        ad_line_rows = index_summary.get("ad_line_rows", [])
        if isinstance(ad_line_rows, list) and ad_line_rows:
            with st.expander(f"{proxy} A/D line detail", expanded=False):
                st.dataframe(pd.DataFrame(ad_line_rows), width="stretch", hide_index=True)

    errors = summary.get("errors", [])
    if isinstance(errors, list) and errors:
        with st.expander("Index breadth source issues", expanded=False):
            st.text("\n".join(str(error) for error in errors[:20]))


def _volume_mini_chart_html(row: dict[str, object]) -> str:
    up_volume = row.get("Up Volume 20")
    down_volume = row.get("Down Volume 20")
    up_value = _market_float(up_volume) or 0
    down_value = _market_float(down_volume) or 0
    total = max(up_value + down_value, 1)
    up_pct = up_value / total * 100
    down_pct = down_value / total * 100
    has_count_data = all(key in row and row.get(key) not in (None, "") for key in ("Up Day Count 20", "Down Day Count 20", "Flat Day Count 20"))
    up_count = int(_market_float(row.get("Up Day Count 20")) or 0)
    down_count = int(_market_float(row.get("Down Day Count 20")) or 0)
    flat_count = int(_market_float(row.get("Flat Day Count 20")) or 0)
    above_avg_count = int(_market_float(row.get("Above Avg Volume Count 20")) or 0)
    above_avg_bullish = int(_market_float(row.get("Above Avg Volume Bullish Count 20")) or 0)
    above_avg_bearish = int(_market_float(row.get("Above Avg Volume Bearish Count 20")) or 0)
    above_avg_flat = int(_market_float(row.get("Above Avg Volume Flat Count 20")) or 0)
    count_html = (
        f'<div><b style="color:#166534;">{up_count}</b><span> Up</span></div>'
        f'<div><b style="color:#991b1b;">{down_count}</b><span> Down</span></div>'
        f'<div><b style="color:#6b7280;">{flat_count}</b><span> Flat</span></div>'
        if has_count_data
        else '<div><b style="color:#6b7280;">Refresh</b><span> market data</span></div>'
    )
    biggest_direction = str(row.get("Biggest Volume Direction 20") or "-")
    biggest_date = str(row.get("Biggest Volume Date 20") or "-")
    biggest_volume_value = _market_float(row.get("Biggest Volume 20"))
    avg_volume_value = _market_float(row.get("Volume SMA 20"))
    biggest_volume = _market_compact(biggest_volume_value)
    avg_volume = _market_compact(avg_volume_value)
    if biggest_volume_value is not None and avg_volume_value not in (None, 0):
        bigger_than_avg = (biggest_volume_value / avg_volume_value - 1) * 100
        bigger_than_avg_text = f"{bigger_than_avg:.0f}% bigger than avg"
    else:
        bigger_than_avg_text = "avg comparison unavailable"
    biggest_color = "#166534" if biggest_direction == "Up" else "#991b1b" if biggest_direction == "Down" else "#6b7280"
    return (
        '<div class="mc-mini-volume">'
        '<div class="mc-mini-volume-head"><span>20-candle volume window</span>'
        f'<b>Up {up_pct:.0f}% / Down {down_pct:.0f}%</b></div>'
        '<div class="mc-mini-volume-body">'
        '<div class="mc-count-box">'
        f"{count_html}"
        "</div>"
        '<div class="mc-mini-bars">'
        f'<div class="mc-mini-line"><span>Up</span><div class="mc-mini-track"><div class="mc-up" style="width:{up_pct:.1f}%"></div></div><b>{escape(_market_compact(up_value))}</b></div>'
        f'<div class="mc-mini-line"><span>Down</span><div class="mc-mini-track"><div class="mc-down" style="width:{down_pct:.1f}%"></div></div><b>{escape(_market_compact(down_value))}</b></div>'
        "</div></div>"
        f'<div class="mc-above-volume">Above avg volume candles: <b>{above_avg_count}/20</b> '
        f'(<span style="color:#166534;font-weight:800;">{above_avg_bullish} bullish</span>, '
        f'<span style="color:#991b1b;font-weight:800;">{above_avg_bearish} bearish</span>, '
        f'<span style="color:#6b7280;font-weight:800;">{above_avg_flat} flat</span>)</div>'
        f'<div class="mc-average-volume">Avg volume: <b>{escape(avg_volume)}</b></div>'
        f'<div class="mc-big-volume">Big volume candle: <b style="color:{biggest_color};">{escape(biggest_direction)}</b> on {escape(biggest_date)} '
        f'({escape(biggest_volume)}, {escape(bigger_than_avg_text)})</div>'
        "</div>"
    )


def format_market_condition_results(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column in ["Volume", "Volume SMA 5", "Volume SMA 20", "Up Volume 20", "Down Volume 20", "OBV Change 20"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").round(0).astype("Int64").map(
                lambda value: f"{int(value):,}" if pd.notna(value) else ""
            )
    columns = [
        "Ticker",
        "Date",
        "Trend",
        "Close",
        "ADX",
        "+DI",
        "-DI",
        "Weekly Date",
        "Weekly ADX",
        "Weekly +DI",
        "Weekly -DI",
        "Swing Structure",
        "Volume Confirmation",
        "Volume SMA 5/20 Ratio",
        "Up Day Count 20",
        "Down Day Count 20",
        "Flat Day Count 20",
        "Above Avg Volume Count 20",
        "Above Avg Volume Bullish Count 20",
        "Above Avg Volume Bearish Count 20",
        "Above Avg Volume Flat Count 20",
        "Biggest Volume Direction 20",
        "Biggest Volume Date 20",
        "Biggest Volume 20",
        "Reason",
    ]
    return out[[column for column in columns if column in out.columns]]


def _market_condition_conclusion(overall_market: str, result_df: pd.DataFrame) -> str:
    market = str(overall_market or "").upper()
    trend_map: dict[str, str] = {}
    if isinstance(result_df, pd.DataFrame) and not result_df.empty and {"Ticker", "Trend"}.issubset(result_df.columns):
        trend_map = {
            str(row["Ticker"]).upper(): str(row["Trend"]).upper()
            for _, row in result_df[["Ticker", "Trend"]].iterrows()
        }

    spy_weak = "DOWNTREND" in trend_map.get("SPY", "")
    qqq_weak = "DOWNTREND" in trend_map.get("QQQ", "")

    if "BEARISH / WEAK DOWNTREND" in market and spy_weak and qqq_weak:
        return (
            "Meaning: market is not crashing, but SPY and QQQ are weak enough that overall condition is "
            "bearish / weak downtrend. For swing trading, avoid aggressive longs and prefer only strongest pullbacks."
        )
    if "STRONG BEARISH" in market:
        return "Meaning: broad market pressure is strong. For swing trading, avoid new long trades until the indexes recover."
    if "BEARISH" in market:
        return "Meaning: sellers have control in key indexes. For swing trading, reduce long exposure and wait for cleaner setups."
    if "STRONG BULLISH" in market:
        return "Meaning: market trend is broadly supportive. For swing trading, pullbacks and breakouts can both be considered."
    if "BULLISH" in market:
        return "Meaning: market is constructive but not perfect. For swing trading, prefer clean pullbacks and relative-strength stocks."
    return "Meaning: market is choppy or rotational. For swing trading, reduce trade count and avoid chasing extended breakouts."


def format_sector_etf_results(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    columns = [
        "Ticker",
        "Sector",
        "Date",
        "Trend",
        "Close",
        "Return 1D %",
        "Return 5D %",
        "Return 20D %",
        "RS 20D vs SPY %",
        "ADX",
        "+DI",
        "-DI",
        "Weekly Date",
        "Weekly ADX",
        "Weekly +DI",
        "Weekly -DI",
        "MA Stack",
        "Above 9 EMA",
        "Volume SMA 5/20 Ratio",
        "Volume Confirmation",
        "Reason",
    ]
    return out[[column for column in columns if column in out.columns]]


def show_sector_etf_panel() -> None:
    st.markdown("**Sector ETFs**")
    st.caption(
        "Tracks the 11 major SPDR sector ETFs to show leadership, laggards, trend, volume support, and 20-day relative strength versus SPY."
    )

    with st.spinner("Checking sector ETF leadership..."):
        summary = cached_sector_etf_scan()

    sector_df = summary.get("results", pd.DataFrame()) if isinstance(summary, dict) else pd.DataFrame()
    if not isinstance(sector_df, pd.DataFrame):
        sector_df = pd.DataFrame()
    if sector_df.empty:
        st.info("Sector ETF data is unavailable right now.")
        errors = summary.get("errors", []) if isinstance(summary, dict) else []
        if isinstance(errors, list) and errors:
            with st.expander("Sector ETF source issues", expanded=False):
                st.text("\n".join(str(error) for error in errors[:20]))
        return

    leaders = summary.get("leaders", []) if isinstance(summary, dict) else []
    laggards = summary.get("laggards", []) if isinstance(summary, dict) else []
    completed_date = str(summary.get("completed_date") or "-") if isinstance(summary, dict) else "-"

    leader_text = ", ".join(
        f"{row.get('Ticker')} ({row.get('Sector')}, {row.get('RS 20D vs SPY %')}%)"
        for row in leaders
        if isinstance(row, dict)
    ) or "-"
    laggard_text = ", ".join(
        f"{row.get('Ticker')} ({row.get('Sector')}, {row.get('RS 20D vs SPY %')}%)"
        for row in laggards
        if isinstance(row, dict)
    ) or "-"
    bullish_sectors = int(sector_df["Trend"].astype(str).str.contains("UPTREND", case=False, na=False).sum())
    above_9ema = int(sector_df["Above 9 EMA"].sum()) if "Above 9 EMA" in sector_df else 0

    metric_cols = st.columns(4)
    metric_cols[0].metric("Sector candle", completed_date)
    metric_cols[1].metric("Bullish sector ETFs", f"{bullish_sectors}/{len(sector_df)}")
    metric_cols[2].metric("Above 9 EMA", f"{above_9ema}/{len(sector_df)}")
    metric_cols[3].metric("Top RS leader", str(sector_df.iloc[0].get("Ticker", "-")))

    st.markdown(
        f"""
        <div class="mc-takeaways" style="margin-top:8px;">
            <div class="mc-takeaways-title">Sector leadership read</div>
            <div class="mc-takeaway-grid">
                <div class="mc-takeaway"><span class="mc-dot" style="background:#16a34a;"></span><div><b>Leaders:</b> {escape(leader_text)}</div></div>
                <div class="mc-takeaway"><span class="mc-dot" style="background:#dc2626;"></span><div><b>Laggards:</b> {escape(laggard_text)}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cards: list[str] = []
    for row in sector_df.to_dict("records"):
        style = trend_style(str(row.get("Trend") or ""))
        rs_value = _market_float(row.get("RS 20D vs SPY %"))
        rs_color = "#166534" if (rs_value or 0) > 0 else "#991b1b" if (rs_value or 0) < 0 else "#374151"
        return_5d = _market_float(row.get("Return 5D %"))
        return_color = "#166534" if (return_5d or 0) > 0 else "#991b1b" if (return_5d or 0) < 0 else "#374151"
        volume_style = _volume_pill_style(str(row.get("Volume Confirmation") or ""))
        weekly_plus_di = _market_number(row.get("Weekly +DI"), 1)
        weekly_minus_di = _market_number(row.get("Weekly -DI"), 1)
        cards.append(
            '<div class="mc-card" style="min-height:210px;">'
            '<div class="mc-card-head">'
            f'<div><div class="mc-ticker">{escape(str(row.get("Ticker") or ""))}</div><div style="font-size:12px;color:#6b7280;margin-top:2px;">{escape(str(row.get("Sector") or ""))}</div></div>'
            f'<div class="mc-pill" style="background:{style["bg"]};color:{style["color"]};">{escape(style["label"])}</div>'
            "</div>"
            f'<div class="mc-close">{escape(_market_number(row.get("Close"), 2))}</div>'
            f'<div class="mc-row"><span>5D return</span><b style="color:{return_color};">{escape(_market_number(row.get("Return 5D %"), 2))}%</b></div>'
            f'<div class="mc-row"><span>20D RS vs SPY</span><b style="color:{rs_color};">{escape(_market_number(row.get("RS 20D vs SPY %"), 2))}%</b></div>'
            f'<div class="mc-row"><span>ADX</span><b>{escape(_market_number(row.get("ADX"), 1))} <span style="font-weight:500;color:#6b7280;">({_adx_reading(row.get("ADX"))})</span></b></div>'
            f'<div class="mc-row"><span>Weekly ADX</span><b>{escape(_market_number(row.get("Weekly ADX"), 1))} <span style="font-weight:500;color:#6b7280;">({_adx_reading(row.get("Weekly ADX"))})</span></b></div>'
            f'<div class="mc-row"><span>W +DI / -DI</span><b><span style="color:#166534;">{escape(weekly_plus_di)}</span> / <span style="color:#991b1b;">{escape(weekly_minus_di)}</span></b></div>'
            f'<div class="mc-row"><span>MA stack</span><b>{escape(str(row.get("MA Stack") or "-"))}</b></div>'
            f'<div class="mc-volume-pill" style="background:{volume_style["bg"]};color:{volume_style["color"]};">{escape(volume_style["label"])}</div>'
            "</div>"
        )
    st.markdown(f"<div class=\"mc-card-grid\">{''.join(cards)}</div>", unsafe_allow_html=True)

    with st.expander("Sector ETF detail table", expanded=False):
        st.dataframe(format_sector_etf_results(sector_df), width="stretch", hide_index=True)
        st.download_button(
            "Export sector ETF CSV",
            data=sector_df.to_csv(index=False).encode("utf-8"),
            file_name="sector_etf_market_condition.csv",
            mime="text/csv",
        )

    errors = summary.get("errors", []) if isinstance(summary, dict) else []
    if isinstance(errors, list) and errors:
        with st.expander("Sector ETF source issues", expanded=False):
            st.text("\n".join(str(error) for error in errors[:20]))


def _format_market_cap_short(value: object) -> str:
    numeric = _market_float(value)
    if numeric is None:
        return "-"
    if numeric >= 1_000_000_000_000:
        return f"${numeric / 1_000_000_000_000:.2f}T"
    if numeric >= 1_000_000_000:
        return f"${numeric / 1_000_000_000:.2f}B"
    return f"${numeric / 1_000_000:.0f}M"


def format_sector_top_gainers(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    if "market_cap" in out.columns:
        out["market cap"] = out["market_cap"].map(_format_market_cap_short)
    for column in ["volume", "avg_volume_20d"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").round(0).astype("Int64").map(
                lambda value: f"{int(value):,}" if pd.notna(value) else "-"
            )
    columns = [
        "sector",
        "sector_rank",
        "ticker",
        "company",
        "industry",
        "date",
        "gain_pct",
        "close",
        "previous_close",
        "volume_ratio",
        "volume",
        "avg_volume_20d",
        "market cap",
    ]
    return out[[column for column in columns if column in out.columns]].rename(
        columns={
            "sector": "Sector",
            "sector_rank": "Rank",
            "ticker": "Ticker",
            "company": "Company",
            "industry": "Industry",
            "date": "Date",
            "gain_pct": "Gain %",
            "close": "Close",
            "previous_close": "Prev close",
            "volume_ratio": "Volume/20D avg",
            "volume": "Volume",
            "avg_volume_20d": "Avg volume 20D",
            "market cap": "Market cap",
        }
    )


def show_sector_top_gainers_panel() -> None:
    st.markdown("**Top 5 Gainers In Each Sector**")
    st.caption(
        "Uses the saved $500M+ ticker dataset, Nasdaq daily mover data, and cached sector profiles. Gain is latest completed U.S. session versus prior close."
    )

    metadata = load_universe_metadata(DB_PATH)
    eligible_count = len(load_eligible_ticker_symbols(DB_PATH))
    universe_signature = f"{metadata.get('last_refreshed', '')}:{eligible_count}"
    if eligible_count == 0:
        st.info("Top sector gainers need the saved $500M+ ticker universe. Download/store that universe first.")
        return

    with st.spinner("Calculating top sector gainers from the saved $500M+ universe..."):
        summary = cached_sector_top_gainers_scan(universe_signature)

    gainers = summary.get("results", pd.DataFrame()) if isinstance(summary, dict) else pd.DataFrame()
    if not isinstance(gainers, pd.DataFrame):
        gainers = pd.DataFrame()
    if gainers.empty:
        st.info("No sector gainers are available yet.")
        errors = summary.get("errors", []) if isinstance(summary, dict) else []
        if isinstance(errors, list) and errors:
            with st.expander("Top sector gainers source issues", expanded=False):
                st.text("\n".join(str(error) for error in errors[:20]))
        return

    sector_summary = summary.get("sector_summary", pd.DataFrame()) if isinstance(summary, dict) else pd.DataFrame()
    completed_date = str(summary.get("completed_date") or "-") if isinstance(summary, dict) else "-"
    requested = int(summary.get("requested_tickers") or 0) if isinstance(summary, dict) else 0
    available = int(summary.get("downloaded_tickers") or 0) if isinstance(summary, dict) else 0
    profile_checked = int(summary.get("profile_checked_tickers") or 0) if isinstance(summary, dict) else 0

    metric_cols = st.columns(4)
    metric_cols[0].metric("Candle date", completed_date)
    metric_cols[1].metric("Sectors", f"{gainers['sector'].nunique():,}")
    metric_cols[2].metric("Movers checked", f"{profile_checked:,}/{available:,}")
    metric_cols[3].metric("Top gain", f"{pd.to_numeric(gainers['gain_pct'], errors='coerce').max():.2f}%")

    if isinstance(sector_summary, pd.DataFrame) and not sector_summary.empty:
        top_sector = sector_summary.iloc[0]
        st.caption(
            f"Strongest average sector today: {top_sector.get('sector', '-')} "
            f"({top_sector.get('avg_gain_pct', '-')}% average move, {top_sector.get('advancers', 0)} advancers)."
        )

    st.dataframe(format_sector_top_gainers(gainers), width="stretch", hide_index=True)
    st.download_button(
        "Export top sector gainers CSV",
        data=gainers.to_csv(index=False).encode("utf-8"),
        file_name="top_5_sector_gainers_500m.csv",
        mime="text/csv",
    )

    if isinstance(sector_summary, pd.DataFrame) and not sector_summary.empty:
        with st.expander("Sector summary", expanded=False):
            st.dataframe(sector_summary, width="stretch", hide_index=True)

    errors = summary.get("errors", []) if isinstance(summary, dict) else []
    if isinstance(errors, list) and errors:
        with st.expander("Top sector gainers source issues", expanded=False):
            st.text("\n".join(str(error) for error in errors[:20]))


def show_market_condition_panel() -> None:
    st.header("Market Condition")
    st.caption("Uses SPY, QQQ, DIA, and IWM with completed U.S. daily candles only.")

    market_condition_ui_version = 3
    if st.session_state.get("market_condition_ui_version") != market_condition_ui_version:
        cached_market_trend_scan.clear()
        cached_sector_etf_scan.clear()
        st.session_state["market_condition_ui_version"] = market_condition_ui_version

    refresh_col, cache_col = st.columns([1.4, 4])
    if refresh_col.button("Refresh Market Condition", width="stretch"):
        cached_market_trend_scan.clear()
        cached_sector_etf_scan.clear()
        cached_sector_top_gainers_scan.clear()
        st.rerun()
    cache_col.caption("Cached for 15 minutes to avoid repeated Yahoo Finance calls on every Streamlit rerun.")

    with st.spinner("Checking market condition from SPY, QQQ, DIA, and IWM..."):
        summary = cached_market_trend_scan()

    result_df = summary.get("results", pd.DataFrame())
    if not isinstance(result_df, pd.DataFrame):
        result_df = pd.DataFrame()

    overall_market = str(summary.get("overall_market") or "UNKNOWN")
    overall_reason = str(summary.get("overall_reason") or "")
    interpretation = str(summary.get("interpretation") or "")
    weighted_score = float(summary.get("weighted_score") or 0)
    palette = market_condition_style(overall_market)
    conclusion = _market_condition_conclusion(overall_market, result_df)

    st.markdown(
        """
        <style>
            .mc-wrap {font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;}
            .mc-overall {
                border: 1px solid var(--mc-border);
                border-left: 8px solid var(--mc-accent);
                background: var(--mc-bg);
                color: var(--mc-text);
                border-radius: 8px;
                padding: 16px 18px;
                margin: 8px 0 18px 0;
            }
            .mc-overall-label {font-size: 12px; font-weight: 800; letter-spacing: .05em; text-transform: uppercase;}
            .mc-overall-title {font-size: 28px; font-weight: 850; line-height: 1.15; margin-top: 3px;}
            .mc-overall-sub {font-size: 15px; margin-top: 7px;}
            .mc-card-grid {display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 8px 0 18px;}
            .mc-card {border: 1px solid #e5e7eb; border-radius: 8px; background: #fff; padding: 14px 15px; min-height: 250px;}
            .mc-card-head {display: flex; justify-content: space-between; gap: 10px; align-items: flex-start;}
            .mc-ticker {font-size: 18px; font-weight: 850; color: #111827;}
            .mc-pill {border-radius: 999px; padding: 3px 8px; font-size: 11px; font-weight: 800; line-height: 1.1;}
            .mc-close {font-size: 28px; font-weight: 850; color: #111827; margin-top: 12px;}
            .mc-momentum {font-size: 12px; margin-top: 2px; padding-bottom: 9px; border-bottom: 1px solid #e5e7eb;}
            .mc-row {display: grid; grid-template-columns: 78px 1fr; gap: 8px; margin-top: 7px; align-items: baseline; font-size: 12px;}
            .mc-row span:first-child {color: #6b7280;}
            .mc-row b {font-weight: 800;}
            .mc-volume-pill {border-radius: 8px; padding: 7px 8px; font-size: 12px; font-weight: 800; margin-top: 10px;}
            .mc-takeaways {background: #fbfaf4; border-radius: 8px; padding: 17px 18px; margin: 10px 0 22px;}
            .mc-takeaways-title {font-size: 14px; font-weight: 800; color: #111827; margin-bottom: 10px;}
            .mc-takeaway-grid {display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px 22px;}
            .mc-takeaway {display: grid; grid-template-columns: 10px 1fr; gap: 8px; font-size: 13px; color: #374151; line-height: 1.35;}
            .mc-dot {width: 6px; height: 6px; border-radius: 50%; margin-top: 6px;}
            .mc-mini-volume {border-top: 1px solid #eef0f3; margin-top: 11px; padding-top: 10px;}
            .mc-mini-volume-head {display:flex; justify-content:space-between; gap:8px; font-size:11px; color:#6b7280; margin-bottom:8px;}
            .mc-mini-volume-head b {color:#111827; font-weight:800;}
            .mc-mini-volume-body {display:grid; grid-template-columns:82px 1fr; gap:10px; align-items:center;}
            .mc-count-box {border:1px solid #e5e7eb; border-radius:8px; padding:7px 8px; background:#fafafa; display:grid; gap:3px; font-size:11px; color:#6b7280;}
            .mc-count-box b {font-size:13px; font-weight:850;}
            .mc-mini-bars {display:grid; gap:5px;}
            .mc-mini-line {display:grid; grid-template-columns:36px 1fr 42px; gap:6px; align-items:center; font-size:11px; color:#6b7280;}
            .mc-mini-track {height:6px; border-radius:999px; background:#e5e7eb; overflow:hidden;}
            .mc-up {height: 100%; background: #5a8f24;}
            .mc-down {height: 100%; background: #ef5b5b;}
            .mc-above-volume {font-size:11px; color:#374151; margin-top:8px; line-height:1.35;}
            .mc-average-volume {font-size:11px; color:#374151; margin-top:5px; line-height:1.35;}
            .mc-big-volume {font-size:11px; color:#6b7280; margin-top:7px; line-height:1.35;}
            .mc-guide {border: 1px solid #e5e7eb; border-radius: 8px; background: #fff; padding: 14px; height: 100%;}
            .mc-guide h4 {font-size: 14px; margin: 0 0 8px; color: #111827;}
            .mc-guide p {font-size: 13px; margin: 0; color: #4b5563; line-height: 1.45;}
            @media (max-width: 1100px) {.mc-card-grid {grid-template-columns: repeat(2, minmax(0, 1fr));}}
            @media (max-width: 700px) {.mc-card-grid, .mc-takeaway-grid {grid-template-columns: 1fr;}}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="mc-overall" style="--mc-border:{palette['border']};--mc-accent:{palette['accent']};--mc-bg:{palette['bg']};--mc-text:{palette['text']};">
            <div class="mc-overall-label">Overall Market</div>
            <div class="mc-overall-title">{escape(overall_market)}</div>
            <div class="mc-overall-sub">{escape(interpretation)}</div>
            <div style="font-size:13px; opacity:.86; margin-top:9px;">{escape(overall_reason)}</div>
            <div style="font-size:14px; font-weight:750; line-height:1.45; margin-top:12px;">{escape(conclusion)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Weighted Score", f"{weighted_score:.2f}")
    metric_cols[1].metric("Indexes Checked", f"{len(result_df):,}/4")
    latest_dates = sorted({str(value) for value in result_df.get("Date", []) if str(value)})
    metric_cols[2].metric("Latest Candle", latest_dates[-1] if latest_dates else "-")
    bullish_count = int(result_df["Trend"].astype(str).str.contains("UPTREND", case=False, na=False).sum()) if not result_df.empty else 0
    metric_cols[3].metric("Bullish Indexes", f"{bullish_count}/4")

    if not result_df.empty:
        cards: list[str] = []
        for row in result_df.to_dict("records"):
            style = trend_style(str(row.get("Trend") or ""))
            volume_style = _volume_pill_style(str(row.get("Volume Confirmation") or ""))
            momentum, momentum_color = _price_momentum_text(row)
            ma_order, ma_color = _ma_order_text(row)
            swing_text, swing_color = _swing_short(row)
            plus_di = _market_number(row.get("+DI"), 1)
            minus_di = _market_number(row.get("-DI"), 1)
            weekly_plus_di = _market_number(row.get("Weekly +DI"), 1)
            weekly_minus_di = _market_number(row.get("Weekly -DI"), 1)
            volume_ratio = _market_float(row.get("Volume SMA 5/20 Ratio"))
            volume_ratio_note = (
                f"5-day average volume is {volume_ratio:.0%} of the 20-day average volume."
                if volume_ratio is not None
                else "5-day average volume ratio is unavailable."
            )
            cards.append(
                '<div class="mc-card">'
                '<div class="mc-card-head">'
                f'<div class="mc-ticker">{escape(str(row.get("Ticker") or ""))}</div>'
                f'<div class="mc-pill" style="background:{style["bg"]};color:{style["color"]};">{escape(style["label"])}</div>'
                "</div>"
                f'<div class="mc-close">{escape(_market_number(row.get("Close"), 2))}</div>'
                f'<div class="mc-momentum" style="color:{momentum_color};">{escape(momentum)}</div>'
                f'<div class="mc-row"><span>ADX</span><b>{escape(_market_number(row.get("ADX"), 1))} <span style="font-weight:500;color:#6b7280;">({_adx_reading(row.get("ADX"))})</span></b></div>'
                f'<div class="mc-row"><span>Weekly ADX</span><b>{escape(_market_number(row.get("Weekly ADX"), 1))} <span style="font-weight:500;color:#6b7280;">({_adx_reading(row.get("Weekly ADX"))})</span></b></div>'
                f'<div class="mc-row"><span>W +DI / -DI</span><b><span style="color:#166534;">{escape(weekly_plus_di)}</span> / <span style="color:#991b1b;">{escape(weekly_minus_di)}</span></b></div>'
                f'<div class="mc-row"><span>+DI / -DI</span><b><span style="color:#166534;">{escape(plus_di)}</span> / <span style="color:#991b1b;">{escape(minus_di)}</span></b></div>'
                f'<div class="mc-row"><span>MA order</span><b style="color:{ma_color};">{escape(ma_order)}</b></div>'
                f'<div class="mc-row"><span>Swing</span><b style="color:{swing_color};">{escape(swing_text)}</b></div>'
                f'<div class="mc-row"><span>5D/20D Avg Vol</span><b>{escape(_market_number(row.get("Volume SMA 5/20 Ratio"), 2))}</b></div>'
                f'<div style="font-size:11px;color:#6b7280;line-height:1.35;margin-top:3px;">{escape(volume_ratio_note)}</div>'
                f'<div class="mc-volume-pill" style="background:{volume_style["bg"]};color:{volume_style["color"]};">{escape(volume_style["label"])}</div>'
                f"{_volume_mini_chart_html(row)}"
                "</div>"
            )
        st.markdown(f"<div class=\"mc-card-grid\">{''.join(cards)}</div>", unsafe_allow_html=True)

        takeaways = _market_takeaways(result_df)
        takeaway_html = "".join(
            f"<div class=\"mc-takeaway\"><span class=\"mc-dot\" style=\"background:{color};\"></span><div>{escape(text)}</div></div>"
            for color, text in takeaways
        )
        st.markdown(
            f'<div class="mc-takeaways"><div class="mc-takeaways-title">Key takeaways</div><div class="mc-takeaway-grid">{takeaway_html}</div></div>',
            unsafe_allow_html=True,
        )

        show_sector_etf_panel()
        show_sector_top_gainers_panel()

        breadth_tickers = load_eligible_ticker_symbols(DB_PATH)
        if breadth_tickers:
            show_market_breadth_panel(breadth_tickers)
        else:
            st.info("Market breadth needs the saved $500M+ ticker universe. Download/store that universe first.")

        st.markdown("**How to read ADX, volume, price momentum, and breadth**")
        guide_cols = st.columns(4)
        guide_cols[0].markdown(
            """
            <div class="mc-guide">
                <h4>ADX / DI</h4>
                <p>ADX below 20 is usually trendless. ADX 20-25 is forming or choppy. ADX above 25 means the trend has strength. +DI above -DI means buyers control momentum; -DI above +DI means sellers control momentum.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        guide_cols[1].markdown(
            """
            <div class="mc-guide">
                <h4>Price Momentum</h4>
                <p>Close above EMA 9 shows short-term strength. Close below EMA 9 shows pullback pressure. A clean bullish stack is Close > EMA 9 > SMA 20 > SMA 50 > SMA 200. HH + HL confirms bullish structure.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        guide_cols[2].markdown(
            """
            <div class="mc-guide">
                <h4>Volume</h4>
                <p>Volume 5/20 above 1.00 means recent volume is expanding. Up-volume stronger than down-volume plus rising OBV supports an uptrend. Down-volume stronger plus falling OBV warns of distribution.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        guide_cols[3].markdown(
            """
            <div class="mc-guide">
                <h4>Breadth</h4>
                <p>Breadth checks how many stocks participate. If SPY rises while few stocks stay above the 200 SMA, the move is fragile. Healthy breadth means more stocks are advancing and holding above key averages.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Market condition detail table", expanded=False):
            st.dataframe(format_market_condition_results(result_df), width="stretch", hide_index=True)
            st.download_button(
                "Export market condition CSV",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="market_trend_with_volume_support.csv",
                mime="text/csv",
            )
    else:
        st.warning("Market condition could not be calculated.")

    errors = summary.get("errors", [])
    if isinstance(errors, list) and errors:
        with st.expander("Market condition data issues", expanded=False):
            st.text("\n".join(str(error) for error in errors))


def show_rule_editor(selected_editor: str) -> None:
    rule_config = load_rule_config()
    message = st.session_state.pop("rule_config_message", None)
    if message:
        st.success(message)

    with st.expander("Rule Editor", expanded=True):
        if selected_editor == "weekly_v6_cup_handle":
            st.markdown("**Weekly Breakout Rule**")
            st.write(
                "This imported scanner uses fixed strategy settings from "
                "`weekly_v6_rs5_52w_ath_cup_handle_backtest.py`."
            )
            st.write(
                "Main checks: weekly stock uptrend, SPY/QQQ/sector ETF uptrend, "
                "RS 5W vs SPY > 5%, RS ratio ROC 5W > 5%, stock RS vs sector > 5%, "
                "52W/ATH breakout, breakout volume >= 1.2x 20-week average, "
                "and avoids oversized green or extreme-volume breakout candles."
            )
            if st.button("Close Rule Details", width="stretch"):
                st.session_state.pop("rule_editor", None)
                st.rerun()
            return

        if selected_editor == "pullback":
            cfg = rule_config.pullback
            with st.form("pullback_rule_form"):
                st.markdown("**Edit Pullback Setup Rule**")
                min_revenue_growth = st.number_input(
                    "Revenue growth greater than (%)",
                    min_value=0.0,
                    max_value=500.0,
                    value=float(cfg.min_revenue_growth * 100),
                    step=1.0,
                )
                min_eps_growth = st.number_input(
                    "EPS growth greater than (%)",
                    min_value=0.0,
                    max_value=500.0,
                    value=float(cfg.min_eps_growth * 100),
                    step=1.0,
                )
                min_avg_volume = st.number_input(
                    "20-day average volume greater than",
                    min_value=0,
                    value=int(cfg.min_avg_volume),
                    step=100_000,
                )
                max_pullback_distance = st.number_input(
                    "Close within this distance of 9 EMA/SMA (%)",
                    min_value=0.0,
                    max_value=50.0,
                    value=float(cfg.max_pullback_distance * 100),
                    step=0.25,
                )
                allow_hammer = st.checkbox("Hammer candle", value=cfg.allow_hammer)
                allow_bullish_engulfing = st.checkbox(
                    "Bullish engulfing candle",
                    value=cfg.allow_bullish_engulfing,
                )
                allow_bullish_rejection = st.checkbox(
                    "Bullish rejection candle",
                    value=cfg.allow_bullish_rejection,
                )
                submitted = st.form_submit_button("Submit Rule", width="stretch")

            if submitted:
                updated = RuleConfig(
                    pullback=PullbackRuleConfig(
                        min_revenue_growth=min_revenue_growth / 100,
                        min_eps_growth=min_eps_growth / 100,
                        min_avg_volume=int(min_avg_volume),
                        max_pullback_distance=max_pullback_distance / 100,
                        allow_hammer=allow_hammer,
                        allow_bullish_engulfing=allow_bullish_engulfing,
                        allow_bullish_rejection=allow_bullish_rejection,
                    ),
                    marubozu=rule_config.marubozu,
                    weekly_ath=rule_config.weekly_ath,
                    weekly_momentum=rule_config.weekly_momentum,
                    morning_star=rule_config.morning_star,
                    technical_strength=rule_config.technical_strength,
                )
                save_rule_config(updated)
                st.session_state["rule_config_message"] = "Pullback setup rule saved."
                st.rerun()

        elif selected_editor == "marubozu":
            cfg = rule_config.marubozu
            with st.form("marubozu_rule_form"):
                st.markdown("**Edit Green Marubozu 52W Breakout Rule**")
                min_body_pct = st.number_input(
                    "Body at least this much of candle range (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(cfg.min_body_pct * 100),
                    step=1.0,
                )
                max_upper_wick_pct = st.number_input(
                    "Upper wick no more than (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(cfg.max_upper_wick_pct * 100),
                    step=1.0,
                )
                max_lower_wick_pct = st.number_input(
                    "Lower wick no more than (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(cfg.max_lower_wick_pct * 100),
                    step=1.0,
                )
                min_volume_ratio = st.number_input(
                    "Volume at least this many times 20-day average",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(cfg.min_volume_ratio),
                    step=0.1,
                )
                require_52w_breakout = st.checkbox(
                    "Require candle to break prior 52-week high",
                    value=cfg.require_52w_breakout,
                )
                submitted = st.form_submit_button("Submit Rule", width="stretch")

            if submitted:
                updated = RuleConfig(
                    pullback=rule_config.pullback,
                    marubozu=MarubozuRuleConfig(
                        min_body_pct=min_body_pct / 100,
                        max_upper_wick_pct=max_upper_wick_pct / 100,
                        max_lower_wick_pct=max_lower_wick_pct / 100,
                        min_volume_ratio=min_volume_ratio,
                        require_52w_breakout=require_52w_breakout,
                    ),
                    weekly_ath=rule_config.weekly_ath,
                    weekly_momentum=rule_config.weekly_momentum,
                    morning_star=rule_config.morning_star,
                    technical_strength=rule_config.technical_strength,
                )
                save_rule_config(updated)
                st.session_state["rule_config_message"] = "Green marubozu rule saved."
                st.rerun()

        elif selected_editor == "weekly_ath":
            cfg = rule_config.weekly_ath
            with st.form("weekly_ath_rule_form"):
                st.markdown("**Edit Weekly ATH Breakout Rule**")
                min_volume_ratio = st.number_input(
                    "Weekly volume at least this many times 20-week average",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(cfg.min_volume_ratio),
                    step=0.1,
                )
                require_uptrend = st.checkbox("Require weekly uptrend", value=cfg.require_uptrend)
                require_fundamentals = st.checkbox("Require fundamentals", value=cfg.require_fundamentals)
                min_revenue_growth = st.number_input(
                    "Revenue growth greater than (%)",
                    min_value=0.0,
                    max_value=500.0,
                    value=float(cfg.min_revenue_growth * 100),
                    step=1.0,
                    disabled=not require_fundamentals,
                )
                min_eps_growth = st.number_input(
                    "EPS growth greater than (%)",
                    min_value=0.0,
                    max_value=500.0,
                    value=float(cfg.min_eps_growth * 100),
                    step=1.0,
                    disabled=not require_fundamentals,
                )
                min_avg_volume = st.number_input(
                    "Daily average volume greater than",
                    min_value=0,
                    value=int(cfg.min_avg_volume),
                    step=100_000,
                    disabled=not require_fundamentals,
                )
                submitted = st.form_submit_button("Submit Rule", width="stretch")

            if submitted:
                updated = RuleConfig(
                    pullback=rule_config.pullback,
                    marubozu=rule_config.marubozu,
                    weekly_ath=WeeklyATHRuleConfig(
                        min_revenue_growth=min_revenue_growth / 100,
                        min_eps_growth=min_eps_growth / 100,
                        min_avg_volume=int(min_avg_volume),
                        min_volume_ratio=min_volume_ratio,
                        require_fundamentals=require_fundamentals,
                        require_uptrend=require_uptrend,
                    ),
                    weekly_momentum=rule_config.weekly_momentum,
                    morning_star=rule_config.morning_star,
                    technical_strength=rule_config.technical_strength,
                )
                save_rule_config(updated)
                st.session_state["rule_config_message"] = "Weekly ATH breakout rule saved."
                st.rerun()

        elif selected_editor == "weekly_momentum":
            cfg = rule_config.weekly_momentum
            with st.form("weekly_momentum_rule_form"):
                st.markdown("**Edit Weekly Price Momentum Rule**")
                min_revenue_growth = st.number_input(
                    "Revenue growth greater than (%)",
                    min_value=0.0,
                    max_value=500.0,
                    value=float(cfg.min_revenue_growth * 100),
                    step=1.0,
                )
                min_eps_growth = st.number_input(
                    "EPS growth greater than (%)",
                    min_value=0.0,
                    max_value=500.0,
                    value=float(cfg.min_eps_growth * 100),
                    step=1.0,
                )
                min_avg_weekly_volume = st.number_input(
                    "20-week average volume greater than",
                    min_value=0,
                    value=int(cfg.min_avg_weekly_volume),
                    step=1_000_000,
                )
                max_pullback_distance = st.number_input(
                    "Full weekly candle within this distance of 9W EMA/SMA (%)",
                    min_value=0.0,
                    max_value=50.0,
                    value=float(cfg.max_pullback_distance * 100),
                    step=0.25,
                )
                allow_hammer = st.checkbox("Hammer candle", value=cfg.allow_hammer)
                allow_bullish_engulfing = st.checkbox(
                    "Bullish engulfing candle",
                    value=cfg.allow_bullish_engulfing,
                )
                submitted = st.form_submit_button("Submit Rule", width="stretch")

            if submitted:
                updated = RuleConfig(
                    pullback=rule_config.pullback,
                    marubozu=rule_config.marubozu,
                    weekly_ath=rule_config.weekly_ath,
                    weekly_momentum=WeeklyMomentumRuleConfig(
                        min_revenue_growth=min_revenue_growth / 100,
                        min_eps_growth=min_eps_growth / 100,
                        min_avg_weekly_volume=int(min_avg_weekly_volume),
                        max_pullback_distance=max_pullback_distance / 100,
                        allow_hammer=allow_hammer,
                        allow_bullish_engulfing=allow_bullish_engulfing,
                    ),
                    morning_star=rule_config.morning_star,
                    technical_strength=rule_config.technical_strength,
                )
                save_rule_config(updated)
                st.session_state["rule_config_message"] = "Weekly price momentum rule saved."
                st.rerun()

        elif selected_editor == "morning_star":
            cfg = rule_config.morning_star
            with st.form("morning_star_rule_form"):
                st.markdown("**Edit Morning Star Rule**")
                min_avg_volume = st.number_input(
                    "20-day average volume greater than",
                    min_value=0,
                    value=int(cfg.min_avg_volume),
                    step=100_000,
                )
                first_long_body_pct = st.number_input(
                    "First candle body at least this much of range (%)",
                    min_value=1.0,
                    max_value=100.0,
                    value=float(cfg.first_long_body_pct * 100),
                    step=1.0,
                )
                small_body_pct = st.number_input(
                    "Middle candle body no more than first candle body (%)",
                    min_value=1.0,
                    max_value=100.0,
                    value=float(cfg.small_body_pct * 100),
                    step=1.0,
                )
                third_long_body_pct = st.number_input(
                    "Third candle body at least this much of range (%)",
                    min_value=1.0,
                    max_value=100.0,
                    value=float(cfg.third_long_body_pct * 100),
                    step=1.0,
                )
                recovery_pct = st.number_input(
                    "Third candle recovers at least this much of first candle body (%)",
                    min_value=1.0,
                    max_value=100.0,
                    value=float(cfg.recovery_pct * 100),
                    step=1.0,
                )
                gap_tolerance_pct = st.number_input(
                    "Flexible gap tolerance (%)",
                    min_value=0.0,
                    max_value=5.0,
                    value=float(cfg.gap_tolerance_pct * 100),
                    step=0.1,
                )
                require_uptrend = st.checkbox("Require daily uptrend", value=cfg.require_uptrend)
                require_second_volume_above_average = st.checkbox(
                    "Require second candle volume above 20-day average",
                    value=cfg.require_second_volume_above_average,
                )
                require_third_volume_above_average = st.checkbox(
                    "Require third candle volume above 20-day average",
                    value=cfg.require_third_volume_above_average,
                )
                submitted = st.form_submit_button("Submit Rule", width="stretch")

            if submitted:
                updated = RuleConfig(
                    pullback=rule_config.pullback,
                    marubozu=rule_config.marubozu,
                    weekly_ath=rule_config.weekly_ath,
                    weekly_momentum=rule_config.weekly_momentum,
                    morning_star=MorningStarRuleConfig(
                        min_avg_volume=int(min_avg_volume),
                        first_long_body_pct=first_long_body_pct / 100,
                        small_body_pct=small_body_pct / 100,
                        recovery_pct=recovery_pct / 100,
                        third_long_body_pct=third_long_body_pct / 100,
                        gap_tolerance_pct=gap_tolerance_pct / 100,
                        require_uptrend=require_uptrend,
                        require_second_volume_above_average=require_second_volume_above_average,
                        require_third_volume_above_average=require_third_volume_above_average,
                    ),
                    technical_strength=rule_config.technical_strength,
                )
                save_rule_config(updated)
                st.session_state["rule_config_message"] = "Morning Star rule saved."
                st.rerun()

        elif selected_editor in {"technical_strength", "technical_breakout", "technical_pullback_9ema"}:
            cfg = rule_config.technical_strength
            with st.form("technical_strength_rule_form"):
                st.markdown("**Edit Technical Breakout / 9 EMA Pullback Shared Settings**")
                min_rsi = st.number_input(
                    "RSI greater than",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(cfg.min_rsi),
                    step=1.0,
                )
                min_rs_20d_vs_spy = st.number_input(
                    "20D relative strength vs SPY greater than (%)",
                    min_value=-100.0,
                    max_value=500.0,
                    value=float(cfg.min_rs_20d_vs_spy * 100),
                    step=1.0,
                )
                min_rs_ratio_roc_20d = st.number_input(
                    "RS ratio ROC 20D greater than (%)",
                    min_value=-100.0,
                    max_value=500.0,
                    value=float(cfg.min_rs_ratio_roc_20d * 100),
                    step=1.0,
                )
                weak_market_min_outperformance_5d = st.number_input(
                    "If weak market, stock 5D outperformance vs SPY greater than (%)",
                    min_value=-100.0,
                    max_value=500.0,
                    value=float(cfg.weak_market_min_outperformance_5d * 100),
                    step=0.5,
                )
                weak_market_min_stock_return_5d = st.number_input(
                    "If weak market, stock 5D return better than (%)",
                    min_value=-100.0,
                    max_value=500.0,
                    value=float(cfg.weak_market_min_stock_return_5d * 100),
                    step=0.5,
                )
                min_breakout_volume_ratio = st.number_input(
                    "Breakout volume at least this many times 20-day average",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(cfg.min_breakout_volume_ratio),
                    step=0.1,
                )
                max_pullback_distance = st.number_input(
                    "Pullback can be within this distance of 9 EMA (%)",
                    min_value=0.0,
                    max_value=50.0,
                    value=float(cfg.max_pullback_distance * 100),
                    step=0.25,
                )
                require_bullish_signal = st.checkbox("Require bullish signal candle on pullback reclaim", value=cfg.require_bullish_signal)
                submitted = st.form_submit_button("Submit Rule", width="stretch")

            if submitted:
                updated = RuleConfig(
                    pullback=rule_config.pullback,
                    marubozu=rule_config.marubozu,
                    weekly_ath=rule_config.weekly_ath,
                    weekly_momentum=rule_config.weekly_momentum,
                    morning_star=rule_config.morning_star,
                    technical_strength=TechnicalStrengthRuleConfig(
                        min_rsi=min_rsi,
                        min_rs_20d_vs_spy=min_rs_20d_vs_spy / 100,
                        min_rs_ratio_roc_20d=min_rs_ratio_roc_20d / 100,
                        weak_market_min_outperformance_5d=weak_market_min_outperformance_5d / 100,
                        weak_market_min_stock_return_5d=weak_market_min_stock_return_5d / 100,
                        min_breakout_volume_ratio=min_breakout_volume_ratio,
                        max_pullback_distance=max_pullback_distance / 100,
                        require_bullish_signal=require_bullish_signal,
                    ),
                )
                save_rule_config(updated)
                st.session_state["rule_config_message"] = "Technical Strength rule saved."
                st.rerun()

        elif selected_editor == "monthly_big_volume":
            st.info(
                "Monthly Big Volume Candle uses fixed rules for now: big green monthly body or green monthly hammer, "
                "with volume above the prior month and above the 12-month average."
            )


def today_scan_page() -> None:
    st.title(APP_NAME)
    st.caption("Daily U.S. swing-trading setup scanner")
    show_rules()
    show_eligible_universe_builder()

    completed_date = latest_completed_us_session()
    st.info(f"Latest completed U.S. daily candle: {completed_date.date()}")

    with st.sidebar:
        st.header("Scan Controls")
        schedule_config = load_schedule_config()
        st.subheader("Rules")
        weekday_lookup = {name: code for code, name in WEEKDAY_NAMES.items()}
        rule_help = {
            "technical_breakout": (
                "Daily trend stack, RSI > 50, relative strength vs SPY, 5D resilience, "
                "and recent ATH/52W breakout on expanded volume."
            ),
            "technical_pullback_9ema": (
                "Daily trend stack, RSI > 50, relative strength vs SPY, 5D resilience, "
                "and 9 EMA pullback/reclaim with bullish signal candle."
            ),
            "weekly_v6_cup_handle": (
                "Generic weekly breakout: stock/SPY/QQQ/sector uptrend, RS 5W checks, "
                "52W/ATH breakout, and volume confirmation."
            ),
            "monthly_big_volume": (
                "Latest completed monthly candle is a big green candle or green hammer, "
                "with volume above both previous month and 12-month average."
            ),
        }
        run_now: dict[str, bool] = {}
        schedule_updates: dict[str, RuleSchedule] = {}
        for rule_key in RULE_ORDER:
            saved_schedule = schedule_config.schedule_for(rule_key)
            with st.expander(RULE_LABELS[rule_key], expanded=rule_key == "pullback"):
                action_col, run_col = st.columns([1, 5])
                with action_col:
                    if st.button("✎", key=f"edit_{rule_key}_rule", help=f"Edit {RULE_LABELS[rule_key]} rule"):
                        st.session_state["rule_editor"] = rule_key
                with run_col:
                    run_now[rule_key] = st.checkbox(
                        "Run now",
                        value=saved_schedule.enabled,
                        key=f"run_now_{rule_key}",
                        help=rule_help.get(rule_key, "Run this saved scanner rule."),
                    )
                enabled = st.checkbox(
                    "Schedule this rule",
                    value=saved_schedule.enabled,
                    key=f"schedule_enabled_{rule_key}",
                )
                frequency_label = st.radio(
                    "Frequency",
                    ["Daily", "Weekly"],
                    index=0 if saved_schedule.frequency == "daily" else 1,
                    horizontal=True,
                    key=f"schedule_frequency_{rule_key}",
                )
                selected_weekday_names: list[str] = []
                if frequency_label == "Weekly":
                    default_weekday_names = [WEEKDAY_NAMES[day] for day in saved_schedule.active_weekdays]
                    selected_weekday_names = st.multiselect(
                        "Weekly run days",
                        options=list(WEEKDAY_NAMES.values()),
                        default=default_weekday_names,
                        key=f"schedule_weekdays_{rule_key}",
                    )
                schedule_time = st.time_input(
                    "Run at",
                    value=pd.to_datetime(saved_schedule.run_time).time(),
                    help="Saved in Asia/Kolkata time.",
                    key=f"schedule_time_{rule_key}",
                )
                selected_weekdays = [weekday_lookup[name] for name in selected_weekday_names]
                schedule_updates[rule_key] = RuleSchedule(
                    enabled=enabled,
                    run_time=schedule_time.strftime("%H:%M"),
                    timezone="Asia/Kolkata",
                    frequency=frequency_label.lower(),
                    weekdays=selected_weekdays or saved_schedule.active_weekdays,
                )
                if enabled:
                    if frequency_label == "Weekly":
                        days = ", ".join(selected_weekday_names or [WEEKDAY_NAMES[day] for day in saved_schedule.active_weekdays])
                        st.caption(f"Scheduled weekly on {days} at {schedule_time.strftime('%H:%M')} IST.")
                    else:
                        st.caption(f"Scheduled daily at {schedule_time.strftime('%H:%M')} IST.")
                else:
                    st.caption("Scheduled run is disabled.")
        selected_editor = st.session_state.get("rule_editor")
        if selected_editor:
            show_rule_editor(selected_editor)
        run_pullback_setup = run_now.get("pullback", False)
        run_marubozu_setup = run_now.get("marubozu", False)
        run_weekly_ath_setup = run_now.get("weekly_ath", False)
        run_weekly_momentum_setup = run_now.get("weekly_momentum", False)
        run_morning_star_setup = run_now.get("morning_star", False)
        run_technical_breakout_setup = run_now.get("technical_breakout", False)
        run_technical_pullback_9ema_setup = run_now.get("technical_pullback_9ema", False)
        run_weekly_v6_cup_handle_setup = run_now.get("weekly_v6_cup_handle", False)
        run_monthly_big_volume_setup = run_now.get("monthly_big_volume", False)
        selected_rule_count = sum(1 for should_run in run_now.values() if should_run)
        background_running = show_background_scan_status()
        if selected_rule_count == 0:
            st.warning("Select at least one rule to run now.")
        run_selected_rules = st.button(
            "Run Selected Scan Rules",
            type="primary",
            disabled=selected_rule_count == 0 or background_running,
            width="stretch",
        )
        if st.button("Save Rule Schedules", width="stretch"):
            save_schedule_config(ScheduleConfig(rules=schedule_updates))
            st.success("Saved separate schedules for each rule.")
        with st.expander("+ Add More Rules"):
            st.write("Use this area to park the next setup idea before we wire it into the scanner.")
            st.text_input("Rule name", placeholder="Example: Tight flag breakout")
            st.text_area("Rule notes", placeholder="Describe candle, volume, trend, and risk rules.")
        st.divider()
        if st.button("Refresh Screen Data", width="stretch"):
            st.rerun()

    tickers = load_eligible_ticker_symbols(DB_PATH)
    universe_source = "stored $500M+"
    if not tickers:
        st.warning("No stored $500M+ ticker universe yet. Click Download $500M+ NASDAQ/NYSE Tickers first.")

    st.write(f"Universe loaded: **{len(tickers):,}** tickers ({universe_source})")
    start_message = st.session_state.pop("manual_scan_start_message", "")
    if start_message:
        st.success(start_message)

    if "latest_stats" not in st.session_state:
        latest_state = load_latest_scan_state()
        if latest_state:
            results, errors, stats, saved_count = latest_state
            st.session_state["latest_results"] = results
            st.session_state["latest_errors"] = errors
            st.session_state["latest_scanned_count"] = int(stats.get("stored_tickers", len(tickers)))
            st.session_state["latest_saved_count"] = saved_count
            st.session_state["latest_stats"] = stats

    if run_selected_rules and tickers:
        selected_rule_keys = [
            rule_key
            for rule_key, should_run in {
                "pullback": run_pullback_setup,
                "marubozu": run_marubozu_setup,
                "morning_star": run_morning_star_setup,
                "weekly_ath": run_weekly_ath_setup,
                "weekly_momentum": run_weekly_momentum_setup,
                "technical_breakout": run_technical_breakout_setup,
                "technical_pullback_9ema": run_technical_pullback_9ema_setup,
                "weekly_v6_cup_handle": run_weekly_v6_cup_handle_setup,
                "monthly_big_volume": run_monthly_big_volume_setup,
            }.items()
            if should_run
        ]
        process_id = start_background_scan(selected_rule_keys)
        st.session_state["manual_scan_start_message"] = (
            f"Started background scan PID {process_id} for: "
            f"{', '.join(RULE_LABELS[rule_key] for rule_key in selected_rule_keys)}."
        )
        st.rerun()

    show_scanner_results()


def market_condition_page() -> None:
    show_market_condition_panel()


def previous_watchlists_page() -> None:
    st.title("Previous Watchlists")
    st.caption("Saved results are grouped by U.S. candle date. A 9 AM IST run usually saves under the prior U.S. trading date.")

    rule_runs = list_rule_watchlists(DB_PATH)
    run_state_path = Path("data") / "schedule_run_state.json"
    if run_state_path.exists():
        try:
            run_state = json.loads(run_state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            run_state = {}
        if run_state:
            st.info(
                "Last scheduled run: "
                f"{run_state.get('last_scan_run_at', '-')} "
                f"(calendar date {run_state.get('last_scan_run_date', '-')}). "
                "Rows below use completed U.S. candle dates, so a morning IST run usually appears under the prior U.S. session."
            )
    if not rule_runs.empty:
        rule_runs = rule_runs[
            rule_runs["rule_name"].astype(str).isin(
                [
                    "Technical Breakout",
                    "9 EMA Pullback",
                    "Score 60+ Setup Scan",
                    "Weekly Breakout",
                    "Weekly V6 Cup Handle",
                    "Top 50 Score Above 60",
                ]
            )
        ].copy()
    if not rule_runs.empty:
        recent_dates = sorted(rule_runs["scan_date"].dropna().astype(str).unique(), reverse=True)[:5]
        rule_runs = rule_runs[rule_runs["scan_date"].astype(str).isin(recent_dates)].copy()
        latest_saved_at = str(rule_runs["saved_at"].max()) if "saved_at" in rule_runs else "-"
        metric_columns = st.columns(3)
        metric_columns[0].metric("Saved rule runs", f"{len(rule_runs):,}")
        metric_columns[1].metric("Latest candle date", str(rule_runs["scan_date"].max()))
        metric_columns[2].metric("Last saved", latest_saved_at)

        st.subheader("Scheduled Run History")
        history = rule_runs.copy()
        if "saved_at" in history.columns:
            history["saved_at"] = history["saved_at"].astype(str)
        history = history.rename(
            columns={
                "rule_name": "Rule",
                "scan_date": "Candle date",
                "ticker_count": "Tickers",
                "saved_at": "Run saved at",
            }
        )
        st.dataframe(history, width="stretch", hide_index=True)

        st.subheader("Open Watchlist")
        selection_columns = st.columns([1, 2])
        scan_dates = recent_dates
        selected_date = selection_columns[0].selectbox("Candle date", scan_dates)
        rules_for_date = rule_runs.loc[rule_runs["scan_date"] == selected_date, "rule_name"].tolist()
        selected_rule = selection_columns[1].selectbox("Rule", rules_for_date)
        selected_row = rule_runs[
            (rule_runs["scan_date"] == selected_date) & (rule_runs["rule_name"] == selected_rule)
        ].iloc[0]

        frame = load_rule_watchlist(selected_row["rule_name"], selected_row["scan_date"], DB_PATH)
        if frame.empty:
            detail_columns = st.columns(3)
            detail_columns[0].metric("Rule", selected_row["rule_name"])
            detail_columns[1].metric("Candle date", selected_row["scan_date"])
            detail_columns[2].metric("Tickers", "0")
            st.info("This rule ran successfully, but it had no final watchlist tickers.")
            return

        if selected_row["rule_name"] == "Green Marubozu 52W Breakout":
            display_frame = format_marubozu_results(frame)
        elif selected_row["rule_name"] == "Morning Star":
            display_frame = format_morning_star_results(frame)
        elif selected_row["rule_name"] == "Weekly ATH Breakout":
            display_frame = format_weekly_ath_results(frame)
        elif selected_row["rule_name"] == "Weekly Price Momentum":
            display_frame = format_weekly_momentum_results(frame)
        elif selected_row["rule_name"] in {"Weekly Breakout", "Weekly V6 Cup Handle"}:
            display_frame = format_weekly_v6_cup_handle_results(frame)
        elif selected_row["rule_name"] in {"Technical Strength", "Technical Breakout", "9 EMA Pullback", "Score 60+ Setup Scan"}:
            display_frame = format_technical_strength_results(frame)
        elif selected_row["rule_name"] == "Top 50 Score Above 60":
            display_frame = _format_top50_score_results(frame)
        elif selected_row["rule_name"] == "Pullback Setup" and {"entry", "stop", "target"}.issubset(frame.columns):
            display_frame = format_dashboard(frame)
        else:
            display_frame = frame

        detail_columns = st.columns(3)
        detail_columns[0].metric("Rule", selected_row["rule_name"])
        detail_columns[1].metric("Candle date", selected_row["scan_date"])
        detail_columns[2].metric("Tickers", f"{len(frame):,}")

        st.dataframe(display_frame, width="stretch", hide_index=True)
        st.download_button(
            "Export selected rule watchlist to CSV",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_row['rule_name'].lower().replace(' ', '_')}_{selected_row['scan_date']}.csv",
            mime="text/csv",
        )
        return

    dates = list_scan_dates(DB_PATH)[:5]
    if dates:
        st.subheader("Pullback Watchlists")
        selected_date = st.selectbox("Scan date", dates)
        frame = load_watchlist(selected_date, DB_PATH)
        metric_columns = st.columns(2)
        metric_columns[0].metric("Candle date", selected_date)
        metric_columns[1].metric("Tickers", f"{len(frame):,}")
        st.dataframe(format_dashboard(frame), width="stretch", hide_index=True)
        st.download_button(
            "Export selected watchlist to CSV",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name=f"kalyani_watchlist_{selected_date}.csv",
            mime="text/csv",
        )
    else:
        st.write("No saved watchlists yet.")


def _format_top50_score_results(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    numeric_columns = [
        "total_points",
        "market_sector_score",
        "technical_trend_score",
        "relative_strength_score",
        "fundamental_score",
        "liquidity_risk_score",
        "close",
        "avg_volume_20d",
        "dollar_volume",
        "20d_rsi",
        "weekly_rsi_14",
        "spy_20d_rsi",
        "rsi_q",
        "sector_rsi",
        "sector_rsi_q",
        "eps_surprise_pct",
        "revenue_yoy_growth_pct",
        "revenue_qoq_growth_pct",
        "eps_yoy_growth_pct",
        "eps_qoq_growth_pct",
        "roe_pct",
        "rs_5w_vs_spy_pct",
        "rs_13w_vs_spy_pct",
        "rs_26w_vs_spy_pct",
        "rs_20d_vs_spy_pct",
        "sector_rs_5w_vs_spy_pct",
        "stock_rs_5w_vs_sector_pct",
        "atr_pct",
        "spread_pct",
    ]
    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").round(2)
    if "market_cap" in out.columns:
        out["market_cap"] = pd.to_numeric(out["market_cap"], errors="coerce").map(format_market_cap)
    if "growth_star" in out.columns and "ticker" in out.columns:
        out["ticker"] = out.apply(
            lambda row: f"{row.get('growth_star')} {row.get('ticker')}".strip()
            if str(row.get("growth_star") or "").strip()
            else row.get("ticker"),
            axis=1,
        )
    columns = [
        "rank",
        "ticker",
        "close",
        "market_cap",
        "total_points",
        "market_sector_score",
        "technical_trend_score",
        "relative_strength_score",
        "fundamental_score",
        "liquidity_risk_score",
        "rsi_q",
        "20d_rsi",
        "weekly_rsi_14",
        "spy_20d_rsi",
        "sector_rsi",
        "sector_rsi_q",
        "avg_volume_20d",
        "dollar_volume",
        "rs_5w_vs_spy_pct",
        "rs_13w_vs_spy_pct",
        "rs_26w_vs_spy_pct",
        "rs_20d_vs_spy_pct",
        "sector_rs_5w_vs_spy_pct",
        "stock_rs_5w_vs_sector_pct",
        "rs_line_near_52w_high",
        "hh_hl_structure",
        "weekly_close_gt_9ema_gt_20ema",
        "weekly_20sma_gt_50sma",
        "sma20_less_than_15pct_above_sma50",
        "near_52w_or_ath",
        "growth_star",
        "eps_growth_trend_up",
        "revenue_growth_trend_up",
        "eps_qoq_growth_trend",
        "revenue_qoq_growth_trend",
        "eps_quarter_values",
        "revenue_quarter_values",
        "eps_surprise_pct",
        "revenue_yoy_growth_pct",
        "revenue_qoq_growth_pct",
        "eps_yoy_growth_pct",
        "eps_qoq_growth_pct",
        "roe_pct",
        "operating_margin_improving",
        "free_cash_flow_positive_or_improving",
        "low_debt_strong_balance_sheet",
        "positive_guidance",
        "fundamental_source",
        "atr_pct",
        "spread_pct",
        "spread_small",
        "sector",
        "sector_etf",
        "latest_date",
    ]
    return out[[column for column in columns if column in out.columns]].rename(
        columns={
            "market_cap": "market cap",
            "total_points": "score / 100",
            "market_sector_score": "market + sector / 30",
            "technical_trend_score": "technical / 15",
            "relative_strength_score": "relative strength / 20",
            "fundamental_score": "fundamentals / 30",
            "liquidity_risk_score": "liquidity + risk / 5",
            "rsi_q": "RSI Q",
            "20d_rsi": "20D RSI",
            "weekly_rsi_14": "Weekly RSI",
            "spy_20d_rsi": "SPY 20D RSI",
            "sector_rsi": "sector ETF RSI",
            "sector_rsi_q": "RSI vs sector",
            "avg_volume_20d": "20D avg volume",
            "dollar_volume": "dollar volume",
            "rs_5w_vs_spy_pct": "Stock RS 5W vs SPY %",
            "rs_13w_vs_spy_pct": "Stock RS 13W vs SPY %",
            "rs_26w_vs_spy_pct": "Stock RS 26W vs SPY %",
            "rs_20d_vs_spy_pct": "Stock RS 20D vs SPY %",
            "sector_rs_5w_vs_spy_pct": "Sector RS 5W vs SPY %",
            "stock_rs_5w_vs_sector_pct": "Stock RS 5W vs sector %",
            "rs_line_near_52w_high": "RS line near high",
            "hh_hl_structure": "HH/HL structure",
            "weekly_close_gt_9ema_gt_20ema": "W close > 9E > 20E",
            "weekly_20sma_gt_50sma": "W 20SMA > 50SMA",
            "sma20_less_than_15pct_above_sma50": "W 20SMA not extended",
            "near_52w_or_ath": "near 52W/ATH",
            "growth_star": "growth star",
            "eps_growth_trend_up": "EPS trend up",
            "revenue_growth_trend_up": "revenue trend up",
            "eps_qoq_growth_trend": "EPS QoQ trend",
            "revenue_qoq_growth_trend": "revenue QoQ trend",
            "eps_quarter_values": "EPS last 4 qtrs",
            "revenue_quarter_values": "revenue last 4 qtrs",
            "eps_surprise_pct": "EPS surprise %",
            "revenue_yoy_growth_pct": "revenue YoY %",
            "revenue_qoq_growth_pct": "revenue QoQ %",
            "eps_yoy_growth_pct": "EPS YoY qtr growth %",
            "eps_qoq_growth_pct": "EPS QoQ %",
            "roe_pct": "ROE %",
            "operating_margin_improving": "margin/EBITDA improving",
            "free_cash_flow_positive_or_improving": "FCF positive/improving",
            "low_debt_strong_balance_sheet": "low debt",
            "positive_guidance": "positive guidance",
            "fundamental_source": "fundamental source",
            "atr_pct": "ATR %",
            "spread_pct": "spread %",
            "spread_small": "small spread",
            "sector_etf": "sector ETF",
            "latest_date": "latest candle",
        }
    )


def _load_current_top50_strength_state() -> bool:
    latest_state = load_latest_top50_strength_score_scan_state()
    if not latest_state:
        return False
    results, errors, stats = latest_state
    if stats.get("score_model_version") != "top50_100_v6_weekly_rsi_1b":
        st.session_state.pop("top50_strength_results", None)
        st.session_state.pop("top50_strength_errors", None)
        st.session_state.pop("top50_strength_stats", None)
        st.warning("The saved Top 50 result is from the old scoring model. Start a new run to generate the 100-point hard-filter result.")
        return False
    st.session_state["top50_strength_results"] = results
    st.session_state["top50_strength_errors"] = errors
    st.session_state["top50_strength_stats"] = stats
    return True


def _load_current_score_above60_setup_state() -> bool:
    latest_state = load_latest_score_above60_setup_scan_state()
    if not latest_state:
        return False
    results, errors, stats = latest_state
    st.session_state["score_above60_setup_results"] = results
    st.session_state["score_above60_setup_errors"] = errors
    st.session_state["score_above60_setup_stats"] = stats
    return True


def show_score_above60_setup_section(background_running: bool) -> None:
    st.subheader("60+ Setup Scanner")
    st.caption(
        "Scans only the saved Score Above 60 database for two technical setups: "
        "9 EMA pullback/reclaim and recent ATH/52-week high breakout."
    )
    control_cols = st.columns([1, 2.5])
    run_setup_scan = control_cols[0].button(
        "Run 60+ Setup Scan",
        type="primary",
        width="stretch",
        disabled=background_running,
    )
    control_cols[1].caption(
        "Uses the latest saved `Top 50 Score Above 60` watchlist as its universe, "
        "then runs the existing breakout and 9 EMA pullback rules."
    )
    if run_setup_scan:
        process_id = start_background_scan(["score_above60_setup"])
        st.session_state["score_above60_setup_start_message"] = f"Started 60+ setup background scan PID {process_id}."
        st.rerun()

    start_message = st.session_state.pop("score_above60_setup_start_message", "")
    if start_message:
        st.success(start_message)

    if "score_above60_setup_stats" not in st.session_state:
        _load_current_score_above60_setup_state()

    stats = st.session_state.get("score_above60_setup_stats", {})
    results = st.session_state.get("score_above60_setup_results", pd.DataFrame())
    if stats:
        metric_cols = st.columns(7)
        metric_cols[0].metric("60+ source date", str(stats.get("score_above60_scan_date", "-")))
        metric_cols[1].metric("60+ tickers scanned", f"{int(stats.get('score_above60_tickers', stats.get('stored_tickers', 0))):,}")
        metric_cols[2].metric("OHLCV loaded", f"{int(stats.get('ohlcv_loaded', 0)):,}")
        metric_cols[3].metric("Breakouts", f"{int(stats.get('breakout_candidates', 0)):,}")
        metric_cols[4].metric("9 EMA pullbacks", f"{int(stats.get('pullback_candidates', 0)):,}")
        metric_cols[5].metric("Total matches", f"{int(stats.get('matches', 0)):,}")
        metric_cols[6].metric("Candle date", str(stats.get("scan_date", "-")))

    if isinstance(results, pd.DataFrame) and not results.empty:
        setup_type = results.get("setup_type", pd.Series(dtype=str)).astype(str)
        breakout_rows = results[setup_type.str.contains("breakout", case=False, na=False)].copy()
        pullback_rows = results[setup_type.str.contains("pullback", case=False, na=False)].copy()
        breakout_tab, pullback_tab, all_tab = st.tabs(["ATH / 52W Breakout", "9 EMA Pullback", "All 60+ Setups"])
        with breakout_tab:
            if breakout_rows.empty:
                st.write("No ATH / 52W breakout matches found.")
            else:
                st.dataframe(format_technical_strength_results(breakout_rows), width="stretch", hide_index=True)
                st.download_button(
                    "Export 60+ Breakouts CSV",
                    data=breakout_rows.to_csv(index=False).encode("utf-8"),
                    file_name="score_above60_breakouts.csv",
                    mime="text/csv",
                )
        with pullback_tab:
            if pullback_rows.empty:
                st.write("No 9 EMA pullback matches found.")
            else:
                st.dataframe(format_technical_strength_results(pullback_rows), width="stretch", hide_index=True)
                st.download_button(
                    "Export 60+ Pullbacks CSV",
                    data=pullback_rows.to_csv(index=False).encode("utf-8"),
                    file_name="score_above60_9ema_pullbacks.csv",
                    mime="text/csv",
                )
        with all_tab:
            st.dataframe(format_technical_strength_results(results), width="stretch", hide_index=True)
            st.download_button(
                "Export All 60+ Setups CSV",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="score_above60_setups.csv",
                mime="text/csv",
            )
    elif stats:
        st.write("No 60+ setup matches found in the latest run.")
    else:
        st.write("No 60+ setup scan has run yet.")

    errors = st.session_state.get("score_above60_setup_errors", [])
    if errors:
        with st.expander(f"60+ setup data issues ({len(errors):,})"):
            st.text("\n".join(errors[:500]))
            if len(errors) > 500:
                st.caption(f"Showing first 500 of {len(errors):,} issues.")


def top50_strength_score_page() -> None:
    st.title("Top 50 Strength Score")
    st.caption(
        "Scores the saved universe with a $1B+ Top 50 hard filter, then ranks stocks on a 100-point "
        "market, technical, relative-strength, fundamental, and liquidity scorecard."
    )
    st.info(
        "Hard filters: close > 50/200 SMA, weekly RSI > 50, price > $10, avg volume > 1M, dollar volume > $20M, "
        "market cap > $1B, common-stock symbol, and stock RSI > SPY RSI. "
        "Revenue growth, EPS growth, and EPS surprise use FMP first when available. "
        "Final score = Market + Sector 30, Technical 15, Relative Strength 20, Fundamentals 30, Liquidity + Risk 5."
    )

    if "top50_strength_stats" not in st.session_state:
        _load_current_top50_strength_state()

    background_running = show_background_scan_status({"top50_strength_score"})
    current_status = load_manual_scan_status()
    top50_status_active = "top50_strength_score" in {
        str(rule_key) for rule_key in current_status.get("rules", []) if str(rule_key)
    }
    if background_running and top50_status_active:
        reset_cols = st.columns([1, 3])
        if reset_cols[0].button("Clear Top 50 Status", width="stretch"):
            MANUAL_SCAN_STATUS_PATH.write_text(
                json.dumps(
                    {
                        "status": "cleared",
                        "rules": [],
                        "rule_labels": [],
                        "message": "Top 50 running status cleared by user.",
                        "summaries": [],
                        "updated_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            st.rerun()
        reset_cols[1].caption("Use this only if the previous Top 50 background status is stale and you want to start a fresh filtered run.")
    control_cols = st.columns([1, 2.5])
    run_scan = control_cols[0].button(
        "Run Top 50 Score",
        type="primary",
        width="stretch",
        disabled=background_running,
    )
    control_cols[1].caption(
        "Runs in the background and checks fundamentals for all valid $1B+ tickers after the technical hard filters. "
        "You can keep using the app while it runs; refresh this page to see status/results."
    )

    if run_scan:
        process_id = start_background_scan(["top50_strength_score"])
        st.session_state["top50_strength_start_message"] = f"Started Top 50 background scan PID {process_id}."
        st.rerun()

    start_message = st.session_state.pop("top50_strength_start_message", "")
    if start_message:
        st.success(start_message)

    if "top50_strength_stats" not in st.session_state:
        _load_current_top50_strength_state()

    stats = st.session_state.get("top50_strength_stats", {})
    if stats and stats.get("score_model_version") != "top50_100_v6_weekly_rsi_1b":
        for key in ("top50_strength_results", "top50_strength_errors", "top50_strength_stats"):
            st.session_state.pop(key, None)
        stats = {}
        st.warning("Cleared old Top 50 result from memory. Start a new run to generate the current 100-point scorecard.")
    if stats:
        metric_cols = st.columns(9)
        metric_cols[0].metric("Stored tickers", f"{int(stats.get('stored_tickers', 0)):,}")
        metric_cols[1].metric("OHLCV loaded", f"{int(stats.get('ohlcv_loaded', 0)):,}")
        metric_cols[2].metric("Common filtered", f"{int(stats.get('common_stock_filtered', 0)):,}")
        metric_cols[3].metric("Hard filtered", f"{int(stats.get('hard_filter_removed', 0)):,}")
        metric_cols[4].metric("RSI < SPY removed", f"{int(stats.get('rsi_filtered', 0)):,}")
        metric_cols[5].metric("Fundamentals checked", f"{int(stats.get('earnings_checked', 0)):,}")
        metric_cols[6].metric("Top rows", f"{int(stats.get('matches', 0)):,}")
        metric_cols[7].metric("Score >= 60", f"{int(stats.get('above60_count', 0)):,}")
        metric_cols[8].metric("Candle date", str(stats.get("scan_date", "-")))
        st.caption(
            f"FMP used only for EPS/revenue fields. FMP-enriched tickers: "
            f"{int(stats.get('fmp_enriched', 0)):,}; all configured FMP keys are rotated if one is rate-limited; "
            "same-day FMP results are cached."
        )

    results = st.session_state.get("top50_strength_results", pd.DataFrame())
    if isinstance(results, pd.DataFrame) and not results.empty:
        st.subheader("Top 50 Ranked Stocks")
        st.dataframe(_format_top50_score_results(results), width="stretch", hide_index=True)
        st.download_button(
            "Export Top 50 Score CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="top50_strength_score.csv",
            mime="text/csv",
        )
        above_60_rows = stats.get("above60_rows", []) if isinstance(stats, dict) else []
        above_60 = pd.DataFrame(above_60_rows) if isinstance(above_60_rows, list) and above_60_rows else pd.DataFrame()
        st.subheader("Score Above 60")
        if not above_60.empty:
            st.caption(f"{len(above_60):,} ticker(s) from the full qualified universe scored 60 or higher.")
            st.dataframe(_format_top50_score_results(above_60), width="stretch", hide_index=True)
            tickers_text = "\n".join(above_60["ticker"].astype(str).tolist()) if "ticker" in above_60.columns else ""
            st.text_area("Tickers score >= 60", value=tickers_text, height=120, key="top50_score_above_60_tickers")
            st.download_button(
                "Export Score Above 60 CSV",
                data=above_60.to_csv(index=False).encode("utf-8"),
                file_name="top50_score_above_60.csv",
                mime="text/csv",
            )
        else:
            st.write("No Top 50 tickers scored 60 or higher in the latest result.")
    elif background_running:
        st.warning("Top 50 score is still running. The table will appear here after the background job writes the finished result.")
    else:
        st.write("No Top 50 score results yet. Click Run Top 50 Score.")

    st.divider()
    show_score_above60_setup_section(background_running)

    errors = st.session_state.get("top50_strength_errors", [])
    if errors:
        with st.expander(f"Data issues ({len(errors):,})"):
            st.text("\n".join(errors[:500]))
            if len(errors) > 500:
                st.caption(f"Showing first 500 of {len(errors):,} issues.")


def _fmt_trade_pct(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "-"
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "-"


def _fmt_trade_number(value: object, decimals: int = 2) -> str:
    try:
        if value is None or pd.isna(value):
            return "-"
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "-"


def _format_trade_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    columns = [
        "id",
        "ticker",
        "direction",
        "trade_type",
        "setup",
        "status",
        "entry_date",
        "entry_price",
        "stop_price",
        "target_price",
        "quantity",
        "planned_risk_amount",
        "planned_rr",
        "exit_date",
        "exit_price",
        "realized_pl",
        "realized_r",
        "outcome",
        "tags",
        "notes",
    ]
    return frame[[column for column in columns if column in frame.columns]].rename(
        columns={
            "id": "ID",
            "ticker": "Ticker",
            "direction": "Side",
            "trade_type": "Trade type",
            "setup": "Setup",
            "status": "Status",
            "entry_date": "Entry date",
            "entry_price": "Entry",
            "stop_price": "Stop",
            "target_price": "Target",
            "quantity": "Shares",
            "planned_risk_amount": "Planned risk $",
            "planned_rr": "Planned R:R",
            "exit_date": "Exit date",
            "exit_price": "Exit",
            "realized_pl": "Realized P/L",
            "realized_r": "Realized R",
            "outcome": "Outcome",
            "tags": "Label",
            "notes": "Notes",
        }
    )


TRADE_SELECTED_ID_KEY = "trade_selected_for_update_id"


def _sync_trade_checkbox_selection(editor_key: str, trade_ids: list[int]) -> None:
    editor_state = st.session_state.get(editor_key, {})
    edited_rows = editor_state.get("edited_rows", {}) if isinstance(editor_state, dict) else {}
    current_id = st.session_state.get(TRADE_SELECTED_ID_KEY)
    selected_id = current_id
    for row_index, changes in edited_rows.items():
        if not isinstance(changes, dict) or "Select" not in changes:
            continue
        try:
            trade_id = trade_ids[int(row_index)]
        except (IndexError, TypeError, ValueError):
            continue
        if changes["Select"]:
            selected_id = trade_id
        elif current_id == trade_id:
            selected_id = None
    if selected_id is None:
        st.session_state.pop(TRADE_SELECTED_ID_KEY, None)
    else:
        st.session_state[TRADE_SELECTED_ID_KEY] = int(selected_id)


def _trade_checkbox_table(frame: pd.DataFrame, key: str) -> list[int]:
    display = _format_trade_table(frame)
    if display.empty:
        return []
    display = display.copy()
    selected_id = st.session_state.get(TRADE_SELECTED_ID_KEY)
    display.insert(0, "Select", display["ID"].astype(int).eq(int(selected_id)) if selected_id else False)
    trade_ids = [int(value) for value in display["ID"].tolist()]
    editor_key = f"{key}_{selected_id or 'none'}"
    st.data_editor(
        display,
        width="stretch",
        hide_index=True,
        key=editor_key,
        on_change=_sync_trade_checkbox_selection,
        args=(editor_key, trade_ids),
        disabled=[column for column in display.columns if column != "Select"],
        column_config={
            "Select": st.column_config.CheckboxColumn("Select", help="Check one trade to update, close, or delete it.", default=False)
        },
    )
    selected_id = st.session_state.get(TRADE_SELECTED_ID_KEY)
    return [int(selected_id)] if selected_id in trade_ids else []


def _trade_type_analytics_frame(trades: pd.DataFrame) -> pd.DataFrame:
    module_func = getattr(trade_journal, "trade_type_analytics", None)
    if callable(module_func):
        return module_func(trades)
    if trades.empty or "trade_type" not in trades.columns:
        return pd.DataFrame()
    closed = trades[trades["status"].astype(str).str.lower().eq("closed")].copy()
    if closed.empty:
        return pd.DataFrame()
    closed["realized_r"] = pd.to_numeric(closed["realized_r"], errors="coerce")
    closed["realized_pl"] = pd.to_numeric(closed["realized_pl"], errors="coerce")
    rows = []
    for trade_type, group in closed.groupby("trade_type", dropna=False):
        wins = int(group["outcome"].astype(str).eq("Win").sum())
        losses = int(group["outcome"].astype(str).eq("Loss").sum())
        total = len(group)
        rows.append(
            {
                "trade type": trade_type or "Swing Trade",
                "closed trades": total,
                "wins": wins,
                "losses": losses,
                "win rate %": round(wins / total * 100, 1) if total else None,
                "avg R": round(float(group["realized_r"].mean()), 2),
                "total R": round(float(group["realized_r"].sum()), 2),
                "total P/L": round(float(group["realized_pl"].sum()), 2),
            }
        )
    return pd.DataFrame(rows).sort_values(["total R", "win rate %"], ascending=[False, False])


def _build_trade_payload_compatible(form_values: dict[str, object]) -> dict[str, object]:
    try:
        return trade_journal.build_trade_payload(**form_values)
    except TypeError as exc:
        if "trade_type" not in str(exc):
            raise
        legacy_values = dict(form_values)
        trade_type = str(legacy_values.pop("trade_type", "") or "Swing Trade")
        payload = trade_journal.build_trade_payload(**legacy_values)
        payload["trade_type"] = trade_type
        return payload


def _trade_select_label(row: pd.Series) -> str:
    trade_type = row.get("trade_type") or "Swing Trade"
    return (
        f"#{int(row['id'])} {row['ticker']} {row['direction']} | "
        f"{trade_type} | {row['setup'] or 'No setup'} | {row['status']} | entry {row['entry_price']}"
    )


def _trade_result_calendar_html(closed_trades: pd.DataFrame, month_key: str) -> str:
    year, month = [int(part) for part in month_key.split("-")]
    frame = closed_trades.copy()
    if not frame.empty and "exit_date" in frame.columns:
        frame["exit_day"] = pd.to_datetime(frame["exit_date"], errors="coerce").dt.date
        frame = frame.dropna(subset=["exit_day"])
        frame["realized_r"] = pd.to_numeric(frame["realized_r"], errors="coerce").fillna(0)
        frame["realized_pl"] = pd.to_numeric(frame["realized_pl"], errors="coerce").fillna(0)
        frame = frame[
            (pd.to_datetime(frame["exit_day"]).dt.year == year)
            & (pd.to_datetime(frame["exit_day"]).dt.month == month)
        ]
    daily: dict[date, dict[str, object]] = {}
    if "exit_day" in frame.columns:
        for exit_day, group in frame.groupby("exit_day"):
            wins = int(group["outcome"].astype(str).eq("Win").sum())
            losses = int(group["outcome"].astype(str).eq("Loss").sum())
            breakeven = int(group["outcome"].astype(str).eq("Breakeven").sum())
            total_r = float(group["realized_r"].sum())
            total_pl = float(group["realized_pl"].sum())
            tickers = ", ".join(group["ticker"].astype(str).head(4).tolist())
            if total_pl > 0.005:
                mood = "win"
                label = "Winning day"
            elif total_pl < -0.005:
                mood = "loss"
                label = "Losing day"
            else:
                mood = "flat"
                label = "Breakeven day"
            daily[exit_day] = {
                "mood": mood,
                "label": label,
                "wins": wins,
                "losses": losses,
                "breakeven": breakeven,
                "count": len(group),
                "total_r": total_r,
                "total_pl": total_pl,
                "tickers": tickers,
            }

    weeks = calendar.Calendar(firstweekday=0).monthdatescalendar(year, month)
    day_names = "".join(f'<div class="tr-cal-dow">{name}</div>' for name in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    cells: list[str] = []
    for week in weeks:
        for day in week:
            is_current_month = day.month == month
            info = daily.get(day)
            muted = " tr-cal-muted" if not is_current_month else ""
            if not info:
                cells.append(
                    f'<div class="tr-cal-cell{muted}"><div class="tr-cal-day">{day.day}</div></div>'
                )
                continue
            mood = str(info["mood"])
            total_r = float(info["total_r"])
            total_pl = float(info["total_pl"])
            cells.append(
                f'<div class="tr-cal-cell tr-cal-{mood}{muted}">'
                f'<div class="tr-cal-day">{day.day}</div>'
                f'<div class="tr-cal-label">{escape(str(info["label"]))}</div>'
                f'<div class="tr-cal-main">{int(info["count"])} trade(s) | {total_r:+.2f}R</div>'
                f'<div class="tr-cal-sub">${total_pl:+,.0f}</div>'
                f'<div class="tr-cal-sub">W {int(info["wins"])} / L {int(info["losses"])} / B {int(info["breakeven"])}</div>'
                f'<div class="tr-cal-tickers">{escape(str(info["tickers"]))}</div>'
                "</div>"
            )
    return (
        """
        <style>
            .tr-cal-wrap {border:1px solid #e5e7eb; border-radius:8px; background:#fff; padding:14px; margin-top:8px;}
            .tr-cal-head {display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:12px;}
            .tr-cal-title {font-weight:850; font-size:16px; color:#111827;}
            .tr-cal-legend {display:flex; gap:8px; flex-wrap:wrap; font-size:12px; color:#6b7280;}
            .tr-cal-dot {width:9px; height:9px; border-radius:50%; display:inline-block; margin-right:4px;}
            .tr-cal-grid {display:grid; grid-template-columns:repeat(7, minmax(0, 1fr)); gap:6px;}
            .tr-cal-dow {font-size:12px; color:#6b7280; font-weight:800; text-align:center; padding:4px 0;}
            .tr-cal-cell {min-height:100px; border:1px solid #eef0f3; border-radius:8px; padding:8px; background:#fafafa;}
            .tr-cal-muted {opacity:.38;}
            .tr-cal-day {font-size:12px; font-weight:850; color:#111827;}
            .tr-cal-label {font-size:12px; font-weight:850; margin-top:6px;}
            .tr-cal-main {font-size:12px; font-weight:800; margin-top:4px;}
            .tr-cal-sub {font-size:11px; color:#374151; margin-top:2px;}
            .tr-cal-tickers {font-size:11px; color:#6b7280; margin-top:4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
            .tr-cal-win {background:#ecfdf5; border-color:#86efac;}
            .tr-cal-win .tr-cal-label, .tr-cal-win .tr-cal-main {color:#166534;}
            .tr-cal-loss {background:#fef2f2; border-color:#fecaca;}
            .tr-cal-loss .tr-cal-label, .tr-cal-loss .tr-cal-main {color:#991b1b;}
            .tr-cal-flat {background:#f9fafb; border-color:#d1d5db;}
            .tr-cal-flat .tr-cal-label, .tr-cal-flat .tr-cal-main {color:#374151;}
            @media (max-width: 900px) {.tr-cal-cell {min-height:82px; padding:6px;} .tr-cal-sub, .tr-cal-tickers {display:none;}}
        </style>
        """
        f'<div class="tr-cal-wrap">'
        f'<div class="tr-cal-head"><div class="tr-cal-title">{calendar.month_name[month]} {year}</div>'
        '<div class="tr-cal-legend">'
        '<span><i class="tr-cal-dot" style="background:#22c55e;"></i>Win</span>'
        '<span><i class="tr-cal-dot" style="background:#ef4444;"></i>Loss</span>'
        '<span><i class="tr-cal-dot" style="background:#9ca3af;"></i>Breakeven</span>'
        "</div></div>"
        f'<div class="tr-cal-grid">{day_names}{"".join(cells)}</div>'
        "</div>"
    )


def _trade_calendar_month_summary(closed_trades: pd.DataFrame, month_key: str) -> dict[str, object]:
    year, month = [int(part) for part in month_key.split("-")]
    frame = closed_trades.copy()
    if frame.empty or "exit_date" not in frame.columns:
        frame = pd.DataFrame()
    else:
        exit_dates = pd.to_datetime(frame["exit_date"], errors="coerce")
        frame = frame[exit_dates.dt.strftime("%Y-%m").eq(month_key)].copy()
    if frame.empty:
        return {
            "month_key": month_key,
            "label": f"{calendar.month_name[month]} {year}",
            "trade_count": 0,
            "wins": 0,
            "losses": 0,
            "breakeven": 0,
            "total_pl": 0.0,
            "total_r": 0.0,
        }

    outcome = frame["outcome"].astype(str)
    realized_pl = pd.to_numeric(frame["realized_pl"], errors="coerce").fillna(0)
    realized_r = pd.to_numeric(frame["realized_r"], errors="coerce").fillna(0)
    return {
        "month_key": month_key,
        "label": f"{calendar.month_name[month]} {year}",
        "trade_count": int(len(frame)),
        "wins": int(outcome.eq("Win").sum()),
        "losses": int(outcome.eq("Loss").sum()),
        "breakeven": int(outcome.eq("Breakeven").sum()),
        "total_pl": round(float(realized_pl.sum()), 2),
        "total_r": round(float(realized_r.sum()), 2),
    }


TRADE_SETUP_OPTIONS = [
    "9 EMA Pullback",
    "ATH / 52W Breakout",
    "Weekly Breakout",
    "Top 50 Score",
    "Morning Star",
    "Green Marubozu",
    "Manual",
]

TRADE_TYPE_OPTIONS = ["Day Trade", "Swing Trade", "Positional Trade"]
TRADE_STATUS_OPTIONS = ["Initiated", "Open", "Closed"]


def _trade_payload_from_form(prefix: str, existing: pd.Series | None = None) -> dict[str, object]:
    existing = existing if existing is not None else pd.Series(dtype=object)
    default_status = str(existing.get("status") or "Initiated")
    if default_status not in TRADE_STATUS_OPTIONS:
        default_status = "Open" if default_status == "Open" else "Initiated"
    default_direction = str(existing.get("direction") or "Long")
    default_trade_type = str(existing.get("trade_type") or "Swing Trade")
    if default_trade_type not in TRADE_TYPE_OPTIONS:
        default_trade_type = "Swing Trade"
    existing_setup = str(existing.get("setup") or "")
    default_setup = existing_setup if existing_setup in TRADE_SETUP_OPTIONS else "Manual"
    default_entry_date = pd.to_datetime(existing.get("entry_date"), errors="coerce")
    default_exit_date = pd.to_datetime(existing.get("exit_date"), errors="coerce")

    top_cols = st.columns([1, 0.75, 1.05, 1.25, 1.2, 0.85])
    ticker = top_cols[0].text_input("Ticker", value=str(existing.get("ticker") or "").upper(), key=f"{prefix}_ticker").upper()
    direction = top_cols[1].selectbox(
        "Side",
        ["Long", "Short"],
        index=0 if default_direction != "Short" else 1,
        key=f"{prefix}_direction",
    )
    trade_type = top_cols[2].selectbox(
        "Trade type",
        TRADE_TYPE_OPTIONS,
        index=TRADE_TYPE_OPTIONS.index(default_trade_type),
        key=f"{prefix}_trade_type",
    )
    setup = top_cols[3].selectbox(
        "Setup",
        TRADE_SETUP_OPTIONS,
        index=TRADE_SETUP_OPTIONS.index(default_setup),
        key=f"{prefix}_setup",
    )
    custom_setup = top_cols[4].text_input(
        "Custom setup",
        value=existing_setup if existing_setup and existing_setup not in TRADE_SETUP_OPTIONS else "",
        key=f"{prefix}_custom_setup",
    )
    setup_value = custom_setup.strip() or setup
    status = top_cols[5].selectbox(
        "Status",
        TRADE_STATUS_OPTIONS,
        index=TRADE_STATUS_OPTIONS.index(default_status),
        key=f"{prefix}_status",
    )

    trade_cols = st.columns([1, 0.9, 0.9, 0.9, 0.75, 0.9, 0.9])
    entry_date = trade_cols[0].date_input(
        "Entry date",
        value=default_entry_date.date() if pd.notna(default_entry_date) else date.today(),
        key=f"{prefix}_entry_date",
    )
    entry_price = trade_cols[1].number_input("Entry", min_value=0.0, value=float(existing.get("entry_price") or 0.0), step=0.01, key=f"{prefix}_entry")
    stop_price = trade_cols[2].number_input("Stop", min_value=0.0, value=float(existing.get("stop_price") or 0.0), step=0.01, key=f"{prefix}_stop")
    target_price = trade_cols[3].number_input("Target", min_value=0.0, value=float(existing.get("target_price") or 0.0), step=0.01, key=f"{prefix}_target")
    quantity = trade_cols[4].number_input("Shares", min_value=1, value=int(existing.get("quantity") or 1), step=1, key=f"{prefix}_quantity")
    exit_date = trade_cols[5].date_input(
        "Exit date",
        value=default_exit_date.date() if pd.notna(default_exit_date) else date.today(),
        key=f"{prefix}_exit_date",
    )
    exit_price = trade_cols[6].number_input(
        "Exit",
        min_value=0.0,
        value=float(existing.get("exit_price") or 0.0),
        step=0.01,
        key=f"{prefix}_exit_price",
    )

    text_cols = st.columns([1, 2])
    tags = text_cols[0].text_input("Label (optional)", value=str(existing.get("tags") or ""), key=f"{prefix}_tags")
    notes = text_cols[1].text_input("Notes", value=str(existing.get("notes") or ""), key=f"{prefix}_notes")
    return {
        "ticker": ticker,
        "direction": direction,
        "trade_type": trade_type,
        "setup": setup_value,
        "status": status,
        "entry_date": entry_date,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price if target_price > 0 else None,
        "quantity": int(quantity),
        "exit_date": exit_date if status == "Closed" or exit_price > 0 else None,
        "exit_price": exit_price if exit_price > 0 else None,
        "fees": 0.0,
        "notes": notes,
        "tags": tags,
    }


def trades_page() -> None:
    st.title("Trade Journal")
    st.caption(f"Storage: {cloud_sqlite_status()}")
    st.caption(
        "Register planned trades, update exits, and track win rate, losses, risk/reward, realized R, open risk, and setup performance."
    )
    st.info(
        "Suggested trade parameters: ticker, side, trade type, setup, entry, stop, target, shares, planned risk, planned R:R, "
        "exit price/date, realized P/L, realized R, outcome, notes, and optional label."
        " Status flow: Initiated = planned, Open = trade taken, Closed = trade done."
    )

    trades = trade_journal.list_trades(DB_PATH)
    analytics = trade_journal.trade_analytics(trades)

    metric_cols = st.columns(9)
    metric_cols[0].metric("Total trades", f"{analytics['total']:,}")
    metric_cols[1].metric("Initiated", f"{analytics.get('initiated', 0):,}")
    metric_cols[2].metric("Open", f"{analytics['open']:,}")
    metric_cols[3].metric("Closed", f"{analytics['closed']:,}")
    metric_cols[4].metric("Win rate", _fmt_trade_pct(analytics["win_rate"]))
    metric_cols[5].metric("Total P/L", f"${_fmt_trade_number(analytics['total_pl'])}")
    metric_cols[6].metric("Avg R", _fmt_trade_number(analytics["avg_r"]))
    metric_cols[7].metric("Profit factor", _fmt_trade_number(analytics["profit_factor"]))
    metric_cols[8].metric("Open risk", f"${_fmt_trade_number(analytics['open_risk'])}")

    initiated_trades = trades[trades["status"].astype(str).str.lower().eq("initiated")].copy() if not trades.empty else pd.DataFrame()
    open_trades = trades[trades["status"].astype(str).str.lower().eq("open")].copy() if not trades.empty else pd.DataFrame()
    closed_trades = trades[trades["status"].astype(str).str.lower().eq("closed")].copy() if not trades.empty else pd.DataFrame()

    trade_sections = ["Open Trades", "Add Trade", "Closed Trades", "Calendar", "Analytics"]
    selected_trade_section = st.radio(
        "Trade Journal Section",
        trade_sections,
        index=0,
        horizontal=True,
        key="trade_journal_section",
    )

    if selected_trade_section == "Add Trade":
        st.subheader("Register Trade")
        st.caption("Fill the trade ticket from left to right. Planned risk and R:R are calculated after Save.")
        with st.form("add_trade_form"):
            form_values = _trade_payload_from_form("add_trade")
            submitted = st.form_submit_button("Save Trade", type="primary", width="stretch")
        if submitted:
            if not form_values["ticker"]:
                st.error("Ticker is required.")
            else:
                try:
                    payload = _build_trade_payload_compatible(form_values)
                    trade_id = trade_journal.add_trade(payload, DB_PATH)
                    st.success(f"Saved trade #{trade_id}.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    elif selected_trade_section == "Open Trades":
        selected_trade_ids: list[int] = []
        st.subheader("Open Trades")
        if open_trades.empty:
            st.write("No open trades yet.")
        else:
            selected_trade_ids.extend(_trade_checkbox_table(open_trades, "open_trade_checkbox_table"))
        st.subheader("Initiated Trades")
        if initiated_trades.empty:
            st.write("No initiated trades yet.")
        else:
            selected_trade_ids.extend(_trade_checkbox_table(initiated_trades, "initiated_trade_checkbox_table"))

        if not open_trades.empty or not initiated_trades.empty:
            st.markdown("**Update / Close / Delete Trade**")
            if not selected_trade_ids:
                st.info("Check one Open or Initiated trade above to update, close, or delete it.")
            else:
                selected_id = selected_trade_ids[0]
                selected = trades.loc[trades["id"].eq(selected_id)].iloc[0]
                st.caption(f"Editing #{selected_id} {selected['ticker']}")
                update_prefix = f"update_trade_{int(selected_id)}"
                with st.form(f"update_trade_form_{int(selected_id)}"):
                    form_values = _trade_payload_from_form(update_prefix, selected)
                    update_cols = st.columns(2)
                    update_submitted = update_cols[0].form_submit_button("Update Trade", type="primary", width="stretch")
                    delete_submitted = update_cols[1].form_submit_button("Delete Trade", width="stretch")
                if update_submitted:
                    try:
                        payload = _build_trade_payload_compatible(form_values)
                        trade_journal.update_trade(int(selected_id), payload, DB_PATH)
                        st.success(f"Updated trade #{selected_id}.")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
                if delete_submitted:
                    trade_journal.delete_trade(int(selected_id), DB_PATH)
                    st.success(f"Deleted trade #{selected_id}.")
                    st.rerun()

    elif selected_trade_section == "Closed Trades":
        st.subheader("Closed Trades")
        if closed_trades.empty:
            st.write("No closed trades yet.")
        else:
            st.dataframe(_format_trade_table(closed_trades), width="stretch", hide_index=True)
            st.markdown("**Update Closed Trade**")
            closed_trade_ids = [int(value) for value in closed_trades["id"].tolist()]
            selected_closed_id = st.selectbox(
                "Select closed trade",
                closed_trade_ids,
                format_func=lambda trade_id: f"#{trade_id} {closed_trades.loc[closed_trades['id'].eq(trade_id), 'ticker'].iloc[0]}",
                key="closed_trade_selector",
            )
            selected_closed = trades.loc[trades["id"].eq(int(selected_closed_id))].iloc[0]
            st.caption(f"Editing #{int(selected_closed_id)} {selected_closed['ticker']}")
            closed_update_prefix = f"closed_update_trade_{int(selected_closed_id)}"
            with st.form(f"closed_update_trade_form_{int(selected_closed_id)}"):
                form_values = _trade_payload_from_form(closed_update_prefix, selected_closed)
                update_submitted = st.form_submit_button("Update Closed Trade", type="primary", width="stretch")
            if update_submitted:
                try:
                    payload = _build_trade_payload_compatible(form_values)
                    trade_journal.update_trade(int(selected_closed_id), payload, DB_PATH)
                    st.success(f"Updated closed trade #{int(selected_closed_id)}.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
            st.download_button(
                "Export Closed Trades CSV",
                data=closed_trades.to_csv(index=False).encode("utf-8"),
                file_name="closed_trades.csv",
                mime="text/csv",
            )

    elif selected_trade_section == "Calendar":
        st.subheader("Win / Loss Calendar")
        st.caption("Calendar is based on exit date and only includes closed Day Trade results.")
        day_trade_closed = (
            closed_trades[closed_trades["trade_type"].astype(str).str.lower().eq("day trade")].copy()
            if not closed_trades.empty and "trade_type" in closed_trades.columns
            else pd.DataFrame()
        )
        calendar_dates = (
            pd.to_datetime(day_trade_closed["exit_date"], errors="coerce").dropna()
            if not day_trade_closed.empty and "exit_date" in day_trade_closed.columns
            else pd.Series(dtype="datetime64[ns]")
        )
        if calendar_dates.empty:
            month_options = [pd.Timestamp.today().strftime("%Y-%m")]
            st.info("No closed Day Trade results with exit dates yet. The calendar will color days after day trades are closed.")
        else:
            month_options = sorted(calendar_dates.dt.strftime("%Y-%m").unique(), reverse=True)
        selected_month = st.selectbox("Calendar month", month_options, key="trade_calendar_month")
        month_summary = _trade_calendar_month_summary(day_trade_closed, selected_month)
        trade_journal.save_trade_calendar_month(month_summary, DB_PATH)
        summary_cols = st.columns(6)
        summary_cols[0].metric("Month P/L", f"${_fmt_trade_number(month_summary['total_pl'])}")
        summary_cols[1].metric("Month R", _fmt_trade_number(month_summary["total_r"]))
        summary_cols[2].metric("Trades", f"{int(month_summary['trade_count']):,}")
        summary_cols[3].metric("Wins", f"{int(month_summary['wins']):,}")
        summary_cols[4].metric("Losses", f"{int(month_summary['losses']):,}")
        summary_cols[5].metric("Breakeven", f"{int(month_summary['breakeven']):,}")
        st.markdown(_trade_result_calendar_html(day_trade_closed, selected_month), unsafe_allow_html=True)
        saved_months = trade_journal.list_trade_calendar_months(DB_PATH)
        if not saved_months.empty:
            history = saved_months.rename(
                columns={
                    "month_key": "Month",
                    "trade_count": "Trades",
                    "wins": "Wins",
                    "losses": "Losses",
                    "breakeven": "Breakeven",
                    "total_pl": "P/L",
                    "total_r": "R",
                    "updated_at": "Saved",
                }
            )
            history["P/L"] = pd.to_numeric(history["P/L"], errors="coerce").map(lambda value: f"${value:,.2f}")
            history["R"] = pd.to_numeric(history["R"], errors="coerce").map(lambda value: f"{value:+.2f}R")
            st.markdown("**Saved Calendar Months**")
            st.dataframe(history, width="stretch", hide_index=True)

    else:
        st.subheader("Performance Analytics")
        detail_cols = st.columns(5)
        detail_cols[0].metric("Wins", f"{analytics['wins']:,}")
        detail_cols[1].metric("Losses", f"{analytics['losses']:,}")
        detail_cols[2].metric("Breakeven", f"{analytics['breakeven']:,}")
        detail_cols[3].metric("Avg win R", _fmt_trade_number(analytics["avg_win_r"]))
        detail_cols[4].metric("Avg loss R", _fmt_trade_number(analytics["avg_loss_r"]))
        type_frame = _trade_type_analytics_frame(trades)
        st.markdown("**Trade Type Performance**")
        if type_frame.empty:
            st.write("Close a few trades to see day trade, swing trade, and positional trade performance.")
        else:
            st.dataframe(type_frame, width="stretch", hide_index=True)
        setup_frame = trade_journal.setup_analytics(trades)
        st.markdown("**Setup Performance**")
        if setup_frame.empty:
            st.write("Close a few trades to see setup performance.")
        else:
            st.dataframe(setup_frame, width="stretch", hide_index=True)
        if not trades.empty:
            st.download_button(
                "Export All Trades CSV",
                data=trades.to_csv(index=False).encode("utf-8"),
                file_name="trade_journal.csv",
                mime="text/csv",
            )


def _growth_table(rows: list[dict[str, object]], label: str) -> None:
    if not rows:
        st.write(f"No {label.lower()} rows stored.")
        return
    frame = pd.DataFrame(rows)
    st.dataframe(frame, width="stretch", hide_index=True)


def _clean_json_value(value: object) -> object:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, dict):
        return {str(key): _clean_json_value(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_clean_json_value(inner) for inner in value]
    if isinstance(value, tuple):
        return [_clean_json_value(inner) for inner in value]
    return value


def _json_script_payload(payload: object) -> str:
    return json.dumps(_clean_json_value(payload), ensure_ascii=False).replace("</", "<\\/")


def _stock_tracker_html(stocks: pd.DataFrame) -> str:
    records: list[dict[str, object]] = []
    for row in stocks.itertuples(index=False):
        details = get_stock_details(row.ticker) or {}
        records.append(
            {
                "ticker": row.ticker,
                "company_name": row.company_name,
                "market_cap": format_market_cap(row.market_cap),
                "description": row.description,
                "fifty_two_week_high": row.fifty_two_week_high,
                "beta": row.beta,
                "summary": row.summary,
                "news": details.get("news", []),
                "earnings_growth": details.get("earnings_growth", []),
                "revenue_growth": details.get("revenue_growth", []),
                "quarterly_summary": details.get("quarterly_summary", []),
            }
        )
    payload = _json_script_payload(records)
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<style>
  body {{
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
    background: #fff5f5;
    color: #3f0d12;
  }}
  .container {{
    max-width: 1100px;
    margin: 0 auto;
    background: #ffffff;
    border: 1px solid #fecaca;
    border-radius: 12px;
    box-shadow: 0 14px 28px rgba(127, 29, 29, 0.12);
    padding: 24px;
  }}
  h1 {{
    margin: 0;
    font-size: 2rem;
    color: #991b1b;
  }}
  .tracker-header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 16px;
  }}
  .subtitle {{
    margin: 8px 0 18px;
    color: #7f1d1d;
  }}
  .delete-button {{
    border: 1px solid #dc2626;
    border-radius: 8px;
    background: #dc2626;
    color: #ffffff;
    cursor: pointer;
    font-weight: 700;
    padding: 10px 14px;
    white-space: nowrap;
  }}
  .delete-button:disabled {{
    background: #fecaca;
    border-color: #fecaca;
    color: #7f1d1d;
    cursor: default;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 16px;
  }}
  th, td {{
    padding: 14px 12px;
    text-align: left;
    border-bottom: 1px solid #e5e7eb;
    vertical-align: top;
  }}
  th {{
    background: #991b1b;
    color: #ffffff;
    font-weight: 600;
  }}
  tr:hover {{
    background: #fff1f2;
  }}
  .ticker-button {{
    background: none;
    border: none;
    color: #b91c1c;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    padding: 0;
  }}
  .ticker-button:hover {{
    text-decoration: underline;
  }}
  .select-cell {{
    width: 36px;
    text-align: center;
  }}
  .delete-checkbox {{
    width: 18px;
    height: 18px;
    accent-color: #dc2626;
  }}
  .details-row {{
    background: #fff1f2;
  }}
  .details-row.hidden {{
    display: none;
  }}
  .details-content {{
    padding: 18px 0;
    display: grid;
    gap: 20px;
  }}
  .details-grid {{
    display: grid;
    gap: 20px;
  }}
  .section-heading {{
    margin: 0 0 12px;
    font-size: 1.05rem;
    color: #991b1b;
    font-weight: 700;
  }}
  .details-table {{
    width: 100%;
    border-collapse: collapse;
    background: #ffffff;
    border: 1px solid #fecaca;
    border-radius: 10px;
    overflow: hidden;
  }}
  .details-table th, .details-table td {{
    padding: 10px 12px;
    border-bottom: 1px solid #e5e7eb;
  }}
  .details-table th {{
    background: #fee2e2;
    color: #7f1d1d;
    font-weight: 700;
  }}
  .summary {{
    display: grid;
    gap: 10px;
    line-height: 1.55;
  }}
  .news-hover {{
    position: relative;
    display: inline-block;
  }}
  .news-link {{
    color: #b91c1c;
    text-decoration: none;
    font-weight: 600;
  }}
  .news-popup {{
    display: none;
    position: absolute;
    top: calc(100% + 10px);
    left: 0;
    min-width: 320px;
    max-width: 520px;
    background: #ffffff;
    border: 1px solid #fecaca;
    border-radius: 12px;
    box-shadow: 0 14px 30px rgba(127, 29, 29, 0.16);
    padding: 12px;
    z-index: 20;
  }}
  .news-hover:hover .news-popup {{
    display: block;
  }}
  .muted {{
    color: #991b1b;
  }}
  @media (max-width: 760px) {{
    table {{
      display: block;
      overflow-x: auto;
    }}
  }}
</style>
</head>
<body>
<div class="container">
  <form method="get" target="_parent" id="delete-form">
  <input type="hidden" name="page" value="Stock Tracker" />
  <div class="tracker-header">
    <div>
      <h1>Stock Tracker</h1>
      <p class="subtitle">Click a ticker to expand company details, latest news, EPS/revenue rows, and quarterly summaries.</p>
    </div>
    <button class="delete-button" id="delete-selected-button" type="submit" disabled>Delete</button>
  </div>
  <table>
    <thead>
      <tr>
        <th class="select-cell"></th>
        <th>Ticker</th>
        <th>Company Name</th>
        <th>Market Cap</th>
        <th>Description</th>
        <th>52 Week High</th>
        <th>News</th>
        <th>Beta</th>
      </tr>
    </thead>
    <tbody id="stock-table-body"></tbody>
  </table>
  </form>
</div>
<script id="stock-data" type="application/json">{payload}</script>
<script>
const stocks = JSON.parse(document.getElementById('stock-data').textContent || '[]');
const tableBody = document.getElementById('stock-table-body');
const deleteButton = document.getElementById('delete-selected-button');

function escapeHtml(value) {{
  return String(value ?? '').replace(/[&<>"']/g, char => ({{
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }}[char]));
}}

function shortText(value, length = 190) {{
  const text = String(value ?? '');
  return text.length > length ? text.slice(0, length) + '...' : text;
}}

function extractHeadline(value) {{
  const text = String(value ?? '');
  const match = text.match(/\\*\\*Headline:\\*\\*(.*)/i);
  return match ? match[1].trim() : shortText(text, 120);
}}

function rowTable(rows, emptyText) {{
  if (!rows || !rows.length) return `<p class="muted">${{emptyText}}</p>`;
  const keys = Object.keys(rows[0]);
  return `<table class="details-table"><thead><tr>${{keys.map(key => `<th>${{escapeHtml(key.replaceAll('_', ' '))}}</th>`).join('')}}</tr></thead><tbody>${{rows.map(row => `<tr>${{keys.map(key => `<td>${{escapeHtml(row[key])}}</td>`).join('')}}</tr>`).join('')}}</tbody></table>`;
}}

function newsBlock(stock) {{
  const news = stock.news || [];
  if (!news.length) return '<span class="muted">No news</span>';
  const latest = news[0];
  const rows = news.slice(0, 5).map(item => `<tr><td>${{escapeHtml(item.datetime || '')}}</td><td>${{escapeHtml(extractHeadline(item.headline))}}</td></tr>`).join('');
  return `<div class="news-hover"><a href="#" class="news-link">${{escapeHtml(extractHeadline(latest.headline))}}</a><div class="news-popup"><table class="details-table"><thead><tr><th>Date</th><th>Headline</th></tr></thead><tbody>${{rows}}</tbody></table></div></div>`;
}}

function detailsHtml(stock) {{
  return `
    <div class="details-content">
      <div class="summary">
        <div><strong>Description</strong></div>
        <div>${{escapeHtml(stock.description || 'No description stored.')}}</div>
      </div>
      <div>
        <div class="section-heading">Latest News</div>
        ${{rowTable((stock.news || []).map(item => ({{ datetime: item.datetime, headline: extractHeadline(item.headline) }})), 'No latest news available.')}}
      </div>
      <div class="details-grid">
        <div>
          <div class="section-heading">EPS Growth</div>
          ${{rowTable(stock.earnings_growth || [], 'No EPS rows stored.')}}
        </div>
        <div>
          <div class="section-heading">Revenue Growth</div>
          ${{rowTable(stock.revenue_growth || [], 'No revenue rows stored.')}}
        </div>
        <div>
          <div class="section-heading">Quarterly Summary</div>
          ${{rowTable(stock.quarterly_summary || [], 'No quarterly summary rows stored.')}}
        </div>
      </div>
    </div>
  `;
}}

stocks.forEach((stock, index) => {{
  const detailsId = `details-${{index}}`;
  const row = document.createElement('tr');
  row.innerHTML = `
    <td class="select-cell"><input class="delete-checkbox" name="stock_tracker_delete" type="checkbox" value="${{escapeHtml(stock.ticker)}}" aria-label="Select ${{escapeHtml(stock.ticker)}} for deletion" /></td>
    <td><button class="ticker-button" type="button" aria-expanded="false" data-target="${{detailsId}}">${{escapeHtml(stock.ticker)}}</button></td>
    <td>${{escapeHtml(stock.company_name || '')}}</td>
    <td>${{escapeHtml(stock.market_cap || '')}}</td>
    <td>${{escapeHtml(shortText(stock.description || 'N/A'))}}</td>
    <td>${{escapeHtml(stock.fifty_two_week_high || '')}}</td>
    <td>${{newsBlock(stock)}}</td>
    <td>${{escapeHtml(stock.beta ?? '')}}</td>
  `;
  const detailsRow = document.createElement('tr');
  detailsRow.className = 'details-row hidden';
  detailsRow.id = detailsId;
  detailsRow.innerHTML = `<td colspan="8">${{detailsHtml(stock)}}</td>`;
  tableBody.appendChild(row);
  tableBody.appendChild(detailsRow);
}});

function selectedTickers() {{
  return Array.from(document.querySelectorAll('.delete-checkbox:checked')).map(input => input.value);
}}

function updateDeleteButton() {{
  const selected = selectedTickers();
  deleteButton.disabled = selected.length === 0;
  deleteButton.textContent = selected.length ? `Delete (${{selected.length}})` : 'Delete';
}}

document.addEventListener('change', event => {{
  if (event.target.closest('.delete-checkbox')) updateDeleteButton();
}});

document.getElementById('delete-form').addEventListener('submit', event => {{
  const selected = selectedTickers();
  if (!selected.length || !window.confirm(`Delete ${{selected.join(', ')}} from Stock Tracker?`)) {{
    event.preventDefault();
  }}
}});

document.addEventListener('click', event => {{
  const button = event.target.closest('.ticker-button');
  if (!button) return;
  const target = document.getElementById(button.dataset.target);
  if (!target) return;
  const isOpen = !target.classList.contains('hidden');
  document.querySelectorAll('.details-row').forEach(row => row.classList.add('hidden'));
  document.querySelectorAll('.ticker-button').forEach(btn => btn.setAttribute('aria-expanded', 'false'));
  if (!isOpen) {{
    target.classList.remove('hidden');
    button.setAttribute('aria-expanded', 'true');
  }}
}});
</script>
</body>
</html>
"""


def _stock_tracker_streamlit_theme() -> None:
    st.markdown(
        """
        <style>
          div[data-testid="stAppViewContainer"] {
            background: #fff5f5;
          }
          .stock-tracker-title {
            color: #991b1b;
            font-size: 2.35rem;
            font-weight: 800;
            line-height: 1.05;
            margin: 0;
          }
          .stock-tracker-subtitle {
            color: #7f1d1d;
            font-size: 1.05rem;
            margin: 0.35rem 0 1rem;
          }
          .stock-table-header {
            background: #991b1b;
            color: white;
            font-weight: 800;
            border-radius: 8px 8px 0 0;
            padding: 0.75rem 0.6rem;
          }
          .stock-row {
            border-bottom: 1px solid #fecaca;
            padding: 0.75rem 0;
          }
          .stock-ticker-text {
            color: #b91c1c;
            font-weight: 800;
            font-size: 1.05rem;
          }
          div[data-testid="stAppViewContainer"] .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
          }
          div[data-testid="stAppViewContainer"] .stTabs [data-baseweb="tab"] {
            color: #7f1d1d;
            font-weight: 700;
          }
          div[data-testid="stAppViewContainer"] .stTabs [aria-selected="true"] {
            color: #dc2626;
          }
          div[data-testid="stAppViewContainer"] .stButton > button {
            border-color: #dc2626;
            color: #991b1b;
          }
          div[data-testid="stAppViewContainer"] .stButton > button[kind="primary"] {
            background: #dc2626;
            border-color: #dc2626;
            color: #ffffff;
          }
          div[data-testid="stAppViewContainer"] [role="radiogroup"] {
            background: #fee2e2;
            border-radius: 10px;
            padding: 0.2rem;
            width: fit-content;
            margin-left: auto;
          }
          div[data-testid="stAppViewContainer"] [role="radiogroup"] label {
            background: transparent;
            border-radius: 8px;
            color: #7f1d1d;
            font-weight: 800;
            padding: 0.2rem 0.8rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_rows_table(rows: list[dict[str, object]], empty_text: str) -> None:
    if not rows:
        st.caption(empty_text)
        return
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_quarterly_summary(rows: object, ticker: str) -> None:
    if not isinstance(rows, list) or not rows:
        st.markdown("**Quarterly Summary**")
        st.caption(f"No quarterly summary available for {ticker}.")
        return

    frame = pd.DataFrame([row for row in rows if isinstance(row, dict)])
    if frame.empty:
        st.markdown("**Quarterly Summary**")
        st.caption(f"No quarterly summary available for {ticker}.")
        return

    quarter = ""
    if "quarter" in frame.columns:
        def quarter_sort_value(value: object) -> tuple[int, int, str]:
            text = str(value or "")
            match = re.search(r"Q([1-4])\s*['-]?\s*(20\d{2}|\d{2})", text, flags=re.IGNORECASE)
            if match:
                year = int(match.group(2))
                if year < 100:
                    year += 2000
                return (year, int(match.group(1)), text)
            parsed = pd.to_datetime(text, errors="coerce")
            if not pd.isna(parsed):
                return (int(parsed.year), ((parsed.month - 1) // 3) + 1, text)
            return (0, 0, text)

        frame["_quarter_sort"] = frame["quarter"].map(quarter_sort_value)
        quarter_values = frame["quarter"].dropna().astype(str)
        quarter = quarter_values.iloc[0] if not quarter_values.empty else ""
    if "section" not in frame.columns and "transcript" in frame.columns:
        frame["section"] = frame["transcript"]
        frame = frame.drop(columns=["transcript"])

    sec_sections = {
        "99.1 Exhibit Link",
        "99.1 Exhibit Description",
        "SEC Earnings Release Summary",
    }
    if "section" in frame.columns:
        frame["_section_sort"] = frame["section"].map(lambda value: 1 if str(value or "") in sec_sections else 0)
    else:
        frame["_section_sort"] = 0
    if "_quarter_sort" in frame.columns:
        frame = frame.sort_values(["_section_sort", "_quarter_sort"], ascending=[True, False])
    else:
        frame = frame.sort_values(["_section_sort"], ascending=[True])
    frame = frame.drop(columns=[column for column in ["_section_sort", "_quarter_sort", "quarter"] if column in frame.columns])

    frame = frame.rename(columns={"section": "Section", "summary": "Analysis", "transcript": "Transcript"})
    column_order = [column for column in ["Section", "Analysis", "Transcript"] if column in frame.columns]
    frame = frame[column_order or frame.columns.tolist()]

    heading = "Quarterly Summary"
    st.markdown(f"**{heading}**")
    st.caption("Takeaways remain the first source. SEC 8-K Exhibit 99.1 rows are appended when available.")
    st.dataframe(frame, width="stretch", hide_index=True)


def _first_metric_value(rows: object, metric_name: str) -> str:
    if not isinstance(rows, list):
        return ""
    for row in rows:
        if not isinstance(row, dict) or row.get("Metric") != metric_name:
            continue
        for key, value in row.items():
            if key in {"Metric", "Quarter Info"}:
                continue
            if value not in (None, ""):
                return str(value)
    return ""


def _merge_missing_metric_rows(
    base_rows: object,
    fallback_rows: object,
) -> list[dict[str, object]]:
    merged = [dict(row) for row in base_rows if isinstance(row, dict)] if isinstance(base_rows, list) else []
    if not isinstance(fallback_rows, list):
        return merged
    for fallback in fallback_rows:
        if not isinstance(fallback, dict):
            continue
        metric = str(fallback.get("Metric") or "").strip()
        if not metric:
            continue
        target = next((row for row in merged if str(row.get("Metric") or "") == metric), None)
        if target is None:
            merged.append(dict(fallback))
            continue
        for key, value in fallback.items():
            if key in {"Metric", "Quarter Info"} or value in (None, ""):
                continue
            if target.get(key) in (None, ""):
                target[key] = value
    return merged


def _render_stock_detail_tables(
    ticker: str,
    description: str | None,
    details: dict[str, object],
    key_context: str,
) -> None:
    news = details.get("news", [])
    st.markdown("**Description**")
    st.write(description or details.get("description") or "No description stored.")

    st.markdown("**Latest News**")
    _render_rows_table(
        [
            {
                "date": item.get("date") or item.get("datetime"),
                "source": item.get("source", ""),
                "headline": extract_feed_headline(str(item.get("headline", ""))),
                "link": item.get("link", ""),
            }
            for item in news
            if isinstance(item, dict)
        ],
        f"No latest news available for {ticker}.",
    )

    period_options = ["Annual", "Quarterly"]
    if hasattr(st, "segmented_control"):
        period = st.segmented_control(
            "EPS and revenue period",
            period_options,
            default="Quarterly",
            key=f"stock_tracker_period_{key_context}_{ticker}",
            label_visibility="collapsed",
        )
    else:
        period = st.radio(
            "EPS and revenue period",
            period_options,
            index=1,
            key=f"stock_tracker_period_{key_context}_{ticker}",
            horizontal=True,
            label_visibility="collapsed",
        )

    result_key = f"manual_transcript_summary_{ticker}"
    name_key = f"manual_transcript_name_{ticker}"
    eps_metric_key = f"manual_transcript_eps_{ticker}"
    revenue_metric_key = f"manual_transcript_revenue_{ticker}"

    show_annual = period == "Annual"
    eps_rows = (
        details.get("eps_annual_table", [])
        if show_annual
        else details.get("eps_quarter_table", []) or details.get("earnings_growth", [])
    )
    revenue_rows = (
        details.get("revenue_annual_table", [])
        if show_annual
        else details.get("revenue_quarter_table", []) or details.get("revenue_growth", [])
    )
    if not show_annual:
        eps_rows = _merge_missing_metric_rows(eps_rows, st.session_state.get(eps_metric_key, []))
        revenue_rows = _merge_missing_metric_rows(revenue_rows, st.session_state.get(revenue_metric_key, []))
    period_label = "annual" if show_annual else "quarterly"

    st.markdown("**EPS Growth**")
    _render_rows_table(
        eps_rows,
        f"No {period_label} EPS rows available for {ticker}.",
    )

    st.markdown("**Revenue Growth**")
    _render_rows_table(
        revenue_rows,
        f"No {period_label} revenue rows available for {ticker}.",
    )

    url_key = f"transcript_url_{key_context}_{ticker}"
    paste_key = f"transcript_paste_{key_context}_{ticker}"
    transcript_url = st.text_input(
        "Transcript URL",
        placeholder="https://www.fool.com/earnings/call-transcripts/...",
        key=url_key,
    ).strip()
    pasted_transcript = st.text_area(
        "Paste transcript text",
        placeholder="Paste earnings transcript, press release, or article text here...",
        height=180,
        key=paste_key,
        help="Paste text is prioritized when you click Analyze Pasted Text. EPS/revenue values are extracted when structured tables are missing.",
    )
    action_cols = st.columns([1, 1, 2])
    if action_cols[0].button(
        "Analyze URL",
        key=f"analyze_transcript_url_{key_context}_{ticker}",
        disabled=not transcript_url,
        width="stretch",
    ):
        try:
            with st.spinner(f"Fetching transcript URL and creating {ticker} summary..."):
                analysis = analyze_transcript_url_with_metrics(ticker, transcript_url)
                st.session_state[result_key] = analysis.get("summary", [])
                st.session_state[eps_metric_key] = analysis.get("eps_quarter_table", [])
                st.session_state[revenue_metric_key] = analysis.get("revenue_quarter_table", [])
                st.session_state[name_key] = transcript_url
                if st.session_state[eps_metric_key] or st.session_state[revenue_metric_key]:
                    save_growth_tables(
                        ticker,
                        st.session_state[eps_metric_key],
                        st.session_state[revenue_metric_key],
                    )
            st.session_state["stock_tracker_message"] = f"Transcript URL analyzed for {ticker}."
            st.rerun()
        except Exception as exc:
            st.error(f"Could not analyze transcript URL: {exc}")
    if action_cols[1].button(
        "Analyze Pasted Text",
        key=f"analyze_transcript_paste_{key_context}_{ticker}",
        disabled=not pasted_transcript.strip(),
        width="stretch",
    ):
        try:
            with st.spinner(f"Creating {ticker} summary from pasted transcript..."):
                analysis = analyze_pasted_transcript_with_metrics(ticker, pasted_transcript)
                st.session_state[result_key] = analysis.get("summary", [])
                st.session_state[eps_metric_key] = analysis.get("eps_quarter_table", [])
                st.session_state[revenue_metric_key] = analysis.get("revenue_quarter_table", [])
                st.session_state[name_key] = "pasted transcript"
                if st.session_state[eps_metric_key] or st.session_state[revenue_metric_key]:
                    save_growth_tables(
                        ticker,
                        st.session_state[eps_metric_key],
                        st.session_state[revenue_metric_key],
                    )
            st.session_state["stock_tracker_message"] = f"Pasted transcript analyzed for {ticker}."
            st.rerun()
        except Exception as exc:
            st.error(f"Could not analyze pasted transcript: {exc}")
    manual_summary = st.session_state.get(result_key)
    if manual_summary:
        source_name = st.session_state.get(name_key, "manual transcript")
        action_cols[2].caption(f"Showing Quarterly Summary from: {source_name}")

    _render_quarterly_summary(manual_summary or details.get("quarterly_summary", []), ticker)


def _render_stock_tracker_native(stocks: pd.DataFrame) -> None:
    tracked_tickers = [str(ticker).upper() for ticker in stocks["ticker"].tolist()] if not stocks.empty else []
    selected_tickers = [
        ticker for ticker in tracked_tickers if st.session_state.get(f"tracker_delete_{ticker}", False)
    ]

    title_col, delete_col = st.columns([4, 1])
    title_col.markdown(
        """
        <div class="stock-tracker-title">Stock Tracker</div>
        <div class="stock-tracker-subtitle">
          Click a ticker row to expand company details, latest news, EPS/revenue rows, and quarterly summaries.
        </div>
        """,
        unsafe_allow_html=True,
    )
    delete_label = f"Delete ({len(selected_tickers)})" if selected_tickers else "Delete"
    if delete_col.button(
        delete_label,
        key="delete_selected_stock_tracker_native",
        type="primary",
        disabled=not selected_tickers,
        width="stretch",
    ):
        deleted = [ticker for ticker in selected_tickers if delete_tracked_stock(ticker)]
        for ticker in selected_tickers:
            st.session_state.pop(f"tracker_delete_{ticker}", None)
        if deleted:
            st.session_state["stock_tracker_message"] = f"Deleted: {', '.join(deleted)}"
        else:
            st.session_state["stock_tracker_error"] = "Selected tickers were not found."
        st.rerun()

    with st.container(border=True):
        add_cols = st.columns([3, 1])
        ticker = add_cols[0].text_input("Ticker", placeholder="AAPL", key="tracker_add_ticker").strip().upper()
        if add_cols[1].button("Add Ticker", width="stretch"):
            try:
                add_tracked_stock(ticker)
                with st.spinner(f"Adding {ticker} and fetching earnings summary..."):
                    calendar_metrics_result = save_yesterday_calendar_metrics_for_ticker(ticker)
                    sec_summary_result = fetch_and_save_sec_earnings_release_summary_for_ticker(ticker)
                    summary_result = fetch_and_save_motley_fool_takeaways_for_ticker(ticker)
                rows_saved = int(summary_result.get("rows_saved") or 0)
                sec_rows_saved = int(sec_summary_result.get("rows_saved") or 0)
                metric_rows_saved = int(calendar_metrics_result.get("eps_rows") or 0) + int(calendar_metrics_result.get("revenue_rows") or 0)
                summary_status = summary_result.get("status", "No summary status returned.")
                sec_status = sec_summary_result.get("status", "No SEC summary status returned.")
                st.session_state["stock_tracker_message"] = (
                    f"{ticker} added to stock tracker. {sec_status} SEC rows saved: {sec_rows_saved}. "
                    f"Calendar EPS/revenue rows saved: {metric_rows_saved}. "
                    f"{summary_status} Rows saved: {rows_saved}."
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Could not add {ticker}: {exc}")

    if stocks.empty:
        st.write("No tracked stocks yet.")
        return

    control_cols = st.columns([2.2, 1])
    selected_detail_ticker = control_cols[0].selectbox(
        "Current ticker details",
        tracked_tickers,
        key="stock_tracker_current_ticker",
    )
    if control_cols[1].button("Refresh Live Data", width="stretch"):
        get_live_stock_tracker_details.cache_clear()
        with st.spinner(f"Refreshing live data for {selected_detail_ticker}..."):
            refresh_live_data_for_tracked_stock(selected_detail_ticker)
            try:
                calendar_metrics_result = save_yesterday_calendar_metrics_for_ticker(selected_detail_ticker)
                sec_summary_result = fetch_and_save_sec_earnings_release_summary_for_ticker(selected_detail_ticker)
                summary_result = fetch_and_save_motley_fool_takeaways_for_ticker(selected_detail_ticker)
                rows_saved = int(summary_result.get("rows_saved") or 0)
                sec_rows_saved = int(sec_summary_result.get("rows_saved") or 0)
                metric_rows_saved = int(calendar_metrics_result.get("eps_rows") or 0) + int(calendar_metrics_result.get("revenue_rows") or 0)
                takeaway_status = str(summary_result.get("status") or "No Takeaways status returned.")
                sec_status = str(sec_summary_result.get("status") or "No SEC summary status returned.")
                motley_saved = takeaway_status == "Saved Motley Fool Takeaways." and rows_saved > 0
                if motley_saved:
                    st.session_state.pop(f"manual_transcript_summary_{selected_detail_ticker}", None)
                    st.session_state.pop(f"manual_transcript_name_{selected_detail_ticker}", None)
                st.session_state["stock_tracker_message"] = (
                    f"Live tracker data refreshed for {selected_detail_ticker}. "
                    f"{sec_status} SEC rows saved: {sec_rows_saved}. "
                    f"Calendar EPS/revenue rows saved: {metric_rows_saved}. "
                    f"{takeaway_status} Rows saved: {rows_saved}."
                    + (" Motley Fool summary is now shown first." if motley_saved else " Existing manual summary preserved if present.")
                )
            except Exception as exc:
                st.session_state["stock_tracker_message"] = (
                    f"Live tracker data refreshed for {selected_detail_ticker}. "
                    f"Takeaways refresh failed: {exc}"
                )
        st.rerun()

    selected_row = stocks.loc[stocks["ticker"].astype(str).str.upper() == selected_detail_ticker]
    selected_description = None
    if not selected_row.empty:
        selected_description = str(selected_row.iloc[0].get("description") or "")
    with st.container(border=True):
        st.subheader(f"{selected_detail_ticker} Tracking Tables")
        selected_details = get_stock_details(selected_detail_ticker) or {}
        _render_stock_detail_tables(selected_detail_ticker, selected_description, selected_details, "selected")

    stock_table_columns = [0.35, 0.7, 1.15, 0.8, 0.9, 1.8, 0.75, 0.75, 0.75, 0.75, 0.7, 0.65]
    header_cols = st.columns(stock_table_columns)
    headers = [
        "",
        "Ticker",
        "Company",
        "Market Cap",
        "Next Earnings",
        "Description",
        "Q Rev",
        "Q EPS",
        "Y Rev",
        "Y EPS",
        "News",
        "Beta",
    ]
    for column, label in zip(header_cols, headers, strict=False):
        column.markdown(f'<div class="stock-table-header">{label}</div>', unsafe_allow_html=True)

    for row in stocks.itertuples(index=False):
        ticker = str(row.ticker).upper()
        details = get_stock_details(ticker) or {}
        news = details.get("news", [])
        news_label = "No news"
        if news:
            news_label = extract_feed_headline(str(news[0].get("headline", ""))) or "Latest news"

        quarter_revenue_growth = _first_metric_value(details.get("revenue_quarter_table", []), "Revenue Growth QoQ")
        quarter_eps_growth = _first_metric_value(details.get("eps_quarter_table", []), "QoQ EPS Growth")
        annual_revenue_growth = _first_metric_value(details.get("revenue_annual_table", []), "Revenue Growth YoY")
        annual_eps_growth = _first_metric_value(details.get("eps_annual_table", []), "EPS Growth YoY")
        next_earnings = str(getattr(row, "next_earnings_date", "") or details.get("next_earnings") or "")

        row_cols = st.columns(stock_table_columns)
        row_cols[0].checkbox("Select", key=f"tracker_delete_{ticker}", label_visibility="collapsed")
        row_cols[1].markdown(f'<div class="stock-ticker-text">{ticker}</div>', unsafe_allow_html=True)
        row_cols[2].write(row.company_name or "")
        row_cols[3].write(format_market_cap(row.market_cap))
        row_cols[4].write(next_earnings or "-")
        row_cols[5].write(str(row.description or "N/A")[:160] + ("..." if len(str(row.description or "")) > 160 else ""))
        row_cols[6].write(quarter_revenue_growth or "-")
        row_cols[7].write(quarter_eps_growth or "-")
        row_cols[8].write(annual_revenue_growth or "-")
        row_cols[9].write(annual_eps_growth or "-")
        row_cols[10].write(news_label)
        row_cols[11].write("" if pd.isna(row.beta) else row.beta)

        with st.expander(f"{ticker} details", expanded=False):
            _render_stock_detail_tables(ticker, row.description, details, "row")


@st.cache_data(ttl=300, show_spinner=False)
def _cached_earnings_calendar_rows(day: date, include_reported_details: bool) -> list[dict[str, object]]:
    return earnings_calendar_rows_for_tracker(day, include_reported_details=include_reported_details)


def _add_calendar_ticker_to_tracker(ticker: str, report_date: date | None = None) -> None:
    normalized = ticker.strip().upper()
    add_tracked_stock(normalized)
    calendar_metrics_result = save_yesterday_calendar_metrics_for_ticker(normalized, report_date=report_date)
    sec_summary_result = fetch_and_save_sec_earnings_release_summary_for_ticker(normalized, report_date=report_date)
    summary_result = fetch_and_save_motley_fool_takeaways_for_ticker(normalized)
    rows_saved = int(summary_result.get("rows_saved") or 0)
    sec_rows_saved = int(sec_summary_result.get("rows_saved") or 0)
    metric_rows_saved = int(calendar_metrics_result.get("eps_rows") or 0) + int(calendar_metrics_result.get("revenue_rows") or 0)
    st.session_state["stock_tracker_message"] = (
        f"{normalized} added to stock tracker. {sec_summary_result.get('status', '')} SEC rows saved: {sec_rows_saved}. "
        f"Calendar EPS/revenue rows saved: {metric_rows_saved}. "
        f"{summary_result.get('status', '')} Rows saved: {rows_saved}."
    )


def _previous_earnings_report_day(current_day: date) -> date:
    day = current_day - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def _next_earnings_report_day(current_day: date) -> date:
    day = current_day + timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


def _render_earnings_calendar_section(
    title: str,
    day: date,
    include_reported_details: bool,
    allow_add_to_tracker: bool = False,
) -> None:
    with st.spinner(f"Loading {title.lower()} earnings calendar..."):
        rows = _cached_earnings_calendar_rows(day, include_reported_details)

    st.markdown(f"**{title} ({day.isoformat()}) - {len(rows)} report{'s' if len(rows) != 1 else ''}**")
    if not rows:
        st.caption("No earnings calendar rows found.")
        return

    frame = pd.DataFrame(rows)
    display_columns = [
        column
        for column in [
            "ticker",
            "company",
            "date",
            "time",
            "eps actual",
            "eps estimate",
            "eps beat",
            "revenue actual",
            "revenue estimate",
            "revenue beat",
            "actual source",
            "earnings filing",
            "exhibit description",
            "quarterly summary",
            "source",
        ]
        if column in frame.columns
    ]
    if not allow_add_to_tracker:
        st.dataframe(frame[display_columns], width="stretch", hide_index=True)
        return

    editor_frame = frame[display_columns].copy()
    editor_frame.insert(0, "Add", False)
    edited_frame = st.data_editor(
        editor_frame,
        key=f"calendar_select_table_{title}_{day.isoformat()}",
        width="stretch",
        hide_index=True,
        disabled=display_columns,
        column_config={
            "Add": st.column_config.CheckboxColumn(
                "Add",
                help="Select ticker to add to Stock Tracker",
                default=False,
            )
        },
        num_rows="fixed",
    )
    selected_tickers = (
        edited_frame.loc[edited_frame["Add"], "ticker"].dropna().astype(str).str.upper().tolist()
        if "Add" in edited_frame and "ticker" in edited_frame
        else []
    )

    button_label = f"Add Selected to Tracker ({len(selected_tickers)})" if selected_tickers else "Add Selected to Tracker"
    if st.button(
        button_label,
        key=f"calendar_add_selected_{title}_{day.isoformat()}",
        type="primary",
        disabled=not selected_tickers,
        width="stretch",
    ):
        added: list[str] = []
        failed: list[str] = []
        with st.spinner(f"Adding {len(selected_tickers)} ticker(s) to Stock Tracker..."):
            for ticker in selected_tickers:
                try:
                    _add_calendar_ticker_to_tracker(ticker, report_date=day)
                    added.append(ticker)
                except Exception:
                    failed.append(ticker)
        message_parts = []
        if added:
            message_parts.append(f"Added: {', '.join(added)}")
        if failed:
            message_parts.append(f"Failed: {', '.join(failed)}")
        st.session_state["stock_tracker_message"] = ". ".join(message_parts) or "No tickers were added."
        st.rerun()


def _render_reported_earnings_scan_controls() -> None:
    with st.container(border=True):
        st.markdown("**Reported Earnings Growth Scan**")
        try:
            auto_scan_status = auto_scan_reported_earnings_if_due()
            reported_date = auto_scan_status.get("reported_date") or "-"
            checked = auto_scan_status.get("checked") or 0
            added = auto_scan_status.get("added") or 0
            if auto_scan_status.get("ran"):
                st.session_state["stock_tracker_earnings_results"] = auto_scan_status.get("results", [])
                st.info(
                    f"Auto reported-earnings scan ran for {reported_date}: "
                    f"checked {checked}, added {added}."
                )
                if int(added or 0) > 0:
                    st.rerun()
            else:
                st.caption(
                    f"Auto reported-earnings scan: {auto_scan_status.get('status', 'not due')}. "
                    f"Latest reported date checked: {reported_date}; checked {checked}, added {added}."
                )
        except Exception as exc:
            st.warning(f"Auto reported-earnings scan could not run: {exc}")

        earnings_cols = st.columns([1.2, 1.2, 1.2, 1.4])
        report_date = earnings_cols[0].date_input(
            "Reported date",
            value=_previous_earnings_report_day(date.today()),
            key="calendar_earnings_reported_date",
        )
        eps_threshold = earnings_cols[1].number_input(
            "Min EPS QoQ %",
            min_value=-500.0,
            max_value=1000.0,
            value=40.0,
            step=5.0,
            key="calendar_eps_qoq_threshold",
        )
        revenue_threshold = earnings_cols[2].number_input(
            "Min Revenue QoQ %",
            min_value=-500.0,
            max_value=1000.0,
            value=30.0,
            step=5.0,
            key="calendar_revenue_qoq_threshold",
        )
        if earnings_cols[3].button("Scan Reported Earnings", width="stretch", key="calendar_scan_reported_earnings"):
            with st.spinner("Checking reported earnings and adding passing tickers..."):
                results = scan_recent_earnings_growth_and_add_to_tracker(
                    report_date=report_date,
                    min_eps_qoq_growth=eps_threshold,
                    min_revenue_qoq_growth=revenue_threshold,
                )
            added_count = sum(1 for item in results if item.get("added"))
            st.session_state["stock_tracker_earnings_results"] = results
            st.success(f"Earnings scan checked {len(results)} tickers. Added {added_count} passing tickers.")
            st.rerun()

    earnings_results = st.session_state.get("stock_tracker_earnings_results")
    if earnings_results:
        with st.expander("Latest Reported Earnings Growth Scan", expanded=False):
            st.dataframe(pd.DataFrame(earnings_results), width="stretch", hide_index=True)


def _render_earnings_calendar_tab() -> None:
    today = date.today()
    yesterday = _previous_earnings_report_day(today)
    tomorrow = _next_earnings_report_day(today)
    refresh_cols = st.columns([4, 1])
    refresh_cols[0].caption("Earnings calendar dates/estimates come from cached Alpha Vantage rows. Actuals and summaries come from SEC 8-K exhibit/earnings-release filings when available.")
    if refresh_cols[1].button("Refresh Calendar", width="stretch", key="refresh_earnings_calendar"):
        _cached_earnings_calendar_rows.clear()
        clear_earnings_calendar_detail_cache()
        st.rerun()
    _render_reported_earnings_scan_controls()
    st.divider()
    calendar_tabs = st.tabs(
        [
            f"Yesterday / Prev Market Day ({yesterday.isoformat()})",
            f"Today ({today.isoformat()})",
            f"Tomorrow / Next Market Day ({tomorrow.isoformat()})",
        ]
    )
    with calendar_tabs[0]:
        st.caption("Uses the previous weekday when today is Sunday or Monday. Includes EPS/revenue beat and Motley Fool Takeaways preview when available.")
        _render_earnings_calendar_section(
            "Yesterday / Previous Market Day Earnings",
            yesterday,
            include_reported_details=True,
            allow_add_to_tracker=True,
        )
    with calendar_tabs[1]:
        _render_earnings_calendar_section("Today Earnings", today, include_reported_details=False)
    with calendar_tabs[2]:
        st.caption("Uses the next weekday when today is Friday or Saturday.")
        _render_earnings_calendar_section("Tomorrow / Next Market Day Earnings", tomorrow, include_reported_details=False)


def stock_tracker_page() -> None:
    _stock_tracker_streamlit_theme()

    if not tracking_database_exists():
        st.warning("Stock tracking database was not found in data/stock_tracking/stocks.db.")
        return

    delete_values = st.query_params.get_all("stock_tracker_delete")
    if delete_values:
        tickers_to_delete = [
            ticker.strip().upper()
            for value in delete_values
            for ticker in str(value).split(",")
            if ticker.strip()
        ]
        deleted = [ticker for ticker in tickers_to_delete if delete_tracked_stock(ticker)]
        if deleted:
            st.session_state["stock_tracker_message"] = f"Deleted: {', '.join(deleted)}"
        else:
            st.session_state["stock_tracker_error"] = "Selected tickers were not found."
        st.query_params.clear()
        st.query_params["page"] = "Stock Tracker"
        st.rerun()

    tracker_message = st.session_state.pop("stock_tracker_message", None)
    tracker_error = st.session_state.pop("stock_tracker_error", None)
    if tracker_message:
        st.success(tracker_message)
    if tracker_error:
        st.error(tracker_error)

    try:
        panel_refresh = auto_refresh_tracker_panel_if_due()
        if panel_refresh.get("ran"):
            st.caption(
                "Tracker panel checked for post-earnings updates today: "
                f"refreshed {panel_refresh.get('post_earnings_refreshed', 0)} due tickers."
            )
    except Exception as exc:
        st.warning(f"Could not refresh tracker panel dates: {exc}")

    stocks = list_tracked_stocks()
    stock_sections = ["Stock Tracker", "Earnings Calendar"]
    query_section = st.query_params.get("stock_tracker_section")
    section_index = stock_sections.index(query_section) if query_section in stock_sections else 0
    selected_section = st.radio(
        "Stock Tracker Section",
        stock_sections,
        index=section_index,
        horizontal=True,
        label_visibility="collapsed",
        key="stock_tracker_section_radio",
    )
    if st.query_params.get("stock_tracker_section") != selected_section:
        st.query_params["page"] = "Stock Tracker"
        st.query_params["stock_tracker_section"] = selected_section

    if selected_section == "Stock Tracker":
        _render_stock_tracker_native(stocks)
    else:
        _render_earnings_calendar_tab()


pages = ["Run Today's Scan", "Market Condition", "Top 50 Strength Score", "Trades", "Previous Watchlists", "Stock Tracker"]
query_page = st.query_params.get("page")
page_index = pages.index(query_page) if query_page in pages else 0
page = st.sidebar.radio(
    "Page",
    pages,
    index=page_index,
)
if st.query_params.get("page") != page:
    st.query_params["page"] = page
if page == "Run Today's Scan":
    today_scan_page()
elif page == "Market Condition":
    market_condition_page()
elif page == "Top 50 Strength Score":
    top50_strength_score_page()
elif page == "Trades":
    trades_page()
elif page == "Previous Watchlists":
    previous_watchlists_page()
else:
    stock_tracker_page()
