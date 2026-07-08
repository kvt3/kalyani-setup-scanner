from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import pandas as pd

from config import DATA_DIR
from database import save_rule_watchlist, save_watchlist
from scan_state import (
    _clean_value,
    save_latest_marubozu_scan_state,
    save_latest_morning_star_scan_state,
    save_latest_monthly_big_volume_scan_state,
    save_latest_scan_state,
    save_latest_technical_breakout_scan_state,
    save_latest_technical_pullback_scan_state,
    save_latest_technical_strength_scan_state,
    save_latest_score_above60_setup_scan_state,
    save_latest_top50_strength_score_scan_state,
    save_latest_weekly_v6_cup_handle_scan_state,
    save_latest_weekly_ath_scan_state,
    save_latest_weekly_momentum_scan_state,
)
from scanner import (
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
from schedule_config import RULE_LABELS, RULE_ORDER, due_rule_keys, is_scan_schedule_due, load_schedule_config, mark_rule_schedule_ran


MANUAL_SCAN_STATUS_PATH = DATA_DIR / "manual_scan_status.json"


def _write_scan_status(status_file: Path | None, payload: dict[str, object]) -> None:
    if not status_file:
        return
    status_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **payload,
        "updated_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
    }
    status_file.write_text(json.dumps(_clean_value(payload), indent=2), encoding="utf-8")


def _parse_rule_keys(raw_rules: str) -> list[str]:
    if not raw_rules.strip():
        return []
    rule_keys = [item.strip() for item in raw_rules.split(",") if item.strip()]
    unknown = [rule_key for rule_key in rule_keys if rule_key not in RULE_LABELS]
    if unknown:
        raise ValueError(f"Unknown rule key(s): {', '.join(unknown)}")
    return rule_keys


def run_configured_scans(
    rule_keys: list[str] | None = None,
    mark_ran: bool = True,
    status_file: Path | None = None,
) -> list[str]:
    config = load_schedule_config()
    summaries: list[str] = []
    selected_rules = set(rule_keys or [rule_key for rule_key in RULE_ORDER if config.schedule_for(rule_key).enabled])

    if "pullback" in selected_rules:
        results, errors, stats = run_saved_database_scan()
        saved = save_watchlist(results)
        stats["saved_count"] = saved
        stats["error_count"] = len(errors)
        save_latest_scan_state(results, errors, stats, saved)
        save_rule_watchlist("Pullback Setup", results, str(stats.get("scan_date", "")))
        summaries.append(
            f"Pullback Setup: scanned={stats.get('stored_tickers', 0)}, "
            f"candidates={stats.get('price_candidates', 0)}, matches={len(results)}, saved={saved}, errors={len(errors)}"
        )
        if mark_ran:
            mark_rule_schedule_ran("pullback", timezone=config.schedule_for("pullback").timezone)

    if "marubozu" in selected_rules:
        results, errors, stats = run_marubozu_breakout_scan()
        stats["error_count"] = len(errors)
        save_latest_marubozu_scan_state(results, errors, stats)
        saved = save_rule_watchlist("Green Marubozu 52W Breakout", results, str(stats.get("scan_date", "")))
        summaries.append(
            f"Green Marubozu 52W Breakout: scanned={stats.get('stored_tickers', 0)}, "
            f"matches={len(results)}, saved={saved}, errors={len(errors)}"
        )
        if mark_ran:
            mark_rule_schedule_ran("marubozu", timezone=config.schedule_for("marubozu").timezone)

    if "morning_star" in selected_rules:
        results, errors, stats = run_morning_star_scan()
        stats["error_count"] = len(errors)
        save_latest_morning_star_scan_state(results, errors, stats)
        saved = save_rule_watchlist("Morning Star", results, str(stats.get("scan_date", "")))
        summaries.append(
            f"Morning Star: scanned={stats.get('stored_tickers', 0)}, "
            f"matches={len(results)}, saved={saved}, errors={len(errors)}"
        )
        if mark_ran:
            mark_rule_schedule_ran("morning_star", timezone=config.schedule_for("morning_star").timezone)

    if "weekly_ath" in selected_rules:
        results, errors, stats = run_weekly_ath_breakout_scan()
        stats["error_count"] = len(errors)
        save_latest_weekly_ath_scan_state(results, errors, stats)
        saved = save_rule_watchlist("Weekly ATH Breakout", results, str(stats.get("scan_date", "")))
        summaries.append(
            f"Weekly ATH Breakout: scanned={stats.get('stored_tickers', 0)}, "
            f"matches={len(results)}, saved={saved}, errors={len(errors)}"
        )
        if mark_ran:
            mark_rule_schedule_ran("weekly_ath", timezone=config.schedule_for("weekly_ath").timezone)

    if "weekly_momentum" in selected_rules:
        results, errors, stats = run_weekly_momentum_scan()
        stats["error_count"] = len(errors)
        save_latest_weekly_momentum_scan_state(results, errors, stats)
        saved = save_rule_watchlist("Weekly Price Momentum", results, str(stats.get("scan_date", "")))
        summaries.append(
            f"Weekly Price Momentum: scanned={stats.get('stored_tickers', 0)}, "
            f"matches={len(results)}, saved={saved}, errors={len(errors)}"
        )
        if mark_ran:
            mark_rule_schedule_ran("weekly_momentum", timezone=config.schedule_for("weekly_momentum").timezone)

    if "technical_strength" in selected_rules:
        results, errors, stats = run_technical_strength_scan()
        stats["error_count"] = len(errors)
        save_latest_technical_strength_scan_state(results, errors, stats)
        saved = save_rule_watchlist("Technical Strength", results, str(stats.get("scan_date", "")))
        summaries.append(
            f"Technical Strength: scanned={stats.get('stored_tickers', 0)}, "
            f"matches={len(results)}, saved={saved}, errors={len(errors)}"
        )
        if mark_ran:
            mark_rule_schedule_ran("technical_strength", timezone=config.schedule_for("technical_strength").timezone)

    if "technical_breakout" in selected_rules:
        results, errors, stats = run_technical_breakout_scan()
        stats["error_count"] = len(errors)
        save_latest_technical_breakout_scan_state(results, errors, stats)
        saved = save_rule_watchlist("Technical Breakout", results, str(stats.get("scan_date", "")))
        summaries.append(
            f"Technical Breakout: scanned={stats.get('stored_tickers', 0)}, "
            f"matches={len(results)}, saved={saved}, errors={len(errors)}"
        )
        if mark_ran:
            mark_rule_schedule_ran("technical_breakout", timezone=config.schedule_for("technical_breakout").timezone)

    if "technical_pullback_9ema" in selected_rules:
        results, errors, stats = run_technical_pullback_scan()
        stats["error_count"] = len(errors)
        save_latest_technical_pullback_scan_state(results, errors, stats)
        saved = save_rule_watchlist("9 EMA Pullback", results, str(stats.get("scan_date", "")))
        summaries.append(
            f"9 EMA Pullback: scanned={stats.get('stored_tickers', 0)}, "
            f"matches={len(results)}, saved={saved}, errors={len(errors)}"
        )
        if mark_ran:
            mark_rule_schedule_ran("technical_pullback_9ema", timezone=config.schedule_for("technical_pullback_9ema").timezone)

    if "weekly_v6_cup_handle" in selected_rules:
        results, errors, stats = run_weekly_v6_cup_handle_scan()
        stats["error_count"] = len(errors)
        save_latest_weekly_v6_cup_handle_scan_state(results, errors, stats)
        saved = save_rule_watchlist("Weekly Breakout", results, str(stats.get("scan_date", "")))
        summaries.append(
            f"Weekly Breakout: scanned={stats.get('stored_tickers', 0)}, "
            f"matches={len(results)}, saved={saved}, errors={len(errors)}"
        )
        if mark_ran:
            mark_rule_schedule_ran("weekly_v6_cup_handle", timezone=config.schedule_for("weekly_v6_cup_handle").timezone)

    if "monthly_big_volume" in selected_rules:
        results, errors, stats = run_monthly_big_volume_scan()
        stats["error_count"] = len(errors)
        save_latest_monthly_big_volume_scan_state(results, errors, stats)
        saved = save_rule_watchlist("Monthly Big Volume Candle", results, str(stats.get("scan_date", "")))
        summaries.append(
            f"Monthly Big Volume Candle: scanned={stats.get('stored_tickers', 0)}, "
            f"matches={len(results)}, saved={saved}, errors={len(errors)}"
        )
        if mark_ran:
            mark_rule_schedule_ran("monthly_big_volume", timezone=config.schedule_for("monthly_big_volume").timezone)

    if "top50_strength_score" in selected_rules:
        def _top50_status(progress: dict[str, object]) -> None:
            _write_scan_status(
                status_file,
                {
                    "status": "running",
                    "rules": ["top50_strength_score"],
                    "rule_labels": [RULE_LABELS["top50_strength_score"]],
                    "message": str(progress.get("message") or "Top 50 Strength Score is running."),
                    "summaries": [],
                    "progress": progress,
                },
            )

        results, errors, stats = run_top50_strength_score_scan(
            earnings_check_limit=None,
            status_callback=_top50_status,
        )
        stats["error_count"] = len(errors)
        save_latest_top50_strength_score_scan_state(results, errors, stats)
        above_60_rows = stats.get("above60_rows", [])
        above_60 = pd.DataFrame(above_60_rows) if isinstance(above_60_rows, list) and above_60_rows else pd.DataFrame()
        saved = save_rule_watchlist("Top 50 Score Above 60", above_60, str(stats.get("scan_date", "")))
        summaries.append(
            f"Top 50 Strength Score: scanned={stats.get('stored_tickers', 0)}, "
            f"earnings_checked={stats.get('earnings_checked', 0)}, rows={len(results)}, "
            f"above60={len(above_60)}, saved={saved}, errors={len(errors)}"
        )

    if "score_above60_setup" in selected_rules:
        results, errors, stats = run_score_above60_setup_scan("both")
        stats["error_count"] = len(errors)
        save_latest_score_above60_setup_scan_state(results, errors, stats)
        saved = save_rule_watchlist("Score 60+ Setup Scan", results, str(stats.get("scan_date", "")))
        summaries.append(
            f"Score 60+ Setup Scan: source={stats.get('score_above60_scan_date', '-')}, "
            f"scanned={stats.get('score_above60_tickers', stats.get('stored_tickers', 0))}, "
            f"breakouts={stats.get('breakout_candidates', 0)}, pullbacks={stats.get('pullback_candidates', 0)}, "
            f"saved={saved}, errors={len(errors)}"
        )

    if not summaries:
        return ["No scheduled scan rules are enabled."]

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run configured Kalyani Setup Scanner rules.")
    parser.add_argument(
        "--scheduled",
        action="store_true",
        help="Only run if the saved daily/weekly schedule is due.",
    )
    parser.add_argument(
        "--rules",
        default="",
        help="Comma-separated rule keys to run manually, e.g. pullback,marubozu,morning_star.",
    )
    parser.add_argument(
        "--status-file",
        default="",
        help="Optional JSON status file to update while the scan runs.",
    )
    args = parser.parse_args()

    config = load_schedule_config()
    due_rules: list[str] | None = None
    status_file = Path(args.status_file) if args.status_file else None
    if args.scheduled:
        due, reason = is_scan_schedule_due(config)
        if not due:
            print(f"Scheduled scan skipped. {reason}")
            _write_scan_status(
                status_file,
                {
                    "status": "skipped",
                    "rules": [],
                    "message": reason,
                },
            )
            return
        due_rules = due_rule_keys(config)
    elif args.rules:
        due_rules = _parse_rule_keys(args.rules)

    selected_rules = due_rules if due_rules is not None else [
        rule_key for rule_key in RULE_ORDER if config.schedule_for(rule_key).enabled
    ]
    selected_labels = [RULE_LABELS[rule_key] for rule_key in selected_rules]
    _write_scan_status(
        status_file,
        {
            "status": "running",
            "rules": selected_rules,
            "rule_labels": selected_labels,
            "message": "Scan is running in the background.",
            "summaries": [],
        },
    )

    try:
        summaries = run_configured_scans(rule_keys=due_rules, mark_ran=bool(args.scheduled), status_file=status_file)
        print("Scheduled scan complete.")
        for summary in summaries:
            print(summary)
        _write_scan_status(
            status_file,
            {
                "status": "completed",
                "rules": selected_rules,
                "rule_labels": selected_labels,
                "message": "Scan complete.",
                "summaries": summaries,
            },
        )
    except Exception as exc:
        _write_scan_status(
            status_file,
            {
                "status": "failed",
                "rules": selected_rules,
                "rule_labels": selected_labels,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "summaries": [],
            },
        )
        raise


if __name__ == "__main__":
    main()
