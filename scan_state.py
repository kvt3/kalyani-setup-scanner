from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config import DATA_DIR


LATEST_SCAN_PATH = DATA_DIR / "latest_scan.json"
LATEST_MARUBOZU_SCAN_PATH = DATA_DIR / "latest_marubozu_scan.json"
LATEST_WEEKLY_ATH_SCAN_PATH = DATA_DIR / "latest_weekly_ath_scan.json"
LATEST_WEEKLY_MOMENTUM_SCAN_PATH = DATA_DIR / "latest_weekly_momentum_scan.json"
LATEST_MORNING_STAR_SCAN_PATH = DATA_DIR / "latest_morning_star_scan.json"
LATEST_TECHNICAL_STRENGTH_SCAN_PATH = DATA_DIR / "latest_technical_strength_scan.json"
LATEST_TECHNICAL_BREAKOUT_SCAN_PATH = DATA_DIR / "latest_technical_breakout_scan.json"
LATEST_TECHNICAL_PULLBACK_SCAN_PATH = DATA_DIR / "latest_technical_pullback_scan.json"
LATEST_WEEKLY_V6_CUP_HANDLE_SCAN_PATH = DATA_DIR / "latest_weekly_v6_cup_handle_scan.json"
LATEST_MONTHLY_BIG_VOLUME_SCAN_PATH = DATA_DIR / "latest_monthly_big_volume_scan.json"
LATEST_TOP50_STRENGTH_SCORE_SCAN_PATH = DATA_DIR / "latest_top50_strength_score_scan.json"
LATEST_SCORE_ABOVE60_SETUP_SCAN_PATH = DATA_DIR / "latest_score_above60_setup_scan.json"


def _clean_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clean_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_value(item) for item in value]
    if isinstance(value, tuple | set):
        return [_clean_value(item) for item in value]
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return _clean_value(value.item())
        except Exception:
            pass
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return None
    except Exception:
        pass
    return value


def _clean_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: _clean_value(value) for key, value in row.items()} for row in records]


def save_latest_scan_state(
    results: pd.DataFrame,
    errors: list[str],
    stats: dict[str, Any],
    saved_count: int,
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": _clean_records(results.to_dict("records")) if not results.empty else [],
        "errors": errors,
        "stats": _clean_value(stats),
        "saved_count": saved_count,
    }
    LATEST_SCAN_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_latest_scan_state() -> tuple[pd.DataFrame, list[str], dict[str, Any], int] | None:
    if not LATEST_SCAN_PATH.exists():
        return None
    payload = json.loads(LATEST_SCAN_PATH.read_text(encoding="utf-8"))
    return (
        pd.DataFrame(payload.get("results", [])),
        payload.get("errors", []),
        payload.get("stats", {}),
        int(payload.get("saved_count", 0)),
    )


def save_latest_marubozu_scan_state(
    results: pd.DataFrame,
    errors: list[str],
    stats: dict[str, Any],
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": _clean_records(results.to_dict("records")) if not results.empty else [],
        "errors": errors,
        "stats": _clean_value(stats),
    }
    LATEST_MARUBOZU_SCAN_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_latest_marubozu_scan_state() -> tuple[pd.DataFrame, list[str], dict[str, Any]] | None:
    if not LATEST_MARUBOZU_SCAN_PATH.exists():
        return None
    payload = json.loads(LATEST_MARUBOZU_SCAN_PATH.read_text(encoding="utf-8"))
    return (
        pd.DataFrame(payload.get("results", [])),
        payload.get("errors", []),
        payload.get("stats", {}),
    )


def save_latest_weekly_ath_scan_state(
    results: pd.DataFrame,
    errors: list[str],
    stats: dict[str, Any],
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": _clean_records(results.to_dict("records")) if not results.empty else [],
        "errors": errors,
        "stats": _clean_value(stats),
    }
    LATEST_WEEKLY_ATH_SCAN_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_latest_weekly_ath_scan_state() -> tuple[pd.DataFrame, list[str], dict[str, Any]] | None:
    if not LATEST_WEEKLY_ATH_SCAN_PATH.exists():
        return None
    payload = json.loads(LATEST_WEEKLY_ATH_SCAN_PATH.read_text(encoding="utf-8"))
    return (
        pd.DataFrame(payload.get("results", [])),
        payload.get("errors", []),
        payload.get("stats", {}),
    )


def save_latest_weekly_momentum_scan_state(
    results: pd.DataFrame,
    errors: list[str],
    stats: dict[str, Any],
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": _clean_records(results.to_dict("records")) if not results.empty else [],
        "errors": errors,
        "stats": _clean_value(stats),
    }
    LATEST_WEEKLY_MOMENTUM_SCAN_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_latest_weekly_momentum_scan_state() -> tuple[pd.DataFrame, list[str], dict[str, Any]] | None:
    if not LATEST_WEEKLY_MOMENTUM_SCAN_PATH.exists():
        return None
    payload = json.loads(LATEST_WEEKLY_MOMENTUM_SCAN_PATH.read_text(encoding="utf-8"))
    return (
        pd.DataFrame(payload.get("results", [])),
        payload.get("errors", []),
        payload.get("stats", {}),
    )


def save_latest_morning_star_scan_state(
    results: pd.DataFrame,
    errors: list[str],
    stats: dict[str, Any],
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": _clean_records(results.to_dict("records")) if not results.empty else [],
        "errors": errors,
        "stats": _clean_value(stats),
    }
    LATEST_MORNING_STAR_SCAN_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_latest_morning_star_scan_state() -> tuple[pd.DataFrame, list[str], dict[str, Any]] | None:
    if not LATEST_MORNING_STAR_SCAN_PATH.exists():
        return None
    payload = json.loads(LATEST_MORNING_STAR_SCAN_PATH.read_text(encoding="utf-8"))
    return (
        pd.DataFrame(payload.get("results", [])),
        payload.get("errors", []),
        payload.get("stats", {}),
    )


def save_latest_technical_strength_scan_state(
    results: pd.DataFrame,
    errors: list[str],
    stats: dict[str, Any],
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": _clean_records(results.to_dict("records")) if not results.empty else [],
        "errors": errors,
        "stats": _clean_value(stats),
    }
    LATEST_TECHNICAL_STRENGTH_SCAN_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_latest_technical_strength_scan_state() -> tuple[pd.DataFrame, list[str], dict[str, Any]] | None:
    if not LATEST_TECHNICAL_STRENGTH_SCAN_PATH.exists():
        return None
    payload = json.loads(LATEST_TECHNICAL_STRENGTH_SCAN_PATH.read_text(encoding="utf-8"))
    return (
        pd.DataFrame(payload.get("results", [])),
        payload.get("errors", []),
        payload.get("stats", {}),
    )


def save_latest_technical_breakout_scan_state(
    results: pd.DataFrame,
    errors: list[str],
    stats: dict[str, Any],
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": _clean_records(results.to_dict("records")) if not results.empty else [],
        "errors": errors,
        "stats": _clean_value(stats),
    }
    LATEST_TECHNICAL_BREAKOUT_SCAN_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_latest_technical_breakout_scan_state() -> tuple[pd.DataFrame, list[str], dict[str, Any]] | None:
    if not LATEST_TECHNICAL_BREAKOUT_SCAN_PATH.exists():
        return None
    payload = json.loads(LATEST_TECHNICAL_BREAKOUT_SCAN_PATH.read_text(encoding="utf-8"))
    return (
        pd.DataFrame(payload.get("results", [])),
        payload.get("errors", []),
        payload.get("stats", {}),
    )


def save_latest_technical_pullback_scan_state(
    results: pd.DataFrame,
    errors: list[str],
    stats: dict[str, Any],
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": _clean_records(results.to_dict("records")) if not results.empty else [],
        "errors": errors,
        "stats": _clean_value(stats),
    }
    LATEST_TECHNICAL_PULLBACK_SCAN_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_latest_technical_pullback_scan_state() -> tuple[pd.DataFrame, list[str], dict[str, Any]] | None:
    if not LATEST_TECHNICAL_PULLBACK_SCAN_PATH.exists():
        return None
    payload = json.loads(LATEST_TECHNICAL_PULLBACK_SCAN_PATH.read_text(encoding="utf-8"))
    return (
        pd.DataFrame(payload.get("results", [])),
        payload.get("errors", []),
        payload.get("stats", {}),
    )


def save_latest_weekly_v6_cup_handle_scan_state(
    results: pd.DataFrame,
    errors: list[str],
    stats: dict[str, Any],
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": _clean_records(results.to_dict("records")) if not results.empty else [],
        "errors": errors,
        "stats": _clean_value(stats),
    }
    LATEST_WEEKLY_V6_CUP_HANDLE_SCAN_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_latest_weekly_v6_cup_handle_scan_state() -> tuple[pd.DataFrame, list[str], dict[str, Any]] | None:
    if not LATEST_WEEKLY_V6_CUP_HANDLE_SCAN_PATH.exists():
        return None
    payload = json.loads(LATEST_WEEKLY_V6_CUP_HANDLE_SCAN_PATH.read_text(encoding="utf-8"))
    return (
        pd.DataFrame(payload.get("results", [])),
        payload.get("errors", []),
        payload.get("stats", {}),
    )


def save_latest_monthly_big_volume_scan_state(
    results: pd.DataFrame,
    errors: list[str],
    stats: dict[str, Any],
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": _clean_records(results.to_dict("records")) if not results.empty else [],
        "errors": errors,
        "stats": _clean_value(stats),
    }
    LATEST_MONTHLY_BIG_VOLUME_SCAN_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_latest_monthly_big_volume_scan_state() -> tuple[pd.DataFrame, list[str], dict[str, Any]] | None:
    if not LATEST_MONTHLY_BIG_VOLUME_SCAN_PATH.exists():
        return None
    payload = json.loads(LATEST_MONTHLY_BIG_VOLUME_SCAN_PATH.read_text(encoding="utf-8"))
    return (
        pd.DataFrame(payload.get("results", [])),
        payload.get("errors", []),
        payload.get("stats", {}),
    )


def save_latest_top50_strength_score_scan_state(
    results: pd.DataFrame,
    errors: list[str],
    stats: dict[str, Any],
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": _clean_records(results.to_dict("records")) if not results.empty else [],
        "errors": errors,
        "stats": _clean_value(stats),
    }
    LATEST_TOP50_STRENGTH_SCORE_SCAN_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_latest_top50_strength_score_scan_state() -> tuple[pd.DataFrame, list[str], dict[str, Any]] | None:
    if not LATEST_TOP50_STRENGTH_SCORE_SCAN_PATH.exists():
        return None
    payload = json.loads(LATEST_TOP50_STRENGTH_SCORE_SCAN_PATH.read_text(encoding="utf-8"))
    return (
        pd.DataFrame(payload.get("results", [])),
        payload.get("errors", []),
        payload.get("stats", {}),
    )


def save_latest_score_above60_setup_scan_state(
    results: pd.DataFrame,
    errors: list[str],
    stats: dict[str, Any],
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": _clean_records(results.to_dict("records")) if not results.empty else [],
        "errors": errors,
        "stats": _clean_value(stats),
    }
    LATEST_SCORE_ABOVE60_SETUP_SCAN_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_latest_score_above60_setup_scan_state() -> tuple[pd.DataFrame, list[str], dict[str, Any]] | None:
    if not LATEST_SCORE_ABOVE60_SETUP_SCAN_PATH.exists():
        return None
    payload = json.loads(LATEST_SCORE_ABOVE60_SETUP_SCAN_PATH.read_text(encoding="utf-8"))
    return (
        pd.DataFrame(payload.get("results", [])),
        payload.get("errors", []),
        payload.get("stats", {}),
    )
