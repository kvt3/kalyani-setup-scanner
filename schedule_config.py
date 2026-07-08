from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from config import DATA_DIR


SCHEDULE_CONFIG_PATH = DATA_DIR / "schedule_config.json"
SCHEDULE_RUN_STATE_PATH = DATA_DIR / "schedule_run_state.json"

WEEKDAY_CODES = ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]
WEEKDAY_NAMES = {
    "MO": "Monday",
    "TU": "Tuesday",
    "WE": "Wednesday",
    "TH": "Thursday",
    "FR": "Friday",
    "SA": "Saturday",
    "SU": "Sunday",
}

RULE_ORDER = ["technical_breakout", "technical_pullback_9ema", "weekly_v6_cup_handle", "monthly_big_volume"]
RULE_LABELS = {
    "pullback": "Pullback Setup",
    "marubozu": "Green Marubozu 52W Breakout",
    "weekly_ath": "Weekly ATH Breakout",
    "weekly_momentum": "Weekly Price Momentum",
    "morning_star": "Morning Star",
    "technical_strength": "Technical Strength",
    "technical_breakout": "Technical Breakout",
    "technical_pullback_9ema": "9 EMA Pullback",
    "weekly_v6_cup_handle": "Weekly Breakout",
    "monthly_big_volume": "Monthly Big Volume Candle",
    "top50_strength_score": "Top 50 Strength Score",
    "score_above60_setup": "Score 60+ Setup Scan",
}


@dataclass(frozen=True)
class RuleSchedule:
    enabled: bool = False
    run_time: str = "09:00"
    timezone: str = "Asia/Kolkata"
    frequency: str = "daily"
    weekdays: list[str] | None = None

    @property
    def active_weekdays(self) -> list[str]:
        return self.weekdays or ["MO", "TU", "WE", "TH", "FR"]


@dataclass(frozen=True)
class ScheduleConfig:
    rules: dict[str, RuleSchedule] = field(default_factory=dict)

    def schedule_for(self, rule_key: str) -> RuleSchedule:
        return self.rules.get(rule_key, RuleSchedule(enabled=rule_key in {"technical_breakout", "technical_pullback_9ema"}))

    @property
    def run_pullback_setup(self) -> bool:
        return self.schedule_for("pullback").enabled

    @property
    def run_marubozu_setup(self) -> bool:
        return self.schedule_for("marubozu").enabled

    @property
    def run_weekly_ath_setup(self) -> bool:
        return self.schedule_for("weekly_ath").enabled

    @property
    def run_weekly_momentum_setup(self) -> bool:
        return self.schedule_for("weekly_momentum").enabled

    @property
    def run_technical_strength_setup(self) -> bool:
        return self.schedule_for("technical_strength").enabled

    @property
    def run_technical_breakout_setup(self) -> bool:
        return self.schedule_for("technical_breakout").enabled

    @property
    def run_technical_pullback_9ema_setup(self) -> bool:
        return self.schedule_for("technical_pullback_9ema").enabled

    @property
    def run_weekly_v6_cup_handle_setup(self) -> bool:
        return self.schedule_for("weekly_v6_cup_handle").enabled

    @property
    def run_monthly_big_volume_setup(self) -> bool:
        return self.schedule_for("monthly_big_volume").enabled

    @property
    def timezone(self) -> str:
        return self.schedule_for("pullback").timezone

    @property
    def run_time(self) -> str:
        return self.schedule_for("pullback").run_time

    @property
    def frequency(self) -> str:
        return self.schedule_for("pullback").frequency

    @property
    def active_weekdays(self) -> list[str]:
        return self.schedule_for("pullback").active_weekdays


def _rule_schedule_from_payload(payload: dict[str, object], default_enabled: bool = False) -> RuleSchedule:
    weekdays = payload.get("weekdays", ["MO", "TU", "WE", "TH", "FR"])
    return RuleSchedule(
        enabled=bool(payload.get("enabled", default_enabled)),
        run_time=str(payload.get("run_time", "09:00")),
        timezone=str(payload.get("timezone", "Asia/Kolkata")),
        frequency=str(payload.get("frequency", "daily")),
        weekdays=[day for day in weekdays if str(day) in WEEKDAY_CODES],
    )


def load_schedule_config() -> ScheduleConfig:
    if not SCHEDULE_CONFIG_PATH.exists():
        return ScheduleConfig(
            rules={
                "technical_breakout": RuleSchedule(enabled=True),
                "technical_pullback_9ema": RuleSchedule(enabled=True),
            }
        )
    payload = json.loads(SCHEDULE_CONFIG_PATH.read_text(encoding="utf-8"))
    if isinstance(payload.get("rules"), dict):
        rules_payload = payload["rules"]
        rules = {
            rule_key: _rule_schedule_from_payload(
                rules_payload.get(rule_key, {}) if isinstance(rules_payload.get(rule_key, {}), dict) else {},
                default_enabled=rule_key in {"technical_breakout", "technical_pullback_9ema"},
            )
            for rule_key in RULE_ORDER
        }
        return ScheduleConfig(rules=rules)

    legacy_weekdays = [day for day in payload.get("weekdays", ["MO", "TU", "WE", "TH", "FR"]) if day in WEEKDAY_CODES]
    legacy_common = {
        "run_time": str(payload.get("run_time", "09:00")),
        "timezone": str(payload.get("timezone", "Asia/Kolkata")),
        "frequency": str(payload.get("frequency", "daily")),
        "weekdays": legacy_weekdays,
    }
    return ScheduleConfig(
        rules={
            "pullback": RuleSchedule(enabled=bool(payload.get("run_pullback_setup", True)), **legacy_common),
            "marubozu": RuleSchedule(enabled=bool(payload.get("run_marubozu_setup", False)), **legacy_common),
            "weekly_ath": RuleSchedule(enabled=bool(payload.get("run_weekly_ath_setup", False)), **legacy_common),
            "weekly_momentum": RuleSchedule(enabled=bool(payload.get("run_weekly_momentum_setup", False)), **legacy_common),
            "morning_star": RuleSchedule(enabled=bool(payload.get("run_morning_star_setup", False)), **legacy_common),
            "technical_strength": RuleSchedule(enabled=bool(payload.get("run_technical_strength_setup", False)), **legacy_common),
            "technical_breakout": RuleSchedule(enabled=bool(payload.get("run_technical_breakout_setup", True)), **legacy_common),
            "technical_pullback_9ema": RuleSchedule(enabled=bool(payload.get("run_technical_pullback_9ema_setup", True)), **legacy_common),
            "weekly_v6_cup_handle": RuleSchedule(enabled=bool(payload.get("run_weekly_v6_cup_handle_setup", False)), **legacy_common),
            "monthly_big_volume": RuleSchedule(enabled=bool(payload.get("run_monthly_big_volume_setup", False)), **legacy_common),
        }
    )


def save_schedule_config(config: ScheduleConfig) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rules = {rule_key: config.schedule_for(rule_key) for rule_key in RULE_ORDER}
    enabled = {rule_key: rules.get(rule_key, RuleSchedule()).enabled for rule_key in RULE_LABELS}
    SCHEDULE_CONFIG_PATH.write_text(
        json.dumps(
            {
                "rules": {
                    rule_key: {
                        "enabled": schedule.enabled,
                        "frequency": schedule.frequency,
                        "run_time": schedule.run_time,
                        "timezone": schedule.timezone,
                        "weekdays": schedule.active_weekdays,
                    }
                    for rule_key, schedule in rules.items()
                },
                "run_pullback_setup": enabled["pullback"],
                "run_marubozu_setup": enabled["marubozu"],
                "run_weekly_ath_setup": enabled["weekly_ath"],
                "run_weekly_momentum_setup": enabled["weekly_momentum"],
                "run_morning_star_setup": enabled["morning_star"],
                "run_technical_strength_setup": enabled["technical_strength"],
                "run_technical_breakout_setup": enabled["technical_breakout"],
                "run_technical_pullback_9ema_setup": enabled["technical_pullback_9ema"],
                "run_weekly_v6_cup_handle_setup": enabled["weekly_v6_cup_handle"],
                "run_monthly_big_volume_setup": enabled["monthly_big_volume"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_run_state() -> dict[str, str]:
    if not SCHEDULE_RUN_STATE_PATH.exists():
        return {}
    return json.loads(SCHEDULE_RUN_STATE_PATH.read_text(encoding="utf-8"))


def _save_run_state(state: dict[str, str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SCHEDULE_RUN_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _parse_run_time(run_time: str) -> time:
    hour, minute = run_time.split(":", maxsplit=1)
    return time(hour=int(hour), minute=int(minute))


def is_scan_schedule_due(config: ScheduleConfig, now: datetime | None = None) -> tuple[bool, str]:
    due_rules = due_rule_keys(config, now)
    if due_rules:
        labels = ", ".join(RULE_LABELS[rule_key] for rule_key in due_rules)
        return True, f"Due rules: {labels}."
    enabled = [rule_key for rule_key in RULE_ORDER if config.schedule_for(rule_key).enabled]
    if not enabled:
        return False, "No scheduled scan rules are enabled."
    return False, "No rule schedule is due right now."


def is_rule_schedule_due(rule_key: str, schedule: RuleSchedule, now: datetime | None = None) -> tuple[bool, str]:
    if not schedule.enabled:
        return False, f"{RULE_LABELS.get(rule_key, rule_key)} is disabled."
    tz = ZoneInfo(schedule.timezone)
    local_now = (now or datetime.now(tz)).astimezone(tz)
    today_key = local_now.date().isoformat()
    state = _load_run_state()

    if state.get(f"{rule_key}_last_scan_run_date") == today_key:
        return False, f"{RULE_LABELS.get(rule_key, rule_key)} already ran on {today_key}."

    scheduled_time = _parse_run_time(schedule.run_time)
    if local_now.time() < scheduled_time:
        return False, f"Not due until {schedule.run_time} {schedule.timezone}."

    if schedule.frequency == "weekly":
        weekday = WEEKDAY_CODES[local_now.weekday()]
        if weekday not in schedule.active_weekdays:
            return False, f"Weekly scan is not scheduled for {WEEKDAY_NAMES[weekday]}."

    return True, "Scan schedule is due."


def due_rule_keys(config: ScheduleConfig, now: datetime | None = None) -> list[str]:
    return [
        rule_key
        for rule_key in RULE_ORDER
        if is_rule_schedule_due(rule_key, config.schedule_for(rule_key), now)[0]
    ]


def mark_rule_schedule_ran(rule_key: str, now: datetime | None = None, timezone: str = "Asia/Kolkata") -> None:
    tz = ZoneInfo(timezone)
    local_now = (now or datetime.now(tz)).astimezone(tz)
    state = _load_run_state()
    state[f"{rule_key}_last_scan_run_date"] = local_now.date().isoformat()
    state[f"{rule_key}_last_scan_run_at"] = local_now.isoformat()
    state["last_scan_run_date"] = local_now.date().isoformat()
    state["last_scan_run_at"] = local_now.isoformat()
    _save_run_state(state)


def mark_scan_schedule_ran(now: datetime | None = None, timezone: str = "Asia/Kolkata") -> None:
    tz = ZoneInfo(timezone)
    local_now = (now or datetime.now(tz)).astimezone(tz)
    state = _load_run_state()
    state["last_scan_run_date"] = local_now.date().isoformat()
    state["last_scan_run_at"] = local_now.isoformat()
    _save_run_state(state)
