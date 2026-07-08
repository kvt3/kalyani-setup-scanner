from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from config import DATA_DIR


RULE_CONFIG_PATH = DATA_DIR / "rule_config.json"


@dataclass
class PullbackRuleConfig:
    min_revenue_growth: float = 0.30
    min_eps_growth: float = 0.30
    min_avg_volume: int = 1_000_000
    max_pullback_distance: float = 0.03
    allow_hammer: bool = True
    allow_bullish_engulfing: bool = True
    allow_bullish_rejection: bool = False


@dataclass
class MarubozuRuleConfig:
    min_body_pct: float = 0.80
    max_upper_wick_pct: float = 0.12
    max_lower_wick_pct: float = 0.12
    min_volume_ratio: float = 1.0
    require_52w_breakout: bool = True


@dataclass
class WeeklyATHRuleConfig:
    min_revenue_growth: float = 0.30
    min_eps_growth: float = 0.30
    min_avg_volume: int = 1_000_000
    min_volume_ratio: float = 1.0
    require_fundamentals: bool = True
    require_uptrend: bool = True


@dataclass
class WeeklyMomentumRuleConfig:
    min_revenue_growth: float = 0.30
    min_eps_growth: float = 0.30
    min_avg_weekly_volume: int = 50_000_000
    max_pullback_distance: float = 0.05
    allow_hammer: bool = True
    allow_bullish_engulfing: bool = True


@dataclass
class MorningStarRuleConfig:
    min_avg_volume: int = 1_000_000
    first_long_body_pct: float = 0.60
    small_body_pct: float = 0.35
    recovery_pct: float = 0.50
    third_long_body_pct: float = 0.60
    gap_tolerance_pct: float = 0.005
    require_uptrend: bool = True
    require_second_volume_above_average: bool = True
    require_third_volume_above_average: bool = True


@dataclass
class TechnicalStrengthRuleConfig:
    min_rsi: float = 50.0
    min_rs_20d_vs_spy: float = 0.05
    min_rs_ratio_roc_20d: float = 0.05
    weak_market_min_outperformance_5d: float = 0.03
    weak_market_min_stock_return_5d: float = -0.02
    min_breakout_volume_ratio: float = 1.20
    max_pullback_distance: float = 0.03
    require_bullish_signal: bool = True


@dataclass
class RuleConfig:
    pullback: PullbackRuleConfig = field(default_factory=PullbackRuleConfig)
    marubozu: MarubozuRuleConfig = field(default_factory=MarubozuRuleConfig)
    weekly_ath: WeeklyATHRuleConfig = field(default_factory=WeeklyATHRuleConfig)
    weekly_momentum: WeeklyMomentumRuleConfig = field(default_factory=WeeklyMomentumRuleConfig)
    morning_star: MorningStarRuleConfig = field(default_factory=MorningStarRuleConfig)
    technical_strength: TechnicalStrengthRuleConfig = field(default_factory=TechnicalStrengthRuleConfig)


def _float_value(payload: dict[str, object], key: str, default: float) -> float:
    try:
        return float(payload.get(key, default))
    except (TypeError, ValueError):
        return default


def _int_value(payload: dict[str, object], key: str, default: int) -> int:
    try:
        return int(payload.get(key, default))
    except (TypeError, ValueError):
        return default


def _bool_value(payload: dict[str, object], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def load_rule_config() -> RuleConfig:
    if not RULE_CONFIG_PATH.exists():
        return RuleConfig()
    try:
        payload = json.loads(RULE_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return RuleConfig()

    pullback_payload = payload.get("pullback", {})
    if not isinstance(pullback_payload, dict):
        pullback_payload = {}
    marubozu_payload = payload.get("marubozu", {})
    if not isinstance(marubozu_payload, dict):
        marubozu_payload = {}
    weekly_ath_payload = payload.get("weekly_ath", {})
    if not isinstance(weekly_ath_payload, dict):
        weekly_ath_payload = {}
    weekly_momentum_payload = payload.get("weekly_momentum", {})
    if not isinstance(weekly_momentum_payload, dict):
        weekly_momentum_payload = {}
    morning_star_payload = payload.get("morning_star", {})
    if not isinstance(morning_star_payload, dict):
        morning_star_payload = {}
    technical_strength_payload = payload.get("technical_strength", {})
    if not isinstance(technical_strength_payload, dict):
        technical_strength_payload = {}

    return RuleConfig(
        pullback=PullbackRuleConfig(
            min_revenue_growth=_float_value(pullback_payload, "min_revenue_growth", 0.30),
            min_eps_growth=_float_value(pullback_payload, "min_eps_growth", 0.30),
            min_avg_volume=_int_value(pullback_payload, "min_avg_volume", 1_000_000),
            max_pullback_distance=_float_value(pullback_payload, "max_pullback_distance", 0.03),
            allow_hammer=_bool_value(pullback_payload, "allow_hammer", True),
            allow_bullish_engulfing=_bool_value(pullback_payload, "allow_bullish_engulfing", True),
            allow_bullish_rejection=_bool_value(pullback_payload, "allow_bullish_rejection", False),
        ),
        marubozu=MarubozuRuleConfig(
            min_body_pct=_float_value(marubozu_payload, "min_body_pct", 0.80),
            max_upper_wick_pct=_float_value(marubozu_payload, "max_upper_wick_pct", 0.12),
            max_lower_wick_pct=_float_value(marubozu_payload, "max_lower_wick_pct", 0.12),
            min_volume_ratio=_float_value(marubozu_payload, "min_volume_ratio", 1.0),
            require_52w_breakout=_bool_value(marubozu_payload, "require_52w_breakout", True),
        ),
        weekly_ath=WeeklyATHRuleConfig(
            min_revenue_growth=_float_value(weekly_ath_payload, "min_revenue_growth", 0.30),
            min_eps_growth=_float_value(weekly_ath_payload, "min_eps_growth", 0.30),
            min_avg_volume=_int_value(weekly_ath_payload, "min_avg_volume", 1_000_000),
            min_volume_ratio=_float_value(weekly_ath_payload, "min_volume_ratio", 1.0),
            require_fundamentals=_bool_value(weekly_ath_payload, "require_fundamentals", True),
            require_uptrend=_bool_value(weekly_ath_payload, "require_uptrend", True),
        ),
        weekly_momentum=WeeklyMomentumRuleConfig(
            min_revenue_growth=_float_value(weekly_momentum_payload, "min_revenue_growth", 0.30),
            min_eps_growth=_float_value(weekly_momentum_payload, "min_eps_growth", 0.30),
            min_avg_weekly_volume=_int_value(weekly_momentum_payload, "min_avg_weekly_volume", 50_000_000),
            max_pullback_distance=_float_value(weekly_momentum_payload, "max_pullback_distance", 0.05),
            allow_hammer=_bool_value(weekly_momentum_payload, "allow_hammer", True),
            allow_bullish_engulfing=_bool_value(weekly_momentum_payload, "allow_bullish_engulfing", True),
        ),
        morning_star=MorningStarRuleConfig(
            min_avg_volume=_int_value(morning_star_payload, "min_avg_volume", 1_000_000),
            first_long_body_pct=_float_value(morning_star_payload, "first_long_body_pct", 0.60),
            small_body_pct=_float_value(morning_star_payload, "small_body_pct", 0.35),
            recovery_pct=_float_value(morning_star_payload, "recovery_pct", 0.50),
            third_long_body_pct=_float_value(morning_star_payload, "third_long_body_pct", 0.60),
            gap_tolerance_pct=_float_value(morning_star_payload, "gap_tolerance_pct", 0.005),
            require_uptrend=_bool_value(morning_star_payload, "require_uptrend", True),
            require_second_volume_above_average=_bool_value(
                morning_star_payload,
                "require_second_volume_above_average",
                _bool_value(morning_star_payload, "require_volume_above_average", True),
            ),
            require_third_volume_above_average=_bool_value(
                morning_star_payload,
                "require_third_volume_above_average",
                _bool_value(morning_star_payload, "require_volume_above_average", True),
            ),
        ),
        technical_strength=TechnicalStrengthRuleConfig(
            min_rsi=_float_value(technical_strength_payload, "min_rsi", 50.0),
            min_rs_20d_vs_spy=_float_value(technical_strength_payload, "min_rs_20d_vs_spy", 0.05),
            min_rs_ratio_roc_20d=_float_value(technical_strength_payload, "min_rs_ratio_roc_20d", 0.05),
            weak_market_min_outperformance_5d=_float_value(
                technical_strength_payload,
                "weak_market_min_outperformance_5d",
                0.03,
            ),
            weak_market_min_stock_return_5d=_float_value(
                technical_strength_payload,
                "weak_market_min_stock_return_5d",
                -0.02,
            ),
            min_breakout_volume_ratio=_float_value(technical_strength_payload, "min_breakout_volume_ratio", 1.20),
            max_pullback_distance=_float_value(technical_strength_payload, "max_pullback_distance", 0.03),
            require_bullish_signal=_bool_value(technical_strength_payload, "require_bullish_signal", True),
        ),
    )


def save_rule_config(config: RuleConfig) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RULE_CONFIG_PATH.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
