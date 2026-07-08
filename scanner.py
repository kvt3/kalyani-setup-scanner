from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import pandas as pd
import yfinance as yf

import fundamentals as fundamentals_module
from config import (
    DATA_DIR,
    MIN_MARKET_CAP,
)
from data_loader import chunked, download_ohlcv, latest_completed_us_session, load_us_tickers
from database import list_rule_watchlists, load_eligible_tickers, load_rule_watchlist
from fundamentals import (
    Fundamentals,
    get_eps_growth_details,
    get_fmp_top50_scorecard,
    get_fundamentals,
    get_quarterly_growth_details,
)
from indicators import add_indicators, is_clear_uptrend, pullback_to_9ma
from patterns import detect_morning_star, detect_signal
from rule_config import (
    MarubozuRuleConfig,
    MorningStarRuleConfig,
    PullbackRuleConfig,
    TechnicalStrengthRuleConfig,
    WeeklyATHRuleConfig,
    WeeklyMomentumRuleConfig,
    load_rule_config,
)


@dataclass
class ScanResult:
    scan_date: str
    ticker: str
    close: float
    entry: float
    stop: float
    target: float
    risk_pct: float
    setup_grade: str
    reason: str
    revenue_growth: float | None
    eps_growth: float | None
    market_cap: float | None
    average_volume: float | None
    volume: float
    avg_volume_20d: float
    signal_date: str


def previous_swing_high(df: pd.DataFrame, lookback: int = 60) -> float | None:
    if len(df) < 5:
        return None
    prior = df.iloc[:-1].tail(lookback)
    if prior.empty:
        return None
    return float(prior["High"].max())


def _target(row: pd.Series, swing_high: float | None) -> float:
    entry = float(row["High"])
    stop = float(row["Low"])
    risk = entry - stop
    two_r = entry + 2 * risk
    if swing_high and swing_high > entry:
        return min(float(swing_high), two_r)
    return two_r


def _fmt_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.0%}"


def _grade(fundamentals: Fundamentals, signal: dict[str, bool | str], row: pd.Series, ma_label: str) -> str:
    volume_above_avg = row["Volume"] > row["Avg_Volume_20D"]
    strong_growth = (
        fundamentals.revenue_growth is not None
        and fundamentals.eps_growth is not None
        and fundamentals.revenue_growth > 0.40
        and fundamentals.eps_growth > 0.40
    )
    if strong_growth and signal["hammer"] and volume_above_avg:
        return "A+"
    if (
        fundamentals.revenue_growth is not None
        and fundamentals.eps_growth is not None
        and fundamentals.revenue_growth > 0.30
        and fundamentals.eps_growth > 0.30
        and (signal["bullish_engulfing"] or signal["bullish_rejection"])
    ):
        return "A"
    return "B"


def _passes_fundamental_rules(
    fundamentals: Fundamentals,
    market_cap: float | None,
    average_volume: float | None,
    pullback_config: PullbackRuleConfig | None = None,
) -> bool:
    cfg = pullback_config or load_rule_config().pullback
    return (
        fundamentals.revenue_growth is not None
        and fundamentals.eps_growth is not None
        and market_cap is not None
        and average_volume is not None
        and fundamentals.revenue_growth > cfg.min_revenue_growth
        and fundamentals.eps_growth > cfg.min_eps_growth
        and market_cap > MIN_MARKET_CAP
        and average_volume > cfg.min_avg_volume
    )


def _build_result(
    ticker: str,
    df: pd.DataFrame,
    fundamentals: Fundamentals,
    market_cap: float,
    average_volume: float,
    scan_date: str,
    signal: dict[str, bool | str],
    ma_label: str,
    distance: float,
) -> ScanResult | None:
    row = df.iloc[-1]
    entry = float(row["High"])
    stop = float(row["Low"])
    if stop <= 0 or entry <= stop:
        return None

    target = _target(row, previous_swing_high(df))
    risk_pct = (entry - stop) / entry
    enriched_fundamentals = Fundamentals(
        ticker=ticker,
        revenue_growth=fundamentals.revenue_growth,
        eps_growth=fundamentals.eps_growth,
        market_cap=market_cap,
        average_volume=average_volume,
        source=fundamentals.source,
        error=fundamentals.error,
    )
    grade = _grade(enriched_fundamentals, signal, row, ma_label)
    volume_note = "volume above 20-day average" if row["Volume"] > row["Avg_Volume_20D"] else "volume below 20-day average"
    reason = (
        f"{signal['pattern']} near {ma_label} ({distance:.1%} away); "
        f"uptrend confirmed; revenue {_fmt_pct(fundamentals.revenue_growth)}, "
        f"EPS {_fmt_pct(fundamentals.eps_growth)}; {volume_note}"
    )

    return ScanResult(
        scan_date=scan_date,
        ticker=ticker,
        close=round(float(row["Close"]), 2),
        entry=round(entry, 2),
        stop=round(stop, 2),
        target=round(target, 2),
        risk_pct=round(float(risk_pct), 4),
        setup_grade=grade,
        reason=reason,
        revenue_growth=fundamentals.revenue_growth,
        eps_growth=fundamentals.eps_growth,
        market_cap=market_cap,
        average_volume=average_volume,
        volume=float(row["Volume"]),
        avg_volume_20d=float(row["Avg_Volume_20D"]),
        signal_date=str(df.index[-1].date()),
    )


def find_price_setup(
    ticker: str,
    prices: pd.DataFrame,
    scan_date: str,
    allow_bullish_rejection: bool = False,
    pullback_config: PullbackRuleConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, bool | str], str, float] | None:
    cfg = pullback_config or load_rule_config().pullback
    if prices.empty or len(prices) < 210:
        return None

    df = add_indicators(prices)
    row = df.iloc[-1]
    if pd.isna(row["Avg_Volume_20D"]) or row["Avg_Volume_20D"] <= cfg.min_avg_volume:
        return None
    if not is_clear_uptrend(row):
        return None

    near_9ma, ma_label, distance = pullback_to_9ma(row, cfg.max_pullback_distance)
    if not near_9ma:
        return None

    signal = detect_signal(df)
    signal_candle = (cfg.allow_hammer and signal["hammer"]) or (
        cfg.allow_bullish_engulfing and signal["bullish_engulfing"]
    )
    if allow_bullish_rejection or cfg.allow_bullish_rejection:
        signal_candle = signal_candle or signal["bullish_rejection"]
    if not (signal_candle and signal["lower_wick_2x_body"]):
        return None
    return df, signal, ma_label, distance


def price_candidate_row(
    ticker: str,
    df: pd.DataFrame,
    signal: dict[str, bool | str],
    ma_label: str,
    distance: float,
    market_cap: float,
) -> dict[str, float | str]:
    row = df.iloc[-1]
    entry = float(row["High"])
    stop = float(row["Low"])
    target = _target(row, previous_swing_high(df))
    risk_per_share = entry - stop
    return {
        "ticker": ticker,
        "signal_date": str(df.index[-1].date()),
        "close": round(float(row["Close"]), 2),
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "risk_per_share": round(risk_per_share, 2),
        "quantity_for_1000_risk": int(1000 // risk_per_share) if risk_per_share > 0 else 0,
        "risk_pct": round(float(risk_per_share / entry), 4) if entry > stop else 0,
        "pattern": str(signal["pattern"]),
        "pullback_ma": ma_label,
        "distance_to_9ma_pct": round(distance * 100, 2),
        "volume": float(row["Volume"]),
        "avg_volume_20d": float(row["Avg_Volume_20D"]),
        "market_cap": market_cap,
    }


def scan_ticker(ticker: str, prices: pd.DataFrame, fundamentals: Fundamentals, scan_date: str) -> ScanResult | None:
    if prices.empty or len(prices) < 210:
        return None

    pullback_config = load_rule_config().pullback
    if not _passes_fundamental_rules(
        fundamentals,
        fundamentals.market_cap,
        fundamentals.average_volume,
        pullback_config,
    ):
        return None

    setup = find_price_setup(
        ticker,
        prices,
        scan_date,
        allow_bullish_rejection=True,
        pullback_config=pullback_config,
    )
    if not setup:
        return None
    df, signal, ma_label, distance = setup
    row = df.iloc[-1]

    return _build_result(
        ticker=ticker,
        df=df,
        fundamentals=fundamentals,
        market_cap=float(fundamentals.market_cap),
        average_volume=float(fundamentals.average_volume),
        scan_date=scan_date,
        signal=signal,
        ma_label=ma_label,
        distance=distance,
    )


def run_scan(limit: int | None = None, tickers: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    completed_date = latest_completed_us_session()
    scan_date = str(completed_date.date())
    universe = tickers if tickers is not None else load_us_tickers(limit=limit)
    price_data = download_ohlcv(universe, completed_date=completed_date)

    results: list[ScanResult] = []
    errors: list[str] = []

    for ticker in universe:
        prices = price_data.get(ticker)
        if prices is None or prices.empty:
            errors.append(f"{ticker}: missing OHLCV data")
            continue

        fundamentals = get_fundamentals(ticker)
        if fundamentals.error:
            errors.append(f"{ticker}: fundamentals unavailable ({fundamentals.error})")

        result = scan_ticker(ticker, prices, fundamentals, scan_date)
        if result:
            results.append(result)

    frame = pd.DataFrame([asdict(result) for result in results])
    if not frame.empty:
        grade_order = {"A+": 0, "A": 1, "B": 2}
        frame["_grade_order"] = frame["setup_grade"].map(grade_order).fillna(3)
        frame = frame.sort_values(
            by=["_grade_order", "risk_pct", "revenue_growth", "eps_growth"],
            ascending=[True, True, False, False],
        ).drop(columns=["_grade_order"])

    return frame, errors


def run_saved_database_scan() -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    from database import load_eligible_tickers

    pullback_config = load_rule_config().pullback
    completed_date = latest_completed_us_session()
    scan_date = str(completed_date.date())
    universe = load_eligible_tickers()
    if universe.empty:
        return pd.DataFrame(), [], {"scan_date": scan_date, "stored_tickers": 0, "price_candidates": 0}

    tickers = universe["ticker"].tolist()
    market_caps = dict(zip(universe["ticker"], universe["market_cap"], strict=False))
    price_data = download_ohlcv(tickers, completed_date=completed_date)

    results: list[ScanResult] = []
    errors: list[str] = []
    price_candidates: list[tuple[str, pd.DataFrame, dict[str, bool | str], str, float]] = []
    price_candidate_rows: list[dict[str, float | str]] = []

    for ticker in tickers:
        prices = price_data.get(ticker)
        if prices is None or prices.empty:
            errors.append(f"{ticker}: missing OHLCV data")
            continue
        setup = find_price_setup(
            ticker,
            prices,
            scan_date,
            allow_bullish_rejection=False,
            pullback_config=pullback_config,
        )
        if setup:
            df, signal, ma_label, distance = setup
            price_candidates.append((ticker, df, signal, ma_label, distance))
            price_candidate_rows.append(
                price_candidate_row(
                    ticker=ticker,
                    df=df,
                    signal=signal,
                    ma_label=ma_label,
                    distance=distance,
                    market_cap=float(market_caps[ticker]),
                )
            )

    for ticker, df, signal, ma_label, distance in price_candidates:
        fundamentals = get_fundamentals(ticker)
        if fundamentals.error:
            errors.append(f"{ticker}: fundamentals unavailable ({fundamentals.error})")
            continue

        market_cap = float(market_caps[ticker])
        average_volume = float(df.iloc[-1]["Avg_Volume_20D"])
        if not _passes_fundamental_rules(fundamentals, market_cap, average_volume, pullback_config):
            continue

        result = _build_result(
            ticker=ticker,
            df=df,
            fundamentals=fundamentals,
            market_cap=market_cap,
            average_volume=average_volume,
            scan_date=scan_date,
            signal=signal,
            ma_label=ma_label,
            distance=distance,
        )
        if result:
            results.append(result)

    frame = pd.DataFrame([asdict(result) for result in results])
    if not frame.empty:
        grade_order = {"A+": 0, "A": 1, "B": 2}
        frame["_grade_order"] = frame["setup_grade"].map(grade_order).fillna(3)
        frame = frame.sort_values(
            by=["_grade_order", "risk_pct", "revenue_growth", "eps_growth"],
            ascending=[True, True, False, False],
        ).drop(columns=["_grade_order"])

    stats = {
        "scan_date": scan_date,
        "stored_tickers": len(tickers),
        "ohlcv_loaded": len(price_data),
        "price_candidates": len(price_candidates),
        "price_candidate_rows": price_candidate_rows,
        "matches": len(frame),
    }
    return frame, errors, stats


def find_green_marubozu_breakout(
    ticker: str,
    prices: pd.DataFrame,
    scan_date: str,
    marubozu_config: MarubozuRuleConfig | None = None,
) -> dict[str, float | str] | None:
    cfg = marubozu_config or load_rule_config().marubozu
    if prices.empty or len(prices) < 210:
        return None

    df = add_indicators(prices)
    df["Prior_52W_High"] = df["High"].shift(1).rolling(252, min_periods=200).max()
    row = df.iloc[-1]
    if str(df.index[-1].date()) != scan_date:
        return None

    candle_range = float(row["High"] - row["Low"])
    if candle_range <= 0 or pd.isna(row["Prior_52W_High"]) or pd.isna(row["Avg_Volume_20D"]):
        return None

    body = float(row["Close"] - row["Open"])
    upper_wick = float(row["High"] - row["Close"])
    lower_wick = float(row["Open"] - row["Low"])
    body_pct = body / candle_range
    upper_wick_pct = upper_wick / candle_range
    lower_wick_pct = lower_wick / candle_range
    volume_ratio = float(row["Volume"] / row["Avg_Volume_20D"]) if row["Avg_Volume_20D"] else 0
    breaks_52w_high = row["High"] > row["Prior_52W_High"] or row["Close"] > row["Prior_52W_High"]

    if not row["Close"] > row["Open"]:
        return None
    if body_pct < cfg.min_body_pct:
        return None
    if upper_wick_pct > cfg.max_upper_wick_pct or lower_wick_pct > cfg.max_lower_wick_pct:
        return None
    if volume_ratio < cfg.min_volume_ratio:
        return None
    if cfg.require_52w_breakout and not breaks_52w_high:
        return None

    return {
        "ticker": ticker,
        "signal_date": str(df.index[-1].date()),
        "open": round(float(row["Open"]), 2),
        "high": round(float(row["High"]), 2),
        "low": round(float(row["Low"]), 2),
        "close": round(float(row["Close"]), 2),
        "volume": float(row["Volume"]),
        "avg_volume_20d": float(row["Avg_Volume_20D"]),
        "volume_ratio": round(volume_ratio, 2),
        "prior_52w_high": round(float(row["Prior_52W_High"]), 2),
        "body_pct": round(body_pct * 100, 1),
        "upper_wick_pct": round(upper_wick_pct * 100, 1),
        "lower_wick_pct": round(lower_wick_pct * 100, 1),
    }


def run_marubozu_breakout_scan() -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    from database import load_eligible_tickers

    marubozu_config = load_rule_config().marubozu
    completed_date = latest_completed_us_session()
    scan_date = str(completed_date.date())
    universe = load_eligible_tickers()
    if universe.empty:
        return pd.DataFrame(), [], {"scan_date": scan_date, "stored_tickers": 0, "price_candidates": 0}

    tickers = universe["ticker"].tolist()
    market_caps = dict(zip(universe["ticker"], universe["market_cap"], strict=False))
    price_data = download_ohlcv(tickers, completed_date=completed_date)
    errors: list[str] = []
    price_candidates: list[dict[str, float | str]] = []

    for ticker in tickers:
        prices = price_data.get(ticker)
        if prices is None or prices.empty:
            errors.append(f"{ticker}: missing OHLCV data")
            continue
        candidate = find_green_marubozu_breakout(ticker, prices, scan_date, marubozu_config)
        if candidate:
            candidate["market_cap"] = float(market_caps[ticker])
            price_candidates.append(candidate)

    frame = pd.DataFrame(price_candidates)
    if not frame.empty:
        frame = frame.sort_values(by=["volume_ratio", "market_cap"], ascending=[False, False])

    stats = {
        "scan_date": scan_date,
        "stored_tickers": len(tickers),
        "ohlcv_loaded": len(price_data),
        "price_candidates": len(price_candidates),
        "matches": len(frame),
    }
    return frame, errors, stats


def _morning_star_target(df: pd.DataFrame) -> float:
    row = df.iloc[-1]
    entry = float(row["High"])
    stop = float(df.tail(3)["Low"].min())
    risk = max(entry - stop, 0)
    swing_high = previous_swing_high(df)
    two_r = entry + (2 * risk)
    if swing_high and swing_high > entry:
        return min(float(swing_high), two_r)
    return two_r


def find_morning_star_setup(
    ticker: str,
    prices: pd.DataFrame,
    scan_date: str,
    config: MorningStarRuleConfig | None = None,
) -> dict[str, float | str] | None:
    cfg = config or load_rule_config().morning_star
    if prices.empty or len(prices) < 210:
        return None

    df = add_indicators(prices)
    row = df.iloc[-1]
    if str(df.index[-1].date()) != scan_date:
        return None
    if pd.isna(row["Avg_Volume_20D"]) or float(row["Avg_Volume_20D"]) <= cfg.min_avg_volume:
        return None
    if cfg.require_uptrend and not is_clear_uptrend(row):
        return None
    first = df.iloc[-3]
    second = df.iloc[-2]
    if cfg.require_second_volume_above_average and (
        pd.isna(second["Avg_Volume_20D"]) or float(second["Volume"]) <= float(second["Avg_Volume_20D"])
    ):
        return None
    if cfg.require_third_volume_above_average and float(row["Volume"]) <= float(row["Avg_Volume_20D"]):
        return None
    if not detect_morning_star(
        df,
        cfg.first_long_body_pct,
        cfg.small_body_pct,
        cfg.recovery_pct,
        cfg.third_long_body_pct,
        cfg.gap_tolerance_pct,
    ):
        return None

    entry = float(row["High"])
    stop = float(df.tail(3)["Low"].min())
    if stop <= 0 or entry <= stop:
        return None
    target = _morning_star_target(df)
    risk = entry - stop
    first_parts_body_pct = abs(float(first["Close"]) - float(first["Open"])) / max(float(first["High"]) - float(first["Low"]), 0.0001)
    third_parts_body_pct = abs(float(row["Close"]) - float(row["Open"])) / max(float(row["High"]) - float(row["Low"]), 0.0001)
    second_volume_ratio = float(second["Volume"] / second["Avg_Volume_20D"]) if second["Avg_Volume_20D"] else 0.0
    volume_ratio = float(row["Volume"] / row["Avg_Volume_20D"]) if row["Avg_Volume_20D"] else 0.0
    return {
        "ticker": ticker,
        "signal_date": str(df.index[-1].date()),
        "pattern": "Morning Star",
        "open_1": round(float(first["Open"]), 2),
        "close_1": round(float(first["Close"]), 2),
        "open_2": round(float(second["Open"]), 2),
        "close_2": round(float(second["Close"]), 2),
        "volume_2": float(second["Volume"]),
        "avg_volume_20d_2": float(second["Avg_Volume_20D"]),
        "volume_ratio_2": round(second_volume_ratio, 2),
        "open": round(float(row["Open"]), 2),
        "high": round(entry, 2),
        "low": round(float(row["Low"]), 2),
        "pattern_low": round(stop, 2),
        "close": round(float(row["Close"]), 2),
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "risk_reward": round((target - entry) / risk, 2) if risk > 0 else None,
        "volume": float(row["Volume"]),
        "avg_volume_20d": float(row["Avg_Volume_20D"]),
        "volume_ratio": round(volume_ratio, 2),
        "first_body_pct": round(first_parts_body_pct * 100, 1),
        "third_body_pct": round(third_parts_body_pct * 100, 1),
        "gap_tolerance_pct": round(cfg.gap_tolerance_pct * 100, 2),
        "ema_21": round(float(row["21_EMA"]), 2),
        "sma_50": round(float(row["50_SMA"]), 2),
        "sma_200": round(float(row["200_SMA"]), 2),
        "uptrend": "Close > 21 EMA > 50 SMA > 200 SMA" if is_clear_uptrend(row) else "Uptrend not required",
        "reason": (
            "Strict Morning Star: long bearish first candle, small/doji star with flexible gap down, "
            "long bullish third candle with flexible gap up and close above midpoint of first body; "
            f"second volume {second_volume_ratio:.2f}x avg, third volume {volume_ratio:.2f}x avg."
        ),
    }


def run_morning_star_scan() -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    from database import load_eligible_tickers

    cfg = load_rule_config().morning_star
    completed_date = latest_completed_us_session()
    scan_date = str(completed_date.date())
    universe = load_eligible_tickers()
    if universe.empty:
        return pd.DataFrame(), [], {"scan_date": scan_date, "stored_tickers": 0, "price_candidates": 0}

    tickers = universe["ticker"].tolist()
    market_caps = dict(zip(universe["ticker"], universe["market_cap"], strict=False))
    price_data = download_ohlcv(tickers, completed_date=completed_date)
    errors: list[str] = []
    candidates: list[dict[str, float | str]] = []

    for ticker in tickers:
        prices = price_data.get(ticker)
        if prices is None or prices.empty:
            errors.append(f"{ticker}: missing OHLCV data")
            continue
        candidate = find_morning_star_setup(ticker, prices, scan_date, cfg)
        if candidate:
            candidate["market_cap"] = float(market_caps[ticker])
            candidates.append(candidate)

    frame = pd.DataFrame(candidates)
    if not frame.empty:
        frame = frame.sort_values(by=["volume_ratio", "risk_reward", "market_cap"], ascending=[False, False, False])

    stats = {
        "scan_date": scan_date,
        "stored_tickers": len(tickers),
        "ohlcv_loaded": len(price_data),
        "price_candidates": len(candidates),
        "matches": len(frame),
    }
    return frame, errors, stats


def _monthly_ohlcv_from_daily(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    frame = prices.copy()
    frame.index = pd.to_datetime(frame.index)
    monthly = frame.resample("ME").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    monthly = monthly.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    monthly["Avg_Volume_12M"] = monthly["Volume"].shift(1).rolling(12, min_periods=6).mean()
    return monthly


def find_monthly_big_volume_setup(
    ticker: str,
    prices: pd.DataFrame,
) -> dict[str, float | str] | None:
    if prices.empty or len(prices) < 180:
        return None
    monthly = _monthly_ohlcv_from_daily(prices)
    if len(monthly) < 8:
        return None

    row = monthly.iloc[-1]
    previous = monthly.iloc[-2]
    candle_range = float(row["High"] - row["Low"])
    if candle_range <= 0:
        return None
    volume = float(row["Volume"])
    previous_volume = float(previous["Volume"])
    avg_volume_12m = float(row["Avg_Volume_12M"]) if pd.notna(row.get("Avg_Volume_12M")) else 0.0
    if previous_volume <= 0 or avg_volume_12m <= 0:
        return None
    if not (volume > previous_volume and volume > avg_volume_12m):
        return None

    open_price = float(row["Open"])
    close = float(row["Close"])
    high = float(row["High"])
    low = float(row["Low"])
    body = abs(close - open_price)
    body_pct = body / candle_range
    upper_wick = high - max(open_price, close)
    lower_wick = min(open_price, close) - low
    close_position = (close - low) / candle_range
    is_green = close > open_price
    is_big_green = bool(is_green and body_pct >= 0.60 and close_position >= 0.70)
    is_green_hammer = bool(
        is_green
        and lower_wick >= body * 2
        and upper_wick <= max(body, candle_range * 0.20)
        and close_position >= 0.60
    )
    if not (is_big_green or is_green_hammer):
        return None

    pattern = "Big Green Monthly Candle" if is_big_green else "Green Monthly Hammer"
    if is_big_green and is_green_hammer:
        pattern = "Big Green + Green Hammer"
    return {
        "ticker": ticker,
        "signal_month": str(pd.Timestamp(monthly.index[-1]).date()),
        "pattern": pattern,
        "open": round(open_price, 2),
        "high": round(high, 2),
        "low": round(low, 2),
        "close": round(close, 2),
        "volume": round(volume, 0),
        "previous_month_volume": round(previous_volume, 0),
        "avg_volume_12m": round(avg_volume_12m, 0),
        "volume_vs_previous": round(volume / previous_volume, 2),
        "volume_vs_12m_avg": round(volume / avg_volume_12m, 2),
        "body_pct": round(body_pct * 100, 1),
        "lower_wick_body_ratio": round(lower_wick / max(body, 0.0001), 2),
        "upper_wick_pct": round((upper_wick / candle_range) * 100, 1),
        "close_position_pct": round(close_position * 100, 1),
        "reason": (
            f"{pattern}; monthly volume {volume / previous_volume:.2f}x previous month "
            f"and {volume / avg_volume_12m:.2f}x 12-month average."
        ),
    }


def run_monthly_big_volume_scan() -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    from database import load_eligible_tickers

    completed_date = latest_completed_us_session()
    scan_date = str(completed_date.date())
    universe = load_eligible_tickers()
    if universe.empty:
        return pd.DataFrame(), [], {"scan_date": scan_date, "stored_tickers": 0, "price_candidates": 0}

    tickers = universe["ticker"].astype(str).str.upper().drop_duplicates().tolist()
    market_caps = dict(zip(universe["ticker"].astype(str).str.upper(), universe["market_cap"], strict=False))
    price_data = download_ohlcv(tickers, completed_date=completed_date)
    errors: list[str] = []
    candidates: list[dict[str, float | str]] = []

    for ticker in tickers:
        prices = price_data.get(ticker)
        if prices is None or prices.empty:
            errors.append(f"{ticker}: missing OHLCV data")
            continue
        candidate = find_monthly_big_volume_setup(ticker, prices)
        if candidate:
            candidate["market_cap"] = float(market_caps.get(ticker, 0) or 0)
            candidates.append(candidate)

    frame = pd.DataFrame(candidates)
    if not frame.empty:
        frame = frame.sort_values(
            by=["volume_vs_12m_avg", "volume_vs_previous", "market_cap"],
            ascending=[False, False, False],
        )

    stats = {
        "scan_date": scan_date,
        "stored_tickers": len(tickers),
        "ohlcv_loaded": len(price_data),
        "price_candidates": len(candidates),
        "matches": len(frame),
    }
    return frame, errors, stats


def _latest_completed_week_end(completed_date: pd.Timestamp) -> pd.Timestamp:
    completed_date = pd.Timestamp(completed_date).normalize()
    days_since_friday = (completed_date.weekday() - 4) % 7
    return completed_date - pd.Timedelta(days=days_since_friday)


def _to_weekly_ohlcv(prices: pd.DataFrame, completed_date: pd.Timestamp) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()

    latest_week_end = _latest_completed_week_end(completed_date)
    weekly = prices.resample("W-FRI").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    weekly = weekly[weekly.index <= latest_week_end].dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if weekly.empty:
        return weekly

    weekly["EMA_10W"] = weekly["Close"].ewm(span=10, adjust=False).mean()
    weekly["EMA_9W"] = weekly["Close"].ewm(span=9, adjust=False).mean()
    weekly["SMA_9W"] = weekly["Close"].rolling(window=9).mean()
    weekly["EMA_21W"] = weekly["Close"].ewm(span=21, adjust=False).mean()
    weekly["SMA_30W"] = weekly["Close"].rolling(window=30).mean()
    weekly["SMA_50W"] = weekly["Close"].rolling(window=50).mean()
    weekly["SMA_200W"] = weekly["Close"].rolling(window=200).mean()
    weekly["Avg_Volume_20W"] = weekly["Volume"].rolling(window=20).mean()
    weekly["Prior_All_Time_High"] = weekly["High"].shift(1).cummax()
    weekly["SMA_30W_Rising"] = weekly["SMA_30W"] > weekly["SMA_30W"].shift(4)
    return weekly


def _weekly_target(weekly: pd.DataFrame) -> float:
    row = weekly.iloc[-1]
    entry = float(row["High"])
    stop = float(row["Low"])
    risk = max(entry - stop, 0)
    prior = weekly.iloc[:-1].tail(26)
    swing_high = float(prior["High"].max()) if not prior.empty else 0.0
    two_r = entry + (2 * risk)
    if swing_high > entry:
        return min(swing_high, two_r)
    return two_r


def _distance_from_candle_to_ma(row: pd.Series, ma_value: float) -> float:
    low = float(row["Low"])
    high = float(row["High"])
    if ma_value <= 0:
        return 1.0
    if low <= ma_value <= high:
        return 0.0
    if high < ma_value:
        return (ma_value - high) / ma_value
    return (low - ma_value) / ma_value


def _round_optional(value: object, digits: int = 2) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return round(float(numeric), digits)


def _add_technical_strength_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    df = add_indicators(prices)
    df["20_EMA"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["20_SMA"] = df["Close"].rolling(20).mean()
    df["RSI_14"] = _rsi(df["Close"], 14)
    df["RSI_20"] = _rsi(df["Close"], 20)
    df["Prior_52W_High"] = df["High"].shift(1).rolling(252, min_periods=200).max()
    df["Prior_Available_High"] = df["High"].shift(1).cummax()
    return df


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def _pct_change(series: pd.Series, periods: int) -> float | None:
    if len(series.dropna()) <= periods:
        return None
    latest = float(series.iloc[-1])
    prior = float(series.iloc[-1 - periods])
    if prior == 0:
        return None
    return latest / prior - 1


def _is_strong_daily_stack(row: pd.Series) -> bool:
    required = ["Close", "9_EMA", "20_SMA", "50_SMA", "200_SMA"]
    if row[required].isna().any():
        return False
    return bool(row["Close"] > row["9_EMA"] > row["20_SMA"] > row["50_SMA"] > row["200_SMA"])


def _is_weak_market(benchmark_data: dict[str, pd.DataFrame]) -> bool:
    for ticker in ("SPY", "QQQ"):
        prices = benchmark_data.get(ticker)
        if prices is None or prices.empty or len(prices) < 210:
            continue
        df = _add_technical_strength_indicators(prices)
        row = df.iloc[-1]
        if row[["Close", "9_EMA", "20_SMA", "50_SMA", "200_SMA"]].isna().any():
            continue
        if not (row["Close"] > row["9_EMA"] > row["20_SMA"] > row["50_SMA"] > row["200_SMA"]):
            return True
    return False


def _bullish_signal_name(df: pd.DataFrame) -> str:
    signal = detect_signal(df)
    row = df.iloc[-1]
    candle_range = max(float(row["High"] - row["Low"]), 0.0001)
    close_position = (float(row["Close"]) - float(row["Low"])) / candle_range
    strong_green = bool(row["Close"] > row["Open"] and close_position >= 0.60)
    if signal["pattern"]:
        return str(signal["pattern"])
    if strong_green:
        return "Strong green candle"
    return ""


def _technical_strength_candidate(
    ticker: str,
    prices: pd.DataFrame,
    spy_prices: pd.DataFrame,
    weak_market: bool,
    scan_date: str,
    cfg: TechnicalStrengthRuleConfig,
) -> dict[str, float | str] | None:
    if prices.empty or spy_prices.empty or len(prices) < 260 or len(spy_prices) < 260:
        return None

    df = _add_technical_strength_indicators(prices).dropna(subset=["9_EMA", "20_SMA", "50_SMA", "200_SMA", "RSI_14", "Avg_Volume_20D"])
    spy = _add_technical_strength_indicators(spy_prices).dropna(subset=["Close"])
    if df.empty or spy.empty or str(df.index[-1].date()) != scan_date:
        return None

    common_index = df.index.intersection(spy.index)
    if len(common_index) < 260:
        return None
    aligned_stock = df.loc[common_index].copy()
    aligned_spy = spy.loc[common_index].copy()
    row = aligned_stock.iloc[-1]

    if not _is_strong_daily_stack(row):
        return None
    rsi = float(row["RSI_14"])
    if rsi <= cfg.min_rsi:
        return None

    stock_return_20d = _pct_change(aligned_stock["Close"], 20)
    spy_return_20d = _pct_change(aligned_spy["Close"], 20)
    stock_return_5d = _pct_change(aligned_stock["Close"], 5)
    spy_return_5d = _pct_change(aligned_spy["Close"], 5)
    if None in (stock_return_20d, spy_return_20d, stock_return_5d, spy_return_5d):
        return None

    rs_20d_vs_spy = float(stock_return_20d - spy_return_20d)
    rs_ratio = aligned_stock["Close"] / aligned_spy["Close"].replace(0, pd.NA)
    rs_ratio_roc_20d = _pct_change(rs_ratio, 20)
    if rs_ratio_roc_20d is None:
        return None
    if rs_20d_vs_spy <= cfg.min_rs_20d_vs_spy:
        return None
    if rs_ratio_roc_20d <= cfg.min_rs_ratio_roc_20d:
        return None

    outperformance_5d = float(stock_return_5d - spy_return_5d)
    if outperformance_5d <= cfg.weak_market_min_outperformance_5d:
        return None
    if float(stock_return_5d) <= cfg.weak_market_min_stock_return_5d:
        return None

    volume_ratio = float(row["Volume"] / row["Avg_Volume_20D"]) if row["Avg_Volume_20D"] else 0.0

    previous = aligned_stock.iloc[-2]
    pullback_distance = abs(float(row["Low"]) - float(row["9_EMA"])) / float(row["9_EMA"]) if row["9_EMA"] else 1.0
    touched_9ema = float(row["Low"]) <= float(row["9_EMA"]) * (1 + cfg.max_pullback_distance)
    reclaimed_9ema = float(previous["Close"]) <= float(previous["9_EMA"]) and float(row["Close"]) > float(row["9_EMA"])
    signal_name = _bullish_signal_name(aligned_stock)
    pullback = touched_9ema and reclaimed_9ema and (bool(signal_name) or not cfg.require_bullish_signal)

    if not pullback:
        return None

    entry = float(row["High"])
    stop = float(row["Low"])
    if stop <= 0 or entry <= stop:
        return None
    risk = entry - stop
    target = entry + 2 * risk
    volume_support = "volume above 20D average" if volume_ratio >= 1 and row["Close"] > row["Open"] else "volume not expanded"

    return {
        "ticker": ticker,
        "signal_date": str(aligned_stock.index[-1].date()),
        "latest_date": str(aligned_stock.index[-1].date()),
        "setup_type": "9 EMA pullback reclaim",
        "close": round(float(row["Close"]), 2),
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target_2r": round(target, 2),
        "risk_reward": 2.0,
        "exit_rule": "Stop loss OR 9 EMA crosses below 20 SMA",
        "rsi_14": round(rsi, 1),
        "stock_return_20d_pct": round(float(stock_return_20d) * 100, 2),
        "spy_return_20d_pct": round(float(spy_return_20d) * 100, 2),
        "rs_20d_vs_spy_pct": round(rs_20d_vs_spy * 100, 2),
        "rs_ratio_roc_20d_pct": round(float(rs_ratio_roc_20d) * 100, 2),
        "stock_return_5d_pct": round(float(stock_return_5d) * 100, 2),
        "spy_return_5d_pct": round(float(spy_return_5d) * 100, 2),
        "outperformance_5d_pct": round(outperformance_5d * 100, 2),
        "weak_market_filter": "Passed",
        "volume": float(row["Volume"]),
        "avg_volume_20d": float(row["Avg_Volume_20D"]),
        "volume_ratio": round(volume_ratio, 2),
        "ema_9": round(float(row["9_EMA"]), 2),
        "distance_to_9ema_pct": round(pullback_distance * 100, 2),
        "signal": signal_name,
        "ma_stack": "Close > 9 EMA > 20 SMA > 50 SMA > 200 SMA",
        "volume_support": volume_support,
        "reason": (
            f"9 EMA pullback reclaim; RSI {rsi:.1f}; RS 20D vs SPY {rs_20d_vs_spy:.1%}; "
            f"RS ratio ROC 20D {float(rs_ratio_roc_20d):.1%}; {volume_support}."
        ),
    }


def _technical_strength_breakout_watch_candidate(
    ticker: str,
    prices: pd.DataFrame,
    spy_prices: pd.DataFrame,
    weak_market: bool,
    scan_date: str,
    cfg: TechnicalStrengthRuleConfig,
    lookback: int = 20,
) -> dict[str, float | str] | None:
    if prices.empty or spy_prices.empty or len(prices) < 260 or len(spy_prices) < 260:
        return None

    df = _add_technical_strength_indicators(prices).dropna(
        subset=["9_EMA", "20_SMA", "50_SMA", "200_SMA", "RSI_14", "Avg_Volume_20D"]
    )
    spy = _add_technical_strength_indicators(spy_prices).dropna(subset=["Close"])
    if df.empty or spy.empty or str(df.index[-1].date()) != scan_date:
        return None

    common_index = df.index.intersection(spy.index)
    if len(common_index) < 260:
        return None
    aligned_stock = df.loc[common_index].copy()
    aligned_spy = spy.loc[common_index].copy()
    row = aligned_stock.iloc[-1]

    if not _is_strong_daily_stack(row):
        return None
    rsi = float(row["RSI_14"])
    if rsi <= cfg.min_rsi:
        return None

    stock_return_20d = _pct_change(aligned_stock["Close"], 20)
    spy_return_20d = _pct_change(aligned_spy["Close"], 20)
    stock_return_5d = _pct_change(aligned_stock["Close"], 5)
    spy_return_5d = _pct_change(aligned_spy["Close"], 5)
    if None in (stock_return_20d, spy_return_20d, stock_return_5d, spy_return_5d):
        return None

    rs_20d_vs_spy = float(stock_return_20d - spy_return_20d)
    rs_ratio = aligned_stock["Close"] / aligned_spy["Close"].replace(0, pd.NA)
    rs_ratio_roc_20d = _pct_change(rs_ratio, 20)
    if rs_ratio_roc_20d is None:
        return None
    if rs_20d_vs_spy <= cfg.min_rs_20d_vs_spy:
        return None
    if rs_ratio_roc_20d <= cfg.min_rs_ratio_roc_20d:
        return None

    outperformance_5d = float(stock_return_5d - spy_return_5d)
    if weak_market:
        if outperformance_5d <= cfg.weak_market_min_outperformance_5d:
            return None
        if float(stock_return_5d) <= cfg.weak_market_min_stock_return_5d:
            return None

    breakout_rows = []
    recent = aligned_stock.tail(lookback)
    for index, candle in recent.iterrows():
        prior_52w_high = float(candle["Prior_52W_High"]) if pd.notna(candle["Prior_52W_High"]) else None
        prior_available_high = (
            float(candle["Prior_Available_High"]) if pd.notna(candle["Prior_Available_High"]) else None
        )
        breakout_levels = [value for value in [prior_52w_high, prior_available_high] if value is not None]
        if not breakout_levels or pd.isna(candle["Avg_Volume_20D"]) or float(candle["Avg_Volume_20D"]) <= 0:
            continue
        breakout_level = max(breakout_levels)
        volume_ratio = float(candle["Volume"] / candle["Avg_Volume_20D"])
        is_breakout = (
            (float(candle["High"]) > breakout_level or float(candle["Close"]) > breakout_level)
            and volume_ratio >= cfg.min_breakout_volume_ratio
            and float(candle["Close"]) > float(candle["Open"])
        )
        if is_breakout:
            breakout_rows.append((index, candle, breakout_level, volume_ratio))

    if not breakout_rows:
        return None

    breakout_index, breakout_candle, breakout_level, breakout_volume_ratio = breakout_rows[-1]
    latest_volume_ratio = float(row["Volume"] / row["Avg_Volume_20D"]) if row["Avg_Volume_20D"] else 0.0
    entry = float(breakout_candle["High"])
    stop = float(breakout_candle["Low"])
    if stop <= 0 or entry <= stop:
        return None
    risk = entry - stop
    target = entry + 2 * risk
    distance_to_9ema = (
        abs(float(row["Close"]) - float(row["9_EMA"])) / float(row["9_EMA"])
        if float(row["9_EMA"]) > 0
        else 1.0
    )
    days_since_breakout = len(aligned_stock.loc[breakout_index:])

    return {
        "ticker": ticker,
        "signal_date": str(breakout_index.date()),
        "latest_date": str(aligned_stock.index[-1].date()),
        "setup_type": "Recent ATH/52W high breakout",
        "breakout_date": str(breakout_index.date()),
        "days_since_breakout": int(days_since_breakout - 1),
        "close": round(float(row["Close"]), 2),
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target_2r": round(target, 2),
        "risk_reward": 2.0,
        "exit_rule": "Stop loss OR 9 EMA crosses below 20 SMA",
        "ema_9": round(float(row["9_EMA"]), 2),
        "distance_to_9ema_pct": round(distance_to_9ema * 100, 2),
        "breakout_high": round(float(breakout_candle["High"]), 2),
        "breakout_close": round(float(breakout_candle["Close"]), 2),
        "breakout_level": round(float(breakout_level), 2),
        "breakout_volume_ratio": round(breakout_volume_ratio, 2),
        "latest_volume_ratio": round(latest_volume_ratio, 2),
        "rsi_14": round(rsi, 1),
        "stock_return_20d_pct": round(float(stock_return_20d) * 100, 2),
        "spy_return_20d_pct": round(float(spy_return_20d) * 100, 2),
        "rs_20d_vs_spy_pct": round(rs_20d_vs_spy * 100, 2),
        "rs_ratio_roc_20d_pct": round(float(rs_ratio_roc_20d) * 100, 2),
        "stock_return_5d_pct": round(float(stock_return_5d) * 100, 2),
        "spy_return_5d_pct": round(float(spy_return_5d) * 100, 2),
        "outperformance_5d_pct": round(outperformance_5d * 100, 2),
        "weak_market_filter": "Passed",
        "ma_stack": "Close > 9 EMA > 20 SMA > 50 SMA > 200 SMA",
        "volume_support": "breakout volume supports/confirms uptrend",
        "reason": (
            f"Recent ATH/52W breakout {days_since_breakout - 1} days ago; "
            f"breakout volume {breakout_volume_ratio:.2f}x avg; RSI {rsi:.1f}; "
            f"RS 20D vs SPY {rs_20d_vs_spy:.1%}; 5D outperformance {outperformance_5d:.1%}."
        ),
    }


def run_technical_strength_scan(
    setup_kind: str = "both",
    universe: pd.DataFrame | None = None,
    universe_label: str = "$500M dataset",
) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    if setup_kind not in {"breakout", "pullback", "both"}:
        raise ValueError(f"Unknown technical strength setup kind: {setup_kind}")

    cfg = load_rule_config().technical_strength
    completed_date = latest_completed_us_session()
    scan_date = str(completed_date.date())
    universe = load_eligible_tickers() if universe is None else universe.copy()
    if universe.empty:
        return pd.DataFrame(), [], {"scan_date": scan_date, "stored_tickers": 0, "price_candidates": 0}

    tickers = sorted({str(ticker).strip().upper() for ticker in universe["ticker"].dropna() if str(ticker).strip()})
    market_caps = {
        str(row["ticker"]).strip().upper(): float(row["market_cap"])
        for _, row in universe.dropna(subset=["ticker"]).iterrows()
        if "market_cap" in universe.columns and pd.notna(row.get("market_cap"))
    }
    requested = sorted(set(tickers + ["SPY", "QQQ"]))
    price_data = download_ohlcv(requested, completed_date=completed_date)
    benchmark_data = {ticker: price_data.get(ticker, pd.DataFrame()) for ticker in ("SPY", "QQQ")}
    weak_market = _is_weak_market(benchmark_data)
    spy_prices = benchmark_data.get("SPY", pd.DataFrame())
    errors: list[str] = []
    candidates: list[dict[str, float | str]] = []
    breakout_candidates: list[dict[str, float | str]] = []
    pullback_candidates: list[dict[str, float | str]] = []

    if spy_prices is None or spy_prices.empty:
        return pd.DataFrame(), ["SPY: missing benchmark OHLCV data"], {
            "scan_date": scan_date,
            "stored_tickers": len(tickers),
            "ohlcv_loaded": len(price_data),
            "price_candidates": 0,
            "matches": 0,
        }

    for ticker in tickers:
        prices = price_data.get(ticker)
        if prices is None or prices.empty:
            errors.append(f"{ticker}: missing OHLCV data")
            continue
        if setup_kind in {"breakout", "both"}:
            breakout_candidate = _technical_strength_breakout_watch_candidate(
                ticker, prices, spy_prices, weak_market, scan_date, cfg
            )
            if breakout_candidate:
                breakout_candidate["market_cap"] = float(market_caps.get(ticker, 0))
                breakout_candidates.append(breakout_candidate)
                candidates.append(breakout_candidate)
        if setup_kind in {"pullback", "both"}:
            pullback_candidate = _technical_strength_candidate(ticker, prices, spy_prices, weak_market, scan_date, cfg)
            if pullback_candidate:
                pullback_candidate["market_cap"] = float(market_caps.get(ticker, 0))
                pullback_candidates.append(pullback_candidate)
                candidates.append(pullback_candidate)

    frame = pd.DataFrame(candidates)
    if not frame.empty:
        for column in ["breakout_volume_ratio", "volume_ratio", "market_cap"]:
            if column not in frame.columns:
                frame[column] = pd.NA
        frame["_sort_volume_ratio"] = pd.to_numeric(frame["breakout_volume_ratio"], errors="coerce").fillna(
            pd.to_numeric(frame["volume_ratio"], errors="coerce")
        )
        frame = frame.sort_values(
            by=["rs_20d_vs_spy_pct", "rs_ratio_roc_20d_pct", "_sort_volume_ratio", "market_cap"],
            ascending=[False, False, False, False],
        ).drop(columns=["_sort_volume_ratio"])

    stats = {
        "scan_date": scan_date,
        "stored_tickers": len(tickers),
        "ohlcv_loaded": len(price_data),
        "price_candidates": len(candidates),
        "breakout_candidates": len(breakout_candidates),
        "breakout_candidate_rows": breakout_candidates,
        "pullback_candidates": len(pullback_candidates),
        "pullback_candidate_rows": pullback_candidates,
        "matches": len(frame),
        "weak_market": weak_market,
        "setup_kind": setup_kind,
        "universe_label": universe_label,
    }
    return frame, errors, stats


def run_technical_breakout_scan() -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    return run_technical_strength_scan("breakout")


def run_technical_pullback_scan() -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    return run_technical_strength_scan("pullback")


def _latest_score_above_60_universe() -> tuple[pd.DataFrame, str]:
    runs = list_rule_watchlists()
    if runs.empty:
        return pd.DataFrame(), ""
    rows = runs[runs["rule_name"].astype(str).eq("Top 50 Score Above 60")].copy()
    if rows.empty:
        return pd.DataFrame(), ""
    rows["scan_date"] = rows["scan_date"].astype(str)
    latest_date = str(rows.sort_values(["scan_date", "saved_at"], ascending=[False, False]).iloc[0]["scan_date"])
    frame = load_rule_watchlist("Top 50 Score Above 60", latest_date)
    if frame.empty or "ticker" not in frame.columns:
        return pd.DataFrame(), latest_date
    out = frame.copy()
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    if "market_cap" not in out.columns:
        out["market_cap"] = pd.NA
    out["market_cap"] = pd.to_numeric(out["market_cap"], errors="coerce")
    out = out.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"])
    return out[["ticker", "market_cap"]], latest_date


def run_score_above60_setup_scan(setup_kind: str = "both") -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    universe, score_date = _latest_score_above_60_universe()
    if universe.empty:
        return pd.DataFrame(), ["Top 50 Score Above 60 watchlist is empty or missing."], {
            "scan_date": str(latest_completed_us_session().date()),
            "score_above60_scan_date": score_date,
            "stored_tickers": 0,
            "matches": 0,
            "setup_kind": setup_kind,
            "universe_label": "Score Above 60",
        }
    results, errors, stats = run_technical_strength_scan(
        setup_kind=setup_kind,
        universe=universe,
        universe_label=f"Score Above 60 from {score_date}",
    )
    stats["score_above60_scan_date"] = score_date
    stats["score_above60_tickers"] = len(universe)
    return results, errors, stats


def _weekly_close_from_daily(prices: pd.DataFrame) -> pd.Series:
    if prices.empty or "Close" not in prices.columns:
        return pd.Series(dtype=float)
    frame = prices.copy()
    frame.index = pd.to_datetime(frame.index)
    return frame["Close"].resample("W-FRI").last().dropna()


def _latest_earnings_points(ticker: str) -> dict[str, object]:
    output: dict[str, object] = {
        "eps_surprise_pct": None,
        "eps_surprise_points": 0.0,
        "latest_reported_eps": None,
        "latest_eps_period": "",
        "same_quarter_last_year_eps": None,
        "same_quarter_last_year_period": "",
        "eps_yoy_growth_pct": None,
        "eps_yoy_points": 0.0,
        "earnings_error": "",
    }
    try:
        earnings = yf.Ticker(ticker).get_earnings_dates(limit=8)
    except Exception as exc:
        output["earnings_error"] = str(exc)
        return output

    if not isinstance(earnings, pd.DataFrame) or earnings.empty:
        output["earnings_error"] = "No earnings rows"
        return output

    frame = earnings.copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame[~frame.index.isna()].sort_index(ascending=False)
    if "Reported EPS" not in frame.columns:
        output["earnings_error"] = "Reported EPS missing"
        return output
    frame["Reported EPS"] = pd.to_numeric(frame["Reported EPS"], errors="coerce")
    frame = frame.dropna(subset=["Reported EPS"])
    if frame.empty:
        output["earnings_error"] = "Reported EPS unavailable"
        return output

    latest = frame.iloc[0]
    latest_date = frame.index[0]
    output["latest_reported_eps"] = round(float(latest["Reported EPS"]), 4)
    output["latest_eps_period"] = f"Q{((latest_date.month - 1) // 3) + 1} {latest_date.year}"

    surprise_value = None
    for surprise_column in ("Surprise(%)", "Surprise %", "EPS Surprise"):
        if surprise_column in frame.columns:
            surprise_value = pd.to_numeric(latest.get(surprise_column), errors="coerce")
            break
    if surprise_value is not None and pd.notna(surprise_value):
        surprise_pct = float(surprise_value)
        if abs(surprise_pct) <= 1:
            surprise_pct *= 100
        output["eps_surprise_pct"] = round(surprise_pct, 2)
        if surprise_pct >= 9:
            output["eps_surprise_points"] = round(surprise_pct / 100, 4)

    if len(frame) >= 5:
        prior = frame.iloc[4]
        prior_date = frame.index[4]
        latest_eps = float(latest["Reported EPS"])
        prior_eps = float(prior["Reported EPS"])
        output["same_quarter_last_year_eps"] = round(prior_eps, 4)
        output["same_quarter_last_year_period"] = f"Q{((prior_date.month - 1) // 3) + 1} {prior_date.year}"
        if prior_eps != 0:
            growth_pct = ((latest_eps - prior_eps) / abs(prior_eps)) * 100
            output["eps_yoy_growth_pct"] = round(growth_pct, 2)
            if growth_pct > 50:
                output["eps_yoy_points"] = round(growth_pct / 100, 4)
    else:
        output["earnings_error"] = "Need 5 EPS rows for YoY quarter comparison"
    return output


def _is_likely_common_stock_symbol(ticker: str) -> bool:
    clean = str(ticker or "").strip().upper()
    if not clean or not clean.replace("-", "").isalpha():
        return False
    suffixes = ("W", "WS", "WT", "WTS", "U", "UN", "RT", "R")
    if len(clean) > 4 and any(clean.endswith(suffix) for suffix in suffixes):
        return False
    preferred_markers = ("-P", "-PA", "-PB", "-PC", "-PD", "-PE", "-PF", "-PG", "-PH", "-PI", "-PJ", "-PK")
    return not any(marker in clean for marker in preferred_markers)


def _weekly_ohlcv_from_daily(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    frame = prices.copy()
    frame.index = pd.to_datetime(frame.index)
    weekly = frame.resample("W-FRI").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    weekly = weekly.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    weekly["EMA_9W"] = weekly["Close"].ewm(span=9, adjust=False).mean()
    weekly["EMA_20W"] = weekly["Close"].ewm(span=20, adjust=False).mean()
    weekly["SMA_20W"] = weekly["Close"].rolling(20).mean()
    weekly["SMA_50W"] = weekly["Close"].rolling(50).mean()
    weekly["RSI_14W"] = _rsi(weekly["Close"], 14)
    return weekly


def _atr_pct(df: pd.DataFrame, period: int = 14) -> float | None:
    if df.empty or len(df) <= period:
        return None
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prior_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prior_close).abs(),
            (low - prior_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(period).mean().iloc[-1]
    latest_close = close.iloc[-1]
    if pd.isna(atr) or latest_close <= 0:
        return None
    return float(atr / latest_close)


def _tier_score(value: float, tiers: list[tuple[float, float]]) -> float:
    score = 0.0
    for threshold, points in tiers:
        if value > threshold:
            score = points
    return score


def _safe_ratio_growth(current: object, prior: object) -> float | None:
    current_value = pd.to_numeric(current, errors="coerce")
    prior_value = pd.to_numeric(prior, errors="coerce")
    if pd.isna(current_value) or pd.isna(prior_value) or float(prior_value) == 0:
        return None
    return (float(current_value) - float(prior_value)) / abs(float(prior_value))


def _row_value(frame: pd.DataFrame, row_names: list[str], column_index: int) -> float | None:
    if not isinstance(frame, pd.DataFrame) or frame.empty or frame.shape[1] <= column_index:
        return None
    for row_name in row_names:
        if row_name in frame.index:
            value = pd.to_numeric(frame.iloc[frame.index.get_loc(row_name), column_index], errors="coerce")
            if pd.notna(value):
                return float(value)
    return None


def _eps_value_from_income(income: pd.DataFrame, column_index: int) -> float | None:
    eps = _row_value(income, ["Diluted EPS", "DilutedEPS", "Basic EPS", "BasicEPS", "Normalized Diluted EPS"], column_index)
    if eps is not None:
        return eps
    net_income = _row_value(income, ["Net Income", "NetIncome"], column_index)
    shares = _row_value(income, ["Diluted Average Shares", "DilutedAverageShares", "Basic Average Shares"], column_index)
    if net_income is not None and shares not in (None, 0):
        return net_income / shares
    return None


def _free_cash_flow_from_cashflow(cashflow: pd.DataFrame, column_index: int) -> float | None:
    fcf = _row_value(cashflow, ["Free Cash Flow", "FreeCashFlow"], column_index)
    if fcf is not None:
        return fcf
    operating = _row_value(cashflow, ["Operating Cash Flow", "OperatingCashFlow", "Total Cash From Operating Activities"], column_index)
    capex = _row_value(cashflow, ["Capital Expenditure", "CapitalExpenditure", "Capital Expenditures"], column_index)
    if operating is not None and capex is not None:
        return operating + capex
    return None


def _latest_earnings_surprise_pct(stock: yf.Ticker) -> float | None:
    try:
        earnings = stock.get_earnings_dates(limit=8)
    except Exception:
        return None
    if not isinstance(earnings, pd.DataFrame) or earnings.empty:
        return None
    frame = earnings.copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame[~frame.index.isna()].sort_index(ascending=False)
    for surprise_column in ("Surprise(%)", "Surprise %", "EPS Surprise"):
        if surprise_column in frame.columns:
            value = pd.to_numeric(frame.iloc[0].get(surprise_column), errors="coerce")
            if pd.notna(value):
                surprise_pct = float(value)
                return surprise_pct * 100 if abs(surprise_pct) <= 1 else surprise_pct
    return None


def _top50_fundamental_points(result: dict[str, object]) -> float:
    points = 0.0
    revenue_yoy = pd.to_numeric(result.get("revenue_yoy_growth_pct"), errors="coerce")
    revenue_qoq = pd.to_numeric(result.get("revenue_qoq_growth_pct"), errors="coerce")
    eps_yoy = pd.to_numeric(result.get("eps_yoy_growth_pct"), errors="coerce")
    eps_qoq = pd.to_numeric(result.get("eps_qoq_growth_pct"), errors="coerce")
    eps_surprise = pd.to_numeric(result.get("eps_surprise_pct"), errors="coerce")
    roe = pd.to_numeric(result.get("roe_pct"), errors="coerce")
    points += 5 if pd.notna(revenue_yoy) and float(revenue_yoy) > 25 else 0
    points += 3 if pd.notna(revenue_qoq) and float(revenue_qoq) > 20 else 0
    points += 5 if pd.notna(eps_yoy) and float(eps_yoy) > 25 else 0
    points += 3 if pd.notna(eps_qoq) and float(eps_qoq) > 20 else 0
    points += 3 if pd.notna(eps_surprise) and float(eps_surprise) > 10 else 0
    points += 3 if pd.notna(roe) and float(roe) > 20 else 0
    points += 3 if bool(result.get("operating_margin_improving")) else 0
    points += 1 if bool(result.get("free_cash_flow_positive_or_improving")) else 0
    points += 1 if bool(result.get("low_debt_strong_balance_sheet")) else 0
    points += 3 if bool(result.get("positive_guidance")) else 0
    return round(points, 2)


def _fundamental_scorecard(ticker: str, use_fmp: bool = True) -> dict[str, object]:
    result: dict[str, object] = {
        "fundamental_score": 0.0,
        "revenue_yoy_growth_pct": None,
        "revenue_qoq_growth_pct": None,
        "eps_yoy_growth_pct": None,
        "eps_qoq_growth_pct": None,
        "eps_surprise_pct": None,
        "growth_star": "",
        "revenue_growth_trend_up": False,
        "eps_growth_trend_up": False,
        "revenue_qoq_growth_trend": [],
        "eps_qoq_growth_trend": [],
        "revenue_quarter_values": [],
        "eps_quarter_values": [],
        "roe_pct": None,
        "operating_margin_improving": False,
        "free_cash_flow_positive_or_improving": False,
        "low_debt_strong_balance_sheet": False,
        "positive_guidance": False,
        "spread_pct": None,
        "fundamental_source": "yfinance",
        "fundamental_error": "",
    }
    try:
        stock = yf.Ticker(ticker)
        info = stock.get_info()
    except Exception as exc:
        result["fundamental_error"] = str(exc)
        info = {}
        stock = yf.Ticker(ticker)

    try:
        income = stock.get_income_stmt(freq="quarterly")
    except Exception:
        income = pd.DataFrame()
    try:
        cashflow = stock.get_cashflow(freq="quarterly")
    except Exception:
        cashflow = pd.DataFrame()
    try:
        balance = stock.get_balance_sheet(freq="quarterly")
    except Exception:
        balance = pd.DataFrame()

    revenue_latest = _row_value(income, ["Total Revenue", "TotalRevenue"], 0)
    revenue_prior_q = _row_value(income, ["Total Revenue", "TotalRevenue"], 1)
    revenue_prior_year = _row_value(income, ["Total Revenue", "TotalRevenue"], 4)
    revenue_yoy = _safe_ratio_growth(revenue_latest, revenue_prior_year)
    revenue_qoq = _safe_ratio_growth(revenue_latest, revenue_prior_q)
    if revenue_yoy is not None:
        result["revenue_yoy_growth_pct"] = round(revenue_yoy * 100, 2)
    if revenue_qoq is not None:
        result["revenue_qoq_growth_pct"] = round(revenue_qoq * 100, 2)

    eps_latest = _eps_value_from_income(income, 0)
    eps_prior_q = _eps_value_from_income(income, 1)
    eps_prior_year = _eps_value_from_income(income, 4)
    eps_yoy = _safe_ratio_growth(eps_latest, eps_prior_year)
    eps_qoq = _safe_ratio_growth(eps_latest, eps_prior_q)
    if eps_yoy is not None:
        result["eps_yoy_growth_pct"] = round(eps_yoy * 100, 2)
    if eps_qoq is not None:
        result["eps_qoq_growth_pct"] = round(eps_qoq * 100, 2)

    surprise_pct = _latest_earnings_surprise_pct(stock)
    if surprise_pct is not None:
        result["eps_surprise_pct"] = round(surprise_pct, 2)

    roe = pd.to_numeric(info.get("returnOnEquity"), errors="coerce") if info else pd.NA
    if pd.notna(roe):
        roe_pct = float(roe) * 100
        result["roe_pct"] = round(roe_pct, 2)

    operating_margin_latest = None
    operating_margin_prior = None
    operating_income_latest = _row_value(income, ["Operating Income", "OperatingIncome"], 0)
    operating_income_prior = _row_value(income, ["Operating Income", "OperatingIncome"], 1)
    if revenue_latest not in (None, 0) and operating_income_latest is not None:
        operating_margin_latest = operating_income_latest / revenue_latest
    if revenue_prior_q not in (None, 0) and operating_income_prior is not None:
        operating_margin_prior = operating_income_prior / revenue_prior_q
    ebitda_latest = _row_value(income, ["EBITDA", "Normalized EBITDA"], 0)
    ebitda_prior = _row_value(income, ["EBITDA", "Normalized EBITDA"], 1)
    margin_improving = (
        operating_margin_latest is not None
        and operating_margin_prior is not None
        and operating_margin_latest > operating_margin_prior
    ) or (ebitda_latest is not None and ebitda_prior is not None and ebitda_latest > ebitda_prior)
    result["operating_margin_improving"] = bool(margin_improving)

    fcf_latest = _free_cash_flow_from_cashflow(cashflow, 0)
    fcf_prior = _free_cash_flow_from_cashflow(cashflow, 1)
    fcf_good = (fcf_latest is not None and fcf_latest > 0) or (
        fcf_latest is not None and fcf_prior is not None and fcf_latest > fcf_prior
    )
    result["free_cash_flow_positive_or_improving"] = bool(fcf_good)

    debt_to_equity = pd.to_numeric(info.get("debtToEquity"), errors="coerce") if info else pd.NA
    total_debt = _row_value(balance, ["Total Debt", "TotalDebt"], 0)
    cash = _row_value(balance, ["Cash And Cash Equivalents", "CashAndCashEquivalents", "Cash Cash Equivalents And Short Term Investments"], 0)
    low_debt = (pd.notna(debt_to_equity) and float(debt_to_equity) < 100) or (
        total_debt is not None and cash is not None and total_debt <= cash * 2
    )
    result["low_debt_strong_balance_sheet"] = bool(low_debt)

    bid = pd.to_numeric(info.get("bid"), errors="coerce") if info else pd.NA
    ask = pd.to_numeric(info.get("ask"), errors="coerce") if info else pd.NA
    if pd.notna(bid) and pd.notna(ask) and float(bid) > 0 and float(ask) > float(bid):
        spread_pct = (float(ask) - float(bid)) / ((float(ask) + float(bid)) / 2)
        result["spread_pct"] = round(spread_pct * 100, 3)

    fmp_result = get_fmp_top50_scorecard(ticker) if use_fmp else {}
    if fmp_result:
        for key in [
            "revenue_yoy_growth_pct",
            "revenue_qoq_growth_pct",
            "eps_yoy_growth_pct",
            "eps_qoq_growth_pct",
            "eps_surprise_pct",
            "revenue_growth_trend_up",
            "eps_growth_trend_up",
            "revenue_qoq_growth_trend",
            "eps_qoq_growth_trend",
            "revenue_quarter_values",
            "eps_quarter_values",
        ]:
            if fmp_result.get(key) is not None:
                result[key] = fmp_result[key]
        trend_count = int(bool(result.get("eps_growth_trend_up"))) + int(bool(result.get("revenue_growth_trend_up")))
        result["growth_star"] = "★★" if trend_count == 2 else "★" if trend_count == 1 else ""
        result["fundamental_source"] = fmp_result.get("fundamental_source") or "FMP"
        if fmp_result.get("fundamental_error"):
            result["fundamental_error"] = fmp_result["fundamental_error"]

    result["fundamental_score"] = _top50_fundamental_points(result)
    return result


def run_top50_strength_score_scan(
    earnings_check_limit: int | None = None,
    max_tickers: int | None = None,
    status_callback: Callable[[dict[str, object]], None] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    score_model_version = "top50_100_v6_weekly_rsi_1b"
    top50_min_market_cap = 1_000_000_000
    completed_date = latest_completed_us_session()
    scan_date = str(completed_date.date())
    universe = load_eligible_tickers()
    if universe.empty:
        return pd.DataFrame(), [], {
            "scan_date": scan_date,
            "score_model_version": score_model_version,
            "stored_tickers": 0,
            "ohlcv_loaded": 0,
            "earnings_checked": 0,
            "matches": 0,
        }

    tickers = universe["ticker"].astype(str).str.upper().drop_duplicates().tolist()
    if max_tickers is not None:
        tickers = tickers[: max(0, int(max_tickers))]
    market_caps = dict(zip(universe["ticker"].astype(str).str.upper(), universe["market_cap"], strict=False))

    universe_path = _write_weekly_v6_universe_file(universe)
    sector_cache_path = DATA_DIR / "weekly_v6_sector_etf_cache.csv"
    sector_map: dict[str, dict[str, str]] = {}
    try:
        import weekly_v8_rs5_52w_ath_sma20_no_cup_breakout_backtest as weekly_breakout_strat

        sector_map = weekly_breakout_strat.build_sector_etf_map(universe_path, tickers, sector_cache_path)
    except Exception:
        sector_map = {}
    sector_etfs = sorted({data.get("Sector ETF", "") for data in sector_map.values() if data.get("Sector ETF")})

    requested = sorted(set(tickers + ["SPY"] + sector_etfs))
    price_data = download_ohlcv(requested, completed_date=completed_date)
    spy_prices = price_data.get("SPY", pd.DataFrame())
    if spy_prices is None or spy_prices.empty:
        return pd.DataFrame(), ["SPY: missing benchmark OHLCV data"], {
            "scan_date": scan_date,
            "score_model_version": score_model_version,
            "stored_tickers": len(tickers),
            "ohlcv_loaded": len(price_data),
            "hard_filter_removed": 0,
            "rsi_filtered": 0,
            "earnings_checked": 0,
            "matches": 0,
        }

    spy_daily = _add_technical_strength_indicators(spy_prices).dropna(subset=["RSI_20"])
    spy_weekly_close = _weekly_close_from_daily(spy_prices)
    if spy_daily.empty or len(spy_weekly_close) <= 5:
        return pd.DataFrame(), ["SPY: not enough indicator data"], {
            "scan_date": scan_date,
            "score_model_version": score_model_version,
            "stored_tickers": len(tickers),
            "ohlcv_loaded": len(price_data),
            "hard_filter_removed": 0,
            "rsi_filtered": 0,
            "earnings_checked": 0,
            "matches": 0,
        }
    spy_rsi_20 = float(spy_daily.iloc[-1]["RSI_20"])
    spy_weekly_return_5w = _pct_change(spy_weekly_close, 5)
    spy_weekly_return_13w = _pct_change(spy_weekly_close, 13)
    spy_weekly_return_26w = _pct_change(spy_weekly_close, 26)
    spy_daily_return_20d = _pct_change(spy_daily["Close"], 20)

    sector_weekly_returns: dict[str, float | None] = {}
    sector_rsi_values: dict[str, float | None] = {}
    for etf in sector_etfs:
        sector_prices = price_data.get(etf, pd.DataFrame())
        sector_weekly_returns[etf] = _pct_change(_weekly_close_from_daily(sector_prices), 5)
        sector_daily = _add_technical_strength_indicators(sector_prices).dropna(subset=["RSI_20"]) if not sector_prices.empty else pd.DataFrame()
        sector_rsi_values[etf] = float(sector_daily.iloc[-1]["RSI_20"]) if not sector_daily.empty else None

    errors: list[str] = []
    rows: list[dict[str, object]] = []
    common_stock_filtered = 0
    hard_filter_removed = 0
    rsi_filtered = 0
    for ticker in tickers:
        if not _is_likely_common_stock_symbol(ticker):
            common_stock_filtered += 1
            continue
        prices = price_data.get(ticker, pd.DataFrame())
        if prices is None or prices.empty:
            errors.append(f"{ticker}: missing OHLCV data")
            continue
        daily = _add_technical_strength_indicators(prices).dropna(
            subset=["9_EMA", "20_EMA", "50_SMA", "200_SMA", "RSI_20"]
        )
        weekly_close = _weekly_close_from_daily(prices)
        weekly = _weekly_ohlcv_from_daily(prices)
        latest_week = weekly.iloc[-1] if not weekly.empty else pd.Series(dtype=float)
        weekly_rsi_14 = pd.to_numeric(latest_week.get("RSI_14W"), errors="coerce") if not latest_week.empty else pd.NA
        if daily.empty or len(weekly_close) <= 5 or pd.isna(weekly_rsi_14):
            errors.append(f"{ticker}: not enough indicator data")
            continue
        row = daily.iloc[-1]
        close = float(row["Close"])
        avg_volume = float(row["Avg_Volume_20D"]) if pd.notna(row.get("Avg_Volume_20D")) else 0.0
        dollar_volume = close * avg_volume
        market_cap = float(market_caps.get(ticker, 0) or 0)
        if (
            close <= 10
            or market_cap <= top50_min_market_cap
            or avg_volume <= 1_000_000
            or dollar_volume <= 20_000_000
            or pd.isna(row.get("50_SMA"))
            or pd.isna(row.get("200_SMA"))
            or close <= float(row["50_SMA"])
            or close <= float(row["200_SMA"])
            or float(weekly_rsi_14) <= 50
        ):
            hard_filter_removed += 1
            continue

        stock_weekly_return_5w = _pct_change(weekly_close, 5)
        stock_weekly_return_13w = _pct_change(weekly_close, 13)
        stock_weekly_return_26w = _pct_change(weekly_close, 26)
        stock_daily_return_20d = _pct_change(daily["Close"], 20)
        rs_ratio_roc_5w = None
        aligned_weekly = pd.concat(
            [weekly_close.rename("stock"), spy_weekly_close.rename("spy")],
            axis=1,
            join="inner",
        ).dropna()
        if len(aligned_weekly) > 5:
            rs_ratio_roc_5w = _pct_change(aligned_weekly["stock"] / aligned_weekly["spy"].replace(0, pd.NA), 5)

        sector_info = sector_map.get(ticker, {})
        sector_etf = str(sector_info.get("Sector ETF") or "")
        sector_rsi = sector_rsi_values.get(sector_etf)
        sector_return_5w = sector_weekly_returns.get(sector_etf)
        rs_5w_vs_spy = (
            float(stock_weekly_return_5w - spy_weekly_return_5w)
            if stock_weekly_return_5w is not None and spy_weekly_return_5w is not None
            else None
        )
        sector_rs_5w_vs_spy = (
            float(sector_return_5w - spy_weekly_return_5w)
            if sector_return_5w is not None and spy_weekly_return_5w is not None
            else None
        )
        stock_rs_5w_vs_sector = (
            float(stock_weekly_return_5w - sector_return_5w)
            if stock_weekly_return_5w is not None and sector_return_5w is not None
            else None
        )

        rsi_20 = float(row["RSI_20"])
        rsi_q = rsi_20 - spy_rsi_20
        if rsi_q <= 0:
            rsi_filtered += 1
            continue

        sector_rsi_q = rsi_20 - sector_rsi if sector_rsi is not None else None
        market_sector_raw = 0.0
        market_sector_raw += _tier_score(rsi_q, [(5, 3), (10, 4), (15, 6), (20, 7)])
        if sector_rsi_q is not None:
            market_sector_raw += _tier_score(sector_rsi_q, [(5, 1), (10, 2), (15, 3)])
        if rsi_20 > 50:
            market_sector_raw += 2
        if sector_rsi is not None and sector_rsi > 50:
            market_sector_raw += 1
        market_sector_score = round((market_sector_raw / 13) * 30, 2)

        technical_score = 0.0
        technical_score += 2 if close > float(row["50_SMA"]) else 0
        technical_score += 2 if close > float(row["200_SMA"]) else 0
        technical_score += 2 if float(row["50_SMA"]) > float(row["200_SMA"]) else 0
        weekly_trend_pass = bool(
            not latest_week.empty
            and pd.notna(latest_week.get("EMA_9W"))
            and pd.notna(latest_week.get("EMA_20W"))
            and float(latest_week["Close"]) > float(latest_week["EMA_9W"]) > float(latest_week["EMA_20W"])
        )
        technical_score += 3 if weekly_trend_pass else 0
        weekly_20_gt_50 = bool(
            not latest_week.empty
            and pd.notna(latest_week.get("SMA_20W"))
            and pd.notna(latest_week.get("SMA_50W"))
            and float(latest_week["SMA_20W"]) > float(latest_week["SMA_50W"])
        )
        technical_score += 3 if weekly_20_gt_50 else 0
        sma20_not_extended = bool(
            not latest_week.empty
            and pd.notna(latest_week.get("SMA_20W"))
            and pd.notna(latest_week.get("SMA_50W"))
            and float(latest_week["SMA_50W"]) > 0
            and float(latest_week["SMA_20W"]) <= float(latest_week["SMA_50W"]) * 1.15
        )
        technical_score += 1 if sma20_not_extended else 0
        high_52w = float(daily["High"].tail(252).max())
        all_time_proxy_high = float(daily["High"].cummax().iloc[-1])
        high_reference = max(high_52w, all_time_proxy_high)
        near_high = high_reference > 0 and close >= high_reference * 0.85
        technical_score += 2 if near_high else 0

        relative_strength_score = 0.0
        rs_13w_vs_spy = (
            float(stock_weekly_return_13w - spy_weekly_return_13w)
            if stock_weekly_return_13w is not None and spy_weekly_return_13w is not None
            else None
        )
        rs_26w_vs_spy = (
            float(stock_weekly_return_26w - spy_weekly_return_26w)
            if stock_weekly_return_26w is not None and spy_weekly_return_26w is not None
            else None
        )
        rs_20d_vs_spy = (
            float(stock_daily_return_20d - spy_daily_return_20d)
            if stock_daily_return_20d is not None and spy_daily_return_20d is not None
            else None
        )
        relative_strength_score += 4 if rs_5w_vs_spy is not None and rs_5w_vs_spy > 0.05 else 0
        relative_strength_score += 4 if rs_13w_vs_spy is not None and rs_13w_vs_spy > 0.05 else 0
        relative_strength_score += 3 if rs_26w_vs_spy is not None and rs_26w_vs_spy > 0.05 else 0
        relative_strength_score += 3 if rs_20d_vs_spy is not None and rs_20d_vs_spy > 0.05 else 0
        relative_strength_score += 3 if stock_rs_5w_vs_sector is not None and stock_rs_5w_vs_sector > 0.05 else 0
        aligned_daily_rs = pd.concat(
            [daily["Close"].rename("stock"), spy_daily["Close"].rename("spy")],
            axis=1,
            join="inner",
        ).dropna()
        rs_line_near_high = False
        if not aligned_daily_rs.empty:
            rs_line = aligned_daily_rs["stock"] / aligned_daily_rs["spy"].replace(0, pd.NA)
            rs_high = float(rs_line.tail(252).max())
            rs_latest = float(rs_line.iloc[-1])
            rs_line_near_high = rs_high > 0 and rs_latest >= rs_high * 0.95
        relative_strength_score += 2 if rs_line_near_high else 0
        recent = daily.tail(40)
        prior = daily.iloc[-80:-40]
        hh_hl = bool(
            not recent.empty
            and not prior.empty
            and float(recent["High"].max()) > float(prior["High"].max())
            and float(recent["Low"].min()) > float(prior["Low"].min())
        )
        relative_strength_score += 1 if hh_hl else 0

        atr_pct = _atr_pct(daily)
        liquidity_score = 0.0
        liquidity_score += 1 if avg_volume > 1_000_000 else 0
        liquidity_score += 1 if dollar_volume > 50_000_000 else 0
        liquidity_score += 1 if atr_pct is not None and atr_pct <= 0.08 else 0
        liquidity_score += 1 if market_cap > 2_000_000_000 else 0

        rows.append(
            {
                "ticker": ticker,
                "company": "",
                "close": round(close, 2),
                "market_cap": market_cap,
                "avg_volume_20d": round(avg_volume, 0),
                "dollar_volume": round(dollar_volume, 0),
                "20d_rsi": round(rsi_20, 2),
                "weekly_rsi_14": round(float(weekly_rsi_14), 2),
                "spy_20d_rsi": round(spy_rsi_20, 2),
                "rsi_q": round(rsi_q, 2),
                "sector_rsi": round(sector_rsi, 2) if sector_rsi is not None else None,
                "sector_rsi_q": round(sector_rsi_q, 2) if sector_rsi_q is not None else None,
                "market_sector_score": market_sector_score,
                "technical_trend_score": round(technical_score, 2),
                "relative_strength_score": round(relative_strength_score, 2),
                "fundamental_score": 0.0,
                "liquidity_risk_score": round(liquidity_score, 2),
                "rs_5w_vs_spy_pct": round(rs_5w_vs_spy * 100, 2) if rs_5w_vs_spy is not None else None,
                "rs_13w_vs_spy_pct": round(rs_13w_vs_spy * 100, 2) if rs_13w_vs_spy is not None else None,
                "rs_26w_vs_spy_pct": round(rs_26w_vs_spy * 100, 2) if rs_26w_vs_spy is not None else None,
                "rs_20d_vs_spy_pct": round(rs_20d_vs_spy * 100, 2) if rs_20d_vs_spy is not None else None,
                "sector_rs_5w_vs_spy_pct": round(sector_rs_5w_vs_spy * 100, 2) if sector_rs_5w_vs_spy is not None else None,
                "stock_rs_5w_vs_sector_pct": round(stock_rs_5w_vs_sector * 100, 2) if stock_rs_5w_vs_sector is not None else None,
                "rs_line_near_52w_high": rs_line_near_high,
                "hh_hl_structure": hh_hl,
                "weekly_close_gt_9ema_gt_20ema": weekly_trend_pass,
                "weekly_20sma_gt_50sma": weekly_20_gt_50,
                "sma20_less_than_15pct_above_sma50": sma20_not_extended,
                "near_52w_or_ath": near_high,
                "atr_pct": round(atr_pct * 100, 2) if atr_pct is not None else None,
                "spread_pct": None,
                "spread_small": False,
            "eps_surprise_pct": None,
            "growth_star": "",
            "revenue_growth_trend_up": False,
            "eps_growth_trend_up": False,
            "revenue_qoq_growth_trend": [],
            "eps_qoq_growth_trend": [],
            "revenue_quarter_values": [],
            "eps_quarter_values": [],
            "revenue_yoy_growth_pct": None,
                "revenue_qoq_growth_pct": None,
                "eps_yoy_growth_pct": None,
                "eps_qoq_growth_pct": None,
                "roe_pct": None,
                "operating_margin_improving": False,
                "free_cash_flow_positive_or_improving": False,
                "low_debt_strong_balance_sheet": False,
            "positive_guidance": False,
            "fundamental_source": "",
            "total_points": round(market_sector_score + technical_score + relative_strength_score + liquidity_score, 2),
                "sector": str(sector_info.get("Sector") or ""),
                "sector_etf": sector_etf,
                "latest_date": str(daily.index[-1].date()),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, errors, {
            "scan_date": scan_date,
            "score_model_version": score_model_version,
            "stored_tickers": len(tickers),
            "ohlcv_loaded": len(price_data),
            "common_stock_filtered": common_stock_filtered,
            "hard_filter_removed": hard_filter_removed,
            "rsi_filtered": rsi_filtered,
            "earnings_checked": 0,
            "matches": 0,
        }

    frame = frame.sort_values(
        ["total_points", "market_sector_score", "relative_strength_score", "technical_trend_score", "market_cap"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    if status_callback:
        status_callback(
            {
                "phase": "price_rs_complete",
                "stored_tickers": len(tickers),
                "ohlcv_loaded": len(price_data),
                "qualified_rows": len(frame),
                "common_stock_filtered": common_stock_filtered,
                "hard_filter_removed": hard_filter_removed,
                "rsi_filtered": rsi_filtered,
                "message": "Hard filters complete. Checking fundamentals for qualified candidates.",
            }
        )

    earnings_limit = None if earnings_check_limit is None else max(0, int(earnings_check_limit))
    earnings_candidates = frame.copy() if earnings_limit is None else frame.head(earnings_limit).copy()
    earnings_checked = 0
    fmp_enriched = 0
    total_earnings_to_check = len(earnings_candidates)
    for index, candidate in earnings_candidates.iterrows():
        ticker = str(candidate["ticker"])
        fundamentals = _fundamental_scorecard(ticker, use_fmp=True)
        if str(fundamentals.get("fundamental_source") or "").upper().startswith("FMP"):
            fmp_enriched += 1
        earnings_checked += 1
        for key, value in fundamentals.items():
            frame.at[index, key] = value
        spread_pct = fundamentals.get("spread_pct")
        spread_small = spread_pct is not None and float(spread_pct) <= 1.0
        frame.loc[index, "spread_small"] = bool(spread_small)
        liquidity_score = float(frame.loc[index, "liquidity_risk_score"] or 0)
        if spread_small:
            liquidity_score = min(liquidity_score + 1, 5)
        frame.loc[index, "liquidity_risk_score"] = round(liquidity_score, 2)
        frame.loc[index, "total_points"] = round(
            float(frame.loc[index, "market_sector_score"] or 0)
            + float(frame.loc[index, "technical_trend_score"] or 0)
            + float(frame.loc[index, "relative_strength_score"] or 0)
            + float(frame.loc[index, "fundamental_score"] or 0)
            + float(frame.loc[index, "liquidity_risk_score"] or 0),
            2,
        )
        if status_callback and (earnings_checked == 1 or earnings_checked % 50 == 0 or earnings_checked == total_earnings_to_check):
            status_callback(
                {
                    "phase": "earnings",
                    "stored_tickers": len(tickers),
                    "ohlcv_loaded": len(price_data),
                    "qualified_rows": len(frame),
                    "common_stock_filtered": common_stock_filtered,
                    "hard_filter_removed": hard_filter_removed,
                    "rsi_filtered": rsi_filtered,
                    "earnings_checked": earnings_checked,
                    "earnings_total": total_earnings_to_check,
                    "message": f"Checking earnings: {earnings_checked:,} of {total_earnings_to_check:,}.",
                }
            )

    scored_frame = frame.sort_values(
        [
            "total_points",
            "market_sector_score",
            "fundamental_score",
            "relative_strength_score",
            "rs_5w_vs_spy_pct",
            "market_cap",
        ],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)
    above_60_frame = scored_frame[pd.to_numeric(scored_frame["total_points"], errors="coerce") >= 60].copy()
    above_60_frame.insert(0, "rank", range(1, len(above_60_frame) + 1))
    frame = scored_frame.head(50).reset_index(drop=True)
    frame.insert(0, "rank", range(1, len(frame) + 1))

    stats = {
        "scan_date": scan_date,
        "score_model_version": score_model_version,
        "stored_tickers": len(tickers),
        "ohlcv_loaded": len(price_data),
        "sector_mapped": len(sector_map),
        "sector_etfs": len(sector_etfs),
        "common_stock_filtered": common_stock_filtered,
        "hard_filter_removed": hard_filter_removed,
        "rsi_filtered": rsi_filtered,
        "qualified_before_fundamentals": len(rows),
        "earnings_checked": earnings_checked,
        "fmp_enriched": fmp_enriched,
        "fmp_call_budget": None,
        "fmp_rate_limit_status": getattr(
            fundamentals_module,
            "get_fmp_rate_limit_status",
            lambda: {"configured_keys": 0, "rate_limited_keys": 0, "available_keys": 0},
        )(),
        "min_market_cap": top50_min_market_cap,
        "above60_count": len(above_60_frame),
        "above60_rows": above_60_frame.to_dict("records") if not above_60_frame.empty else [],
        "matches": len(frame),
        "error_count": len(errors),
    }
    return frame, errors, stats


def _write_weekly_v6_universe_file(eligible: pd.DataFrame) -> Path:
    universe_path = DATA_DIR / "weekly_v6_500m_universe.csv"
    out = eligible.copy()
    out["Ticker"] = out["ticker"].astype(str).str.upper()
    if "market_cap" in out.columns:
        out["Market Cap"] = pd.to_numeric(out["market_cap"], errors="coerce")

    sector_cache_path = DATA_DIR / "sector_profile_cache.json"
    if sector_cache_path.exists():
        try:
            import json

            sector_cache = json.loads(sector_cache_path.read_text(encoding="utf-8"))
            if isinstance(sector_cache, dict):
                out["Sector"] = out["Ticker"].map(
                    lambda ticker: str(sector_cache.get(ticker, {}).get("sector") or "")
                    if isinstance(sector_cache.get(ticker, {}), dict)
                    else ""
                )
        except Exception:
            pass

    columns = [column for column in ["Ticker", "Market Cap", "Sector"] if column in out.columns]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out[columns].drop_duplicates(subset=["Ticker"]).to_csv(universe_path, index=False)
    return universe_path


def _download_weekly_v6_batch(
    tickers: list[str],
    start: str,
    end: str | None,
    weekly_v6_scanner_module,
    weekly_v6_strategy_module,
) -> dict[str, pd.DataFrame]:
    weekly_data: dict[str, pd.DataFrame] = {}
    for batch in chunked(tickers, size=40):
        try:
            downloaded = yf.download(
                tickers=batch,
                start=start,
                end=end,
                interval="1wk",
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception:
            continue

        if downloaded is None or downloaded.empty:
            continue

        if isinstance(downloaded.columns, pd.MultiIndex):
            first_level = set(downloaded.columns.get_level_values(0))
            second_level = set(downloaded.columns.get_level_values(1))
            for ticker in batch:
                frame = pd.DataFrame()
                if ticker in first_level:
                    frame = downloaded[ticker].copy()
                elif ticker in second_level:
                    frame = downloaded.xs(ticker, axis=1, level=1).copy()
                if frame.empty:
                    continue
                frame = weekly_v6_strategy_module.flatten_yfinance_columns(frame)
                needed = ["Open", "High", "Low", "Close", "Volume"]
                if not set(needed).issubset(frame.columns):
                    continue
                frame = frame[needed].copy()
                frame.index = pd.to_datetime(frame.index)
                frame = frame.sort_index().replace([float("inf"), float("-inf")], pd.NA)
                frame = frame.dropna(subset=needed)
                frame = frame[frame["Volume"] > 0]
                frame = weekly_v6_scanner_module.drop_incomplete_latest_week(frame)
                if not frame.empty:
                    weekly_data[ticker] = frame
        elif len(batch) == 1:
            ticker = batch[0]
            frame = weekly_v6_strategy_module.flatten_yfinance_columns(downloaded.copy())
            needed = ["Open", "High", "Low", "Close", "Volume"]
            if set(needed).issubset(frame.columns):
                frame = frame[needed].copy()
                frame.index = pd.to_datetime(frame.index)
                frame = frame.sort_index().replace([float("inf"), float("-inf")], pd.NA)
                frame = frame.dropna(subset=needed)
                frame = frame[frame["Volume"] > 0]
                frame = weekly_v6_scanner_module.drop_incomplete_latest_week(frame)
                if not frame.empty:
                    weekly_data[ticker] = frame
    return weekly_data


def run_weekly_v6_cup_handle_scan(max_tickers: int | None = None) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    import traceback

    import weekly_v9_v8_breakout_plus_cup_handle_scanner as weekly_v9
    import weekly_v8_rs5_52w_ath_sma20_no_cup_breakout_backtest as weekly_breakout_strat

    eligible = load_eligible_tickers()
    if eligible.empty:
        return pd.DataFrame(), ["Saved $500M+ ticker universe is empty."], {
            "scan_date": "",
            "stored_tickers": 0,
            "ohlcv_loaded": 0,
            "price_candidates": 0,
        }

    universe_path = _write_weekly_v6_universe_file(eligible)
    tickers = weekly_breakout_strat.load_universe(universe_path, max_tickers)
    errors: list[str] = []
    breakout_candidates: list[dict[str, object]] = []
    cup_handle_candidates: list[dict[str, object]] = []

    try:
        spy_raw = weekly_breakout_strat.download_weekly(
            weekly_breakout_strat.SPY_TICKER,
            weekly_breakout_strat.START_DATE,
            weekly_breakout_strat.END_DATE,
        )
        qqq_raw = weekly_breakout_strat.download_weekly(
            weekly_breakout_strat.QQQ_TICKER,
            weekly_breakout_strat.START_DATE,
            weekly_breakout_strat.END_DATE,
        )
        spy_raw = weekly_v9.drop_incomplete_latest_week(spy_raw)
        qqq_raw = weekly_v9.drop_incomplete_latest_week(qqq_raw)
        if spy_raw.empty or qqq_raw.empty:
            raise RuntimeError("Could not download enough SPY/QQQ weekly data.")
        spy, qqq = weekly_breakout_strat.prepare_etf_data(spy_raw, qqq_raw)
    except Exception as exc:
        return pd.DataFrame(), [f"Market ETF setup failed: {exc}"], {
            "scan_date": "",
            "stored_tickers": len(tickers),
            "ohlcv_loaded": 0,
            "price_candidates": 0,
        }

    sector_cache_path = DATA_DIR / "weekly_v6_sector_etf_cache.csv"
    sector_map = weekly_breakout_strat.build_sector_etf_map(universe_path, tickers, sector_cache_path)
    unique_sector_etfs = sorted({data["Sector ETF"] for data in sector_map.values() if data.get("Sector ETF")})
    sector_data: dict[str, pd.DataFrame] = {}
    for etf in unique_sector_etfs:
        try:
            raw = weekly_breakout_strat.download_weekly(
                etf,
                weekly_breakout_strat.START_DATE,
                weekly_breakout_strat.END_DATE,
            )
            raw = weekly_v9.drop_incomplete_latest_week(raw)
            if raw.empty:
                errors.append(f"{etf}: missing sector ETF weekly data")
                continue
            sector_data[etf] = weekly_breakout_strat.prepare_sector_etf_data(raw, etf)
        except Exception as exc:
            errors.append(f"{etf}: sector ETF data unavailable ({exc})")

    stock_weekly_data = _download_weekly_v6_batch(
        tickers,
        weekly_breakout_strat.START_DATE,
        weekly_breakout_strat.END_DATE,
        weekly_v9,
        weekly_breakout_strat,
    )
    ohlcv_loaded = 0
    scan_date = str(pd.Timestamp(spy.index[-1]).date()) if not spy.empty else ""
    for index, ticker in enumerate(tickers, start=1):
        try:
            sector_info = sector_map.get(ticker)
            if not sector_info:
                errors.append(f"{ticker}: missing sector ETF mapping")
                continue
            sector_name = sector_info.get("Sector", "")
            sector_etf = sector_info.get("Sector ETF", "")
            sector_df = sector_data.get(sector_etf)
            if sector_df is None or sector_df.empty:
                errors.append(f"{ticker}: missing sector ETF data for {sector_etf}")
                continue

            raw = stock_weekly_data.get(ticker, pd.DataFrame())
            if raw.empty or len(raw) < weekly_breakout_strat.SMA_LONG + 20:
                errors.append(f"{ticker}: not enough weekly data")
                continue
            ohlcv_loaded += 1

            df = weekly_breakout_strat.add_relative_strength(raw, spy, qqq, sector_df, sector_name, sector_etf)
            if df.empty:
                continue
            row = df.iloc[-1]
            scan_date = str(pd.Timestamp(row.name).date())
            if weekly_breakout_strat.setup_passes(row):
                cup = weekly_v9.detect_cup_handle_at(
                    df=df,
                    i=len(df) - 1,
                    min_cup_depth_pct=weekly_v9.CUP_MIN_DEPTH_PCT,
                    max_cup_depth_pct=weekly_v9.CUP_MAX_DEPTH_PCT,
                    big_green_volume_ratio=weekly_v9.CUP_BIG_GREEN_VOLUME_RATIO,
                )
                record = weekly_v9.build_candidate_record(ticker, row, cup)
                breakout_candidates.append(record)
                if bool(cup.get("Cup Handle Pattern", False)):
                    cup_handle_candidates.append(record)
        except Exception as exc:
            errors.append(f"{ticker}: {exc}\n{traceback.format_exc(limit=2)}")

    results = pd.DataFrame(breakout_candidates)
    if not results.empty:
        results = results.sort_values(
            [
                "Cup Handle Pattern",
                "Right Side Big Green Volume Count",
                "Cup Big Green Volume Count",
                "Stock RS 5W vs Sector",
                "RS 5W vs SPY",
                "Breakout Volume Ratio",
            ],
            ascending=[False, False, False, False, False, False],
        ).reset_index(drop=True)

    cup_handle_results = pd.DataFrame(cup_handle_candidates)
    if not cup_handle_results.empty:
        cup_handle_results = cup_handle_results.sort_values(
            [
                "Right Side Big Green Volume Count",
                "Cup Big Green Volume Count",
                "Stock RS 5W vs Sector",
                "RS 5W vs SPY",
                "Breakout Volume Ratio",
            ],
            ascending=[False, False, False, False, False],
        ).reset_index(drop=True)

    stats: dict[str, object] = {
        "scan_date": scan_date,
        "stored_tickers": len(tickers),
        "sector_mapped": len(sector_map),
        "sector_etfs": len(unique_sector_etfs),
        "ohlcv_loaded": ohlcv_loaded,
        "price_candidates": len(results),
        "cup_handle_candidates": len(cup_handle_results),
        "cup_handle_rows": cup_handle_results.to_dict("records") if not cup_handle_results.empty else [],
        "matches": len(results),
        "error_count": len(errors),
        "rule": "Weekly Breakout",
    }
    return results, errors, stats


def _weekly_momentum_candidate(
    ticker: str,
    prices: pd.DataFrame,
    completed_date: pd.Timestamp,
    cfg: WeeklyMomentumRuleConfig,
) -> dict[str, float | str] | None:
    if prices.empty or len(prices) < 120:
        return None

    weekly = _to_weekly_ohlcv(prices, completed_date)
    if weekly.empty or len(weekly) < 30:
        return None

    row = weekly.iloc[-1]
    required = ["EMA_9W", "SMA_9W", "EMA_21W", "Avg_Volume_20W"]
    if any(pd.isna(row[column]) for column in required):
        return None

    avg_volume_20w = float(row["Avg_Volume_20W"])
    if avg_volume_20w <= cfg.min_avg_weekly_volume:
        return None

    close = float(row["Close"])
    if close <= float(row["EMA_21W"]):
        return None

    trend_parts = ["Close > 21W EMA"]
    has_50w = not pd.isna(row["SMA_50W"])
    has_200w = not pd.isna(row["SMA_200W"])
    if has_50w:
        if float(row["EMA_21W"]) <= float(row["SMA_50W"]):
            return None
        trend_parts.append("21W EMA > 50W SMA")
    if has_50w and has_200w:
        if float(row["SMA_50W"]) <= float(row["SMA_200W"]):
            return None
        trend_parts.append("50W SMA > 200W SMA")
    elif not has_50w:
        trend_parts.append("50W SMA unavailable; ignored")
    if not has_200w:
        trend_parts.append("200W SMA unavailable; ignored")

    distance_ema = _distance_from_candle_to_ma(row, float(row["EMA_9W"]))
    distance_sma = _distance_from_candle_to_ma(row, float(row["SMA_9W"]))
    if min(distance_ema, distance_sma) > cfg.max_pullback_distance:
        return None

    signal = detect_signal(weekly)
    pattern_allowed = (cfg.allow_hammer and signal["hammer"]) or (
        cfg.allow_bullish_engulfing and signal["bullish_engulfing"]
    )
    if not pattern_allowed:
        return None

    ma_label = "9W EMA" if distance_ema <= distance_sma else "9W SMA"
    distance = min(distance_ema, distance_sma)
    entry = float(row["High"])
    stop = float(row["Low"])
    if stop <= 0 or entry <= stop:
        return None
    target = _weekly_target(weekly)
    risk = entry - stop

    return {
        "ticker": ticker,
        "signal_week": str(weekly.index[-1].date()),
        "open": round(float(row["Open"]), 2),
        "high": round(entry, 2),
        "low": round(stop, 2),
        "close": round(close, 2),
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "risk_reward": round((target - entry) / risk, 2) if risk > 0 else None,
        "pattern": str(signal["pattern"]),
        "pullback_ma": ma_label,
        "distance_to_9w_ma_pct": round(distance * 100, 2),
        "volume": float(row["Volume"]),
        "avg_volume_20w": avg_volume_20w,
        "volume_ratio": round(float(row["Volume"] / avg_volume_20w), 2) if avg_volume_20w else 0.0,
        "ema_9w": round(float(row["EMA_9W"]), 2),
        "sma_9w": round(float(row["SMA_9W"]), 2),
        "ema_21w": round(float(row["EMA_21W"]), 2),
        "sma_50w": _round_optional(row["SMA_50W"]),
        "sma_200w": _round_optional(row["SMA_200W"]),
        "uptrend": "; ".join(trend_parts),
    }


def run_weekly_momentum_scan() -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    from database import load_eligible_tickers

    cfg = load_rule_config().weekly_momentum
    completed_date = latest_completed_us_session()
    scan_date = str(_latest_completed_week_end(completed_date).date())
    universe = load_eligible_tickers()
    if universe.empty:
        return pd.DataFrame(), [], {"scan_date": scan_date, "stored_tickers": 0, "price_candidates": 0}

    tickers = universe["ticker"].tolist()
    market_caps = dict(zip(universe["ticker"], universe["market_cap"], strict=False))
    price_data = download_ohlcv(
        tickers,
        period="max",
        completed_date=completed_date,
        use_nasdaq_fallback=False,
    )
    errors: list[str] = []
    technical_candidates: list[dict[str, float | str]] = []
    final_matches: list[dict[str, float | str | None]] = []

    for ticker in tickers:
        prices = price_data.get(ticker)
        if prices is None or prices.empty:
            errors.append(f"{ticker}: missing all-time OHLCV data")
            continue
        candidate = _weekly_momentum_candidate(ticker, prices, completed_date, cfg)
        if candidate:
            candidate["market_cap"] = float(market_caps[ticker])
            technical_candidates.append(candidate)

    for candidate in technical_candidates:
        ticker = str(candidate["ticker"])
        fundamentals = get_fundamentals(ticker)
        if fundamentals.error:
            errors.append(f"{ticker}: fundamentals unavailable ({fundamentals.error})")
            candidate["fundamental_status"] = f"Fundamentals unavailable: {fundamentals.error}"
            continue

        market_cap = float(candidate["market_cap"])
        avg_weekly_volume = float(candidate["avg_volume_20w"])
        candidate["revenue_growth"] = fundamentals.revenue_growth
        candidate["eps_growth"] = fundamentals.eps_growth
        candidate["average_volume_weekly"] = avg_weekly_volume
        candidate.update(get_eps_growth_details(ticker))
        candidate.update(get_quarterly_growth_details(ticker))

        failed_reasons: list[str] = []
        if fundamentals.revenue_growth is None:
            failed_reasons.append("revenue growth missing")
        elif fundamentals.revenue_growth <= cfg.min_revenue_growth:
            failed_reasons.append(f"revenue growth {fundamentals.revenue_growth:.1%} <= {cfg.min_revenue_growth:.0%}")
        if fundamentals.eps_growth is None:
            failed_reasons.append("EPS growth missing")
        elif fundamentals.eps_growth <= cfg.min_eps_growth:
            failed_reasons.append(f"EPS growth {fundamentals.eps_growth:.1%} <= {cfg.min_eps_growth:.0%}")
        if market_cap <= MIN_MARKET_CAP:
            failed_reasons.append("market cap <= $500M")
        if avg_weekly_volume <= cfg.min_avg_weekly_volume:
            failed_reasons.append(f"20-week average volume <= {cfg.min_avg_weekly_volume:,}")

        if not (
            fundamentals.revenue_growth is not None
            and fundamentals.eps_growth is not None
            and fundamentals.revenue_growth > cfg.min_revenue_growth
            and fundamentals.eps_growth > cfg.min_eps_growth
            and market_cap > MIN_MARKET_CAP
            and avg_weekly_volume > cfg.min_avg_weekly_volume
        ):
            candidate["fundamental_status"] = "; ".join(failed_reasons)
            continue

        candidate["fundamental_status"] = "Passed fundamentals"
        final_matches.append(candidate)

    frame = pd.DataFrame(final_matches)
    if not frame.empty:
        frame = frame.sort_values(
            by=["volume_ratio", "revenue_growth", "eps_growth", "market_cap"],
            ascending=[False, False, False, False],
        )

    stats = {
        "scan_date": scan_date,
        "stored_tickers": len(tickers),
        "ohlcv_loaded": len(price_data),
        "price_candidates": len(technical_candidates),
        "technical_candidate_rows": technical_candidates,
        "matches": len(frame),
    }
    return frame, errors, stats


def find_weekly_ath_breakout(
    ticker: str,
    prices: pd.DataFrame,
    completed_date: pd.Timestamp,
    cfg: WeeklyATHRuleConfig,
) -> dict[str, float | str] | None:
    if prices.empty or len(prices) < 260:
        return None

    weekly = _to_weekly_ohlcv(prices, completed_date)
    if weekly.empty or len(weekly) < 52:
        return None

    row = weekly.iloc[-1]
    required = ["Prior_All_Time_High", "Avg_Volume_20W", "EMA_10W", "SMA_30W"]
    if any(pd.isna(row[column]) for column in required):
        return None

    breaks_ath = float(row["High"]) > float(row["Prior_All_Time_High"]) or float(row["Close"]) > float(row["Prior_All_Time_High"])
    volume_ratio = float(row["Volume"] / row["Avg_Volume_20W"]) if row["Avg_Volume_20W"] else 0.0
    uptrend = (
        float(row["Close"]) > float(row["EMA_10W"])
        and float(row["EMA_10W"]) > float(row["SMA_30W"])
        and bool(row["SMA_30W_Rising"])
    )

    if not breaks_ath:
        return None
    if volume_ratio <= cfg.min_volume_ratio:
        return None
    if cfg.require_uptrend and not uptrend:
        return None

    return {
        "ticker": ticker,
        "signal_week": str(weekly.index[-1].date()),
        "open": round(float(row["Open"]), 2),
        "high": round(float(row["High"]), 2),
        "low": round(float(row["Low"]), 2),
        "close": round(float(row["Close"]), 2),
        "prior_all_time_high": round(float(row["Prior_All_Time_High"]), 2),
        "volume": float(row["Volume"]),
        "avg_volume_20w": float(row["Avg_Volume_20W"]),
        "volume_ratio": round(volume_ratio, 2),
        "ema_10w": round(float(row["EMA_10W"]), 2),
        "sma_30w": round(float(row["SMA_30W"]), 2),
        "uptrend": "Close > 10W EMA > 30W SMA; 30W SMA rising",
    }


def run_weekly_ath_breakout_scan() -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    from database import load_eligible_tickers

    cfg = load_rule_config().weekly_ath
    completed_date = latest_completed_us_session()
    scan_date = str(_latest_completed_week_end(completed_date).date())
    universe = load_eligible_tickers()
    if universe.empty:
        return pd.DataFrame(), [], {"scan_date": scan_date, "stored_tickers": 0, "price_candidates": 0}

    tickers = universe["ticker"].tolist()
    market_caps = dict(zip(universe["ticker"], universe["market_cap"], strict=False))
    price_data = download_ohlcv(
        tickers,
        period="max",
        completed_date=completed_date,
        use_nasdaq_fallback=False,
    )
    errors: list[str] = []
    technical_candidates: list[dict[str, float | str]] = []
    final_matches: list[dict[str, float | str | None]] = []

    for ticker in tickers:
        prices = price_data.get(ticker)
        if prices is None or prices.empty:
            errors.append(f"{ticker}: missing all-time OHLCV data")
            continue
        candidate = find_weekly_ath_breakout(ticker, prices, completed_date, cfg)
        if candidate:
            candidate["market_cap"] = float(market_caps[ticker])
            technical_candidates.append(candidate)

    for candidate in technical_candidates:
        ticker = str(candidate["ticker"])
        fundamentals = get_fundamentals(ticker)
        if fundamentals.error:
            errors.append(f"{ticker}: fundamentals unavailable ({fundamentals.error})")
            continue

        market_cap = float(candidate["market_cap"])
        if cfg.require_fundamentals and not (
            fundamentals.revenue_growth is not None
            and fundamentals.eps_growth is not None
            and fundamentals.average_volume is not None
            and fundamentals.revenue_growth > cfg.min_revenue_growth
            and fundamentals.eps_growth > cfg.min_eps_growth
            and market_cap > MIN_MARKET_CAP
            and fundamentals.average_volume > cfg.min_avg_volume
        ):
            continue

        candidate["revenue_growth"] = fundamentals.revenue_growth
        candidate["eps_growth"] = fundamentals.eps_growth
        candidate["average_volume"] = fundamentals.average_volume
        candidate.update(get_eps_growth_details(ticker))
        final_matches.append(candidate)

    frame = pd.DataFrame(final_matches)
    if not frame.empty:
        frame = frame.sort_values(
            by=["volume_ratio", "revenue_growth", "eps_growth", "market_cap"],
            ascending=[False, False, False, False],
        )

    stats = {
        "scan_date": scan_date,
        "stored_tickers": len(tickers),
        "ohlcv_loaded": len(price_data),
        "price_candidates": len(technical_candidates),
        "technical_candidate_rows": technical_candidates,
        "matches": len(frame),
    }
    return frame, errors, stats


def explain_rules() -> dict[str, str]:
    rule_config = load_rule_config()
    pullback = rule_config.pullback
    marubozu = rule_config.marubozu
    weekly_ath = rule_config.weekly_ath
    weekly_momentum = rule_config.weekly_momentum
    morning_star = rule_config.morning_star
    technical_strength = rule_config.technical_strength
    allowed_patterns = []
    if pullback.allow_hammer:
        allowed_patterns.append("hammer")
    if pullback.allow_bullish_engulfing:
        allowed_patterns.append("bullish engulfing")
    if pullback.allow_bullish_rejection:
        allowed_patterns.append("bullish rejection")
    pattern_text = ", ".join(allowed_patterns) if allowed_patterns else "no candle pattern enabled"
    return {
        "fundamentals": (
            f"Revenue growth > {pullback.min_revenue_growth:.0%}, EPS growth > {pullback.min_eps_growth:.0%}, "
            f"market cap > ${MIN_MARKET_CAP / 1_000_000:.0f}M, average volume > {pullback.min_avg_volume:,}"
        ),
        "trend": "Close > 21 EMA > 50 SMA > 200 SMA",
        "setup": (
            f"Close within {pullback.max_pullback_distance:.1%} of 9 EMA or 9 SMA with {pattern_text} candle"
        ),
        "marubozu": (
            f"Green marubozu body >= {marubozu.min_body_pct:.0%} of range, "
            f"upper wick <= {marubozu.max_upper_wick_pct:.0%}, lower wick <= {marubozu.max_lower_wick_pct:.0%}, "
            f"volume >= {marubozu.min_volume_ratio:.2f}x 20-day average"
        ),
        "weekly_ath": (
            "Completed weekly candle breaks prior all-time high, "
            f"weekly volume >= {weekly_ath.min_volume_ratio:.2f}x 20-week average"
            + (
                ", weekly uptrend is Close > 10W EMA > 30W SMA with 30W SMA rising"
                if weekly_ath.require_uptrend
                else ""
            )
            + (
                f", fundamentals pass revenue > {weekly_ath.min_revenue_growth:.0%}, "
                f"EPS > {weekly_ath.min_eps_growth:.0%}, average volume > {weekly_ath.min_avg_volume:,}"
                if weekly_ath.require_fundamentals
                else ""
            )
        ),
        "weekly_momentum": (
            f"Weekly setup: revenue growth > {weekly_momentum.min_revenue_growth:.0%}, "
            f"EPS growth > {weekly_momentum.min_eps_growth:.0%}, market cap > $500M, "
            f"20-week average volume > {weekly_momentum.min_avg_weekly_volume:,}, "
            "Close > 21W EMA, 50W/200W SMA trend checks only when available, and full weekly candle range "
            f"must be within {weekly_momentum.max_pullback_distance:.1%} of 9W EMA/SMA"
        ),
        "morning_star": (
            "Daily Morning Star pattern on the saved $500M universe, "
            f"20-day average volume > {morning_star.min_avg_volume:,}, "
            f"first candle body >= {morning_star.first_long_body_pct:.0%} of range, "
            f"middle candle body <= {morning_star.small_body_pct:.0%} of first candle body, "
            f"third candle body >= {morning_star.third_long_body_pct:.0%} of range, "
            f"third candle recovers at least {morning_star.recovery_pct:.0%} of first candle body, "
            f"gap tolerance {morning_star.gap_tolerance_pct:.1%}"
            + (", requires uptrend" if morning_star.require_uptrend else "")
            + (", requires second candle volume above average" if morning_star.require_second_volume_above_average else "")
            + (", requires third candle volume above average" if morning_star.require_third_volume_above_average else "")
        ),
        "technical_strength": (
            "Daily technical strength setup on the saved $500M universe: "
            "Close > 9 EMA > 20 SMA > 50 SMA > 200 SMA, "
            f"RSI > {technical_strength.min_rsi:.0f}, "
            f"20D relative strength vs SPY > {technical_strength.min_rs_20d_vs_spy:.0%}, "
            f"RS ratio ROC 20D > {technical_strength.min_rs_ratio_roc_20d:.0%}, "
            f"breakout volume >= {technical_strength.min_breakout_volume_ratio:.1f}x 20D average, "
            "or pullback/reclaim of 9 EMA with bullish signal candle. "
            "If SPY or QQQ is weak, stock must outperform SPY over 5D and hold better than -2%."
        ),
        "technical_breakout": (
            "Daily technical breakout on the saved $500M universe: "
            "Close > 9 EMA > 20 SMA > 50 SMA > 200 SMA, "
            f"RSI > {technical_strength.min_rsi:.0f}, "
            f"20D relative strength vs SPY > {technical_strength.min_rs_20d_vs_spy:.0%}, "
            f"RS ratio ROC 20D > {technical_strength.min_rs_ratio_roc_20d:.0%}, "
            f"5D outperformance vs SPY > {technical_strength.weak_market_min_outperformance_5d:.0%}, "
            f"5D return > {technical_strength.weak_market_min_stock_return_5d:.0%}, "
            f"and recent ATH/52W breakout volume >= {technical_strength.min_breakout_volume_ratio:.1f}x 20D average."
        ),
        "technical_pullback_9ema": (
            "Daily 9 EMA pullback on the saved $500M universe: "
            "Close > 9 EMA > 20 SMA > 50 SMA > 200 SMA, "
            f"RSI > {technical_strength.min_rsi:.0f}, "
            f"20D relative strength vs SPY > {technical_strength.min_rs_20d_vs_spy:.0%}, "
            f"RS ratio ROC 20D > {technical_strength.min_rs_ratio_roc_20d:.0%}, "
            f"5D outperformance vs SPY > {technical_strength.weak_market_min_outperformance_5d:.0%}, "
            f"5D return > {technical_strength.weak_market_min_stock_return_5d:.0%}, "
            f"and pullback/reclaim within {technical_strength.max_pullback_distance:.1%} of 9 EMA with bullish signal candle."
        ),
        "weekly_v6_cup_handle": (
            "Generic weekly breakout on the saved $500M universe: "
            "weekly stock uptrend, SPY/QQQ/sector ETF confirmation, "
            "RS 5W vs SPY > 5%, RS ratio ROC 5W > 5%, stock RS vs sector > 5%, "
            "52W/ATH breakout, breakout volume >= 1.2x 20-week average, "
            "and avoids oversized green or extreme-volume breakout candles."
        ),
        "monthly_big_volume": (
            "Monthly candle setup on the saved $500M universe: latest completed monthly candle is either "
            "a big green candle or green hammer, with monthly volume greater than the prior month and "
            "greater than the 12-month average."
        ),
    }
