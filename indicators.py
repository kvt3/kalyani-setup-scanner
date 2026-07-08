from __future__ import annotations

import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["9_EMA"] = out["Close"].ewm(span=9, adjust=False).mean()
    out["9_SMA"] = out["Close"].rolling(9).mean()
    out["21_EMA"] = out["Close"].ewm(span=21, adjust=False).mean()
    out["50_SMA"] = out["Close"].rolling(50).mean()
    out["200_SMA"] = out["Close"].rolling(200).mean()
    out["Avg_Volume_20D"] = out["Volume"].rolling(20).mean()
    out["52W_High"] = out["High"].rolling(252, min_periods=100).max()
    return out


def is_clear_uptrend(row: pd.Series) -> bool:
    needed = ["Close", "21_EMA", "50_SMA", "200_SMA"]
    if row[needed].isna().any():
        return False
    return bool(row["Close"] > row["21_EMA"] > row["50_SMA"] > row["200_SMA"])


def pullback_to_9ma(row: pd.Series, max_distance: float = 0.03) -> tuple[bool, str, float]:
    close = float(row["Close"])
    candidates = {
        "9 EMA": float(row["9_EMA"]),
        "9 SMA": float(row["9_SMA"]),
    }
    distances = {
        label: abs(close - value) / close
        for label, value in candidates.items()
        if close > 0 and pd.notna(value)
    }
    if not distances:
        return False, "", float("nan")
    label = min(distances, key=distances.get)
    return distances[label] <= max_distance, label, distances[label]

