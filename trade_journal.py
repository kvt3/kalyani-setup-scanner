from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from cloud_sqlite_store import connect_synced_sqlite, reset_cloud_dirty
from config import DB_PATH


TRADE_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL DEFAULT 'Long',
    trade_type TEXT NOT NULL DEFAULT 'Swing Trade',
    setup TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'Open',
    entry_date TEXT NOT NULL,
    entry_price REAL NOT NULL,
    stop_price REAL NOT NULL,
    target_price REAL,
    quantity INTEGER NOT NULL,
    planned_risk_per_share REAL NOT NULL,
    planned_risk_amount REAL NOT NULL,
    planned_reward_per_share REAL,
    planned_rr REAL,
    exit_date TEXT,
    exit_price REAL,
    fees REAL NOT NULL DEFAULT 0,
    realized_pl REAL,
    realized_r REAL,
    outcome TEXT,
    notes TEXT,
    tags TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trade_calendar_months (
    month_key TEXT PRIMARY KEY,
    trade_count INTEGER NOT NULL DEFAULT 0,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,
    breakeven INTEGER NOT NULL DEFAULT 0,
    total_pl REAL NOT NULL DEFAULT 0,
    total_r REAL NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def _ensure_trade_columns(conn: sqlite3.Connection) -> None:
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(trades)").fetchall()}
    if "trade_type" not in existing_columns:
        conn.execute("ALTER TABLE trades ADD COLUMN trade_type TEXT NOT NULL DEFAULT 'Swing Trade'")


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect_synced_sqlite(db_path)
    conn.executescript(TRADE_SCHEMA)
    _ensure_trade_columns(conn)
    reset_cloud_dirty(conn)
    return conn


def _clean_text(value: object) -> str:
    return str(value or "").strip()


def _direction_sign(direction: str) -> int:
    return -1 if direction.lower() == "short" else 1


def calculate_trade_fields(
    direction: str,
    entry_price: float,
    stop_price: float,
    target_price: float | None,
    quantity: int,
    exit_price: float | None = None,
    fees: float = 0.0,
) -> dict[str, float | str | None]:
    sign = _direction_sign(direction)
    risk_per_share = (entry_price - stop_price) * sign
    if risk_per_share <= 0:
        raise ValueError("Stop price must be below entry for Long trades and above entry for Short trades.")

    planned_reward_per_share = None
    planned_rr = None
    if target_price is not None and target_price > 0:
        planned_reward_per_share = (target_price - entry_price) * sign
        planned_rr = planned_reward_per_share / risk_per_share if risk_per_share else None

    planned_risk_amount = risk_per_share * quantity
    realized_pl = None
    realized_r = None
    outcome = None
    if exit_price is not None and exit_price > 0:
        realized_pl = ((exit_price - entry_price) * sign * quantity) - fees
        realized_r = realized_pl / planned_risk_amount if planned_risk_amount else None
        if realized_r > 0.05:
            outcome = "Win"
        elif realized_r < -0.05:
            outcome = "Loss"
        else:
            outcome = "Breakeven"

    return {
        "planned_risk_per_share": round(float(risk_per_share), 4),
        "planned_risk_amount": round(float(planned_risk_amount), 2),
        "planned_reward_per_share": round(float(planned_reward_per_share), 4) if planned_reward_per_share is not None else None,
        "planned_rr": round(float(planned_rr), 2) if planned_rr is not None else None,
        "realized_pl": round(float(realized_pl), 2) if realized_pl is not None else None,
        "realized_r": round(float(realized_r), 2) if realized_r is not None else None,
        "outcome": outcome,
    }


def add_trade(trade: dict[str, Any], db_path: Path = DB_PATH) -> int:
    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO trades (
                ticker, direction, trade_type, setup, status, entry_date, entry_price, stop_price,
                target_price, quantity, planned_risk_per_share, planned_risk_amount,
                planned_reward_per_share, planned_rr, exit_date, exit_price, fees,
                realized_pl, realized_r, outcome, notes, tags
            )
            VALUES (
                :ticker, :direction, :trade_type, :setup, :status, :entry_date, :entry_price, :stop_price,
                :target_price, :quantity, :planned_risk_per_share, :planned_risk_amount,
                :planned_reward_per_share, :planned_rr, :exit_date, :exit_price, :fees,
                :realized_pl, :realized_r, :outcome, :notes, :tags
            )
            """,
            trade,
        )
        return int(cursor.lastrowid)


def update_trade(trade_id: int, updates: dict[str, Any], db_path: Path = DB_PATH) -> None:
    allowed = {
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
        "planned_risk_per_share",
        "planned_risk_amount",
        "planned_reward_per_share",
        "planned_rr",
        "exit_date",
        "exit_price",
        "fees",
        "realized_pl",
        "realized_r",
        "outcome",
        "notes",
        "tags",
    }
    clean_updates = {key: value for key, value in updates.items() if key in allowed}
    if not clean_updates:
        return
    assignments = ", ".join(f"{key}=:{key}" for key in clean_updates)
    clean_updates["id"] = trade_id
    with _connect(db_path) as conn:
        conn.execute(
            f"UPDATE trades SET {assignments}, updated_at=CURRENT_TIMESTAMP WHERE id=:id",
            clean_updates,
        )


def delete_trade(trade_id: int, db_path: Path = DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM trades WHERE id = ?", (trade_id,))


def list_trades(db_path: Path = DB_PATH) -> pd.DataFrame:
    with _connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM trades ORDER BY entry_date DESC, id DESC", conn)


def save_trade_calendar_month(summary: dict[str, Any], db_path: Path = DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO trade_calendar_months (
                month_key, trade_count, wins, losses, breakeven, total_pl, total_r
            )
            VALUES (
                :month_key, :trade_count, :wins, :losses, :breakeven, :total_pl, :total_r
            )
            ON CONFLICT(month_key) DO UPDATE SET
                trade_count=excluded.trade_count,
                wins=excluded.wins,
                losses=excluded.losses,
                breakeven=excluded.breakeven,
                total_pl=excluded.total_pl,
                total_r=excluded.total_r,
                updated_at=CURRENT_TIMESTAMP
            """,
            summary,
        )


def list_trade_calendar_months(db_path: Path = DB_PATH) -> pd.DataFrame:
    with _connect(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT month_key, trade_count, wins, losses, breakeven, total_pl, total_r, updated_at
            FROM trade_calendar_months
            ORDER BY month_key DESC
            """,
            conn,
        )


def build_trade_payload(
    ticker: str,
    direction: str,
    trade_type: str,
    setup: str,
    status: str,
    entry_date: object,
    entry_price: float,
    stop_price: float,
    target_price: float | None,
    quantity: int,
    exit_date: object | None = None,
    exit_price: float | None = None,
    fees: float = 0.0,
    notes: str = "",
    tags: str = "",
) -> dict[str, Any]:
    fields = calculate_trade_fields(
        direction=direction,
        entry_price=float(entry_price),
        stop_price=float(stop_price),
        target_price=target_price,
        quantity=int(quantity),
        exit_price=exit_price,
        fees=float(fees or 0),
    )
    is_closed = status == "Closed" or (exit_price is not None and exit_price > 0)
    status_value = _clean_text(status)
    if status_value not in {"Initiated", "Open", "Closed"}:
        status_value = "Open"
    return {
        "ticker": _clean_text(ticker).upper(),
        "direction": direction,
        "trade_type": _clean_text(trade_type) or "Swing Trade",
        "setup": _clean_text(setup),
        "status": "Closed" if is_closed else status_value,
        "entry_date": str(entry_date),
        "entry_price": float(entry_price),
        "stop_price": float(stop_price),
        "target_price": float(target_price) if target_price is not None and target_price > 0 else None,
        "quantity": int(quantity),
        "exit_date": str(exit_date) if is_closed and exit_date is not None else None,
        "exit_price": float(exit_price) if is_closed and exit_price is not None and exit_price > 0 else None,
        "fees": float(fees or 0),
        "notes": _clean_text(notes),
        "tags": _clean_text(tags),
        **fields,
    }


def trade_analytics(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "total": 0,
            "initiated": 0,
            "open": 0,
            "closed": 0,
            "wins": 0,
            "losses": 0,
            "breakeven": 0,
            "win_rate": None,
            "total_pl": 0.0,
            "avg_r": None,
            "expectancy_r": None,
            "profit_factor": None,
            "avg_win_r": None,
            "avg_loss_r": None,
            "open_risk": 0.0,
    }
    frame = trades.copy()
    status = frame["status"].astype(str).str.lower()
    closed = frame[status.eq("closed")]
    initiated = frame[status.eq("initiated")]
    open_trades = frame[status.eq("open")]
    wins = closed[closed["outcome"].astype(str).eq("Win")]
    losses = closed[closed["outcome"].astype(str).eq("Loss")]
    breakeven = closed[closed["outcome"].astype(str).eq("Breakeven")]
    realized_r = pd.to_numeric(closed["realized_r"], errors="coerce").dropna()
    realized_pl = pd.to_numeric(closed["realized_pl"], errors="coerce").dropna()
    win_rate = len(wins) / len(closed) if len(closed) else None
    gross_profit = pd.to_numeric(wins["realized_pl"], errors="coerce").clip(lower=0).sum() if not wins.empty else 0.0
    gross_loss = abs(pd.to_numeric(losses["realized_pl"], errors="coerce").clip(upper=0).sum()) if not losses.empty else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss else None
    loss_r = pd.to_numeric(losses["realized_r"], errors="coerce").dropna()
    win_r = pd.to_numeric(wins["realized_r"], errors="coerce").dropna()
    return {
        "total": len(frame),
        "initiated": len(initiated),
        "open": len(open_trades),
        "closed": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": len(breakeven),
        "win_rate": win_rate,
        "total_pl": float(realized_pl.sum()) if not realized_pl.empty else 0.0,
        "avg_r": float(realized_r.mean()) if not realized_r.empty else None,
        "expectancy_r": float(realized_r.mean()) if not realized_r.empty else None,
        "profit_factor": float(profit_factor) if profit_factor is not None else None,
        "avg_win_r": float(win_r.mean()) if not win_r.empty else None,
        "avg_loss_r": float(loss_r.mean()) if not loss_r.empty else None,
        "open_risk": float(pd.to_numeric(open_trades["planned_risk_amount"], errors="coerce").sum()) if not open_trades.empty else 0.0,
    }


def setup_analytics(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    closed = trades[trades["status"].astype(str).str.lower().eq("closed")].copy()
    if closed.empty:
        return pd.DataFrame()
    closed["realized_r"] = pd.to_numeric(closed["realized_r"], errors="coerce")
    closed["realized_pl"] = pd.to_numeric(closed["realized_pl"], errors="coerce")
    grouped = closed.groupby("setup", dropna=False)
    rows = []
    for setup, group in grouped:
        wins = int(group["outcome"].astype(str).eq("Win").sum())
        losses = int(group["outcome"].astype(str).eq("Loss").sum())
        total = len(group)
        rows.append(
            {
                "setup": setup or "Unlabeled",
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


def trade_type_analytics(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "trade_type" not in trades.columns:
        return pd.DataFrame()
    closed = trades[trades["status"].astype(str).str.lower().eq("closed")].copy()
    if closed.empty:
        return pd.DataFrame()
    closed["realized_r"] = pd.to_numeric(closed["realized_r"], errors="coerce")
    closed["realized_pl"] = pd.to_numeric(closed["realized_pl"], errors="coerce")
    grouped = closed.groupby("trade_type", dropna=False)
    rows = []
    for trade_type, group in grouped:
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
