from __future__ import annotations

import sqlite3
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from cloud_sqlite_store import connect_synced_sqlite, reset_cloud_dirty
from config import DB_PATH


SCHEMA = """
CREATE TABLE IF NOT EXISTS watchlists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    close REAL NOT NULL,
    entry REAL NOT NULL,
    stop REAL NOT NULL,
    target REAL NOT NULL,
    risk_pct REAL NOT NULL,
    setup_grade TEXT NOT NULL,
    reason TEXT NOT NULL,
    revenue_growth REAL,
    eps_growth REAL,
    market_cap REAL,
    average_volume REAL,
    volume REAL,
    avg_volume_20d REAL,
    signal_date TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(scan_date, ticker)
);

CREATE TABLE IF NOT EXISTS eligible_tickers (
    ticker TEXT PRIMARY KEY,
    market_cap REAL NOT NULL,
    source TEXT NOT NULL,
    downloaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS universe_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rule_watchlists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_name TEXT NOT NULL,
    scan_date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(rule_name, scan_date, ticker)
);

CREATE TABLE IF NOT EXISTS rule_runs (
    rule_name TEXT NOT NULL,
    scan_date TEXT NOT NULL,
    ticker_count INTEGER NOT NULL DEFAULT 0,
    saved_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(rule_name, scan_date)
);
"""


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect_synced_sqlite(db_path)
    conn.executescript(SCHEMA)
    reset_cloud_dirty(conn)
    return conn


def save_watchlist(results: pd.DataFrame, db_path: Path = DB_PATH) -> int:
    if results.empty:
        return 0
    columns = [
        "scan_date",
        "ticker",
        "close",
        "entry",
        "stop",
        "target",
        "risk_pct",
        "setup_grade",
        "reason",
        "revenue_growth",
        "eps_growth",
        "market_cap",
        "average_volume",
        "volume",
        "avg_volume_20d",
        "signal_date",
    ]
    rows = results[columns].to_dict("records")
    with get_connection(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO watchlists (
                scan_date, ticker, close, entry, stop, target, risk_pct, setup_grade,
                reason, revenue_growth, eps_growth, market_cap, average_volume,
                volume, avg_volume_20d, signal_date
            )
            VALUES (
                :scan_date, :ticker, :close, :entry, :stop, :target, :risk_pct, :setup_grade,
                :reason, :revenue_growth, :eps_growth, :market_cap, :average_volume,
                :volume, :avg_volume_20d, :signal_date
            )
            ON CONFLICT(scan_date, ticker) DO UPDATE SET
                close=excluded.close,
                entry=excluded.entry,
                stop=excluded.stop,
                target=excluded.target,
                risk_pct=excluded.risk_pct,
                setup_grade=excluded.setup_grade,
                reason=excluded.reason,
                revenue_growth=excluded.revenue_growth,
                eps_growth=excluded.eps_growth,
                market_cap=excluded.market_cap,
                average_volume=excluded.average_volume,
                volume=excluded.volume,
                avg_volume_20d=excluded.avg_volume_20d,
                signal_date=excluded.signal_date,
                created_at=CURRENT_TIMESTAMP
            """,
            rows,
        )
    return len(rows)


def list_scan_dates(db_path: Path = DB_PATH) -> list[str]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT scan_date FROM watchlists ORDER BY scan_date DESC"
        ).fetchall()
    return [row[0] for row in rows]


def load_watchlist(scan_date: str | None = None, db_path: Path = DB_PATH) -> pd.DataFrame:
    with get_connection(db_path) as conn:
        if scan_date:
            query = "SELECT * FROM watchlists WHERE scan_date = ? ORDER BY setup_grade, risk_pct ASC"
            return pd.read_sql_query(query, conn, params=(scan_date,))
        return pd.read_sql_query(
            "SELECT * FROM watchlists ORDER BY scan_date DESC, setup_grade, risk_pct ASC",
            conn,
        )


def save_eligible_tickers(
    rows: list[dict[str, float | str]],
    source: str,
    min_market_cap: int,
    total_tickers: int | None = None,
    error_count: int | None = None,
    db_path: Path = DB_PATH,
) -> int:
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM eligible_tickers")
        conn.executemany(
            """
            INSERT INTO eligible_tickers (ticker, market_cap, source)
            VALUES (:ticker, :market_cap, :source)
            ON CONFLICT(ticker) DO UPDATE SET
                market_cap=excluded.market_cap,
                source=excluded.source,
                downloaded_at=CURRENT_TIMESTAMP
            """,
            rows,
        )
        metadata = {
            "last_refreshed": pd.Timestamp.utcnow().isoformat(),
            "source": source,
            "min_market_cap": str(min_market_cap),
            "eligible_count": str(len(rows)),
            "total_tickers_checked": str(total_tickers if total_tickers is not None else ""),
            "market_cap_error_count": str(error_count if error_count is not None else ""),
        }
        conn.executemany(
            """
            INSERT INTO universe_metadata (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            metadata.items(),
        )
    return len(rows)


def load_eligible_tickers(db_path: Path = DB_PATH) -> pd.DataFrame:
    with get_connection(db_path) as conn:
        return pd.read_sql_query(
            "SELECT ticker, market_cap, source, downloaded_at FROM eligible_tickers ORDER BY market_cap DESC",
            conn,
        )


def load_eligible_ticker_symbols(db_path: Path = DB_PATH) -> list[str]:
    frame = load_eligible_tickers(db_path)
    if frame.empty:
        return []
    return frame["ticker"].tolist()


def load_universe_metadata(db_path: Path = DB_PATH) -> dict[str, str]:
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT key, value FROM universe_metadata").fetchall()
    return dict(rows)


def _clean_payload_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clean_payload_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_payload_value(item) for item in value]
    if isinstance(value, tuple | set):
        return [_clean_payload_value(item) for item in value]
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return _clean_payload_value(value.item())
        except Exception:
            pass
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return None
    except Exception:
        pass
    return value


def save_rule_watchlist(
    rule_name: str,
    results: pd.DataFrame,
    scan_date: str,
    db_path: Path = DB_PATH,
) -> int:
    rows = []
    if not results.empty:
        for record in results.to_dict("records"):
            ticker = str(record.get("ticker") or record.get("Ticker") or "").strip()
            if not ticker:
                continue
            payload = {key: _clean_payload_value(value) for key, value in record.items()}
            rows.append(
                {
                    "rule_name": rule_name,
                    "scan_date": scan_date,
                    "ticker": ticker,
                    "payload_json": json.dumps(payload, sort_keys=True, default=str),
                }
            )

    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO rule_runs (rule_name, scan_date, ticker_count)
            VALUES (?, ?, ?)
            ON CONFLICT(rule_name, scan_date) DO UPDATE SET
                ticker_count=excluded.ticker_count,
                saved_at=CURRENT_TIMESTAMP
            """,
            (rule_name, scan_date, len(rows)),
        )
        if rows:
            conn.executemany(
                """
                INSERT INTO rule_watchlists (rule_name, scan_date, ticker, payload_json)
                VALUES (:rule_name, :scan_date, :ticker, :payload_json)
                ON CONFLICT(rule_name, scan_date, ticker) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    created_at=CURRENT_TIMESTAMP
                """,
                rows,
            )
    return len(rows)


def list_rule_watchlists(db_path: Path = DB_PATH) -> pd.DataFrame:
    with get_connection(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT rule_name, scan_date, ticker_count, saved_at
            FROM rule_runs
            UNION ALL
            SELECT w.rule_name, w.scan_date, COUNT(*) AS ticker_count, MAX(w.created_at) AS saved_at
            FROM rule_watchlists w
            WHERE NOT EXISTS (
                SELECT 1
                FROM rule_runs r
                WHERE r.rule_name = w.rule_name
                  AND r.scan_date = w.scan_date
            )
            GROUP BY w.rule_name, w.scan_date
            ORDER BY scan_date DESC, rule_name ASC
            """,
            conn,
        )


def load_rule_watchlist(rule_name: str, scan_date: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT payload_json
            FROM rule_watchlists
            WHERE rule_name = ? AND scan_date = ?
            ORDER BY ticker ASC
            """,
            (rule_name, scan_date),
        ).fetchall()
    return pd.DataFrame([json.loads(row[0]) for row in rows])
