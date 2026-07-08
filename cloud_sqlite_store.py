from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any


CLOUD_SQLITE_TABLE = "app_sqlite_blobs"
MUTATING_SQL_PREFIXES = (
    "alter",
    "create",
    "delete",
    "drop",
    "insert",
    "replace",
    "update",
    "vacuum",
)


def _database_url() -> str:
    raw_url = os.getenv("DATABASE_URL", "").strip()
    if raw_url:
        return raw_url
    try:
        import streamlit as st

        return str(st.secrets.get("DATABASE_URL", "")).strip()
    except Exception:
        return ""


def cloud_sqlite_enabled() -> bool:
    return bool(_database_url())


def _cloud_connect() -> Any:
    import psycopg

    return psycopg.connect(_database_url(), autocommit=True)


def _ensure_cloud_table() -> None:
    with _cloud_connect() as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {CLOUD_SQLITE_TABLE} (
                name TEXT PRIMARY KEY,
                content BYTEA NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )


def sync_sqlite_from_cloud(db_path: Path) -> None:
    if not cloud_sqlite_enabled():
        return

    db_path = Path(db_path)
    name = str(db_path)
    _ensure_cloud_table()
    with _cloud_connect() as conn:
        row = conn.execute(
            f"SELECT content FROM {CLOUD_SQLITE_TABLE} WHERE name = %s",
            (name,),
        ).fetchone()
        if row:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            db_path.write_bytes(bytes(row[0]))
            return

    if db_path.exists():
        sync_sqlite_to_cloud(db_path)


def sync_sqlite_to_cloud(db_path: Path) -> None:
    if not cloud_sqlite_enabled():
        return

    db_path = Path(db_path)
    if not db_path.exists():
        return

    _ensure_cloud_table()
    with _cloud_connect() as conn:
        conn.execute(
            f"""
            INSERT INTO {CLOUD_SQLITE_TABLE} (name, content, updated_at)
            VALUES (%s, %s, now())
            ON CONFLICT (name) DO UPDATE SET
                content = EXCLUDED.content,
                updated_at = now()
            """,
            (str(db_path), db_path.read_bytes()),
        )


class CloudSyncedSQLiteConnection(sqlite3.Connection):
    _cloud_db_path: Path | None
    _cloud_dirty: bool

    def _mark_dirty(self, sql: object) -> None:
        text = str(sql or "").lstrip().lower()
        if text.startswith(MUTATING_SQL_PREFIXES):
            self._cloud_dirty = True

    def execute(self, sql: str, parameters: Any = (), /) -> sqlite3.Cursor:
        self._mark_dirty(sql)
        return super().execute(sql, parameters)

    def executemany(self, sql: str, parameters: Any, /) -> sqlite3.Cursor:
        self._mark_dirty(sql)
        return super().executemany(sql, parameters)

    def executescript(self, sql_script: str, /) -> sqlite3.Cursor:
        lowered = str(sql_script or "").lower()
        if any(prefix in lowered for prefix in MUTATING_SQL_PREFIXES):
            self._cloud_dirty = True
        return super().executescript(sql_script)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        result = super().__exit__(exc_type, exc_value, traceback)
        if exc_type is None and getattr(self, "_cloud_dirty", False) and self._cloud_db_path:
            sync_sqlite_to_cloud(self._cloud_db_path)
            self._cloud_dirty = False
        return result


def connect_synced_sqlite(db_path: Path, *, row_factory: Any | None = None) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    sync_sqlite_from_cloud(db_path)
    conn = sqlite3.connect(db_path, factory=CloudSyncedSQLiteConnection)
    conn._cloud_db_path = db_path
    conn._cloud_dirty = False
    if row_factory is not None:
        conn.row_factory = row_factory
    return conn


def reset_cloud_dirty(conn: sqlite3.Connection) -> None:
    if isinstance(conn, CloudSyncedSQLiteConnection):
        conn._cloud_dirty = False
