from __future__ import annotations

import argparse
from pathlib import Path

from cloud_sqlite_store import (
    cloud_sqlite_enabled,
    cloud_sqlite_status,
    sync_sqlite_from_cloud,
    sync_sqlite_to_cloud,
)
from config import DATA_DIR, DB_PATH


TRACKING_DB_PATH = DATA_DIR / "stock_tracking" / "stocks.db"
SYNC_PATHS = (DB_PATH, TRACKING_DB_PATH)


def _sync_push(paths: tuple[Path, ...]) -> None:
    for path in paths:
        sync_sqlite_to_cloud(path)
        print(f"Uploaded {path}")


def _sync_pull(paths: tuple[Path, ...]) -> None:
    for path in paths:
        sync_sqlite_from_cloud(path)
        print(f"Restored {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync local SQLite app databases with DATABASE_URL storage.")
    parser.add_argument("direction", choices=("push", "pull", "status"), help="push uploads local DBs; pull restores them")
    parser.add_argument(
        "--if-configured",
        action="store_true",
        help="Exit successfully when DATABASE_URL is not configured.",
    )
    args = parser.parse_args()

    status = cloud_sqlite_status()
    print(f"Storage: {status}")

    if not cloud_sqlite_enabled():
        if args.if_configured:
            print("DATABASE_URL is not configured; skipping cloud DB sync.")
            return
        raise SystemExit("DATABASE_URL is not configured.")

    if args.direction == "status":
        return
    if args.direction == "push":
        _sync_push(SYNC_PATHS)
    else:
        _sync_pull(SYNC_PATHS)


if __name__ == "__main__":
    main()
