from __future__ import annotations

import os
from pathlib import Path


def _load_local_env() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_local_env()


APP_NAME = "Kalyani Setup Scanner"
DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "watchlists.sqlite3"

MIN_REVENUE_GROWTH = 0.30
MIN_EPS_GROWTH = 0.30
MIN_MARKET_CAP = 500_000_000
MIN_AVG_VOLUME = 1_000_000

PULLBACK_DISTANCE = 0.03
HISTORY_PERIOD = "1y"
PRICE_CHUNK_SIZE = 40

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
