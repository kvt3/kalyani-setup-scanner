from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import yfinance as yf


def _parse_news_item(item: dict[str, Any]) -> dict[str, str] | None:
    content = item.get("content", item)
    title = content.get("title")
    provider = content.get("provider", {}).get("displayName", "")
    url = (
        content.get("canonicalUrl", {}).get("url")
        or content.get("clickThroughUrl", {}).get("url")
        or content.get("link")
        or ""
    )
    published = content.get("pubDate") or content.get("displayTime") or item.get("providerPublishTime")

    if not title:
        return None

    if isinstance(published, int | float):
        published_text = datetime.fromtimestamp(published, tz=timezone.utc).isoformat()
    else:
        published_text = str(published or "")

    return {
        "catalyst_title": str(title),
        "catalyst_source": str(provider),
        "catalyst_date": published_text[:10],
        "catalyst_url": str(url),
    }


def _is_recent(date_text: str, signal_date: str, lookback_days: int) -> bool:
    if not date_text:
        return False
    published = pd.to_datetime(date_text, utc=True, errors="coerce")
    signal = pd.to_datetime(signal_date, utc=True, errors="coerce")
    if pd.isna(published) or pd.isna(signal):
        return False
    start = signal - timedelta(days=lookback_days)
    end = datetime.now(timezone.utc) + timedelta(days=1)
    return start <= published <= end


def get_recent_catalyst(ticker: str, signal_date: str, lookback_days: int = 7) -> dict[str, str] | None:
    try:
        news = yf.Ticker(ticker).news or []
    except Exception:
        return None

    for item in news:
        parsed = _parse_news_item(item)
        if parsed and _is_recent(parsed["catalyst_date"], signal_date, lookback_days):
            return parsed
    return None
