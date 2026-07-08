from __future__ import annotations

from config import MIN_MARKET_CAP
from database import save_eligible_tickers
from universe_builder import build_eligible_market_cap_universe


def main() -> None:
    result = build_eligible_market_cap_universe(min_market_cap=MIN_MARKET_CAP)
    saved = save_eligible_tickers(
        result.rows,
        source=result.source,
        min_market_cap=MIN_MARKET_CAP,
        total_tickers=result.total_tickers,
        error_count=len(result.errors),
    )
    print(
        f"Market-cap universe refreshed. source={result.source}, checked={result.total_tickers}, "
        f"stored={saved}, market_cap_gaps={len(result.errors)}"
    )


if __name__ == "__main__":
    main()
