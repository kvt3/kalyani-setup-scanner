from __future__ import annotations

import argparse

from stock_tracking import poll_sec_8k_feed_once, run_sec_8k_feed_poller


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll SEC current 8-K and Form 4 filings for saved $500M+ tickers.")
    parser.add_argument("--once", action="store_true", help="Run one poll and exit.")
    parser.add_argument("--interval", type=int, default=300, help="Polling interval in seconds. Default: 300.")
    parser.add_argument("--count", type=int, default=100, help="SEC current-feed row count. Default: 100.")
    parser.add_argument("--force", action="store_true", help="Regenerate summaries even when an accession was already summarized.")
    args = parser.parse_args()

    if args.once:
        result = poll_sec_8k_feed_once(count=args.count, progress=True, force_resummarize=args.force)
        print(
            f"checked={result['checked']} matched={result['matched']} "
            f"inserted={result['inserted']} status={result['status']}"
        )
        return

    run_sec_8k_feed_poller(interval_seconds=args.interval, count=args.count, force_resummarize=args.force)


if __name__ == "__main__":
    main()
