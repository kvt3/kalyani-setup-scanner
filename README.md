# Kalyani Setup Scanner

A Python Streamlit web app for scanning U.S. stocks after the U.S. market close for swing-trading setups. It also includes the merged stock tracking and SEC feed features from `stock_tracking_app.zip`.

## What It Scans

- Fundamentals: revenue growth YoY > 30%, EPS growth YoY > 30%, market cap > $500M, average volume > 1M shares.
- Trend: close > 21 EMA > 50 SMA > 200 SMA.
- Setup: close within 3% of the 9 EMA or 9 SMA, with a hammer, bullish engulfing, or bullish rejection candle.
- Trade levels: entry is the signal candle high, stop is the signal candle low, target is the previous swing high or 2R.

The app uses daily candles only and filters out incomplete U.S. market sessions, so it does not use premarket data.

## Files

- `app.py`: Streamlit dashboard.
- `data_loader.py`: Loads NASDAQ/NYSE tickers and downloads daily OHLCV data.
- `indicators.py`: Calculates moving averages, average volume, and trend checks.
- `patterns.py`: Detects hammer, bullish engulfing, and bullish rejection candles.
- `fundamentals.py`: Loads revenue growth, EPS growth, market cap, and volume.
- `scanner.py`: Combines all rules and ranks matches.
- `database.py`: Saves and loads daily watchlists from SQLite.
- `config.py`: App constants and thresholds.
- `stock_tracking.py`: Reads and updates the merged stock tracker database.
- `imported_stock_tracking_app/`: Original source files copied from `stock_tracking_app.zip` for reference.

## Pages

- **Run Today's Scan**: Runs selected scanner rules against the stored $500M+ ticker universe.
- **Previous Watchlists**: Opens saved scanner results by date and rule.
- **Stock Tracker**: Shows tracked stocks from the merged tracker database, lets you add/delete tickers, and opens company/news/details views.
- **SEC Stock Feeds**: Shows searchable SEC/news analysis rows from the merged tracker database.

## Scanner Rules

- **Pullback Setup**: Daily pullback near the 9 EMA/SMA with trend and fundamental filters.
- **Green Marubozu 52W Breakout**: Daily green marubozu candle breaking a prior 52-week high on above-average volume.
- **Weekly ATH Breakout**: Completed weekly candle breaks a prior all-time high, weekly volume is above the 20-week average, and weekly trend is up.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional: set a Financial Modeling Prep API key for fundamentals. Without it, the app uses `yfinance`.

```bash
export FMP_API_KEY="your_api_key_here"
```

## Run

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in your terminal, then click **Run Today's Scan**.

## Daily 9 AM IST Scheduling

Streamlit itself is interactive, so the dashboard runs scans when you click the button. For a local reminder/workflow, open the app at 9 AM IST after the U.S. market close and run the scan.

If you want a fully automated background scan later, add a small CLI wrapper around `scanner.run_scan()` and schedule it with cron or launchd for `0 9 * * 2-6` Asia/Kolkata time.

## Deploy Remotely

The simplest remote deployment path is Streamlit Community Cloud:

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create a new app from the repo.
3. Set the main file to `app.py`.
4. Add any required secrets in the Cloud dashboard instead of committing `.streamlit/secrets.toml`.

Notes:

- This app reads `.env` locally, but hosted deployments should use Streamlit secrets or provider environment variables.
- Background scheduled scans that depend on the local filesystem are better suited to a VM/container host than to a purely interactive Streamlit app.

## Notes

- Full NASDAQ + NYSE scans can take a long time because fundamentals are checked per ticker.
- Start with the default limited universe, confirm everything works, then enable the full exchange scan.
- Saved watchlists are stored in `data/watchlists.sqlite3`.
- Merged stock tracker data is stored in `data/stock_tracking/stocks.db`.
- Data providers may occasionally omit fundamentals for some tickers; those symbols are skipped and listed under data issues.
