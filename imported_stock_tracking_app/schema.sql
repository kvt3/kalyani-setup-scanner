BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS stocks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL UNIQUE,
    company_name TEXT,
    market_cap TEXT,
    description TEXT,
    fifty_two_week_high TEXT,
    beta REAL,
    latest_news TEXT,
    summary TEXT,
    cik TEXT NOT NULL UNIQUE,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sec_companies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    company_name TEXT NOT NULL,
    cik TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Faster lookup indexes
CREATE INDEX IF NOT EXISTS idx_sec_ticker
ON sec_companies(ticker);

CREATE INDEX IF NOT EXISTS idx_sec_company
ON sec_companies(company_name);

CREATE TABLE IF NOT EXISTS stock_news (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cik TEXT NOT NULL,
    headline TEXT NOT NULL,
    datetime TEXT,
    FOREIGN KEY(cik) REFERENCES sec_companies(cik) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS earnings_growth (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cik TEXT NOT NULL,
    metric TEXT NOT NULL CHECK(metric IN ('EPS', 'Revenue')),
    report_type TEXT NOT NULL CHECK(report_type IN ('annual', 'quarterly')),
    period_label TEXT NOT NULL,
    reported TEXT,
    estimate TEXT,
    surprise TEXT,
    FOREIGN KEY(cik) REFERENCES sec_companies(cik) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS revenue_growth (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cik TEXT NOT NULL,
    report_type TEXT NOT NULL CHECK(report_type IN ('annual', 'quarterly')),
    period_label TEXT NOT NULL,
    reported TEXT,
    estimate TEXT,
    surprise TEXT,
    FOREIGN KEY(cik) REFERENCES sec_companies(cik) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS quarterly_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cik TEXT NOT NULL,
    quarter TEXT NOT NULL,
    transcript TEXT,
    summary TEXT,
    FOREIGN KEY(cik) REFERENCES sec_companies(cik) ON DELETE CASCADE
);

COMMIT;
