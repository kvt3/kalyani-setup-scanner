import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List

import requests




BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / 'stocks.db'
SCHEMA_PATH = BASE_DIR / 'schema.sql'
API_KEY = 'E0XRCLUA3ANRR2T1'
OVERVIEW_URL = 'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={api_key}'

REQUIRED_COLUMNS = {
    'company_name': 'TEXT',
    'description': 'TEXT',
    'fifty_two_week_high': 'TEXT',
    'beta': 'REAL',
}


def load_schema() -> str:
    return SCHEMA_PATH.read_text(encoding='utf-8')


def initialize_database() -> None:
    schema_sql = load_schema()
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(schema_sql)
        conn.commit()


def get_existing_columns(cursor: sqlite3.Cursor, table_name: str) -> List[str]:
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [row[1] for row in cursor.fetchall()]


def ensure_schema() -> None:
    if not DB_PATH.exists():
        initialize_database()
        return

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        existing_columns = get_existing_columns(cursor, 'stocks')
        for column_name, column_type in REQUIRED_COLUMNS.items():
            if column_name not in existing_columns:
                cursor.execute(
                    f'ALTER TABLE stocks ADD COLUMN {column_name} {column_type}'
                )
        existing_columns = get_existing_columns(cursor, 'stock_news')
        for column_name, column_type in REQUIRED_COLUMNS.items():
            if column_name not in existing_columns:
                cursor.execute(
                    f'ALTER TABLE stock_news ADD COLUMN {column_name} {column_type}'
                )
        existing_columns = get_existing_columns(cursor, 'revenue_growth')
        for column_name, column_type in REQUIRED_COLUMNS.items():
            if column_name not in existing_columns:
                cursor.execute(
                    f'ALTER TABLE revenue_growth ADD COLUMN {column_name} {column_type}'
                )
        existing_columns = get_existing_columns(cursor, 'earnings_growth')
        for column_name, column_type in REQUIRED_COLUMNS.items():
            if column_name not in existing_columns:
                cursor.execute(
                    f'ALTER TABLE earnings_growth ADD COLUMN {column_name} {column_type}'
                )
        existing_columns = get_existing_columns(cursor, 'quarterly_summary')
        for column_name, column_type in REQUIRED_COLUMNS.items():
            if column_name not in existing_columns:
                cursor.execute(
                    f'ALTER TABLE quarterly_summary ADD COLUMN {column_name} {column_type}'
                )
        conn.commit()


def fetch_stock_overview(symbol: str) -> Dict[str, Any]:
    url = OVERVIEW_URL.format(symbol=symbol, api_key=API_KEY)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    if not data or 'Error Message' in data or 'Note' in data or 'Symbol' not in data:
        error_text = data.get('Note') or data.get('Error Message') or 'No valid overview data returned.'
        raise RuntimeError(f'Alpha Vantage error for {symbol}: {error_text}')

    return data


def map_overview_to_stock(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'cik': data.get('CIK').zfill(10),
        'ticker': data.get('Symbol', '').strip().upper(),
        'company_name': data.get('Name'),
        'market_cap': data.get('MarketCapitalization'),
        'description': data.get('Description'),
        'fifty_two_week_high': data.get('52WeekHigh'),
        'beta': _safe_float(data.get('Beta')),
    }


def _safe_float(value: Any) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def get_company_name_from_cik(cik: str) -> str: 
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT company_name FROM sec_companies WHERE cik = ?", (cik,))
        result = cursor.fetchone()
        return result[0] if result else None
    
def get_company_ticker_from_cik(cik: str) -> str:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT ticker FROM sec_companies WHERE cik = ?", (cik,))
        result = cursor.fetchone()
        return result[0] if result else None

def get_cik_from_ticker(ticker: str) -> str:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT cik FROM sec_companies WHERE ticker = ?", (ticker,))
        result = cursor.fetchone()
        return result[0] if result else None

def save_stock(data: Dict[str, Any]) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            '''
            INSERT INTO stocks (ticker, company_name, market_cap, description, fifty_two_week_high, beta)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                company_name = excluded.company_name,
                market_cap = excluded.market_cap,
                description = excluded.description,
                fifty_two_week_high = excluded.fifty_two_week_high,
                beta = excluded.beta,
                updated_at = datetime('now')
            ''',
            (
                data['ticker'],
                data['company_name'],
                data['market_cap'],
                data['description'],
                data['fifty_two_week_high'],
                data['beta'],
            )
        )
        conn.commit()

def save_stock_news(cik: str, headline: str, datetime_str: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            '''
            INSERT INTO stock_news (cik, headline, datetime)
            VALUES (?, ?, ?)
            ''',
            (cik, headline, datetime_str)
        )
        conn.commit()

def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python fetch_and_store_stock.py <TICKER>')
        return 1

    symbol = sys.argv[1].strip().upper()
    if not symbol:
        print('Ticker symbol is required.')
        return 1

    ensure_schema()
    try:
        overview_data = fetch_stock_overview(symbol)
        stock_data = map_overview_to_stock(overview_data)
        print(f"Fetched overview for {overview_data}:")
        save_stock(stock_data)
        print(f"Stored data for {symbol}:")
        print(f"  Company name: {stock_data['company_name']}")
        print(f"  Market cap: {stock_data['market_cap']}")
        print(f"  52-week high: {stock_data['fifty_two_week_high']}")
        print(f"  Beta: {stock_data['beta']}")
        print(f"  Description: {stock_data['description']}")
        print(f"  CIK: {stock_data['cik']}")
    except Exception as exc:
        print(f'Failed to store {symbol}: {exc}')
        return 1
    
    ''' try:
        eps_revenue_data = fetch_quarterly_data(symbol)
        stock_data = process_and_analyze_data(eps_revenue_data)
        save_stock(stock_data)
        print(f"Stored data for {symbol}:")
        print(f"  Company name: {stock_data['company_name']}")
        print(f"  Market cap: {stock_data['market_cap']}")
        print(f"  52-week high: {stock_data['fifty_two_week_high']}")
        print(f"  Beta: {stock_data['beta']}")
        print(f"  Description: {stock_data['description']}")
    except Exception as exc:
        print(f'Failed to store {symbol}: {exc}')
        return 1'''

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
