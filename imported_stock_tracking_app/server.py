import json
import sqlite3
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

from fetch_and_store_stock import fetch_stock_overview, map_overview_to_stock, ensure_schema

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / 'stocks.db'
PORT = 8000


def fetch_rows(cursor, query, params=()):
    cursor.execute(query, params)
    rows = cursor.fetchall()
    return [dict(row) for row in rows]


def get_stock_data():
    if not DB_PATH.exists():
        return {'stocks': []}

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        stocks = fetch_rows(cursor, 'SELECT id, ticker, company_name, market_cap, description, fifty_two_week_high, beta, latest_news, summary,cik FROM stocks ORDER BY ticker')
        print(f'Fetched {len(stocks)} stocks from database')
        for stock in stocks:
            stock_id = stock['cik']
            stock['news'] = fetch_rows(
                cursor,
                'SELECT headline, datetime FROM stock_news WHERE cik = ? ORDER BY datetime DESC LIMIT 5',
                (stock_id,)
            )
            stock['earnings_growth'] = fetch_rows(
                cursor,
                'SELECT report_type, period_label, reported, estimate, surprise FROM earnings_growth WHERE cik = ? ORDER BY report_type, period_label',
                (stock_id,)
            )
            stock['revenue_growth'] = fetch_rows(
                cursor,
                'SELECT report_type, period_label, reported, estimate, surprise FROM revenue_growth WHERE cik = ? ORDER BY report_type, period_label',
                (stock_id,)
            )
            stock['quarterly_summary'] = fetch_rows(
                cursor,
                'SELECT quarter, transcript, summary FROM quarterly_summary WHERE cik = ? ORDER BY quarter',
                (stock_id,)
            )

        return {'stocks': stocks}


def create_stock(cursor, data):
    ticker = data.get('ticker', '').strip().upper()
    if not ticker:
        raise ValueError('Ticker is required')

    overview_data = fetch_stock_overview(ticker)
    stock_data = map_overview_to_stock(overview_data)

    cursor.execute(
        '''
        INSERT INTO stocks (ticker, cik, company_name, market_cap, description, fifty_two_week_high, beta)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            company_name = excluded.company_name,
            market_cap = excluded.market_cap,
            description = excluded.description,
            fifty_two_week_high = excluded.fifty_two_week_high,
            beta = excluded.beta,
            cik = excluded.cik,
            updated_at = datetime('now')
        ''',
        (
            stock_data['ticker'],
            stock_data['cik'],
            stock_data['company_name'],
            stock_data['market_cap'],
            stock_data['description'],
            stock_data['fifty_two_week_high'],
            stock_data['beta'],
        )
    )

    cursor.execute('SELECT id, ticker, company_name, market_cap, description, fifty_two_week_high, beta, latest_news, summary FROM stocks WHERE ticker = ?', (ticker,))
    row = cursor.fetchone()
    return dict(row) if row else None


def get_news_data():

    conn = sqlite3.connect("stocks.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""

        SELECT
            s.ticker,
            s.company_name,
            sn.headline,
            sn.datetime

        FROM stock_news sn

        LEFT JOIN sec_companies s
        ON sn.cik = s.cik

        ORDER BY sn.datetime DESC

        LIMIT 100

    """)

    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def delete_stock(cursor, ticker):
    cursor.execute('DELETE FROM stocks WHERE ticker = ?', (ticker.strip().upper(),))
    return cursor.rowcount > 0


class StockRequestHandler(SimpleHTTPRequestHandler):
    def _send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get('Content-Length', 0))
        raw = self.rfile.read(length) if length > 0 else b'{}'
        return json.loads(raw.decode('utf-8'))

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/stocks':
            self._send_json(get_stock_data())
            return
        # -----------------------------------
        # NEWS API
        # -----------------------------------

        if parsed.path == '/api/news':
            self._send_json(
            get_news_data()
            )
            return
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != '/api/stocks':
            self.send_error(404)
            return

        try:
            payload = self._read_json()
        except json.JSONDecodeError:
            self.send_error(400, 'Invalid JSON')
            return

        try:
            ensure_schema()
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                stock = create_stock(cursor, payload)
                conn.commit()

            if stock:
                self._send_json({'stock': stock}, status=201)
            else:
                self.send_error(500, 'Could not create stock')
        except ValueError as exc:
            self.send_error(400, str(exc))

    def do_DELETE(self):
        parsed = urlparse(self.path)
        if not parsed.path.startswith('/api/stocks'):
            self.send_error(404)
            return

        path_parts = parsed.path.strip('/').split('/')
        ticker = None
        if len(path_parts) == 2 and path_parts[0] == 'api' and path_parts[1] == 'stocks':
            query = parsed.query
            params = dict(item.split('=') for item in query.split('&') if item)
            ticker = params.get('ticker')
        elif len(path_parts) == 3 and path_parts[0] == 'api' and path_parts[1] == 'stocks':
            ticker = path_parts[2]

        if not ticker:
            self.send_error(400, 'Ticker is required')
            return

        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            deleted = delete_stock(cursor, ticker)
            conn.commit()

        if deleted:
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404, 'Ticker not found')


if __name__ == '__main__':
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, StockRequestHandler)
    print(f'Serving on http://localhost:{PORT}')
    httpd.serve_forever()
