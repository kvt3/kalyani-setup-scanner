with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(schema_sql)
        conn.row_factory = sqlite3.Row

        cursor = conn.cursor()
        for stock in SAMPLE_STOCKS:
            cursor.execute(
                '''
                INSERT OR IGNORE INTO stocks (ticker, market_cap, company_name, fifty_two_week_high, latest_news, beta, summary)
                VALUES (?, ?, ?, ?, ?, ?,?)
                ''',
                (stock['ticker'], stock['market_cap'], stock['company_name'], stock['fifty_two_week_high'], stock['latest_news'], stock['beta'],stock['summary'])
            )

        for ticker, headline, news_datetime in NEWS_ROWS:
            cursor.execute('SELECT id FROM stocks WHERE ticker = ?', (ticker,))
            stock_id = cursor.fetchone()['id']
            cursor.execute(
                '''
                INSERT INTO stock_news (stock_id, headline, datetime)
                VALUES (?, ?, ?)
                ''',
                (stock_id, headline, news_datetime)
            )

        for ticker, report_type, period_label, reported, estimate, surprise in EARNINGS_ROWS:
            cursor.execute('SELECT id FROM stocks WHERE ticker = ?', (ticker,))
            stock_id = cursor.fetchone()['id']
            cursor.execute(
                '''
                INSERT INTO earnings_growth (stock_id, metric, report_type, period_label, reported, estimate, surprise)
                VALUES (?, 'EPS', ?, ?, ?, ?, ?)
                ''',
                (stock_id, report_type, period_label, reported, estimate, surprise)
            )

        for ticker, report_type, period_label, reported, estimate, surprise in REVENUE_ROWS:
            cursor.execute('SELECT id FROM stocks WHERE ticker = ?', (ticker,))
            stock_id = cursor.fetchone()['id']
            cursor.execute(
                '''
                INSERT INTO revenue_growth (stock_id, report_type, period_label, reported, estimate, surprise)
                VALUES (?, ?, ?, ?, ?, ?)
                ''',
                (stock_id, report_type, period_label, reported, estimate, surprise)
            )

        for ticker, quarter, transcript, summary in QUARTERLY_SUMMARY_ROWS:
            cursor.execute('SELECT id FROM stocks WHERE ticker = ?', (ticker,))
            stock_id = cursor.fetchone()['id']
            cursor.execute(
                '''
                INSERT INTO quarterly_summary (stock_id, quarter, transcript, summary)
                VALUES (?, ?, ?, ?)
                ''',
                (stock_id, quarter, transcript, summary)
            )

        conn.commit()
    print(f'Created SQLite database at: {DB_PATH}')
