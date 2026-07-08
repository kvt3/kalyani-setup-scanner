
import sqlite3
import requests

headers = {
    "User-Agent": "MyApp myemail@example.com"
}

url = "https://www.sec.gov/files/company_tickers.json"

data = requests.get(url, headers=headers).json()

# ---------------------------------------
# SQLITE
# ---------------------------------------

conn = sqlite3.connect("stocks.db")

cursor = conn.cursor()


# ---------------------------------------
# INSERT DATA
# ---------------------------------------

for item in data.values():

    ticker = item.get("ticker")

    company_name = item.get("title")

    cik = str(item.get("cik_str")).zfill(10)

    cursor.execute("""
        INSERT OR IGNORE INTO sec_companies
        (ticker, company_name, cik)
        VALUES (?, ?, ?)
    """, (ticker, company_name, cik))

conn.commit()

print("DONE")

conn.close()

