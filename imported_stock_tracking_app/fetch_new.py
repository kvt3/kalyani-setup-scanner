
import re
import sqlite3
import requests
import feedparser
import trafilatura

from bs4 import BeautifulSoup
from datetime import datetime

from prompts import prompts
from summries import summarize_content_with_ollama

from fetch_and_store_stock import (
    get_company_name_from_cik,
    get_company_ticker_from_cik,
    save_stock_news
)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DATABASE = "stocks.db"

HEADERS = {
    "User-Agent": "MyApp myemail@example.com"
}

SEC_FEED_URL = (
    "https://www.sec.gov/cgi-bin/"
    "browse-edgar?action=getcurrent"
    "&type=8-K&output=atom"
)

IMPORTANT_ITEMS = {
    '1.01',
    '1.02',
    '1.03',
    '1.05',
    '2.01',
    '2.02',
    '2.03',
    '2.05',
    '2.06',
    '3.01',
    '4.02',
    '5.02',
    '8.01'
}


# --------------------------------------------------
# DATABASE
# --------------------------------------------------

def get_db():

    conn = sqlite3.connect(DATABASE)

    conn.row_factory = sqlite3.Row

    return conn


# --------------------------------------------------
# CREATE TABLE
# --------------------------------------------------

def create_tables():

    conn = get_db()

    cursor = conn.cursor()

    cursor.execute("""

        CREATE TABLE IF NOT EXISTS processed_filings (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            accession_number TEXT,

            cik TEXT,

            item TEXT,

            filing_url TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            UNIQUE(accession_number, item)

        )

    """)

    conn.commit()

    conn.close()


# --------------------------------------------------
# EXTRACT ACCESSION NUMBER
# --------------------------------------------------

def extract_accession_number(filing_url):

    match = re.search(
        r'/data/\d+/(\d+)/',
        filing_url
    )

    if not match:

        return None

    accession_raw = match.group(1)

    accession_number = (
        f"{accession_raw[:10]}-"
        f"{accession_raw[10:12]}-"
        f"{accession_raw[12:]}"
    )

    return accession_number


# --------------------------------------------------
# CHECK DUPLICATE
# --------------------------------------------------

def filing_exists(
    accession_number,
    item
):

    conn = get_db()

    cursor = conn.cursor()

    cursor.execute("""

        SELECT id
        FROM processed_filings
        WHERE accession_number = ?
        AND item = ?

    """, (
        accession_number,
        item
    ))

    exists = cursor.fetchone() is not None

    conn.close()

    return exists


# --------------------------------------------------
# MARK PROCESSED
# --------------------------------------------------

def mark_filing_processed(
    accession_number,
    cik,
    item,
    filing_url
):

    conn = get_db()

    cursor = conn.cursor()

    cursor.execute("""

        INSERT OR IGNORE INTO processed_filings (

            accession_number,
            cik,
            item,
            filing_url

        )

        VALUES (?, ?, ?, ?)

    """, (
        accession_number,
        cik,
        item,
        filing_url
    ))

    conn.commit()

    conn.close()


# --------------------------------------------------
# NORMALIZE URL
# --------------------------------------------------

def normalize_sec_url(url):

    if "ix?doc=" in url or "?doc=" in url:

        path = url.split("?doc=")[1]

        return "https://www.sec.gov" + path

    return url


# --------------------------------------------------
# EXTRACT ITEM TEXT
# --------------------------------------------------

def extract_items(html):

    text = trafilatura.extract(html)

    if not text:

        return {}

    text = re.sub(r'[ \t]+', ' ', text)

    item_pattern = re.compile(
        r'Item\s*\n?\s*(\d+\.\d+)',
        re.IGNORECASE
    )

    matches = list(item_pattern.finditer(text))

    items = {}

    for i, match in enumerate(matches):

        item_number = match.group(1)

        if item_number not in IMPORTANT_ITEMS:

            continue

        start = match.end()

        if i + 1 < len(matches):

            end = matches[i + 1].start()

        else:

            sig_match = re.search(
                r'SIGNATURES?',
                text[start:],
                re.IGNORECASE
            )

            if sig_match:

                end = start + sig_match.start()

            else:

                end = len(text)

        section = text[start:end].strip()

        if section:

            items[item_number] = section

    return items


# --------------------------------------------------
# FETCH FILING HTML
# --------------------------------------------------

def fetch_filing_html(url):

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30
    )

    return response.text


# --------------------------------------------------
# PROCESS SINGLE FILING
# --------------------------------------------------

def process_filing(filing_url):

    print(f"\nProcessing filing:\n{filing_url}\n")

    accession_number = extract_accession_number(
        filing_url
    )

    if not accession_number:

        print("Could not extract accession number")

        return

    html = fetch_filing_html(filing_url)

    items = extract_items(html)

    cik_match = re.search(
        r'/data/(\d+)/',
        filing_url
    )

    if not cik_match:

        return

    cik = cik_match.group(1).zfill(10)

    company_name = get_company_name_from_cik(cik)

    ticker = get_company_ticker_from_cik(cik)

    print(
        f"{company_name} ({ticker})"
    )

    print(
        f"Accession Number: {accession_number}"
    )

    for item, text in items.items():

        # --------------------------------------
        # SKIP DUPLICATES
        # --------------------------------------

        if filing_exists(
            accession_number,
            item
        ):

            print(
                f"Skipping duplicate Item {item}"
            )

            continue

        prompt = prompts.get(item)

        if not prompt:

            continue

        print(f"Summarizing Item {item}")

        summary = summarize_content_with_ollama(
            text,
            prompt
        )

        if not summary:

            continue

        save_stock_news(
            cik,
            summary,
            datetime.now().isoformat()
        )

        mark_filing_processed(
            accession_number,
            cik,
            item,
            filing_url
        )

        print(f"Saved Item {item}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    #create_tables()

    response = requests.get(
        SEC_FEED_URL,
        headers=HEADERS
    )

    feed = feedparser.parse(response.text)

    print(
        f"Feed entries: {len(feed.entries)}"
    )

    for entry in feed.entries:

        try:

            filing_index_url = entry.link

            index_html = fetch_filing_html(
                filing_index_url
            )

            soup = BeautifulSoup(
                index_html,
                "html.parser"
            )

            for a in soup.find_all("a"):

                href = a.get("href", "")

                if (
                    ".htm" in href and
                    "Archives" in href
                ):

                    filing_url = (
                        "https://www.sec.gov" + href
                    )

                    filing_url = normalize_sec_url(
                        filing_url
                    )

                    process_filing(filing_url)

                    break

        except Exception as e:

            print(
                f"ERROR: {str(e)}"
            )


# --------------------------------------------------

if __name__ == "__main__":

    main()
