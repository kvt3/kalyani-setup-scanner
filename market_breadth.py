from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Any

import pandas as pd
import requests

from config import DATA_DIR
from data_loader import download_ohlcv, latest_completed_us_session


ISHARES_IWM_HOLDINGS_URL = (
    "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
)
ISHARES_IWM_PRODUCT_URL = "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf"
ISHARES_IWM_HOLDINGS_DOCUMENT_URL = (
    "https://www.blackrock.com/varnish-api/blk-one01-product-data/product-data/api/v1/get-fund-document"
    "?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=us-ishares&locale=en_US"
    "&portfolioId=239710&userType=individual&asOfDate={as_of_date}&component=holdings"
)
IWM_HOLDINGS_CACHE_PATH = DATA_DIR / "iwm_holdings_cache.csv"
QQQ_HOLDINGS_CACHE_PATH = DATA_DIR / "qqq_holdings_cache.csv"
DOW_HOLDINGS_CACHE_PATH = DATA_DIR / "dow_holdings_cache.csv"
NASDAQ100_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"

NASDAQ100_FALLBACK_ROWS = [
    {"Ticker": "ADBE", "Name": "Adobe Inc.", "Sector": "Technology", "Industry": "Software"},
    {"Ticker": "AMD", "Name": "Advanced Micro Devices", "Sector": "Technology", "Industry": "Semiconductors"},
    {"Ticker": "ABNB", "Name": "Airbnb", "Sector": "Consumer Discretionary", "Industry": "Diversified Commercial Services"},
    {"Ticker": "ALNY", "Name": "Alnylam Pharmaceuticals", "Sector": "Health Care", "Industry": "Biotechnology"},
    {"Ticker": "GOOGL", "Name": "Alphabet Inc. (Class A)", "Sector": "Technology", "Industry": "Software"},
    {"Ticker": "GOOG", "Name": "Alphabet Inc. (Class C)", "Sector": "Technology", "Industry": "Software"},
    {"Ticker": "AMZN", "Name": "Amazon", "Sector": "Consumer Discretionary", "Industry": "Catalog/Specialty Distribution"},
    {"Ticker": "AEP", "Name": "American Electric Power", "Sector": "Utilities", "Industry": "Electric Utilities"},
    {"Ticker": "AMGN", "Name": "Amgen", "Sector": "Health Care", "Industry": "Biotechnology"},
    {"Ticker": "ADI", "Name": "Analog Devices", "Sector": "Technology", "Industry": "Semiconductors"},
    {"Ticker": "AAPL", "Name": "Apple Inc.", "Sector": "Technology", "Industry": "Computer Hardware"},
    {"Ticker": "AMAT", "Name": "Applied Materials", "Sector": "Technology", "Industry": "Semiconductors"},
    {"Ticker": "APP", "Name": "AppLovin", "Sector": "Technology", "Industry": "Software"},
    {"Ticker": "ARM", "Name": "Arm Holdings", "Sector": "Technology", "Industry": "Semiconductors"},
    {"Ticker": "ASML", "Name": "ASML Holding", "Sector": "Technology", "Industry": "Industrial Machinery"},
    {"Ticker": "ALAB", "Name": "Astera Labs", "Sector": "Technology", "Industry": "Semiconductors"},
    {"Ticker": "ADSK", "Name": "Autodesk", "Sector": "Technology", "Industry": "Software"},
    {"Ticker": "ADP", "Name": "Automatic Data Processing", "Sector": "Industrials", "Industry": "Diversified Commercial Services"},
    {"Ticker": "AXON", "Name": "Axon Enterprise", "Sector": "Industrials", "Industry": "Ordnance & Accessories"},
    {"Ticker": "BKR", "Name": "Baker Hughes", "Sector": "Energy", "Industry": "Oil Equipment & Services"},
    {"Ticker": "BKNG", "Name": "Booking Holdings", "Sector": "Consumer Discretionary", "Industry": "Transportation Services"},
    {"Ticker": "AVGO", "Name": "Broadcom", "Sector": "Technology", "Industry": "Semiconductors"},
    {"Ticker": "CDNS", "Name": "Cadence Design Systems", "Sector": "Technology", "Industry": "Software"},
    {"Ticker": "CTAS", "Name": "Cintas", "Sector": "Industrials", "Industry": "Garments & Clothing"},
    {"Ticker": "CSCO", "Name": "Cisco", "Sector": "Telecommunications", "Industry": "Computer Communications Equipment"},
    {"Ticker": "CCEP", "Name": "Coca-Cola Europacific Partners", "Sector": "Consumer Staples", "Industry": "Soft Drinks"},
    {"Ticker": "CMCSA", "Name": "Comcast", "Sector": "Telecommunications", "Industry": "Cable & Other Pay Television Services"},
    {"Ticker": "CEG", "Name": "Constellation Energy", "Sector": "Utilities", "Industry": "Electric Utilities"},
    {"Ticker": "CPRT", "Name": "Copart", "Sector": "Consumer Discretionary", "Industry": "Retail"},
    {"Ticker": "CRWV", "Name": "CoreWeave", "Sector": "Technology", "Industry": "Computer Services"},
    {"Ticker": "COST", "Name": "Costco", "Sector": "Consumer Discretionary", "Industry": "Department/Specialty Retail Stores"},
    {"Ticker": "CRWD", "Name": "CrowdStrike", "Sector": "Technology", "Industry": "Software"},
    {"Ticker": "CSX", "Name": "CSX Corporation", "Sector": "Industrials", "Industry": "Railroads"},
    {"Ticker": "DDOG", "Name": "Datadog", "Sector": "Technology", "Industry": "Software"},
    {"Ticker": "DXCM", "Name": "DexCom", "Sector": "Health Care", "Industry": "Medical/Dental Instruments"},
    {"Ticker": "FANG", "Name": "Diamondback Energy", "Sector": "Energy", "Industry": "Oil & Gas Production"},
    {"Ticker": "DASH", "Name": "DoorDash", "Sector": "Technology", "Industry": "Software"},
    {"Ticker": "EA", "Name": "Electronic Arts", "Sector": "Consumer Discretionary", "Industry": "Miscellaneous Amusement & Recreation Services"},
    {"Ticker": "EXC", "Name": "Exelon", "Sector": "Utilities", "Industry": "Power Generation"},
    {"Ticker": "FAST", "Name": "Fastenal", "Sector": "Industrials", "Industry": "Construction & Materials"},
    {"Ticker": "FER", "Name": "Ferrovial", "Sector": "Industrials", "Industry": "Military, Government, Technical"},
    {"Ticker": "FTNT", "Name": "Fortinet", "Sector": "Technology", "Industry": "Computer Peripheral Equipment"},
    {"Ticker": "GEHC", "Name": "GE HealthCare", "Sector": "Health Care", "Industry": "Medical Electronics"},
    {"Ticker": "GILD", "Name": "Gilead Sciences", "Sector": "Health Care", "Industry": "Biotechnology"},
    {"Ticker": "HONA", "Name": "Honeywell Aerospace", "Sector": "Industrials", "Industry": "Aerospace"},
    {"Ticker": "HON", "Name": "Honeywell Technologies", "Sector": "Industrials", "Industry": "Diversified Industrials"},
    {"Ticker": "IDXX", "Name": "Idexx Laboratories", "Sector": "Health Care", "Industry": "Biotechnology"},
    {"Ticker": "INTC", "Name": "Intel", "Sector": "Technology", "Industry": "Semiconductors"},
    {"Ticker": "INTU", "Name": "Intuit", "Sector": "Technology", "Industry": "Computer Software"},
    {"Ticker": "ISRG", "Name": "Intuitive Surgical", "Sector": "Health Care", "Industry": "Industrial Specialties"},
    {"Ticker": "KDP", "Name": "Keurig Dr Pepper", "Sector": "Consumer Staples", "Industry": "Soft Drinks"},
    {"Ticker": "KLAC", "Name": "KLA Corporation", "Sector": "Technology", "Industry": "Electronic Components"},
    {"Ticker": "KHC", "Name": "Kraft Heinz", "Sector": "Consumer Staples", "Industry": "Packaged Foods"},
    {"Ticker": "LRCX", "Name": "Lam Research", "Sector": "Technology", "Industry": "Industrial Machinery"},
    {"Ticker": "LIN", "Name": "Linde plc", "Sector": "Basic Materials", "Industry": "Major Chemicals"},
    {"Ticker": "LITE", "Name": "Lumentum", "Sector": "Technology", "Industry": "Communication Equipment"},
    {"Ticker": "MAR", "Name": "Marriott International", "Sector": "Consumer Discretionary", "Industry": "Hotels/Resorts"},
    {"Ticker": "MRVL", "Name": "Marvell Technology", "Sector": "Technology", "Industry": "Semiconductors"},
    {"Ticker": "MELI", "Name": "Mercado Libre", "Sector": "Consumer Discretionary", "Industry": "Catalog/Specialty Distribution"},
    {"Ticker": "META", "Name": "Meta Platforms", "Sector": "Technology", "Industry": "Software"},
    {"Ticker": "MCHP", "Name": "Microchip Technology", "Sector": "Technology", "Industry": "Semiconductors"},
    {"Ticker": "MU", "Name": "Micron Technology", "Sector": "Technology", "Industry": "Semiconductors"},
    {"Ticker": "MSFT", "Name": "Microsoft", "Sector": "Technology", "Industry": "Software"},
    {"Ticker": "MSTR", "Name": "MicroStrategy", "Sector": "Technology", "Industry": "Software"},
    {"Ticker": "MDLZ", "Name": "Mondelez International", "Sector": "Consumer Staples", "Industry": "Packaged Foods"},
    {"Ticker": "MPWR", "Name": "Monolithic Power Systems", "Sector": "Technology", "Industry": "Semiconductors"},
    {"Ticker": "MNST", "Name": "Monster Beverage", "Sector": "Consumer Staples", "Industry": "Soft Drinks"},
    {"Ticker": "NBIS", "Name": "Nebius Group", "Sector": "Technology", "Industry": "Computer Services"},
    {"Ticker": "NFLX", "Name": "Netflix, Inc.", "Sector": "Consumer Discretionary", "Industry": "Consumer Electronics"},
    {"Ticker": "NVDA", "Name": "Nvidia", "Sector": "Technology", "Industry": "Semiconductors"},
    {"Ticker": "NXPI", "Name": "NXP Semiconductors", "Sector": "Technology", "Industry": "Semiconductors"},
    {"Ticker": "ORLY", "Name": "O'Reilly Automotive", "Sector": "Consumer Discretionary", "Industry": "Specialty Retailers"},
    {"Ticker": "ODFL", "Name": "Old Dominion Freight Line", "Sector": "Industrials", "Industry": "Trucking"},
    {"Ticker": "PCAR", "Name": "Paccar", "Sector": "Consumer Discretionary", "Industry": "Motor Vehicles"},
    {"Ticker": "PLTR", "Name": "Palantir Technologies", "Sector": "Technology", "Industry": "Software"},
    {"Ticker": "PANW", "Name": "Palo Alto Networks", "Sector": "Technology", "Industry": "Computer Peripheral Equipment"},
    {"Ticker": "PAYX", "Name": "Paychex", "Sector": "Industrials", "Industry": "Diversified Commercial Services"},
    {"Ticker": "PYPL", "Name": "PayPal", "Sector": "Industrials", "Industry": "Diversified Commercial Services"},
    {"Ticker": "PDD", "Name": "PDD Holdings", "Sector": "Technology", "Industry": "EDP Services"},
    {"Ticker": "PEP", "Name": "PepsiCo", "Sector": "Consumer Staples", "Industry": "Soft Drinks"},
    {"Ticker": "QCOM", "Name": "Qualcomm", "Sector": "Technology", "Industry": "Semiconductors"},
    {"Ticker": "REGN", "Name": "Regeneron Pharmaceuticals", "Sector": "Health Care", "Industry": "Biotechnology"},
    {"Ticker": "RKLB", "Name": "Rocket Lab", "Sector": "Industrials", "Industry": "Aerospace"},
    {"Ticker": "ROP", "Name": "Roper Technologies", "Sector": "Technology", "Industry": "Software"},
    {"Ticker": "ROST", "Name": "Ross Stores", "Sector": "Consumer Discretionary", "Industry": "Clothing/Shoe/Accessory Stores"},
    {"Ticker": "SNDK", "Name": "Sandisk", "Sector": "Technology", "Industry": "Electronic Components"},
    {"Ticker": "STX", "Name": "Seagate Technology", "Sector": "Technology", "Industry": "Electronic Components"},
    {"Ticker": "SHOP", "Name": "Shopify", "Sector": "Technology", "Industry": "Software"},
    {"Ticker": "SPCX", "Name": "SpaceX", "Sector": "Telecommunications", "Industry": "Telecommunications Services"},
    {"Ticker": "SBUX", "Name": "Starbucks", "Sector": "Consumer Discretionary", "Industry": "Restaurants"},
    {"Ticker": "SNPS", "Name": "Synopsys", "Sector": "Technology", "Industry": "Software"},
    {"Ticker": "TMUS", "Name": "T-Mobile US", "Sector": "Telecommunications", "Industry": "Telecommunications Services"},
    {"Ticker": "TTWO", "Name": "Take-Two Interactive", "Sector": "Consumer Discretionary", "Industry": "Miscellaneous Amusement & Recreation Services"},
    {"Ticker": "TER", "Name": "Teradyne", "Sector": "Technology", "Industry": "Semiconductors"},
    {"Ticker": "TSLA", "Name": "Tesla, Inc.", "Sector": "Consumer Discretionary", "Industry": "Automobiles & Parts"},
    {"Ticker": "TXN", "Name": "Texas Instruments", "Sector": "Technology", "Industry": "Semiconductors"},
    {"Ticker": "TRI", "Name": "Thomson Reuters", "Sector": "Technology", "Industry": "Software"},
    {"Ticker": "VRTX", "Name": "Vertex Pharmaceuticals", "Sector": "Health Care", "Industry": "Biotechnology"},
    {"Ticker": "WMT", "Name": "Walmart", "Sector": "Consumer Discretionary", "Industry": "Department/Specialty Retail Stores"},
    {"Ticker": "WBD", "Name": "Warner Bros. Discovery", "Sector": "Consumer Discretionary", "Industry": "Entertainment"},
    {"Ticker": "WDC", "Name": "Western Digital", "Sector": "Technology", "Industry": "Electronic Components"},
    {"Ticker": "WDAY", "Name": "Workday, Inc.", "Sector": "Technology", "Industry": "Software"},
    {"Ticker": "XEL", "Name": "Xcel Energy", "Sector": "Utilities", "Industry": "Conventional Electricity"},
]
NASDAQ100_FALLBACK_TICKERS = [row["Ticker"] for row in NASDAQ100_FALLBACK_ROWS]

DOW30_FALLBACK_ROWS = [
    {"Ticker": "MMM", "Name": "3M", "Sector": "Industrials"},
    {"Ticker": "AXP", "Name": "American Express", "Sector": "Financials"},
    {"Ticker": "AMGN", "Name": "Amgen", "Sector": "Health Care"},
    {"Ticker": "AMZN", "Name": "Amazon", "Sector": "Consumer Discretionary"},
    {"Ticker": "AAPL", "Name": "Apple", "Sector": "Information Technology"},
    {"Ticker": "BA", "Name": "Boeing", "Sector": "Industrials"},
    {"Ticker": "CAT", "Name": "Caterpillar", "Sector": "Industrials"},
    {"Ticker": "CVX", "Name": "Chevron", "Sector": "Energy"},
    {"Ticker": "CSCO", "Name": "Cisco Systems", "Sector": "Information Technology"},
    {"Ticker": "KO", "Name": "Coca-Cola", "Sector": "Consumer Staples"},
    {"Ticker": "DIS", "Name": "Disney", "Sector": "Communication Services"},
    {"Ticker": "GS", "Name": "Goldman Sachs", "Sector": "Financials"},
    {"Ticker": "HD", "Name": "Home Depot", "Sector": "Consumer Discretionary"},
    {"Ticker": "HON", "Name": "Honeywell", "Sector": "Industrials"},
    {"Ticker": "IBM", "Name": "IBM", "Sector": "Information Technology"},
    {"Ticker": "JNJ", "Name": "Johnson & Johnson", "Sector": "Health Care"},
    {"Ticker": "JPM", "Name": "JPMorgan Chase", "Sector": "Financials"},
    {"Ticker": "MCD", "Name": "McDonald's", "Sector": "Consumer Discretionary"},
    {"Ticker": "MRK", "Name": "Merck", "Sector": "Health Care"},
    {"Ticker": "MSFT", "Name": "Microsoft", "Sector": "Information Technology"},
    {"Ticker": "NVDA", "Name": "Nvidia", "Sector": "Information Technology"},
    {"Ticker": "NKE", "Name": "Nike", "Sector": "Consumer Discretionary"},
    {"Ticker": "PG", "Name": "Procter & Gamble", "Sector": "Consumer Staples"},
    {"Ticker": "CRM", "Name": "Salesforce", "Sector": "Information Technology"},
    {"Ticker": "SHW", "Name": "Sherwin-Williams", "Sector": "Materials"},
    {"Ticker": "TRV", "Name": "Travelers", "Sector": "Financials"},
    {"Ticker": "UNH", "Name": "UnitedHealth Group", "Sector": "Health Care"},
    {"Ticker": "V", "Name": "Visa", "Sector": "Financials"},
    {"Ticker": "WMT", "Name": "Walmart", "Sector": "Consumer Staples"},
    {"Ticker": "GOOGL", "Name": "Alphabet", "Sector": "Communication Services"},
]


@dataclass(frozen=True)
class IndexUniverse:
    key: str
    label: str
    proxy: str
    tickers: list[str]
    sectors: dict[str, str]
    industries: dict[str, str]
    names: dict[str, str]
    source: str


def _normalize_symbol(value: Any) -> str:
    symbol = str(value or "").strip().upper()
    if not symbol or symbol in {"NAN", "-", "--"}:
        return ""
    return symbol.replace(".", "-")


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_html_table(url: str, required_columns: tuple[str, ...]) -> pd.DataFrame:
    headers = {
        "User-Agent": (
            "KalyaniSetupScanner/1.0 "
            "(mailto:research@example.com) Python requests"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
    except requests.exceptions.SSLError:
        response = requests.get(url, headers=headers, timeout=30, verify=False)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))
    for table in tables:
        columns = {str(column).strip(): column for column in table.columns}
        if all(column in columns for column in required_columns):
            return table
    raise RuntimeError(f"Could not find table with columns: {required_columns}")


def _first_matching_column(
    table: pd.DataFrame,
    *,
    exact: tuple[str, ...] = (),
    contains: tuple[str, ...] = (),
) -> Any:
    columns = {str(column).strip(): column for column in table.columns}
    for candidate in exact:
        if candidate in columns:
            return columns[candidate]
    if contains:
        needles = tuple(needle.lower() for needle in contains)
        for column in table.columns:
            normalized = str(column).lower()
            if all(needle in normalized for needle in needles):
                return column
    return ""


def _read_ishares_holdings_csv(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    header_index = None
    for index, line in enumerate(lines):
        first_cell = line.split(",", 1)[0].strip().strip('"').lstrip("\ufeff").upper()
        if first_cell == "TICKER":
            header_index = index
            break
    if header_index is None:
        preview = " ".join(line.strip() for line in lines[:3] if line.strip())[:200]
        raise RuntimeError(f"Could not find iShares holdings CSV header. Response starts: {preview or 'empty response'}")

    table = pd.read_csv(StringIO("\n".join(lines[header_index:])))
    columns = {str(column).strip(): column for column in table.columns}
    if "Ticker" not in columns:
        raise RuntimeError(f"iShares holdings CSV missing Ticker column. Columns: {list(table.columns)[:8]}")
    return table.rename(columns={columns["Ticker"]: "Ticker"})


def _read_cached_holdings_csv(path: Any, symbol_columns: tuple[str, ...] = ("Ticker", "Symbol")) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = text.splitlines()
    header_index = None
    symbol_column = ""
    for index, line in enumerate(lines):
        cells = [cell.strip().strip('"').lstrip("\ufeff") for cell in line.split(",")]
        upper_cells = {cell.upper(): cell for cell in cells}
        for candidate in symbol_columns:
            if candidate.upper() in upper_cells:
                header_index = index
                symbol_column = upper_cells[candidate.upper()]
                break
        if header_index is not None:
            break
    if header_index is None:
        raise RuntimeError(f"Could not find symbol header in {path}")

    table = pd.read_csv(StringIO("\n".join(lines[header_index:])))
    columns = {str(column).strip(): column for column in table.columns}
    if symbol_column not in columns:
        raise RuntimeError(f"Cached holdings missing symbol column. Columns: {list(table.columns)[:8]}")
    return table.rename(columns={columns[symbol_column]: "Ticker"})


def _universe_from_table(
    *,
    key: str,
    label: str,
    proxy: str,
    table: pd.DataFrame,
    symbol_col: str,
    source: str,
    sector_col: str = "",
    industry_col: str = "",
    name_col: str = "",
) -> IndexUniverse:
    tickers = [_normalize_symbol(symbol) for symbol in table[symbol_col]]
    sectors = {
        _normalize_symbol(row[symbol_col]): str(row.get(sector_col) or "Unknown")
        for _, row in table.iterrows()
        if _normalize_symbol(row[symbol_col])
    }
    industries = {
        _normalize_symbol(row[symbol_col]): str(row.get(industry_col) or "")
        for _, row in table.iterrows()
        if _normalize_symbol(row[symbol_col])
    }
    names = {
        _normalize_symbol(row[symbol_col]): str(row.get(name_col) or "")
        for _, row in table.iterrows()
        if _normalize_symbol(row[symbol_col])
    }
    return IndexUniverse(
        key=key,
        label=label,
        proxy=proxy,
        tickers=sorted({ticker for ticker in tickers if ticker}),
        sectors=sectors,
        industries=industries,
        names=names,
        source=source,
    )


def _load_cached_universe(path: Any, key: str, label: str, proxy: str, source: str) -> IndexUniverse:
    table = _read_cached_holdings_csv(path)
    sector_col = "Sector" if "Sector" in table.columns else "GICS Sector" if "GICS Sector" in table.columns else ""
    industry_col = (
        "Industry"
        if "Industry" in table.columns
        else "Sub-Industry"
        if "Sub-Industry" in table.columns
        else "GICS Sub-Industry"
        if "GICS Sub-Industry" in table.columns
        else ""
    )
    name_col = "Name" if "Name" in table.columns else "Company" if "Company" in table.columns else "Security" if "Security" in table.columns else ""
    return _universe_from_table(
        key=key,
        label=label,
        proxy=proxy,
        table=table,
        symbol_col="Ticker",
        source=source,
        sector_col=sector_col,
        industry_col=industry_col,
        name_col=name_col,
    )


def _iwm_holdings_document_urls() -> list[str]:
    completed_date = latest_completed_us_session()
    dates = [
        (completed_date - pd.Timedelta(days=offset)).strftime("%Y%m%d")
        for offset in range(0, 8)
    ]
    urls = [ISHARES_IWM_HOLDINGS_DOCUMENT_URL.format(as_of_date=date) for date in dates]
    urls.append(ISHARES_IWM_HOLDINGS_URL)
    return urls


def _load_sp500_universe() -> IndexUniverse:
    table = _read_html_table("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", ("Symbol",))
    tickers = [_normalize_symbol(symbol) for symbol in table["Symbol"]]
    sector_col = "GICS Sector" if "GICS Sector" in table.columns else ""
    industry_col = "GICS Sub-Industry" if "GICS Sub-Industry" in table.columns else ""
    name_col = "Security" if "Security" in table.columns else ""
    sectors = {
        _normalize_symbol(row["Symbol"]): str(row.get(sector_col) or "Unknown")
        for _, row in table.iterrows()
        if _normalize_symbol(row["Symbol"])
    }
    industries = {
        _normalize_symbol(row["Symbol"]): str(row.get(industry_col) or "")
        for _, row in table.iterrows()
        if _normalize_symbol(row["Symbol"])
    }
    names = {
        _normalize_symbol(row["Symbol"]): str(row.get(name_col) or "")
        for _, row in table.iterrows()
        if _normalize_symbol(row["Symbol"])
    }
    return IndexUniverse(
        key="sp500",
        label="S&P 500",
        proxy="SPY",
        tickers=sorted({ticker for ticker in tickers if ticker}),
        sectors=sectors,
        industries=industries,
        names=names,
        source="Wikipedia S&P 500 constituents",
    )


def _load_nasdaq100_universe() -> IndexUniverse:
    symbol_col = "Ticker"
    source = "Wikipedia Nasdaq 100 constituents"
    try:
        table = _read_html_table(NASDAQ100_WIKIPEDIA_URL, ("Ticker",))
    except Exception as ticker_exc:
        try:
            table = _read_html_table(NASDAQ100_WIKIPEDIA_URL, ("Symbol",))
            symbol_col = "Symbol"
        except Exception as symbol_exc:
            if QQQ_HOLDINGS_CACHE_PATH.exists():
                return _load_cached_universe(
                    QQQ_HOLDINGS_CACHE_PATH,
                    key="nasdaq100",
                    label="QQQ / Nasdaq 100",
                    proxy="QQQ",
                    source=f"Cached QQQ/Nasdaq 100 holdings; live source unavailable: {ticker_exc}; {symbol_exc}",
                )
            return _universe_from_table(
                key="nasdaq100",
                label="QQQ / Nasdaq 100",
                proxy="QQQ",
                table=pd.DataFrame(NASDAQ100_FALLBACK_ROWS),
                symbol_col="Ticker",
                sector_col="Sector",
                industry_col="Industry",
                name_col="Name",
                source=(
                    "Fallback QQQ/Nasdaq 100 tickers from Wikipedia snapshot; live source unavailable: "
                    f"{ticker_exc}; {symbol_exc}"
                ),
            )

    sector_col = _first_matching_column(
        table,
        exact=("GICS Sector", "Sector"),
        contains=("industry",),
    )
    industry_col = _first_matching_column(
        table,
        exact=("GICS Sub-Industry", "Sub-Industry", "Industry"),
        contains=("subsector",),
    )
    name_col = _first_matching_column(table, exact=("Company", "Security", "Name"))
    cache = pd.DataFrame(
        {
            "Ticker": table[symbol_col].map(_normalize_symbol),
            "Name": table[name_col] if name_col else "",
            "Sector": table[sector_col] if sector_col else "",
            "Industry": table[industry_col] if industry_col else "",
        }
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache.dropna(subset=["Ticker"]).to_csv(QQQ_HOLDINGS_CACHE_PATH, index=False)
    return _universe_from_table(
        key="nasdaq100",
        label="QQQ / Nasdaq 100",
        proxy="QQQ",
        table=table,
        symbol_col=symbol_col,
        sector_col=sector_col,
        industry_col=industry_col,
        name_col=name_col,
        source=source,
    )


def _load_dow_universe() -> IndexUniverse:
    try:
        table = _read_html_table("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average", ("Symbol",))
        source = "Wikipedia Dow Jones Industrial Average constituents"
    except Exception as exc:
        if DOW_HOLDINGS_CACHE_PATH.exists():
            return _load_cached_universe(
                DOW_HOLDINGS_CACHE_PATH,
                key="dow30",
                label="Dow 30",
                proxy="DIA",
                source=f"Cached Dow 30 constituents; live source unavailable: {exc}",
            )
        table = pd.DataFrame(DOW30_FALLBACK_ROWS)
        source = f"Static Dow 30 fallback; live source unavailable: {exc}"

    sector_col = "Sector" if "Sector" in table.columns else "Industry" if "Industry" in table.columns else ""
    name_col = "Company" if "Company" in table.columns else "Name" if "Name" in table.columns else ""
    cache = pd.DataFrame(
        {
            "Ticker": table["Symbol"].map(_normalize_symbol) if "Symbol" in table.columns else table["Ticker"].map(_normalize_symbol),
            "Name": table[name_col] if name_col else "",
            "Sector": table[sector_col] if sector_col else "",
            "Industry": table[sector_col] if sector_col else "",
        }
    )
    if source.startswith("Wikipedia"):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        cache.dropna(subset=["Ticker"]).to_csv(DOW_HOLDINGS_CACHE_PATH, index=False)
    symbol_col = "Symbol" if "Symbol" in table.columns else "Ticker"
    return _universe_from_table(
        key="dow30",
        label="Dow 30",
        proxy="DIA",
        table=table,
        symbol_col=symbol_col,
        sector_col=sector_col,
        industry_col=sector_col,
        name_col=name_col,
        source=source,
    )


def _load_iwm_universe(fallback_tickers: list[str]) -> IndexUniverse:
    eligible_tickers = {_normalize_symbol(ticker) for ticker in fallback_tickers}
    eligible_tickers.discard("")
    source_error = ""
    try:
        headers = {
            "Accept": "text/csv,application/csv,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": ISHARES_IWM_PRODUCT_URL,
            "X-Requested-With": "XMLHttpRequest",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0 Safari/537.36"
            ),
        }
        session = requests.Session()
        try:
            session.get(ISHARES_IWM_PRODUCT_URL, headers=headers, timeout=30)
            response = None
            last_error: Exception | None = None
            for url in _iwm_holdings_document_urls():
                try:
                    candidate = session.get(url, headers=headers, timeout=30)
                    candidate.raise_for_status()
                    _read_ishares_holdings_csv(candidate.text)
                    response = candidate
                    break
                except Exception as exc:
                    last_error = exc
            if response is None:
                raise RuntimeError(f"No iShares holdings CSV response found: {last_error}")
        except requests.exceptions.SSLError:
            session.get(ISHARES_IWM_PRODUCT_URL, headers=headers, timeout=30, verify=False)
            response = None
            last_error = None
            for url in _iwm_holdings_document_urls():
                try:
                    candidate = session.get(url, headers=headers, timeout=30, verify=False)
                    candidate.raise_for_status()
                    _read_ishares_holdings_csv(candidate.text)
                    response = candidate
                    break
                except Exception as exc:
                    last_error = exc
            if response is None:
                raise RuntimeError(f"No iShares holdings CSV response found: {last_error}")
        response.raise_for_status()
        table = _read_ishares_holdings_csv(response.text)
        source = "iShares IWM holdings filtered to saved $500M+ universe"
    except Exception as exc:
        source_error = str(exc)
        if IWM_HOLDINGS_CACHE_PATH.exists():
            table = _read_ishares_holdings_csv(IWM_HOLDINGS_CACHE_PATH.read_text(encoding="utf-8-sig", errors="replace"))
            source = f"Cached iShares IWM holdings filtered to saved $500M+ universe; live source unavailable: {exc}"
        else:
            table = pd.DataFrame()
            source = f"iShares IWM holdings unavailable; Russell 2000 fallback disabled to avoid non-Russell tickers: {exc}"

    if table.empty or "Ticker" not in table.columns:
        tickers = []
        sectors = {}
        industries = {}
        names = {}
    else:
        table = table[table.get("Asset Class", "").astype(str).str.upper().eq("EQUITY")] if "Asset Class" in table.columns else table
        table["_normalized_ticker"] = table["Ticker"].map(_normalize_symbol)
        if eligible_tickers:
            table = table[table["_normalized_ticker"].isin(eligible_tickers)]
        tickers = [ticker for ticker in table["_normalized_ticker"] if ticker]
        sector_col = "Sector" if "Sector" in table.columns else ""
        sectors = {
            _normalize_symbol(row["Ticker"]): str(row.get(sector_col) or "Unknown")
            for _, row in table.iterrows()
            if _normalize_symbol(row["Ticker"])
        }
        industries = sectors.copy()
        name_col = "Name" if "Name" in table.columns else ""
        names = {
            _normalize_symbol(row["Ticker"]): str(row.get(name_col) or "")
            for _, row in table.iterrows()
            if _normalize_symbol(row["Ticker"])
        }

    if not tickers and IWM_HOLDINGS_CACHE_PATH.exists() and not source.startswith("Cached iShares"):
        table = _read_ishares_holdings_csv(IWM_HOLDINGS_CACHE_PATH.read_text(encoding="utf-8-sig", errors="replace"))
        table = table[table.get("Asset Class", "").astype(str).str.upper().eq("EQUITY")] if "Asset Class" in table.columns else table
        table["_normalized_ticker"] = table["Ticker"].map(_normalize_symbol)
        if eligible_tickers:
            table = table[table["_normalized_ticker"].isin(eligible_tickers)]
        tickers = [ticker for ticker in table["_normalized_ticker"] if ticker]
        sector_col = "Sector" if "Sector" in table.columns else ""
        sectors = {
            _normalize_symbol(row["Ticker"]): str(row.get(sector_col) or "Unknown")
            for _, row in table.iterrows()
            if _normalize_symbol(row["Ticker"])
        }
        industries = sectors.copy()
        name_col = "Name" if "Name" in table.columns else ""
        names = {
            _normalize_symbol(row["Ticker"]): str(row.get(name_col) or "")
            for _, row in table.iterrows()
            if _normalize_symbol(row["Ticker"])
        }
        if tickers:
            source = "Cached iShares IWM holdings filtered to saved $500M+ universe; live source had no matching $500M+ tickers"

    if tickers and source == "iShares IWM holdings filtered to saved $500M+ universe":
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        table.drop(columns=["_normalized_ticker"], errors="ignore").to_csv(IWM_HOLDINGS_CACHE_PATH, index=False)

    if source_error and not tickers:
        source = f"{source}; no cached IWM holdings available"

    return IndexUniverse(
        key="russell2000",
        label="Russell 2000 / IWM",
        proxy="IWM",
        tickers=sorted({ticker for ticker in tickers if ticker}),
        sectors=sectors,
        industries=industries,
        names=names,
        source=source,
    )


def load_index_universes(fallback_tickers: list[str]) -> tuple[list[IndexUniverse], list[str]]:
    universes: list[IndexUniverse] = []
    errors: list[str] = []
    for loader in (_load_sp500_universe, _load_nasdaq100_universe, _load_dow_universe):
        try:
            universes.append(loader())
        except Exception as exc:
            errors.append(str(exc))
    universes.append(_load_iwm_universe(fallback_tickers))
    return universes, errors


def _breadth_status(value: float | None, bullish_level: float, caution_level: float) -> str:
    if value is None:
        return "Unknown"
    if value >= bullish_level:
        return "Healthy"
    if value >= caution_level:
        return "Mixed"
    return "Weak"


def _breadth_label(pct_above_200: float | None, ad_ratio: float | None) -> str:
    if pct_above_200 is None:
        return "Unknown breadth"
    if pct_above_200 >= 60 and (ad_ratio or 0) >= 1:
        return "Healthy breadth"
    if pct_above_200 >= 45:
        return "Mixed / rotational breadth"
    return "Weak breadth"


def _breadth_interpretation(label: str, pct_above_200: float | None, ad_ratio: float | None) -> str:
    if pct_above_200 is None:
        return "Breadth data is unavailable."
    if "Healthy" in label:
        return "Participation is broad. Index strength is supported by many constituents."
    if "Mixed" in label:
        return "Participation is rotational. Long setups need relative strength and clean pullbacks."
    return "Participation is narrow or weak. Index gains may be fragile; reduce aggressive long exposure."


def _summarize_sector_breadth(records: pd.DataFrame, sectors: dict[str, str]) -> list[dict[str, Any]]:
    if records.empty or not sectors:
        return []
    frame = records.copy()
    frame["sector"] = frame["ticker"].map(sectors).fillna("Unknown")
    rows: list[dict[str, Any]] = []
    for sector, group in frame.groupby("sector"):
        total = len(group)
        if total == 0:
            continue
        rows.append(
            {
                "sector": str(sector),
                "advancers": int(group["advanced"].sum()),
                "decliners": int(group["declined"].sum()),
                "pct_above_50": float(group["above_50"].mean() * 100),
                "pct_above_200": float(group["above_200"].mean() * 100),
            }
        )
    return sorted(rows, key=lambda row: (row["advancers"] - row["decliners"]), reverse=True)


def _summarize_index(universe: IndexUniverse, price_data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    ad_line_parts: list[pd.Series] = []
    available_price_tickers = sum(1 for ticker in universe.tickers if ticker in price_data)

    for ticker in universe.tickers:
        frame = price_data.get(ticker)
        if frame is None or frame.empty or len(frame) < 210:
            continue
        df = frame.copy().sort_index()
        close = pd.to_numeric(df["Close"], errors="coerce")
        high = pd.to_numeric(df.get("High"), errors="coerce")
        low = pd.to_numeric(df.get("Low"), errors="coerce")
        if close.dropna().shape[0] < 210:
            continue

        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        latest = pd.DataFrame(
            {
                "close": close,
                "previous_close": close.shift(1),
                "high": high,
                "low": low,
                "sma20": sma20,
                "previous_sma20": sma20.shift(1),
                "sma50": sma50,
                "previous_sma50": sma50.shift(1),
                "sma200": sma200,
                "previous_sma200": sma200.shift(1),
                "prior_52w_high": high.shift(1).rolling(252, min_periods=200).max(),
                "prior_52w_low": low.shift(1).rolling(252, min_periods=200).min(),
            }
        ).dropna().tail(1)
        if latest.empty:
            continue

        row = latest.iloc[0]
        records.append(
            {
                "ticker": ticker,
                "date": latest.index[-1].date(),
                "name": universe.names.get(ticker, ""),
                "sector": universe.sectors.get(ticker, "Unknown"),
                "industry": universe.industries.get(ticker, ""),
                "advanced": row["close"] > row["previous_close"],
                "declined": row["close"] < row["previous_close"],
                "unchanged": row["close"] == row["previous_close"],
                "new_high": row["high"] >= row["prior_52w_high"],
                "new_low": row["low"] <= row["prior_52w_low"],
                "close": float(row["close"]),
                "high": float(row["high"]),
                "prior_52w_high": float(row["prior_52w_high"]),
                "above_20": row["close"] > row["sma20"],
                "below_20": row["close"] < row["sma20"],
                "above_50": row["close"] > row["sma50"],
                "below_50": row["close"] < row["sma50"],
                "above_200": row["close"] > row["sma200"],
                "below_200": row["close"] < row["sma200"],
                "sma20_above_50": row["sma20"] > row["sma50"],
                "sma50_above_200": row["sma50"] > row["sma200"],
                "bull_cross_20_50": row["previous_sma20"] <= row["previous_sma50"] and row["sma20"] > row["sma50"],
                "bear_cross_20_50": row["previous_sma20"] >= row["previous_sma50"] and row["sma20"] < row["sma50"],
                "bull_cross_50_200": row["previous_sma50"] <= row["previous_sma200"] and row["sma50"] > row["sma200"],
                "bear_cross_50_200": row["previous_sma50"] >= row["previous_sma200"] and row["sma50"] < row["sma200"],
            }
        )
        direction = close.diff().apply(lambda value: 1 if value > 0 else -1 if value < 0 else 0)
        ad_line_parts.append(direction.rename(ticker).tail(40))

    if not records:
        if not universe.tickers:
            error = f"No constituents loaded. Source: {universe.source}"
        elif available_price_tickers == 0:
            error = (
                "No price history downloaded for these constituents. "
                f"Source: {universe.source}"
            )
        else:
            error = (
                f"{available_price_tickers:,} constituents had downloaded data, "
                "but none had enough clean OHLCV history for breadth."
            )
        return {
            "key": universe.key,
            "label": universe.label,
            "proxy": universe.proxy,
            "source": universe.source,
            "constituents": len(universe.tickers),
            "downloaded_tickers": available_price_tickers,
            "processed_tickers": 0,
            "error": error,
        }

    breadth = pd.DataFrame(records)
    new_high_frame = breadth.loc[breadth["new_high"]].copy()
    if not new_high_frame.empty:
        new_high_frame["pct_above_prior_52w_high"] = (
            (new_high_frame["high"] / new_high_frame["prior_52w_high"] - 1) * 100
        )
    new_high_rows = [
        {
            "Ticker": str(row["ticker"]),
            "Name": str(row.get("name") or ""),
            "Sector": str(row.get("sector") or "Unknown"),
            "Industry": str(row.get("industry") or ""),
            "Close": round(float(row["close"]), 2),
            "High": round(float(row["high"]), 2),
            "Prior 52W High": round(float(row["prior_52w_high"]), 2),
            "% Above Prior 52W High": round(float(row["pct_above_prior_52w_high"]), 2),
        }
        for _, row in new_high_frame.sort_values(
            by=["sector", "ticker"],
            ascending=[True, True],
        ).iterrows()
    ]
    processed = len(breadth)
    advancers = int(breadth["advanced"].sum())
    decliners = int(breadth["declined"].sum())
    unchanged = int(breadth["unchanged"].sum())
    ad_ratio = advancers / decliners if decliners else None
    pct_above_20 = float(breadth["above_20"].mean() * 100)
    pct_above_50 = float(breadth["above_50"].mean() * 100)
    pct_above_200 = float(breadth["above_200"].mean() * 100)
    label = _breadth_label(pct_above_200, ad_ratio)

    ad_line_rows: list[dict[str, Any]] = []
    ad_line_trend = "Unknown"
    if ad_line_parts:
        ad_frame = pd.concat(ad_line_parts, axis=1).fillna(0)
        daily_net = ad_frame.sum(axis=1)
        ad_line = daily_net.cumsum()
        recent = ad_line.tail(10)
        if len(recent) >= 2:
            ad_line_trend = "Rising" if recent.iloc[-1] > recent.iloc[0] else "Falling" if recent.iloc[-1] < recent.iloc[0] else "Flat"
        ad_line_rows = [
            {"date": str(index.date()), "net_advancers": int(daily_net.loc[index]), "ad_line": int(ad_line.loc[index])}
            for index in ad_line.tail(20).index
        ]

    return {
        "key": universe.key,
        "label": universe.label,
        "proxy": universe.proxy,
        "source": universe.source,
        "constituents": len(universe.tickers),
        "downloaded_tickers": available_price_tickers,
        "processed_tickers": processed,
        "date": str(breadth["date"].max()),
        "breadth_label": label,
        "interpretation": _breadth_interpretation(label, pct_above_200, ad_ratio),
        "advancers": advancers,
        "decliners": decliners,
        "unchanged": unchanged,
        "advance_decline_ratio": ad_ratio,
        "new_highs": int(breadth["new_high"].sum()),
        "new_high_tickers": sorted(breadth.loc[breadth["new_high"], "ticker"].astype(str).tolist()),
        "new_high_rows": new_high_rows,
        "new_lows": int(breadth["new_low"].sum()),
        "new_low_tickers": sorted(breadth.loc[breadth["new_low"], "ticker"].astype(str).tolist()),
        "above_20": int(breadth["above_20"].sum()),
        "below_20": int(breadth["below_20"].sum()),
        "above_50": int(breadth["above_50"].sum()),
        "below_50": int(breadth["below_50"].sum()),
        "above_200": int(breadth["above_200"].sum()),
        "below_200": int(breadth["below_200"].sum()),
        "pct_above_20": pct_above_20,
        "pct_above_50": pct_above_50,
        "pct_above_200": pct_above_200,
        "pct_above_20_status": _breadth_status(pct_above_20, 60, 45),
        "pct_above_50_status": _breadth_status(pct_above_50, 60, 45),
        "pct_above_200_status": _breadth_status(pct_above_200, 60, 45),
        "sma20_above_50": int(breadth["sma20_above_50"].sum()),
        "sma50_above_200": int(breadth["sma50_above_200"].sum()),
        "bull_cross_20_50": int(breadth["bull_cross_20_50"].sum()),
        "bear_cross_20_50": int(breadth["bear_cross_20_50"].sum()),
        "bull_cross_50_200": int(breadth["bull_cross_50_200"].sum()),
        "bear_cross_50_200": int(breadth["bear_cross_50_200"].sum()),
        "ad_line_trend": ad_line_trend,
        "ad_line_rows": ad_line_rows,
        "sector_breadth": _summarize_sector_breadth(breadth, universe.sectors),
    }


def run_market_breadth_scan(fallback_tickers: list[str], max_tickers: int | None = None) -> dict[str, Any]:
    completed_date = latest_completed_us_session()
    universes, universe_errors = load_index_universes(fallback_tickers)
    if not universes:
        return {
            "completed_date": str(completed_date.date()),
            "indexes": [],
            "errors": universe_errors or ["No index universes found."],
        }

    all_symbols = sorted({ticker for universe in universes for ticker in universe.tickers})
    if max_tickers:
        all_symbols = all_symbols[:max_tickers]
    price_data = download_ohlcv(all_symbols, completed_date=completed_date, use_nasdaq_fallback=False)

    index_results = [_summarize_index(universe, price_data) for universe in universes]
    return {
        "completed_date": str(completed_date.date()),
        "indexes": index_results,
        "downloaded_tickers": len(price_data),
        "requested_tickers": len(all_symbols),
        "errors": universe_errors,
    }
