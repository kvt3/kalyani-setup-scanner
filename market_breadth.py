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

NASDAQ100_FALLBACK_TICKERS = [
    "AAPL",
    "ABNB",
    "ADBE",
    "ADI",
    "ADP",
    "ADSK",
    "AEP",
    "ALNY",
    "AMAT",
    "AMD",
    "AMGN",
    "AMZN",
    "APP",
    "ARM",
    "ASML",
    "AVGO",
    "AXON",
    "BKR",
    "CCEP",
    "CHTR",
    "CMCSA",
    "COST",
    "CPRT",
    "CRWD",
    "CSCO",
    "CSGP",
    "CSX",
    "CTAS",
    "DASH",
    "DDOG",
    "DXCM",
    "EA",
    "EXC",
    "FANG",
    "FAST",
    "FER",
    "FTNT",
    "GEHC",
    "GILD",
    "GOOG",
    "GOOGL",
    "HON",
    "IDXX",
    "INSM",
    "INTC",
    "INTU",
    "ISRG",
    "KDP",
    "KHC",
    "KLAC",
    "LIN",
    "LRCX",
    "MAR",
    "MCHP",
    "MDLZ",
    "MELI",
    "META",
    "MNST",
    "MRVL",
    "MSFT",
    "MSTR",
    "MU",
    "NFLX",
    "NVDA",
    "NXPI",
    "ODFL",
    "ORLY",
    "PANW",
    "PAYX",
    "PCAR",
    "PDD",
    "PEP",
    "PLTR",
    "PYPL",
    "QCOM",
    "REGN",
    "ROP",
    "ROST",
    "SBUX",
    "SNDK",
    "SNPS",
    "STX",
    "TEAM",
    "TMUS",
    "TSLA",
    "TTD",
    "TTWO",
    "TXN",
    "VRSK",
    "VRTX",
    "WBD",
    "WDAY",
    "WMT",
    "XEL",
    "ZS",
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
        table = _read_html_table("https://en.wikipedia.org/wiki/Nasdaq-100", ("Ticker",))
    except Exception as ticker_exc:
        try:
            table = _read_html_table("https://en.wikipedia.org/wiki/Nasdaq-100", ("Symbol",))
            symbol_col = "Symbol"
        except Exception as symbol_exc:
            tickers = NASDAQ100_FALLBACK_TICKERS
            return IndexUniverse(
                key="nasdaq100",
                label="QQQ / Nasdaq 100",
                proxy="QQQ",
                tickers=sorted({ticker for ticker in tickers if ticker}),
                sectors={ticker: "Nasdaq 100 fallback" for ticker in tickers if ticker},
                industries={ticker: "" for ticker in tickers if ticker},
                names={ticker: "" for ticker in tickers if ticker},
                source=(
                    "Fallback QQQ/Nasdaq 100 tickers; live source unavailable: "
                    f"{ticker_exc}; {symbol_exc}"
                ),
            )

    tickers = [_normalize_symbol(symbol) for symbol in table[symbol_col]]
    sector_col = "GICS Sector" if "GICS Sector" in table.columns else "Sector" if "Sector" in table.columns else ""
    industry_col = "GICS Sub-Industry" if "GICS Sub-Industry" in table.columns else "Sub-Industry" if "Sub-Industry" in table.columns else ""
    name_col = "Company" if "Company" in table.columns else "Security" if "Security" in table.columns else "Name" if "Name" in table.columns else ""
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
        key="nasdaq100",
        label="QQQ / Nasdaq 100",
        proxy="QQQ",
        tickers=sorted({ticker for ticker in tickers if ticker}),
        sectors=sectors,
        industries=industries,
        names=names,
        source=source,
    )


def _load_dow_universe() -> IndexUniverse:
    table = _read_html_table("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average", ("Symbol",))
    tickers = [_normalize_symbol(symbol) for symbol in table["Symbol"]]
    sector_col = "Industry" if "Industry" in table.columns else ""
    name_col = "Company" if "Company" in table.columns else ""
    sectors = {
        _normalize_symbol(row["Symbol"]): str(row.get(sector_col) or "Unknown")
        for _, row in table.iterrows()
        if _normalize_symbol(row["Symbol"])
    }
    names = {
        _normalize_symbol(row["Symbol"]): str(row.get(name_col) or "")
        for _, row in table.iterrows()
        if _normalize_symbol(row["Symbol"])
    }
    return IndexUniverse(
        key="dow30",
        label="Dow 30",
        proxy="DIA",
        tickers=sorted({ticker for ticker in tickers if ticker}),
        sectors=sectors,
        industries=sectors,
        names=names,
        source="Wikipedia Dow Jones Industrial Average constituents",
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
