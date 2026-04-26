"""
Price and volume data for Nordic stocks via yfinance.

Nasdaq Stockholm tickers use the ".ST" suffix (e.g. "ERIC-B.ST").
This module handles ISIN → ticker mapping, OHLCV fetching, and
ADV-based position sizing caps.

Position size cap (from strategy.yaml):
    max_position_sek = min(
        portfolio_value * max_portfolio_pct,
        adv_sek * max_pct_of_adv / 100
    )
This links sizing directly to liquidity rather than an arbitrary multiplier.
"""

import logging
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "price_cache.db"

# Known ISIN → Yahoo ticker mappings for common Nordic stocks.
# yfinance does not support ISIN lookup natively — extend this as needed.
# For unknown ISINs, the resolver tries common suffix patterns.
ISIN_TICKER_MAP: dict[str, str] = {
    "SE0000108656": "ERIC-B.ST",
    "SE0000667925": "SAND.ST",
    "SE0000115446": "VOLV-B.ST",
    "SE0015988019": "SBB-B.ST",
}


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS isin_ticker (
            isin    TEXT PRIMARY KEY,
            ticker  TEXT NOT NULL,
            source  TEXT,
            updated TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv_cache (
            ticker TEXT NOT NULL,
            date   TEXT NOT NULL,
            open   REAL,
            high   REAL,
            low    REAL,
            close  REAL,
            volume REAL,
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.commit()


def _get_conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    _init_db(conn)
    return conn


# ── ISIN → Ticker resolution ────────────────────────────────────────────────

def resolve_ticker(isin: str, db_path: Path = DB_PATH) -> Optional[str]:
    """
    Resolve an ISIN to a Yahoo Finance ticker.
    Order: hardcoded map → DB cache → yfinance search → None.
    """
    if isin in ISIN_TICKER_MAP:
        return ISIN_TICKER_MAP[isin]

    conn = _get_conn(db_path)
    row = conn.execute(
        "SELECT ticker FROM isin_ticker WHERE isin = ?", (isin,)
    ).fetchone()
    conn.close()
    if row:
        return row[0]

    ticker = _search_yfinance(isin)
    if ticker:
        _cache_ticker(isin, ticker, "yfinance_search", db_path)
    return ticker


def _search_yfinance(isin: str) -> Optional[str]:
    """Attempt yfinance ticker lookup by ISIN. Best-effort, not guaranteed."""
    try:
        t = yf.Ticker(isin)
        info = t.info
        if info and info.get("regularMarketPrice"):
            return isin
    except Exception:
        pass

    # Try common Stockholm suffix patterns based on ISIN prefix
    if isin.startswith("SE"):
        # yfinance search is unreliable for small Nordic names
        # Return None and let caller flag for manual mapping
        pass
    return None


def register_ticker(isin: str, ticker: str, db_path: Path = DB_PATH) -> None:
    """Manually register an ISIN→ticker mapping (called from CLI or setup script)."""
    _cache_ticker(isin, ticker, "manual", db_path)
    logger.info("Registered %s → %s", isin, ticker)


def _cache_ticker(isin: str, ticker: str, source: str, db_path: Path) -> None:
    conn = _get_conn(db_path)
    conn.execute("""
        INSERT OR REPLACE INTO isin_ticker (isin, ticker, source, updated)
        VALUES (?, ?, ?, ?)
    """, (isin, ticker, source, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


# ── OHLCV fetching ──────────────────────────────────────────────────────────

def fetch_ohlcv(
    ticker: str,
    start: date,
    end: date,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """
    Fetch OHLCV for a ticker between start and end dates (inclusive).
    Uses local cache first; fetches from yfinance for missing ranges.
    Returns DataFrame indexed by date with columns: open high low close volume.
    """
    conn = _get_conn(db_path)
    cached = pd.read_sql_query(
        "SELECT * FROM ohlcv_cache WHERE ticker=? AND date>=? AND date<=? ORDER BY date",
        conn,
        params=(ticker, start.isoformat(), end.isoformat()),
    )

    if not cached.empty:
        cached["date"] = pd.to_datetime(cached["date"]).dt.date
        cached = cached.set_index("date").drop(columns=["ticker"])
        # Check if cache covers the full requested range
        if cached.index.min() <= start and cached.index.max() >= end:
            conn.close()
            return cached

    # Fetch from yfinance
    try:
        raw = yf.download(
            ticker,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        logger.error("yfinance fetch failed for %s: %s", ticker, e)
        conn.close()
        return pd.DataFrame()

    if raw.empty:
        logger.warning("No price data returned for %s (%s to %s)", ticker, start, end)
        conn.close()
        return pd.DataFrame()

    # Newer yfinance returns MultiIndex even for a single ticker
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.xs(ticker, axis=1, level=1) if ticker in raw.columns.get_level_values(1) \
              else raw.xs(ticker, axis=1, level=0)
    raw.columns = [c.lower() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index).date

    # Cache to DB
    for dt, row in raw.iterrows():
        conn.execute("""
            INSERT OR REPLACE INTO ohlcv_cache
            (ticker, date, open, high, low, close, volume)
            VALUES (?,?,?,?,?,?,?)
        """, (
            ticker, dt.isoformat(),
            float(row.get("open", 0)),
            float(row.get("high", 0)),
            float(row.get("low", 0)),
            float(row.get("close", 0)),
            float(row.get("volume", 0)),
        ))
    conn.commit()
    conn.close()

    return raw


# ── ADV-based position sizing ────────────────────────────────────────────────

def get_adv_sek(
    ticker: str,
    as_of_date: date,
    lookback_days: int = 20,
    db_path: Path = DB_PATH,
) -> Optional[float]:
    """
    Average daily volume in SEK over the last `lookback_days` trading days,
    as of a given date. Uses close price × volume as a proxy for daily turnover.

    Returns None if insufficient data.
    """
    end = as_of_date
    start = as_of_date - timedelta(days=lookback_days * 2)  # buffer for weekends/holidays
    df = fetch_ohlcv(ticker, start, end, db_path)

    if df.empty or len(df) < 5:
        return None

    df = df[df.index <= as_of_date].tail(lookback_days)
    if len(df) < 5:
        return None

    df["turnover_sek"] = df["close"] * df["volume"]
    return float(df["turnover_sek"].mean())


def max_position_sek(
    ticker: str,
    as_of_date: date,
    portfolio_value_sek: float,
    max_pct_of_adv: float = 2.0,
    max_portfolio_pct: float = 7.0,
    adv_lookback_days: int = 20,
    db_path: Path = DB_PATH,
) -> Optional[float]:
    """
    Returns the maximum position size in SEK, as the minimum of:
      (a) portfolio_value × max_portfolio_pct/100
      (b) ADV × max_pct_of_adv/100

    Directly links position size to realized liquidity — not an arbitrary multiplier.
    Returns None if ADV cannot be computed (ticker not found, insufficient data).
    """
    adv = get_adv_sek(ticker, as_of_date, adv_lookback_days, db_path)
    if adv is None:
        logger.warning("Cannot compute ADV for %s on %s — skipping position", ticker, as_of_date)
        return None

    cap_from_portfolio = portfolio_value_sek * (max_portfolio_pct / 100.0)
    cap_from_liquidity = adv * (max_pct_of_adv / 100.0)

    result = min(cap_from_portfolio, cap_from_liquidity)
    logger.debug(
        "%s | ADV=%.0f SEK | portfolio cap=%.0f | liquidity cap=%.0f | max position=%.0f",
        ticker, adv, cap_from_portfolio, cap_from_liquidity, result,
    )
    return result


def get_latest_close(
    ticker: str,
    as_of_date: date,
    db_path: Path = DB_PATH,
) -> Optional[float]:
    """Last close price on or before as_of_date."""
    df = fetch_ohlcv(ticker, as_of_date - timedelta(days=10), as_of_date, db_path)
    if df.empty:
        return None
    df = df[df.index <= as_of_date]
    return float(df["close"].iloc[-1]) if not df.empty else None


def passes_liquidity_gate(
    ticker: str,
    as_of_date: date,
    min_adv_sek: float = 500_000,
    db_path: Path = DB_PATH,
) -> bool:
    """Universe filter: reject stocks below minimum average daily volume."""
    adv = get_adv_sek(ticker, as_of_date, db_path=db_path)
    if adv is None:
        return False
    return adv >= min_adv_sek


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    today = date.today()
    test_ticker = "ERIC-B.ST"
    adv = get_adv_sek(test_ticker, today)
    print(f"{test_ticker} ADV (SEK): {adv:,.0f}" if adv else "ADV unavailable")
    cap = max_position_sek(test_ticker, today, portfolio_value_sek=50_000)
    print(f"Max position size: {cap:,.0f} SEK" if cap else "Cannot size position")
