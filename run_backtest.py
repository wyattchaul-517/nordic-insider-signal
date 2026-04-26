"""
run_backtest.py — Full pipeline orchestration.

Steps:
  1. Fetch 2 years of FI insider data
  2. Identify all unique Swedish ISINs from that data
  3. Batch-resolve ISIN → Yahoo ticker via OpenFIGI (up to 100 per API call)
  4. Bulk-download price history for all resolved tickers via yfinance
  5. Run the backtest engine over the validate period
  6. Print results

Run once to set up data, then run again for instant results (all data cached).
Expected first-run time: 5–15 minutes (dominated by yfinance downloads).
"""

import logging
import sqlite3
import sys
import time
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

# Force UTF-8 output on Windows to avoid encoding errors with special chars
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import requests
import yaml
import yfinance as yf

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from data.fi_scraper import refresh as fi_refresh, DB_PATH as FI_DB
from backtest.universe import (
    DB_PATH as UNIVERSE_DB, _init_db, _get_conn, _cache_ticker,
    TickerRecord, _figi_to_yahoo_candidates, _verify_ticker_in_yfinance,
)
from backtest.engine import BacktestEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config" / "strategy.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

BT = CFG.get("backtest", {})
VALIDATE_START = date.fromisoformat(BT.get("validate_start", "2023-01-01"))
VALIDATE_END   = date.fromisoformat(BT.get("validate_end",   "2024-12-31"))

# Price download covers validate period + 1 year buffer for MA calculations
PRICE_DOWNLOAD_START = (VALIDATE_START - timedelta(days=365)).isoformat()
PRICE_DOWNLOAD_END   = (VALIDATE_END   + timedelta(days=5)).isoformat()

OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"
OPENFIGI_BATCH_SIZE = 10       # anonymous API limit is 10 per request (100 with free key)
OPENFIGI_DELAY = 2.5           # seconds between batch calls (safe for anon key)


# ── Step 1: FI data ───────────────────────────────────────────────────────────

def step1_fetch_fi(days_back: int = 730) -> int:
    print(f"\n{'='*60}")
    print(f"STEP 1: Fetching FI insider data (last {days_back} days)")
    print(f"{'='*60}")
    n = fi_refresh(days_back=days_back)
    print(f"  → {n} new records stored in {FI_DB}")
    return n


# ── Step 2: Collect Swedish ISINs ─────────────────────────────────────────────

def step2_get_swedish_isins() -> list[str]:
    print(f"\n{'='*60}")
    print("STEP 2: Collecting Swedish ISINs from FI data")
    print(f"{'='*60}")
    conn = sqlite3.connect(FI_DB)
    # Swedish ISINs start with SE. We also want ISINs active in validate period.
    rows = conn.execute("""
        SELECT DISTINCT isin FROM insider_transactions
        WHERE UPPER(isin) LIKE 'SE%'
          AND DATE(report_date) >= ?
          AND DATE(report_date) <= ?
          AND UPPER(isin) NOT IN ('', 'NONE', 'NAN', 'N/A')
          AND instrument_type = 'Aktie'
    """, (
        (VALIDATE_START - timedelta(days=90)).isoformat(),
        (VALIDATE_END + timedelta(days=30)).isoformat(),
    )).fetchall()
    conn.close()
    isins = [r[0] for r in rows]
    print(f"  → {len(isins)} unique Swedish ISINs found in validate window")
    return isins


# ── Step 3: Batch OpenFIGI resolution ─────────────────────────────────────────

def _openfigi_batch(items: list[dict]) -> list[dict]:
    """Send one batch to OpenFIGI. Returns list of results (same length as items)."""
    time.sleep(OPENFIGI_DELAY)
    try:
        resp = requests.post(
            OPENFIGI_URL,
            json=items,
            headers={"Content-Type": "application/json"},
            timeout=20,
        )
        if resp.status_code == 429:
            logger.warning("OpenFIGI rate limit — sleeping 60s")
            time.sleep(60)
            resp = requests.post(OPENFIGI_URL, json=items,
                                 headers={"Content-Type": "application/json"}, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error("OpenFIGI batch failed: %s", e)
        return [{"error": str(e)}] * len(items)


def step3_resolve_tickers(isins: list[str]) -> dict[str, str]:
    """
    Batch-resolve ISINs to Yahoo tickers via OpenFIGI.
    Returns {isin: yahoo_ticker} for all successfully resolved ISINs.
    Caches results (hits and misses) to avoid re-querying.
    """
    print(f"\n{'='*60}")
    print(f"STEP 3: Resolving {len(isins)} ISINs → Yahoo tickers (OpenFIGI batch)")
    print(f"{'='*60}")

    conn = _get_conn(UNIVERSE_DB)

    # Check which ISINs are already cached
    already = {}
    needs_resolve = []
    for isin in isins:
        row = conn.execute(
            "SELECT ticker FROM ticker_cache WHERE isin = ?", (isin,)
        ).fetchone()
        if row is not None:
            if row[0]:  # cached hit
                already[isin] = row[0]
        else:
            needs_resolve.append(isin)
    conn.close()

    print(f"  Cache hits: {len(already)}  |  Need OpenFIGI: {len(needs_resolve)}")

    resolved = dict(already)

    if not needs_resolve:
        print("  All ISINs already cached.")
        return resolved

    # Batch requests: try Stockholm first, then any exchange for misses
    for exch_code, pass_name in [("SS", "Stockholm"), (None, "any-exchange fallback")]:
        remaining = [i for i in needs_resolve if i not in resolved]
        if not remaining:
            break

        print(f"\n  Pass: {pass_name} ({len(remaining)} ISINs) …")
        batches = [remaining[i:i+OPENFIGI_BATCH_SIZE]
                   for i in range(0, len(remaining), OPENFIGI_BATCH_SIZE)]

        for batch_idx, batch in enumerate(batches):
            items = [
                {"idType": "ID_ISIN", "idValue": isin,
                 **({"exchCode": exch_code} if exch_code else {})}
                for isin in batch
            ]
            results = _openfigi_batch(items)
            print(f"    Batch {batch_idx+1}/{len(batches)} ({len(batch)} ISINs) …", end=" ")

            hits = 0
            for isin, result in zip(batch, results):
                if not result or "error" in result or not result.get("data"):
                    continue
                hit = result["data"][0]
                figi_ticker = hit.get("ticker", "")
                ecode = exch_code or "SS"
                candidates = _figi_to_yahoo_candidates(figi_ticker, ecode)
                # Probe with a date inside the validate period — avoids excluding
                # stocks that were active in 2023-2024 but have since been delisted.
                verified = _verify_ticker_in_yfinance(candidates, probe_date=VALIDATE_END)
                if verified:
                    record = TickerRecord(
                        isin=isin, ticker=verified,
                        figi=hit.get("figi"), company_name=hit.get("name"),
                        source="openfigi_batch",
                        resolved_at=datetime.now(timezone.utc).isoformat(),
                    )
                    _cache_ticker(record, isin, UNIVERSE_DB)
                    resolved[isin] = verified
                    hits += 1

            print(f"{hits} resolved")

    # Cache definitive misses
    conn = _get_conn(UNIVERSE_DB)
    for isin in needs_resolve:
        if isin not in resolved:
            conn.execute("""
                INSERT OR IGNORE INTO ticker_cache
                (isin, ticker, figi, company_name, source, resolved_at)
                VALUES (?,NULL,NULL,NULL,'openfigi_miss',?)
            """, (isin, datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()

    print(f"\n  Total resolved: {len(resolved)}/{len(isins)}")
    missed = len(isins) - len(resolved)
    if missed:
        print(f"  Unresolved: {missed} (cached as misses — likely delisted or non-ST)")

    return resolved


# ── Step 4: Bulk price download ───────────────────────────────────────────────

def step4_download_prices(tickers: list[str]) -> None:
    """
    Bulk-download OHLCV for all tickers and store in universe_cache price_cache.
    Uses yfinance batch download (all tickers in one call) for speed.
    """
    print(f"\n{'='*60}")
    print(f"STEP 4: Bulk downloading prices for {len(tickers)} tickers")
    print(f"        {PRICE_DOWNLOAD_START} → {PRICE_DOWNLOAD_END}")
    print(f"{'='*60}")

    if not tickers:
        print("  No tickers to download.")
        return

    # Check which tickers already have sufficient cached data
    conn = sqlite3.connect(UNIVERSE_DB)
    needs_download = []
    for ticker in tickers:
        count = conn.execute(
            "SELECT COUNT(*) FROM price_cache WHERE ticker=? AND date>=? AND date<=?",
            (ticker, PRICE_DOWNLOAD_START, PRICE_DOWNLOAD_END)
        ).fetchone()[0]
        if count < 100:   # fewer than 100 cached rows → re-download
            needs_download.append(ticker)
    conn.close()

    print(f"  Already cached: {len(tickers) - len(needs_download)}")
    print(f"  Need download:  {len(needs_download)}")

    if not needs_download:
        print("  All tickers already cached.")
        return

    # yfinance batch download — all tickers at once
    print(f"\n  Downloading via yfinance (this may take a few minutes) …")
    try:
        raw = yf.download(
            tickers=needs_download,
            start=PRICE_DOWNLOAD_START,
            end=PRICE_DOWNLOAD_END,
            progress=True,
            auto_adjust=True,
            group_by="ticker",
        )
    except Exception as e:
        logger.error("yfinance batch download failed: %s", e)
        return

    if raw.empty:
        print("  No data returned from yfinance.")
        return

    # Store in price_cache
    conn = sqlite3.connect(UNIVERSE_DB)
    stored = 0

    if isinstance(raw.columns, pd.MultiIndex):
        # In newer yfinance: level 0 = ticker, level 1 = field (Price)
        # In older yfinance: level 0 = field, level 1 = ticker
        # Detect which level holds tickers by checking sample ticker presence
        sample = needs_download[0] if needs_download else ""
        ticker_level = 0 if sample in raw.columns.get_level_values(0) else 1
        for ticker in needs_download:
            try:
                if ticker not in raw.columns.get_level_values(ticker_level):
                    continue
                df = raw.xs(ticker, axis=1, level=ticker_level)
                df.columns = [c.lower() for c in df.columns]
                df.index = pd.to_datetime(df.index).date
                _store_prices(conn, ticker, df)
                stored += 1
            except Exception as e:
                logger.debug("Failed to store %s: %s", ticker, e)
    else:
        # Single ticker
        raw.columns = [c.lower() for c in raw.columns]
        raw.index = pd.to_datetime(raw.index).date
        if needs_download:
            _store_prices(conn, needs_download[0], raw)
            stored = 1

    conn.commit()
    conn.close()
    print(f"  Stored price data for {stored} tickers.")


def _store_prices(conn: sqlite3.Connection, ticker: str, df: pd.DataFrame) -> None:
    if "close" not in df.columns:
        return
    for dt, row in df.iterrows():
        close = float(row.get("close", 0) or 0)
        volume = float(row.get("volume", 0) or 0)
        if close > 0:
            conn.execute(
                "INSERT OR REPLACE INTO price_cache (ticker,date,close,volume) VALUES (?,?,?,?)",
                (ticker, dt.isoformat(), close, volume)
            )


# ── Step 5 & 6: Run backtest ──────────────────────────────────────────────────

def step5_run_backtest(initial_capital: float = 50_000.0):
    print(f"\n{'='*60}")
    print(f"STEP 5: Running backtest")
    print(f"        Validate period: {VALIDATE_START} → {VALIDATE_END}")
    print(f"        Capital: {initial_capital:,.0f} SEK")
    print(f"{'='*60}\n")

    engine = BacktestEngine()
    result = engine.run(VALIDATE_START, VALIDATE_END, initial_capital)

    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(result.summary())

    if result.trades:
        print(f"\nAll {len(result.trades)} trades:")
        header = f"  {'Entry':<12} {'Exit':<12} {'ISIN':<14} {'Score':>5} {'Return':>8} {'Days':>5}  Reason"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for t in sorted(result.trades, key=lambda x: x.entry_date):
            print(
                f"  {str(t.entry_date):<12} {str(t.exit_date):<12} "
                f"{t.isin:<14} {t.conviction_score:>5} "
                f"{t.pct_return:>+7.1f}% {t.trading_days_held:>5}d  {t.exit_reason}"
            )
    else:
        print("\nNo trades executed in this period.")
        print("Possible reasons:")
        print("  - FI data is sparse for the validate period")
        print("  - Conviction scorer filters are too strict")
        print("  - No ticker mappings available (run step 3 again with more ISINs)")

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    capital = float(sys.argv[1]) if len(sys.argv) > 1 else 50_000.0

    # Step 1: FI data — need ~1300 days to cover validate_start=2023-01-01 from today
    step1_fetch_fi(days_back=1300)

    # Step 2: Swedish ISINs
    isins = step2_get_swedish_isins()

    # Step 3: Resolve tickers
    resolved = step3_resolve_tickers(isins)
    tickers = list(set(resolved.values()))

    # Step 4: Bulk price download
    step4_download_prices(tickers)

    # Step 5: Run backtest
    result = step5_run_backtest(capital)

    elapsed = time.time() - t0
    print(f"\nTotal pipeline time: {elapsed/60:.1f} minutes")
