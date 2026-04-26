"""
backtest/universe.py — CandidateValidator

Architecture decision:
  This is NOT a "list of all stocks." It is a point validator for FI events.
  The strategy is event-driven: every candidate arrives via an FI insider report.
  We never scan a full market universe to generate signals.

  The only question we need to answer is:
    "Was this specific FI-flagged ISIN tradeable on this specific signal date?"

  Three sequential checks:
    1. ISIN → ticker  : OpenFIGI API (free, covers delisted securities via FIGI persistence)
    2. Price + volume  : yfinance returns close > 0 AND volume > 0 within ±2 trading days
                         Volume > 0 required — catches suspended stocks with frozen prices
    3. Liquidity       : ADV ≥ 500K SEK over prior 20 trading days

  get_symbols(as_of_date) exists for the backtest engine but returns current Nasdaq
  Stockholm listings only — survivorship-biased, used exclusively for macro context
  display, never as a hard filter or trade gate.

What was deliberately excluded and why:
  - Full historical universe: requires Börsdata (paid). Upgrade path is wiring in
    BorsdataClient in resolve_ticker() when subscription is available.
  - Opportunity cost vs. full market: not computable with free data, not critical
    for event-driven strategy validation.
  - False-positive rate vs. full market: within FI-observable subset is sufficient.
    Companies that never generated FI events are outside our strategy's scope.

Known bias:
  Signals from ISINs that were delisted before the backtest run date are excluded
  if OpenFIGI or yfinance returns no data. Bias direction: unknown (depends on
  delisting cause — M&A vs distress). Estimated impact logged per backtest run.
"""

import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "universe_cache.db"

OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"
OPENFIGI_RATE_LIMIT = 25       # requests per minute (no API key)
OPENFIGI_DELAY = 60 / OPENFIGI_RATE_LIMIT  # seconds between calls

PRICE_CHECK_WINDOW_DAYS = 3    # ±3 calendar days around signal_date
MIN_ADV_SEK = 500_000
ADV_LOOKBACK_DAYS = 20


# ── Result types ─────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    valid: bool
    isin: str
    signal_date: date
    ticker: Optional[str]
    reason: str             # human-readable: why valid or why rejected
    adv_sek: Optional[float]

    def __str__(self) -> str:
        status = "VALID" if self.valid else "REJECTED"
        adv = f"ADV={self.adv_sek:,.0f}SEK" if self.adv_sek else "ADV=unknown"
        return f"{status} | {self.isin} | {self.ticker or 'no-ticker'} | {adv} | {self.reason}"


@dataclass
class TickerRecord:
    isin: str
    ticker: str             # Yahoo Finance format, e.g. "ERIC-B.ST"
    figi: Optional[str]
    company_name: Optional[str]
    source: str             # "openfigi" | "manual" | "fallback"
    resolved_at: str


# ── Database ──────────────────────────────────────────────────────────────────

def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ticker_cache (
            isin         TEXT PRIMARY KEY,
            ticker       TEXT,           -- NULL means definitively unresolvable
            figi         TEXT,
            company_name TEXT,
            source       TEXT,
            resolved_at  TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS price_cache (
            ticker  TEXT NOT NULL,
            date    TEXT NOT NULL,
            close   REAL,
            volume  REAL,
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nasdaq_stockholm (
            isin        TEXT PRIMARY KEY,
            ticker      TEXT,
            name        TEXT,
            fetched_at  TEXT NOT NULL
        )
    """)
    conn.commit()


def _get_conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    _init_db(conn)
    return conn


# ── OpenFIGI ticker resolution ────────────────────────────────────────────────

EXCH_SUFFIX = {"SS": ".ST", "OM": ".ST", "FH": ".HE", "DC": ".CO", "NO": ".OL"}


def _figi_to_yahoo_candidates(figi_ticker: str, exch_code: str) -> list[str]:
    """
    Return candidate Yahoo Finance ticker strings for a Bloomberg/FIGI ticker.

    Swedish share-class tickers follow a consistent pattern:
      Bloomberg: "ERIC B"  or  "ERICB"    (with or without space)
      Yahoo:     "ERIC-B.ST"              (hyphen before last letter)

    Rule: always try hyphen before the last character as a candidate.
    This correctly handles all single-letter share classes (A, B, C, D).
    Also try before last 2 characters for two-letter share classes (PR, SDB).
    """
    if not figi_ticker:
        return []

    t = figi_ticker.strip()
    suffix = EXCH_SUFFIX.get(exch_code, ".ST")
    seen: list[str] = []

    def add(candidate: str) -> None:
        if candidate not in seen:
            seen.append(candidate)

    if " " in t:
        # "ERIC B" → "ERIC-B.ST"  (high confidence — space marks share class boundary)
        add(t.replace(" ", "-") + suffix)
        add(t.replace(" ", "") + suffix)
    else:
        # "ERICB" → try as-is, then with hyphen before last 1 char, then last 2 chars
        add(t + suffix)
        if len(t) > 1:
            add(t[:-1] + "-" + t[-1] + suffix)   # "ERICB" → "ERIC-B.ST"
        if len(t) > 2:
            add(t[:-2] + "-" + t[-2:] + suffix)  # "ERICPR" → "ERIC-PR.ST"

    return seen


def _verify_ticker_in_yfinance(candidates: list[str],
                                 probe_date: Optional[date] = None) -> Optional[str]:
    """
    Try each candidate ticker in yfinance until one returns data.
    Uses a lightweight 5-day probe to minimise API calls.
    Returns the first working ticker, or None if all fail.
    """
    if probe_date is None:
        probe_date = date.today() - timedelta(days=10)
    probe_start = (probe_date - timedelta(days=7)).isoformat()
    probe_end = (probe_date + timedelta(days=2)).isoformat()

    for ticker in candidates:
        try:
            raw = yf.download(ticker, start=probe_start, end=probe_end,
                              progress=False, auto_adjust=True)
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            close_col = next((c for c in raw.columns if "close" in c.lower()), None)
            if close_col and (raw[close_col] > 0).any():
                return ticker
        except Exception:
            continue
    return None


def _query_openfigi(isin: str,
                    probe_date: Optional[date] = None) -> Optional[TickerRecord]:
    """
    Query OpenFIGI for ISIN → ticker mapping.
    Tries Stockholm exchange first, then any Nordic exchange.
    Returns None if unresolvable (not an error — stock may be delisted beyond coverage).

    probe_date: date to use for yfinance verification. Pass the signal date
                so stocks active during the backtest period but since delisted
                are not incorrectly excluded. Defaults to recent past.
    """
    attempts = [
        {"idType": "ID_ISIN", "idValue": isin, "exchCode": "SS"},   # Stockholm
        {"idType": "ID_ISIN", "idValue": isin, "exchCode": "FH"},   # Helsinki
        {"idType": "ID_ISIN", "idValue": isin, "exchCode": "DC"},   # Copenhagen
        {"idType": "ID_ISIN", "idValue": isin, "exchCode": "NO"},   # Oslo
        {"idType": "ID_ISIN", "idValue": isin},                      # any exchange
    ]

    for attempt in attempts:
        try:
            time.sleep(OPENFIGI_DELAY)
            resp = requests.post(
                OPENFIGI_URL,
                json=[attempt],
                headers={"Content-Type": "application/json"},
                timeout=15,
            )
            if resp.status_code == 429:
                logger.warning("OpenFIGI rate limit hit — sleeping 60s")
                time.sleep(60)
                resp = requests.post(OPENFIGI_URL, json=[attempt],
                                     headers={"Content-Type": "application/json"}, timeout=15)

            resp.raise_for_status()
            data = resp.json()

            if not data or "error" in data[0] or not data[0].get("data"):
                continue

            # Take first match
            hit = data[0]["data"][0]
            figi_ticker = hit.get("ticker", "")
            exch_code = attempt.get("exchCode", "SS")
            candidates = _figi_to_yahoo_candidates(figi_ticker, exch_code)
            if not candidates:
                continue

            # Verify against yfinance using probe_date so stocks active during
            # the signal period but since delisted are not wrongly excluded.
            verified = _verify_ticker_in_yfinance(candidates, probe_date=probe_date)
            if not verified:
                logger.debug("OpenFIGI candidates %s all failed yfinance probe", candidates)
                continue

            return TickerRecord(
                isin=isin,
                ticker=verified,
                figi=hit.get("figi"),
                company_name=hit.get("name"),
                source="openfigi",
                resolved_at=datetime.now(timezone.utc).isoformat(),
            )
        except requests.RequestException as e:
            logger.debug("OpenFIGI request failed for %s: %s", isin, e)
            continue

    return None


def _cache_ticker(record: Optional[TickerRecord], isin: str,
                  db_path: Path = DB_PATH) -> None:
    conn = _get_conn(db_path)
    if record:
        conn.execute("""
            INSERT OR REPLACE INTO ticker_cache
            (isin, ticker, figi, company_name, source, resolved_at)
            VALUES (?,?,?,?,?,?)
        """, (record.isin, record.ticker, record.figi,
              record.company_name, record.source, record.resolved_at))
    else:
        # Cache the negative result so we don't re-query OpenFIGI repeatedly
        conn.execute("""
            INSERT OR REPLACE INTO ticker_cache
            (isin, ticker, figi, company_name, source, resolved_at)
            VALUES (?,NULL,NULL,NULL,'openfigi_miss',?)
        """, (isin, datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()


def resolve_ticker(isin: str, db_path: Path = DB_PATH,
                   use_openfigi: bool = True,
                   probe_date: Optional[date] = None) -> Optional[str]:
    """
    Resolve ISIN → Yahoo Finance ticker.
    Order: DB cache → OpenFIGI API → None.
    Caches both hits and misses to avoid redundant API calls.

    probe_date: passed to _query_openfigi for yfinance verification.
                Use the signal date to avoid excluding historically-active but
                now-delisted stocks.
    """
    conn = _get_conn(db_path)
    row = conn.execute(
        "SELECT ticker FROM ticker_cache WHERE isin = ?", (isin,)
    ).fetchone()
    conn.close()

    if row is not None:
        return row[0]  # may be None for cached misses

    if not use_openfigi:
        return None

    record = _query_openfigi(isin, probe_date=probe_date)
    _cache_ticker(record, isin, db_path)
    ticker = record.ticker if record else None
    if ticker:
        logger.info("OpenFIGI resolved %s → %s", isin, ticker)
    else:
        logger.debug("OpenFIGI: no ticker for %s (cached miss)", isin)
    return ticker


def register_ticker(isin: str, ticker: str, db_path: Path = DB_PATH) -> None:
    """Manually register or override an ISIN → ticker mapping."""
    record = TickerRecord(
        isin=isin, ticker=ticker, figi=None, company_name=None,
        source="manual", resolved_at=datetime.now(timezone.utc).isoformat(),
    )
    _cache_ticker(record, isin, db_path)
    logger.info("Manual mapping registered: %s → %s", isin, ticker)


# ── Price + volume existence check ────────────────────────────────────────────

def _fetch_ohlcv_window(ticker: str, signal_date: date,
                         window: int = PRICE_CHECK_WINDOW_DAYS,
                         db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Fetch OHLCV for ticker in a window around signal_date.
    Uses local cache to avoid redundant yfinance calls.
    Returns DataFrame with columns [close, volume], indexed by date.
    """
    start = signal_date - timedelta(days=window + 5)
    end = signal_date + timedelta(days=window + 5)

    conn = _get_conn(db_path)
    cached = pd.read_sql_query(
        "SELECT date, close, volume FROM price_cache WHERE ticker=? AND date>=? AND date<=?",
        conn, params=(ticker, start.isoformat(), end.isoformat())
    )
    conn.close()

    if not cached.empty:
        cached["date"] = pd.to_datetime(cached["date"]).dt.date
        return cached.set_index("date")

    try:
        raw = yf.download(
            ticker,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        logger.warning("yfinance download failed for %s: %s", ticker, e)
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    # yfinance 1.x returns MultiIndex columns: ('Close', 'TICK.ST'), ('Volume', 'TICK.ST')
    # Flatten to single level before further processing.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.columns = [c.lower() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index).date

    if "close" not in raw.columns:
        return pd.DataFrame()

    conn = _get_conn(db_path)
    for dt, row in raw.iterrows():
        conn.execute(
            "INSERT OR REPLACE INTO price_cache (ticker, date, close, volume) VALUES (?,?,?,?)",
            (ticker, dt.isoformat(), float(row.get("close", 0) or 0),
             float(row.get("volume", 0) or 0))
        )
    conn.commit()
    conn.close()

    return raw[["close", "volume"]]


def _check_price_and_volume(
    ticker: str,
    signal_date: date,
    window: int = PRICE_CHECK_WINDOW_DAYS,
    db_path: Path = DB_PATH,
) -> tuple[bool, str]:
    """
    Returns (tradeable: bool, reason: str).

    Requires within ±window trading days of signal_date:
      - close > 0  (price exists)
      - volume > 0  (not suspended / frozen price)

    Volume = 0 with price > 0 indicates suspension or OTC-only trading —
    both are untradeable scenarios for this strategy.
    """
    df = _fetch_ohlcv_window(ticker, signal_date, window, db_path)
    if df.empty:
        return False, "no price data in yfinance"

    # Filter to window around signal_date
    candidates = df[
        (df.index >= signal_date - timedelta(days=window)) &
        (df.index <= signal_date + timedelta(days=window))
    ]

    if candidates.empty:
        return False, f"no price data within ±{window} days of signal date"

    # Require at least one day with close > 0 AND volume > 0
    active = candidates[
        (candidates["close"] > 0) &
        (candidates["volume"] > 0)
    ]

    if active.empty:
        has_price = (candidates["close"] > 0).any()
        has_volume = (candidates["volume"] > 0).any()
        if has_price and not has_volume:
            return False, "price exists but volume=0 on all nearby days — likely suspended"
        if not has_price:
            return False, "close price is zero or missing on all nearby days"
        return False, "price and volume checks both failed"

    closest_idx = min(active.index, key=lambda d: abs((d - signal_date).days))
    days_off = abs((closest_idx - signal_date).days)
    return True, f"price+volume confirmed on {closest_idx} ({days_off}d from signal)"


# ── ADV liquidity check ───────────────────────────────────────────────────────

def _check_adv(
    ticker: str,
    signal_date: date,
    min_adv_sek: float = MIN_ADV_SEK,
    lookback: int = ADV_LOOKBACK_DAYS,
    db_path: Path = DB_PATH,
) -> tuple[bool, Optional[float], str]:
    """
    Returns (passes: bool, adv_sek: float | None, reason: str).
    """
    start = signal_date - timedelta(days=lookback * 2)
    df = _fetch_ohlcv_window(ticker, signal_date,
                              window=lookback * 2, db_path=db_path)

    if df.empty:
        return False, None, "no data for ADV calculation"

    hist = df[df.index <= signal_date].tail(lookback)
    if len(hist) < 5:
        return False, None, f"only {len(hist)} trading days available for ADV"

    # Filter days with volume > 0 (exclude suspended days from ADV calculation)
    active_days = hist[(hist["close"] > 0) & (hist["volume"] > 0)]
    if len(active_days) < 5:
        return False, None, f"only {len(active_days)} active trading days for ADV"

    adv = float((active_days["close"] * active_days["volume"]).mean())
    passes = adv >= min_adv_sek
    reason = (f"ADV={adv:,.0f}SEK >= {min_adv_sek:,.0f}" if passes
              else f"ADV={adv:,.0f}SEK < {min_adv_sek:,.0f} (too illiquid)")
    return passes, adv, reason


# ── Main validator ────────────────────────────────────────────────────────────

class CandidateValidator:
    """
    Validates FI-event candidates for tradeability on a given signal date.
    Use validate() for each candidate before passing to the scorer.
    """

    def __init__(
        self,
        db_path: Path = DB_PATH,
        min_adv_sek: float = MIN_ADV_SEK,
        price_window_days: int = PRICE_CHECK_WINDOW_DAYS,
        use_openfigi: bool = True,
    ):
        self.db_path = db_path
        self.min_adv_sek = min_adv_sek
        self.price_window_days = price_window_days
        self.use_openfigi = use_openfigi
        conn = _get_conn(db_path)
        conn.close()

    def validate(self, isin: str, signal_date: date) -> ValidationResult:
        """
        Run all three checks in sequence. Fail fast — stop at first rejection.
        Returns ValidationResult with valid=True only if all three pass.
        """
        # ── 1. Ticker resolution ────────────────────────────────────────────
        # Use signal_date as yfinance probe so stocks active then but since
        # delisted are not wrongly excluded.
        ticker = resolve_ticker(isin, self.db_path, self.use_openfigi,
                                probe_date=signal_date)
        if not ticker:
            return ValidationResult(
                valid=False, isin=isin, signal_date=signal_date,
                ticker=None, reason="no ticker resolved (ISIN unrecognised by OpenFIGI)",
                adv_sek=None,
            )

        # ── 2. Price + volume existence ─────────────────────────────────────
        tradeable, pv_reason = _check_price_and_volume(
            ticker, signal_date, self.price_window_days, self.db_path
        )
        if not tradeable:
            return ValidationResult(
                valid=False, isin=isin, signal_date=signal_date,
                ticker=ticker, reason=f"price/volume check failed: {pv_reason}",
                adv_sek=None,
            )

        # ── 3. ADV liquidity gate ────────────────────────────────────────────
        passes_adv, adv_sek, adv_reason = _check_adv(
            ticker, signal_date, self.min_adv_sek,
            ADV_LOOKBACK_DAYS, self.db_path
        )
        if not passes_adv:
            return ValidationResult(
                valid=False, isin=isin, signal_date=signal_date,
                ticker=ticker, reason=f"liquidity gate: {adv_reason}",
                adv_sek=adv_sek,
            )

        return ValidationResult(
            valid=True, isin=isin, signal_date=signal_date,
            ticker=ticker, reason=f"all checks passed | {pv_reason} | {adv_reason}",
            adv_sek=adv_sek,
        )

    def validate_batch(
        self, candidates: list[tuple[str, date]]
    ) -> list[ValidationResult]:
        """
        Validate multiple (isin, signal_date) pairs.
        Ticker resolution is cached so repeated ISINs are fast.
        """
        results = []
        for isin, signal_date in candidates:
            result = self.validate(isin, signal_date)
            logger.info(str(result))
            results.append(result)
        return results

    # ── Macro context (survivorship-biased, display only) ───────────────────

    def get_symbols(self, as_of_date: date) -> list[str]:
        """
        Returns currently listed Nasdaq Stockholm ISINs.

        !! SURVIVORSHIP BIAS WARNING !!
        This list reflects current survivors only. Use EXCLUSIVELY for:
          - Macro breadth display (informational, never as a trade filter)
          - Engine scaffolding during development

        Do NOT use as a signal source or entry condition.
        """
        conn = _get_conn(self.db_path)
        rows = conn.execute(
            "SELECT isin FROM nasdaq_stockholm WHERE isin IS NOT NULL"
        ).fetchall()
        conn.close()

        if rows:
            return [r[0] for r in rows]

        # Fallback: return ISINs we've successfully resolved, as a proxy
        conn = _get_conn(self.db_path)
        rows = conn.execute(
            "SELECT isin FROM ticker_cache WHERE ticker IS NOT NULL "
            "AND ticker LIKE '%.ST'"
        ).fetchall()
        conn.close()
        logger.warning(
            "get_symbols(): Nasdaq Stockholm list not populated. "
            "Returning %d cached .ST tickers as proxy. "
            "Run fetch_nasdaq_stockholm() to populate properly.", len(rows)
        )
        return [r[0] for r in rows]

    def compute_breadth(self, as_of_date: date, ma_period: int = 200) -> Optional[float]:
        """
        Fraction of get_symbols() stocks with close above their MA.
        SURVIVORSHIP-BIASED — logged as informational metric only.
        Never use the return value as a trade gate.
        """
        symbols = self.get_symbols(as_of_date)
        if not symbols:
            return None

        above = 0
        total = 0
        for isin in symbols:
            ticker = resolve_ticker(isin, self.db_path, use_openfigi=False)
            if not ticker:
                continue
            start = as_of_date - timedelta(days=ma_period * 2)
            df = _fetch_ohlcv_window(ticker, as_of_date, window=ma_period, db_path=self.db_path)
            if df.empty or "close" not in df.columns:
                continue
            hist = df[df.index <= as_of_date]
            if len(hist) < ma_period:
                continue
            ma = hist["close"].tail(ma_period).mean()
            latest = hist["close"].iloc[-1]
            if latest > 0 and ma > 0:
                total += 1
                if latest > ma:
                    above += 1

        if total == 0:
            return None
        breadth = above / total
        logger.info(
            "Market breadth (BIASED — current survivors only): %.1f%% above %dMA "
            "(%d/%d stocks)", breadth * 100, ma_period, above, total
        )
        return breadth


# ── Nasdaq Stockholm population ───────────────────────────────────────────────

def fetch_nasdaq_stockholm(db_path: Path = DB_PATH) -> int:
    """
    Populate the nasdaq_stockholm table by scraping Nasdaq Nordic's public
    company list. Returns count of ISINs stored.

    Nasdaq Nordic publishes a filterable company list at:
    https://www.nasdaqomxnordic.com/shares/listed-companies/stockholm
    This is the authoritative source for currently listed companies.
    """
    url = (
        "https://www.nasdaqomxnordic.com/shares/listed-companies/stockholm"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        from io import StringIO as _SIO
        tables = pd.read_html(_SIO(resp.text))
        if not tables:
            logger.warning(
                "No HTML tables found on Nasdaq Nordic page — page may be JS-rendered. "
                "Try fetching manually from %s and saving as CSV, then call "
                "load_nasdaq_stockholm_csv(path).", url
            )
            return 0
        df = max(tables, key=len)
        logger.info("Nasdaq Nordic table: %d rows, columns: %s",
                    len(df), list(df.columns)[:6])
    except Exception as e:
        logger.error("Nasdaq Nordic fetch failed: %s — page may require JS rendering.", e)
        return 0

    # Try to find ISIN column (column names vary by page version)
    isin_col = next((c for c in df.columns if "isin" in str(c).lower()), None)
    name_col = next((c for c in df.columns
                     if any(k in str(c).lower() for k in ["name", "company", "namn"])), None)
    ticker_col = next((c for c in df.columns
                       if any(k in str(c).lower() for k in ["symbol", "ticker"])), None)

    if not isin_col:
        logger.warning("Could not find ISIN column in Nasdaq Nordic table. "
                       "Columns: %s", list(df.columns))
        return 0

    fetched_at = datetime.now(timezone.utc).isoformat()
    conn = _get_conn(db_path)
    # Clear stale data
    conn.execute("DELETE FROM nasdaq_stockholm")

    count = 0
    for _, row in df.iterrows():
        isin = str(row.get(isin_col, "")).strip().upper()
        if not isin or len(isin) != 12:
            continue
        name = str(row.get(name_col, "")).strip() if name_col else None
        ticker_raw = str(row.get(ticker_col, "")).strip() if ticker_col else None
        ticker = (ticker_raw + ".ST") if ticker_raw and not ticker_raw.endswith(".ST") else ticker_raw
        conn.execute("""
            INSERT OR REPLACE INTO nasdaq_stockholm (isin, ticker, name, fetched_at)
            VALUES (?,?,?,?)
        """, (isin, ticker, name, fetched_at))
        count += 1

    conn.commit()
    conn.close()
    logger.info("Nasdaq Stockholm: %d ISINs stored", count)
    return count


def load_nasdaq_stockholm_csv(file_path: Path, db_path: Path = DB_PATH) -> int:
    """
    Load Nasdaq Stockholm listings from a manually downloaded CSV/Excel.

    If the website scraper fails (JS-rendered page), download the company list
    manually from https://www.nasdaqomxnordic.com/shares/listed-companies/stockholm
    (use the Export button on the page) and pass the file path here.

    The file must contain at minimum an ISIN column.
    """
    path = Path(file_path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype=str)
    else:
        df = pd.read_csv(path, dtype=str, sep=None, engine="python")

    isin_col = next((c for c in df.columns if "isin" in c.lower()), None)
    if not isin_col:
        raise ValueError(f"No ISIN column found. Columns: {list(df.columns)}")

    name_col = next((c for c in df.columns
                     if any(k in c.lower() for k in ["name", "company", "namn"])), None)
    ticker_col = next((c for c in df.columns
                       if any(k in c.lower() for k in ["symbol", "ticker"])), None)

    fetched_at = datetime.now(timezone.utc).isoformat()
    conn = _get_conn(db_path)
    conn.execute("DELETE FROM nasdaq_stockholm")
    count = 0
    for _, row in df.iterrows():
        isin = str(row.get(isin_col, "")).strip().upper()
        if not isin or len(isin) != 12:
            continue
        name = str(row.get(name_col, "")).strip() if name_col else None
        ticker_raw = str(row.get(ticker_col, "")).strip() if ticker_col else None
        ticker = (ticker_raw + ".ST") if (ticker_raw and not ticker_raw.endswith(".ST")) else ticker_raw
        conn.execute(
            "INSERT OR REPLACE INTO nasdaq_stockholm (isin, ticker, name, fetched_at) VALUES (?,?,?,?)",
            (isin, ticker, name, fetched_at)
        )
        count += 1
    conn.commit()
    conn.close()
    logger.info("Loaded %d ISINs from %s into nasdaq_stockholm", count, path)
    return count


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    validator = CandidateValidator()

    # Populate Nasdaq Stockholm list (one-time)
    print("Fetching Nasdaq Stockholm listing...")
    n = fetch_nasdaq_stockholm()
    print(f"  Stored {n} listed ISINs\n")

    # Test with a known Swedish stock
    test_cases = [
        ("SE0000108656", date(2024, 6, 1)),   # Ericsson — should be valid
        ("SE0000667925", date(2024, 6, 1)),   # Sandvik — should be valid
        ("CA2548481043", date(2024, 6, 1)),   # Canadian company — should be rejected
    ]

    if len(sys.argv) > 1:
        # Accept ISIN from command line: python universe.py SE0000108656
        test_cases = [(sys.argv[1], date.today())]

    print("Running validation tests:")
    for isin, sig_date in test_cases:
        result = validator.validate(isin, sig_date)
        print(f"  {result}")

    # Breadth (informational only)
    breadth = validator.compute_breadth(date.today())
    if breadth is not None:
        print(f"\nMarket breadth proxy (BIASED): {breadth:.1%} above 200MA")
