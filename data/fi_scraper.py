"""
Finansinspektionen insider transaction scraper.

FI requires insiders to report within 3 business days. We store both
transaction_date (when the trade happened) and report_date (when FI
published it). All signal logic MUST use report_date — never transaction_date.
This is the primary pullback-trap guard for this data source.

FI API (marknadssok.fi.se):
  Base: https://marknadssok.fi.se/Publiceringsklient/Swedish/Search/Search
  Params: SearchFunctionType=Insyn + date range + button=export
  Returns: UTF-16 LE CSV, semicolon-delimited, Swedish decimal commas.
  Source: reverse-engineered from github.com/w3stling/insynsregistret (MIT).
"""

import sqlite3
import logging
import urllib.parse
from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── FI API endpoint ──────────────────────────────────────────────────────────
FI_BASE_URL = (
    "https://marknadssok.fi.se/Publiceringsklient/Swedish/Search/Search"
)
FI_MAX_DAYS_PER_REQUEST = 14   # FI API hard-caps at 1000 rows; 14d ≈ 400-600 rows, well below limit

# Actual column names from FI's CSV export (verified 2026-04-24)
FI_COLUMN_MAP = {
    "Publiceringsdatum":          "report_date",       # ← signal uses THIS
    "Emittent":                   "company_name",
    "LEI-kod":                    "lei_code",
    "Person i ledande ställning": "insider_name",
    "Befattning":                 "role",
    "Närstående":                 "is_close_associate",
    "Karaktär":                   "transaction_type",  # Förvärv=buy, Avyttring=sell
    "Instrumenttyp":              "instrument_type",
    "Instrumentnamn":             "instrument_name",
    "ISIN":                       "isin",
    "Transaktionsdatum":          "transaction_date",  # ← do NOT use for signals
    "Volym":                      "volume",            # Swedish decimal: "1 234,56"
    "Pris":                       "unit_price",        # Swedish decimal: "56,00"
    "Valuta":                     "currency",
    "Handelsplats":               "trading_venue",
    "Status":                     "status",
}

SENIOR_ROLES = {
    "vd", "ceo", "verkställande direktör",
    "cfo", "finanschef", "ekonomichef",
    "styrelseordförande", "chairman",
    "styrelseledamot",
}

DB_PATH = Path(__file__).parent / "insider_transactions.db"


@dataclass
class InsiderTransaction:
    report_date: date          # When FI published — use this for signals
    transaction_date: date     # When trade actually occurred — DO NOT use for signals
    company_name: str
    isin: str
    lei_code: str
    insider_name: str
    role: str
    transaction_type: str      # "Förvärv" = buy, "Avyttring" = sell
    instrument_type: str
    instrument_name: str
    volume: float
    unit_price: float
    currency: str
    trading_venue: str
    status: str
    is_close_associate: bool = False

    @property
    def is_buy(self) -> bool:
        t = self.transaction_type.lower()
        return "förvärv" in t or "acquisition" in t or "buy" in t

    @property
    def is_sell(self) -> bool:
        t = self.transaction_type.lower()
        return "avyttring" in t or "disposal" in t or "sell" in t

    @property
    def total_value_sek(self) -> float:
        return self.volume * self.unit_price  # assumes SEK; caller handles FX

    @property
    def role_normalized(self) -> str:
        return self.role.strip().lower()

    @property
    def is_senior(self) -> bool:
        return any(s in self.role_normalized for s in SENIOR_ROLES)


# ── Database ──────────────────────────────────────────────────────────────────

def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS insider_transactions (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            report_date        TEXT NOT NULL,
            transaction_date   TEXT NOT NULL,
            company_name       TEXT NOT NULL,
            isin               TEXT,
            lei_code           TEXT,
            insider_name       TEXT NOT NULL,
            role               TEXT,
            transaction_type   TEXT NOT NULL,
            instrument_type    TEXT,
            instrument_name    TEXT,
            volume             REAL,
            unit_price         REAL,
            currency           TEXT,
            trading_venue      TEXT,
            status             TEXT,
            is_close_associate INTEGER DEFAULT 0,
            ingested_at        TEXT NOT NULL,
            UNIQUE(report_date, isin, insider_name, transaction_date, volume, unit_price)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_isin_report ON insider_transactions(isin, report_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_report_date ON insider_transactions(report_date)")
    conn.commit()


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _parse_date(val) -> Optional[date]:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    s = str(val).strip()
    # FI timestamps look like "2026-04-24 17:37:15" — strip time part
    s = s.split(" ")[0].split("T")[0]
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _parse_float(val) -> float:
    """Handle Swedish decimal format: '1 234,56' → 1234.56"""
    if val is None:
        return 0.0
    try:
        if pd.isna(val):
            return 0.0
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    # Remove thousands separators (space or non-breaking space), swap comma→dot
    s = s.replace("\xa0", "").replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return 0.0


# ── FI fetch ──────────────────────────────────────────────────────────────────

def _build_fi_url(from_pub: date, to_pub: date) -> str:
    params = {
        "SearchFunctionType": "Insyn",
        "Utgivare": "",
        "PersonILedandeStällningNamn": "",
        "Transaktionsdatum.From": "",
        "Transaktionsdatum.To": "",
        "Publiceringsdatum.From": from_pub.strftime("%Y-%m-%d"),
        "Publiceringsdatum.To": to_pub.strftime("%Y-%m-%d"),
        "button": "export",
    }
    return FI_BASE_URL + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)


def _fetch_fi_csv(from_pub: date, to_pub: date, timeout: int = 45) -> pd.DataFrame:
    """
    Fetch FI insider transactions for a publication date range.
    FI returns UTF-16 LE CSV with semicolon delimiter and Swedish decimal commas.
    """
    url = _build_fi_url(from_pub, to_pub)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://marknadssok.fi.se/",
    }
    logger.info("Fetching FI data: %s to %s", from_pub, to_pub)
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    if len(resp.content) < 50:
        logger.warning("FI response too small (%d bytes) — empty result set", len(resp.content))
        return pd.DataFrame()

    text = resp.content.decode("utf-16")
    df = pd.read_csv(StringIO(text), sep=";", dtype=str)
    # Drop unnamed trailing columns (artefact of trailing semicolon)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    logger.info("FI CSV fetched: %d rows (%s to %s)", len(df), from_pub, to_pub)
    return df


def _fetch_fi_range(from_pub: date, to_pub: date) -> pd.DataFrame:
    """Fetch FI data, chunking into FI_MAX_DAYS_PER_REQUEST-day windows with retries."""
    import time as _time
    chunks = []
    cursor = from_pub
    while cursor <= to_pub:
        chunk_end = min(cursor + timedelta(days=FI_MAX_DAYS_PER_REQUEST - 1), to_pub)
        for attempt in range(3):
            try:
                chunk = _fetch_fi_csv(cursor, chunk_end)
                if not chunk.empty:
                    chunks.append(chunk)
                break
            except Exception as e:
                if attempt < 2:
                    _time.sleep(3 * (attempt + 1))
                else:
                    logger.warning("Chunk %s–%s failed after 3 attempts: %s — skipping",
                                   cursor, chunk_end, e)
        cursor = chunk_end + timedelta(days=1)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


# ── Normalise + store ─────────────────────────────────────────────────────────

def _normalize(df: pd.DataFrame) -> list[InsiderTransaction]:
    """Map FI CSV columns → InsiderTransaction objects."""
    known = {k: v for k, v in FI_COLUMN_MAP.items() if k in df.columns}
    if "report_date" not in known.values():
        raise ValueError(
            f"'Publiceringsdatum' not found in FI data. "
            f"Columns present: {list(df.columns)[:10]}"
        )
    df = df[list(known.keys())].rename(columns=known)

    records: list[InsiderTransaction] = []
    for _, row in df.iterrows():
        rd = _parse_date(row.get("report_date"))
        td = _parse_date(row.get("transaction_date"))
        if rd is None or td is None:
            continue
        try:
            close_assoc_raw = str(row.get("is_close_associate", "")).strip().lower()
            is_close = close_assoc_raw in ("ja", "yes", "true", "1")
            records.append(InsiderTransaction(
                report_date=rd,
                transaction_date=td,
                company_name=str(row.get("company_name", "")).strip(),
                isin=str(row.get("isin", "")).strip().upper(),
                lei_code=str(row.get("lei_code", "")).strip(),
                insider_name=str(row.get("insider_name", "")).strip(),
                role=str(row.get("role", "")).strip(),
                transaction_type=str(row.get("transaction_type", "")).strip(),
                instrument_type=str(row.get("instrument_type", "")).strip(),
                instrument_name=str(row.get("instrument_name", "")).strip(),
                volume=_parse_float(row.get("volume")),
                unit_price=_parse_float(row.get("unit_price")),
                currency=str(row.get("currency", "SEK")).strip(),
                trading_venue=str(row.get("trading_venue", "")).strip(),
                status=str(row.get("status", "")).strip(),
                is_close_associate=is_close,
            ))
        except (ValueError, TypeError) as e:
            logger.debug("Skipping malformed row: %s", e)
    return records


def _store(records: list[InsiderTransaction], conn: sqlite3.Connection) -> int:
    ingested_at = datetime.now(timezone.utc).isoformat()
    inserted = 0
    for r in records:
        try:
            conn.execute("""
                INSERT OR IGNORE INTO insider_transactions
                (report_date, transaction_date, company_name, isin, lei_code,
                 insider_name, role, transaction_type, instrument_type,
                 instrument_name, volume, unit_price, currency,
                 trading_venue, status, is_close_associate, ingested_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                r.report_date.isoformat(), r.transaction_date.isoformat(),
                r.company_name, r.isin, r.lei_code, r.insider_name,
                r.role, r.transaction_type, r.instrument_type,
                r.instrument_name, r.volume, r.unit_price, r.currency,
                r.trading_venue, r.status, int(r.is_close_associate), ingested_at,
            ))
            if conn.execute("SELECT changes()").fetchone()[0]:
                inserted += 1
        except sqlite3.Error as e:
            logger.warning("DB insert error: %s", e)
    conn.commit()
    return inserted


# ── Public API ────────────────────────────────────────────────────────────────

def refresh(days_back: int = 90, db_path: Path = DB_PATH) -> int:
    """
    Fetch FI data for the last `days_back` days and store new records.
    Use days_back=365 for initial historical load, days_back=7 for daily updates.
    """
    to_pub = date.today()
    from_pub = to_pub - timedelta(days=days_back)
    conn = sqlite3.connect(db_path)
    _init_db(conn)
    df = _fetch_fi_range(from_pub, to_pub)
    if df.empty:
        logger.warning("No data returned from FI")
        conn.close()
        return 0
    records = _normalize(df)
    n = _store(records, conn)
    conn.close()
    logger.info("FI refresh: %d new records (last %d days)", n, days_back)
    return n


def load_from_file(file_path: Path, db_path: Path = DB_PATH) -> int:
    """Load from a manually downloaded FI CSV/Excel file."""
    conn = sqlite3.connect(db_path)
    _init_db(conn)
    path = Path(file_path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path, engine="openpyxl", dtype=str)
    else:
        with open(path, "rb") as f:
            text = f.read().decode("utf-16")
        df = pd.read_csv(StringIO(text), sep=";", dtype=str)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    logger.info("Loaded %d rows from %s", len(df), path)
    records = _normalize(df)
    n = _store(records, conn)
    conn.close()
    return n


def query_buys(
    isin: str,
    as_of_date: date,
    window_days: int = 30,
    db_path: Path = DB_PATH,
) -> list[InsiderTransaction]:
    """
    Return buy transactions for an ISIN visible as of a given date.
    Filters on report_date — enforces point-in-time correctness.
    """
    window_start = (as_of_date - timedelta(days=window_days)).isoformat()
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT report_date, transaction_date, company_name, isin, lei_code,
               insider_name, role, transaction_type, instrument_type,
               instrument_name, volume, unit_price, currency,
               trading_venue, status, is_close_associate
        FROM insider_transactions
        WHERE isin = ?
          AND report_date >= ?
          AND report_date <= ?
          AND (LOWER(transaction_type) LIKE '%förvärv%'
               OR LOWER(transaction_type) LIKE '%acquisition%'
               OR LOWER(transaction_type) LIKE '%buy%')
          AND instrument_type = 'Aktie'
        ORDER BY report_date
    """, (isin, window_start, as_of_date.isoformat())).fetchall()
    conn.close()
    return [_row_to_tx(r) for r in rows]


def query_sells(
    isin: str,
    as_of_date: date,
    window_days: int = 180,
    db_path: Path = DB_PATH,
) -> list[InsiderTransaction]:
    """Return sell transactions. Same point-in-time rule as query_buys."""
    window_start = (as_of_date - timedelta(days=window_days)).isoformat()
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT report_date, transaction_date, company_name, isin, lei_code,
               insider_name, role, transaction_type, instrument_type,
               instrument_name, volume, unit_price, currency,
               trading_venue, status, is_close_associate
        FROM insider_transactions
        WHERE isin = ?
          AND report_date >= ?
          AND report_date <= ?
          AND (LOWER(transaction_type) LIKE '%avyttring%'
               OR LOWER(transaction_type) LIKE '%disposal%'
               OR LOWER(transaction_type) LIKE '%sell%')
        ORDER BY report_date
    """, (isin, window_start, as_of_date.isoformat())).fetchall()
    conn.close()
    return [_row_to_tx(r) for r in rows]


def get_all_active_isins(db_path: Path = DB_PATH) -> list[str]:
    """Return ISINs with at least one buy published in the last 90 days."""
    cutoff = (date.today() - timedelta(days=90)).isoformat()
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT DISTINCT isin FROM insider_transactions
        WHERE report_date >= ?
          AND (LOWER(transaction_type) LIKE '%förvärv%'
               OR LOWER(transaction_type) LIKE '%acquisition%')
          AND UPPER(isin) NOT IN ('', 'NONE', 'NAN', 'N/A')
    """, (cutoff,)).fetchall()
    conn.close()
    return [r[0] for r in rows]


def _row_to_tx(r: tuple) -> InsiderTransaction:
    return InsiderTransaction(
        report_date=_parse_date(r[0]),
        transaction_date=_parse_date(r[1]),
        company_name=r[2], isin=r[3], lei_code=r[4],
        insider_name=r[5], role=r[6], transaction_type=r[7],
        instrument_type=r[8], instrument_name=r[9],
        volume=r[10] or 0.0, unit_price=r[11] or 0.0,
        currency=r[12] or "SEK", trading_venue=r[13] or "",
        status=r[14] or "", is_close_associate=bool(r[15]),
    )


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        n = load_from_file(Path(sys.argv[1]))
    else:
        n = refresh(days_back=90)

    print(f"\nStored {n} new transactions.")
    isins = get_all_active_isins()
    print(f"Active ISINs with recent buys: {len(isins)}")
    if isins:
        print("Sample ISINs:", isins[:5])
        # Show a sample transaction
        sample = query_buys(isins[0], date.today())
        if sample:
            t = sample[-1]
            print(f"\nLatest buy — {t.company_name} ({t.isin})")
            print(f"  {t.insider_name} | {t.role}")
            print(f"  {t.volume:,.0f} shares @ {t.unit_price:,.2f} {t.currency}")
            print(f"  Reported: {t.report_date}  |  Traded: {t.transaction_date}")
