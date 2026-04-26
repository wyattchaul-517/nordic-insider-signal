"""
signals/sector_filter.py — Layer 1 sector health score (0–20 pts).

Uses yfinance sector metadata for the resolved ticker. Checks whether the
sector has been in a severe drawdown over the prior 3 months.

Limitations:
  - yfinance sector classification is coarse and sometimes wrong for Swedish stocks
  - No sector ETF proxy available without paid data — uses sector peer average instead
  - Hard veto for "systemic regulatory risk" requires a manual flag file

Upgrade path: replace peer-average drawdown with actual sector ETF data
(XACT funds on Stockholm) when Börsdata subscription is available.
"""

import logging
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import yfinance as yf

logger = logging.getLogger(__name__)

UNIVERSE_DB = Path(__file__).parent.parent / "data" / "universe_cache.db"

# Manually maintained: sectors with active systemic regulatory risk
# Add sector strings here when regulatory crackdowns are known.
# Format: lowercase yfinance sector strings.
VETOED_SECTORS: set[str] = set()


def get_sector_score(
    isin: str,
    as_of_date: date,
    cfg: dict,
) -> tuple[int, bool, list[str]]:
    """
    Returns (score: 0–20, hard_veto: bool, notes: list[str]).
    Called by ConvictionScorer._score_sector().
    """
    notes: list[str] = []

    ticker = _get_ticker(isin)
    if not ticker:
        notes.append("sector: no ticker mapping — neutral score applied")
        return 10, False, notes

    sector = _get_sector(ticker)
    if not sector:
        notes.append(f"sector: no yfinance sector for {ticker} — neutral score")
        return 10, False, notes

    # Hard veto for flagged sectors
    if sector.lower() in VETOED_SECTORS:
        notes.append(f"sector: '{sector}' is vetoed (systemic regulatory risk)")
        return 0, True, notes

    # Sector drawdown check using cached price data for the ticker itself
    # as a single-stock proxy (not ideal, but avoids needing sector ETFs)
    drawdown = _compute_3m_drawdown(ticker, as_of_date)
    max_drawdown = cfg.get("sector", {}).get("max_drawdown_pct_3m", -15.0)

    if drawdown is None:
        notes.append(f"sector: '{sector}' — no drawdown data, neutral score")
        return 10, False, notes

    if drawdown <= max_drawdown:
        score = 5
        notes.append(
            f"sector: '{sector}' 3m drawdown={drawdown:.1f}% (severe) → {score}pts"
        )
    elif drawdown <= max_drawdown / 2:
        score = 12
        notes.append(
            f"sector: '{sector}' 3m drawdown={drawdown:.1f}% (moderate) → {score}pts"
        )
    else:
        score = 20
        notes.append(
            f"sector: '{sector}' 3m drawdown={drawdown:.1f}% (healthy) → {score}pts"
        )

    return score, False, notes


def _get_ticker(isin: str) -> Optional[str]:
    if not UNIVERSE_DB.exists():
        return None
    conn = sqlite3.connect(UNIVERSE_DB)
    row = conn.execute(
        "SELECT ticker FROM ticker_cache WHERE isin = ? AND ticker IS NOT NULL",
        (isin,)
    ).fetchone()
    conn.close()
    return row[0] if row else None


def _get_sector(ticker: str) -> Optional[str]:
    try:
        info = yf.Ticker(ticker).info or {}
        return info.get("sector") or info.get("industryDisp")
    except Exception:
        return None


def _compute_3m_drawdown(ticker: str, as_of_date: date) -> Optional[float]:
    """
    3-month peak-to-trough drawdown for a ticker as of a given date.
    Uses the universe_cache price_cache table if available.
    """
    start = (as_of_date - timedelta(days=100)).isoformat()
    end = as_of_date.isoformat()

    try:
        conn = sqlite3.connect(UNIVERSE_DB)
        rows = conn.execute("""
            SELECT close FROM price_cache
            WHERE ticker = ? AND date >= ? AND date <= ? AND close > 0
            ORDER BY date
        """, (ticker, start, end)).fetchall()
        conn.close()

        if len(rows) < 20:
            return None

        closes = [r[0] for r in rows]
        peak = max(closes)
        trough = min(closes)
        if peak <= 0:
            return None
        return (trough - peak) / peak * 100.0
    except Exception as e:
        logger.debug("Drawdown calc failed for %s: %s", ticker, e)
        return None
