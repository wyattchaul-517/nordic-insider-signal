"""
signals/macro_filter.py — Layer 0 macro environment score (0–15 pts).

Current implementation: uses cached .ST price data to compute approximate
market breadth (% of resolved tickers above their 200-day MA).

Limitations:
  - Only covers ISINs we've already resolved via OpenFIGI (not the full market)
  - Survivorship-biased: delisted stocks aren't in the cache
  - This score is informational guidance, not a veto

Upgrade path: wire in a proper breadth feed (Riksbank composite index,
Nasdaq Nordic API) when available. The function signature stays fixed.
"""

import logging
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

UNIVERSE_DB = Path(__file__).parent.parent / "data" / "universe_cache.db"
PRICE_CACHE_DB = Path(__file__).parent.parent / "data" / "universe_cache.db"


def get_breadth_score(as_of_date: date, cfg: dict) -> tuple[int, list[str]]:
    """
    Returns (score: 0–15, notes: list[str]).
    Called by ConvictionScorer._score_macro().

    Score mapping:
      breadth >= 0.50  →  10 pts  (market healthy)
      breadth >= 0.35  →   7 pts  (mixed)
      breadth <  0.35  →   3 pts  (weak, reduce confidence)
      no data          →   8 pts  (neutral — don't penalise for data gaps)
    Riksbank crisis veto adds/removes no points; it's handled as a hard veto
    in the scorer itself (requires a separate data feed not yet available).
    """
    notes: list[str] = []
    breadth = _compute_breadth(as_of_date)

    if breadth is None:
        notes.append("macro: no breadth data — neutral score applied")
        return 8, notes

    if breadth >= 0.50:
        score = 10
    elif breadth >= 0.35:
        score = 7
    else:
        score = 3

    notes.append(
        f"macro: breadth={breadth:.1%} (biased — cached tickers only) → {score}pts"
    )

    # Riksbank crisis veto placeholder — no data feed yet
    riksbank_pts = cfg.get("macro", {}).get("riksbank_crisis_pts", 5)
    score += riksbank_pts   # adds the full allowance since we can't detect crises yet
    notes.append("macro: Riksbank veto not wired — full bonus awarded")

    return min(score, 15), notes


def _compute_breadth(as_of_date: date, ma_period: int = 200) -> Optional[float]:
    """
    Compute fraction of cached .ST tickers with close > MA on as_of_date.
    Returns None if fewer than 10 tickers have sufficient history.
    """
    if not PRICE_CACHE_DB.exists():
        return None

    conn = sqlite3.connect(PRICE_CACHE_DB)

    # Get all .ST tickers with enough history
    tickers = conn.execute(
        "SELECT DISTINCT ticker FROM price_cache WHERE ticker LIKE '%.ST'"
    ).fetchall()
    tickers = [r[0] for r in tickers]

    if not tickers:
        conn.close()
        return None

    lookback_start = (as_of_date - timedelta(days=ma_period * 2)).isoformat()
    as_of_str = as_of_date.isoformat()

    above = 0
    counted = 0

    for ticker in tickers:
        rows = conn.execute("""
            SELECT date, close FROM price_cache
            WHERE ticker = ? AND date >= ? AND date <= ?
              AND close > 0
            ORDER BY date
        """, (ticker, lookback_start, as_of_str)).fetchall()

        if len(rows) < ma_period:
            continue

        closes = [r[1] for r in rows]
        ma = sum(closes[-ma_period:]) / ma_period
        latest = closes[-1]

        counted += 1
        if latest > ma:
            above += 1

    conn.close()

    if counted < 10:
        return None

    return above / counted
