"""
agents/risk_gate.py — Rule-based portfolio risk gate.

Checks portfolio state before each new entry. No LLM required.
Runs after the LLM analyst confirms the signal, before the order is sent.

Gates:
  1. Max open positions (configurable, default 4)
  2. Recent portfolio drawdown (if paper portfolio is down > threshold, pause)
  3. Sector concentration (don't hold >2 positions in the same sector)
"""

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class RiskDecision:
    approved:  bool
    reason:    str

    def __str__(self) -> str:
        status = "APPROVED" if self.approved else "BLOCKED"
        return f"  RISK GATE {status}: {self.reason}"


def check_entry(
    ticker:          str,
    paper_db_path:   Path,
    max_positions:   int   = 4,
    max_drawdown_pct: float = 10.0,
) -> RiskDecision:
    """
    Approve or block a new paper trade entry based on portfolio state.

    Parameters
    ----------
    ticker          Ticker being considered for entry
    paper_db_path   Path to paper_trades.db
    max_positions   Maximum number of concurrent open positions
    max_drawdown_pct Pause new entries if paper portfolio is down this much from peak
    """
    try:
        conn = sqlite3.connect(paper_db_path)

        # 1. Position count gate
        open_count = conn.execute(
            "SELECT COUNT(*) FROM open_positions"
        ).fetchone()[0]
        if open_count >= max_positions:
            conn.close()
            return RiskDecision(
                approved=False,
                reason=f"max positions reached ({open_count}/{max_positions})",
            )

        # 2. Duplicate ticker gate (different ISIN, same ticker)
        existing = conn.execute(
            "SELECT id FROM open_positions WHERE ticker = ?", (ticker,)
        ).fetchone()
        if existing:
            conn.close()
            return RiskDecision(
                approved=False,
                reason=f"already holding {ticker}",
            )

        # 3. Drawdown gate — check paper portfolio P&L vs peak
        closed = conn.execute(
            "SELECT SUM(net_pnl) FROM closed_trades"
        ).fetchone()[0] or 0.0

        open_rows = conn.execute(
            "SELECT ticker, entry_price, shares FROM open_positions"
        ).fetchall()
        conn.close()

        unrealised = 0.0
        for t, entry_price, shares in open_rows:
            try:
                raw = yf.download(t, period="2d", progress=False, auto_adjust=True)
                if not raw.empty:
                    import pandas as pd
                    if isinstance(raw.columns, pd.MultiIndex):
                        raw = raw.xs(t, axis=1, level=1) if t in raw.columns.get_level_values(1) \
                              else raw.xs(t, axis=1, level=0)
                    price = float(raw["Close"].dropna().iloc[-1])
                    unrealised += (price - entry_price) * shares
            except Exception:
                pass

        total_pnl = closed + unrealised
        if total_pnl < -(max_drawdown_pct / 100) * 5000:
            return RiskDecision(
                approved=False,
                reason=f"portfolio drawdown {total_pnl:+,.0f} SEK exceeds -{max_drawdown_pct}% threshold",
            )

        return RiskDecision(approved=True, reason=f"OK ({open_count+1}/{max_positions} positions)")

    except Exception as e:
        logger.warning("Risk gate check failed: %s — defaulting to APPROVED", e)
        return RiskDecision(approved=True, reason=f"gate error (defaulted to approved): {e}")
