"""
risk/exits.py — Exit rule evaluator.

Priority order when multiple conditions are met simultaneously:
  1. Thesis invalidation  (insider sells after our entry — signal reversed)
  2. Profit target        (take gains before they evaporate in illiquid names)
  3. Stop loss            (hard capital protection)
  4. Time stop            (thesis has expired — signal should have played out)

All thresholds from config/strategy.yaml. Do not tune based on backtest output.

Timing convention (matches engine.py):
  Exit condition detected on day d using close price → exit recorded at day d close.
  "Next open" fills would be more conservative but close prices give better coverage
  for Nordic small caps where open prices are often stale.
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class ExitDecision:
    should_exit: bool
    reason: str
    pct_change: float


def evaluate(
    entry_date: date,
    entry_price: float,
    current_date: date,
    current_price: float,
    trading_days_held: int,
    new_insider_sell_detected: bool = False,
    stop_loss_pct: float = -10.0,
    profit_target_pct: float = 15.0,
    time_stop_trading_days: int = 30,
) -> ExitDecision:
    """
    Evaluate all exit conditions for an open position.

    trading_days_held: count of days with valid prices since entry —
                       more accurate than calendar days for the time stop.
    new_insider_sell_detected: True if any insider in the original buying
                               cluster has published a sell report since entry.
    """
    if current_price <= 0:
        # No price data — do not force exit; wait for next day with data.
        return ExitDecision(should_exit=False, reason="no_price", pct_change=0.0)

    pct = (current_price - entry_price) / entry_price * 100.0

    # ── 1. Thesis invalidation ──────────────────────────────────────────────
    if new_insider_sell_detected:
        return ExitDecision(
            should_exit=True,
            reason=f"thesis_invalidation:insider_sell ({pct:+.1f}%)",
            pct_change=pct,
        )

    # ── 2. Profit target ────────────────────────────────────────────────────
    if pct >= profit_target_pct:
        return ExitDecision(
            should_exit=True,
            reason=f"profit_target:{pct:+.1f}%",
            pct_change=pct,
        )

    # ── 3. Stop loss ────────────────────────────────────────────────────────
    if pct <= stop_loss_pct:
        return ExitDecision(
            should_exit=True,
            reason=f"stop_loss:{pct:+.1f}%",
            pct_change=pct,
        )

    # ── 4. Time stop ────────────────────────────────────────────────────────
    if trading_days_held >= time_stop_trading_days:
        return ExitDecision(
            should_exit=True,
            reason=f"time_stop:{trading_days_held}d ({pct:+.1f}%)",
            pct_change=pct,
        )

    return ExitDecision(should_exit=False, reason="hold", pct_change=pct)
