"""
risk/sizing.py — Conviction-scaled position sizer.

Position value is the minimum of two hard caps:
  (a) portfolio_value × max_portfolio_pct
  (b) adv_sek × max_pct_of_adv          ← liquidity-linked cap (user's improvement)

Then scaled by conviction tier and half-Kelly to prevent overbetting.
Both caps come from config — never change them mid-backtest.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SizingResult:
    shares: int
    position_value_sek: float
    binding_constraint: str    # "portfolio" | "liquidity"
    kelly_scale: float


def conviction_scale(score: int, use_half_kelly: bool = True) -> float:
    """
    Map conviction score → position scale factor.
    Tiers chosen by logic (signal strength), not by optimising backtest results.
    Half-Kelly halves the scale to reduce variance at the cost of expected return.
    """
    if score >= 85:
        raw = 1.00
    elif score >= 75:
        raw = 0.75
    else:          # 65–74 — minimum tradeable
        raw = 0.50
    return raw * (0.5 if use_half_kelly else 1.0)


def size_position(
    conviction_score: int,
    portfolio_value_sek: float,
    adv_sek: float,
    price_sek: float,
    max_portfolio_pct: float = 7.0,
    max_pct_of_adv: float = 2.0,
    use_half_kelly: bool = True,
    min_position_sek: float = 1_000.0,
) -> Optional[SizingResult]:
    """
    Returns None when the position would be too small to be meaningful,
    or when inputs are degenerate.

    Shares are always rounded down (never round up into a constraint).
    """
    if price_sek <= 0 or adv_sek <= 0 or portfolio_value_sek <= 0:
        return None

    cap_portfolio = portfolio_value_sek * (max_portfolio_pct / 100.0)
    cap_liquidity = adv_sek * (max_pct_of_adv / 100.0)

    base_value = min(cap_portfolio, cap_liquidity)
    binding = "portfolio" if cap_portfolio <= cap_liquidity else "liquidity"

    scale = conviction_scale(conviction_score, use_half_kelly)
    target_value = base_value * scale

    if target_value < min_position_sek:
        return None

    shares = int(target_value / price_sek)
    if shares < 1:
        return None

    actual_value = shares * price_sek
    return SizingResult(
        shares=shares,
        position_value_sek=actual_value,
        binding_constraint=binding,
        kelly_scale=scale,
    )
