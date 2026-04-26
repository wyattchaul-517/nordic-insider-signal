"""
Fundamental data for Nordic stocks.

Primary source: Yahoo Finance (free, via yfinance).
Coverage gaps for Nordic micro-caps are expected — the system degrades
gracefully by returning None and the scorer applies a conservative penalty
rather than crashing.

Upgrade path: set FUNDAMENTAL_SOURCE = "borsdata" in strategy.yaml once
a Börsdata subscription is available. Add BorsdataClient below.
"""

import logging
from datetime import date, datetime
from typing import Optional

import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)


class FundamentalSnapshot:
    """Point-in-time fundamental snapshot for a single stock."""

    def __init__(
        self,
        ticker: str,
        as_of_date: date,
        revenue_yoy_growth: Optional[float],   # e.g. 0.05 = +5%
        debt_equity: Optional[float],
        cash_sek: Optional[float],
        annual_revenue_sek: Optional[float],
        has_fraud_investigation: bool = False,
        data_source: str = "yahoo_finance",
        data_complete: bool = True,
    ):
        self.ticker = ticker
        self.as_of_date = as_of_date
        self.revenue_yoy_growth = revenue_yoy_growth
        self.debt_equity = debt_equity
        self.cash_sek = cash_sek
        self.annual_revenue_sek = annual_revenue_sek
        self.has_fraud_investigation = has_fraud_investigation
        self.data_source = data_source
        self.data_complete = data_complete

    # ── Derived properties used by scorer ───────────────────────────────────

    @property
    def revenue_trend_ok(self) -> Optional[bool]:
        """True if YoY revenue growth is flat or positive. None if unknown."""
        if self.revenue_yoy_growth is None:
            return None
        return self.revenue_yoy_growth >= -0.02  # small negative acceptable

    @property
    def debt_equity_ok(self) -> Optional[bool]:
        if self.debt_equity is None:
            return None
        return self.debt_equity <= 3.0

    @property
    def cash_runway_months(self) -> Optional[float]:
        """
        Estimate months of cash runway. Requires both cash and revenue.
        Uses revenue as a proxy for burn rate (conservative for pre-profit companies).
        """
        if self.cash_sek is None or self.annual_revenue_sek is None:
            return None
        if self.annual_revenue_sek <= 0:
            return None
        monthly_burn_estimate = self.annual_revenue_sek / 12
        return self.cash_sek / monthly_burn_estimate if monthly_burn_estimate > 0 else None

    def estimated_annual_salary_sek(self, role: str) -> float:
        """
        Rough salary estimate by role (SEK) for insider purchase significance check.
        These are conservative Swedish market estimates.
        """
        role_lower = role.lower()
        if any(r in role_lower for r in ["vd", "ceo", "verkställande"]):
            return 3_000_000
        if any(r in role_lower for r in ["cfo", "finanschef", "ekonomichef"]):
            return 2_500_000
        if "styrelseordförande" in role_lower or "chairman" in role_lower:
            return 1_500_000
        if "styrelseledamot" in role_lower or "board" in role_lower:
            return 500_000
        return 1_000_000  # default for other senior roles


def _safe_float(val) -> Optional[float]:
    try:
        f = float(val)
        return f if not pd.isna(f) else None
    except (TypeError, ValueError):
        return None


def fetch_yahoo(ticker: str, as_of_date: date) -> FundamentalSnapshot:
    """
    Fetch fundamental snapshot from Yahoo Finance.

    Note on point-in-time correctness: yfinance returns the most recent
    reported figures, not historical point-in-time values. For backtesting,
    this introduces a mild look-ahead bias on fundamentals (you'd see
    more recent data than was available at signal time). This is acceptable
    for a first version — Börsdata provides proper point-in-time data.
    Flag this in any backtest results.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
    except Exception as e:
        logger.warning("Yahoo Finance fetch failed for %s: %s", ticker, e)
        return FundamentalSnapshot(
            ticker=ticker, as_of_date=as_of_date,
            revenue_yoy_growth=None, debt_equity=None,
            cash_sek=None, annual_revenue_sek=None,
            data_complete=False,
        )

    # Revenue YoY growth
    revenue_yoy = None
    try:
        financials = t.financials
        if financials is not None and not financials.empty:
            rev_row = financials.loc["Total Revenue"] if "Total Revenue" in financials.index else None
            if rev_row is not None and len(rev_row) >= 2:
                r_now = _safe_float(rev_row.iloc[0])
                r_prev = _safe_float(rev_row.iloc[1])
                if r_now and r_prev and r_prev != 0:
                    revenue_yoy = (r_now - r_prev) / abs(r_prev)
    except Exception as e:
        logger.debug("Revenue YoY calc failed for %s: %s", ticker, e)

    # Debt/Equity
    debt_equity = _safe_float(info.get("debtToEquity"))
    if debt_equity is not None:
        debt_equity = debt_equity / 100.0  # Yahoo returns as percentage

    # Cash
    cash = _safe_float(info.get("totalCash"))
    # Yahoo returns in USD by default for non-USD stocks — adjust if currency known
    currency = info.get("currency", "SEK")
    if currency != "SEK" and cash is not None:
        logger.debug("%s: cash reported in %s, not SEK — conversion not applied", ticker, currency)

    # Revenue
    revenue = _safe_float(info.get("totalRevenue"))

    data_complete = not any(v is None for v in [debt_equity, cash, revenue])

    return FundamentalSnapshot(
        ticker=ticker,
        as_of_date=as_of_date,
        revenue_yoy_growth=revenue_yoy,
        debt_equity=debt_equity,
        cash_sek=cash,
        annual_revenue_sek=revenue,
        data_complete=data_complete,
    )


def fetch(
    ticker: str,
    as_of_date: date,
    source: str = "yahoo_finance",
) -> FundamentalSnapshot:
    """
    Dispatcher. Add "borsdata" branch here when Börsdata subscription is available.
    Börsdata provides proper point-in-time data, removing the look-ahead caveat.
    """
    if source == "yahoo_finance":
        return fetch_yahoo(ticker, as_of_date)
    raise ValueError(f"Unknown fundamental source: {source}. Supported: yahoo_finance")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    snap = fetch("ERIC-B.ST", date.today())
    print(f"Ericsson revenue YoY: {snap.revenue_yoy_growth}")
    print(f"Debt/Equity: {snap.debt_equity}")
    print(f"Cash runway: {snap.cash_runway_months} months")
    print(f"Data complete: {snap.data_complete}")
