"""
backtest/engine.py — Event-driven walk-forward backtest engine.

Timing convention (critical for point-in-time correctness):
  Day d:  FI report published → signal detected → validation + scoring
  Day d+1: position entered at d+1 close price (conservative fill assumption)
           This models the realistic lag of seeing the report, doing checks,
           and placing an order the following market day.

Exit fills:
  Exit condition met at day d close → position closed at day d close.
  Slightly optimistic (assumes fill at close) but necessary for Nordic
  small caps where next-day open prices are often missing in yfinance.

Walk-forward discipline:
  The engine takes explicit start/end dates. The caller is responsible for
  never running the validate period more than once and for never adjusting
  config parameters based on validate results. See backtest/walk_forward.py.

Known limitations:
  - Price data from yfinance; coverage gaps treated as "no fill" (position held).
  - No transaction costs modelled in v1. Add in fill_model.py later.
  - Macro/sector filter scores are approximate (see macro_filter.py).
  - Survivorship bias in universe (see universe.py documentation).
"""

import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from data.fi_scraper import query_buys, query_sells, DB_PATH as FI_DB
from backtest.universe import CandidateValidator, DB_PATH as UNIVERSE_DB
from signals.scorer import ConvictionScorer
from risk.sizing import size_position, SizingResult
from risk.exits import evaluate as evaluate_exit

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "strategy.yaml"
PRICE_DB = UNIVERSE_DB  # price_cache lives in universe_cache.db


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Position:
    isin: str
    ticker: str
    entry_date: date
    entry_price: float
    shares: int
    conviction_score: int
    sizing: SizingResult
    trading_days_held: int = 0
    buyers_at_entry: frozenset = field(default_factory=frozenset)


@dataclass
class ClosedTrade:
    isin: str
    ticker: str
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    shares: int
    conviction_score: int
    exit_reason: str
    pct_return: float
    sek_return: float
    trading_days_held: int


@dataclass
class BacktestResult:
    trades: list[ClosedTrade]
    daily_portfolio_value: pd.Series   # indexed by date
    initial_capital: float
    start_date: date
    end_date: date

    # ── Metrics ──────────────────────────────────────────────────────────────

    @property
    def total_return_pct(self) -> float:
        if self.daily_portfolio_value.empty:
            return 0.0
        return (self.daily_portfolio_value.iloc[-1] / self.initial_capital - 1) * 100

    @property
    def annualised_return_pct(self) -> float:
        n_days = (self.end_date - self.start_date).days
        if n_days <= 0:
            return 0.0
        r = self.daily_portfolio_value.iloc[-1] / self.initial_capital
        return (r ** (365 / n_days) - 1) * 100

    @property
    def sharpe(self) -> float:
        if len(self.daily_portfolio_value) < 2:
            return 0.0
        daily_returns = self.daily_portfolio_value.pct_change().dropna()
        if daily_returns.std() == 0:
            return 0.0
        return (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5)

    @property
    def max_drawdown_pct(self) -> float:
        if self.daily_portfolio_value.empty:
            return 0.0
        roll_max = self.daily_portfolio_value.cummax()
        drawdown = (self.daily_portfolio_value - roll_max) / roll_max * 100
        return float(drawdown.min())

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pct_return > 0)
        return wins / len(self.trades)

    @property
    def avg_win_pct(self) -> float:
        wins = [t.pct_return for t in self.trades if t.pct_return > 0]
        return sum(wins) / len(wins) if wins else 0.0

    @property
    def avg_loss_pct(self) -> float:
        losses = [t.pct_return for t in self.trades if t.pct_return <= 0]
        return sum(losses) / len(losses) if losses else 0.0

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.sek_return for t in self.trades if t.sek_return > 0)
        gross_loss = abs(sum(t.sek_return for t in self.trades if t.sek_return < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    def summary(self) -> str:
        lines = [
            f"Period:          {self.start_date} → {self.end_date}",
            f"Initial capital: {self.initial_capital:,.0f} SEK",
            f"Final value:     {self.daily_portfolio_value.iloc[-1]:,.0f} SEK" if not self.daily_portfolio_value.empty else "",
            f"Total return:    {self.total_return_pct:+.2f}%",
            f"Annualised:      {self.annualised_return_pct:+.2f}%",
            f"Sharpe:          {self.sharpe:.2f}",
            f"Max drawdown:    {self.max_drawdown_pct:.2f}%",
            f"Trades:          {len(self.trades)}",
            f"Win rate:        {self.win_rate:.1%}",
            f"Avg win:         {self.avg_win_pct:+.2f}%",
            f"Avg loss:        {self.avg_loss_pct:+.2f}%",
            f"Profit factor:   {self.profit_factor:.2f}",
        ]
        return "\n".join(l for l in lines if l)


# ── Price fetching ────────────────────────────────────────────────────────────

def _get_price(ticker: str, d: date, conn: sqlite3.Connection) -> Optional[float]:
    """Fetch close price from cache. Returns None if missing."""
    row = conn.execute(
        "SELECT close, volume FROM price_cache WHERE ticker=? AND date=?",
        (ticker, d.isoformat())
    ).fetchone()
    if row and row[0] and row[0] > 0 and row[1] and row[1] > 0:
        return float(row[0])
    return None


def _price_in_window(ticker: str, target: date, conn: sqlite3.Connection,
                      window: int = 3) -> Optional[float]:
    """Find the nearest price within ±window days. Returns None if not found."""
    for delta in range(0, window + 1):
        for sign in ([0] if delta == 0 else [1, -1]):
            d = target + timedelta(days=delta * sign)
            p = _get_price(ticker, d, conn)
            if p:
                return p
    return None


# ── FI signal detection ───────────────────────────────────────────────────────

def _get_new_fi_isins(d: date, fi_db: Path) -> list[str]:
    """Return ISINs that had at least one share-buy report published exactly on day d."""
    conn = sqlite3.connect(fi_db)
    rows = conn.execute("""
        SELECT DISTINCT isin FROM insider_transactions
        WHERE DATE(report_date) = ?
          AND (LOWER(transaction_type) LIKE '%förvärv%'
               OR LOWER(transaction_type) LIKE '%acquisition%')
          AND instrument_type = 'Aktie'
          AND UPPER(isin) NOT IN ('', 'NONE', 'NAN', 'N/A')
    """, (d.isoformat(),)).fetchall()
    conn.close()
    return [r[0] for r in rows]


def _get_buying_insiders(isin: str, as_of: date, window: int, fi_db: Path) -> frozenset[str]:
    """Return frozenset of distinct insider names who bought within window days."""
    buys = query_buys(isin, as_of, window_days=window, db_path=fi_db)
    return frozenset(t.insider_name.lower().strip() for t in buys)


def _has_new_insider_sell(
    isin: str,
    entry_date: date,
    current_date: date,
    original_buyers: frozenset[str],
    fi_db: Path,
) -> bool:
    """
    True if any of the original buyers has filed a sell report since entry.
    Only checks insiders who were part of the original buy cluster — filters
    out routine sells by unrelated insiders.
    """
    sells = query_sells(isin, current_date, window_days=180, db_path=fi_db)
    post_entry_sellers = {
        t.insider_name.lower().strip()
        for t in sells
        if t.report_date >= entry_date
    }
    return bool(post_entry_sellers & original_buyers)


# ── Main engine ───────────────────────────────────────────────────────────────

class BacktestEngine:

    def __init__(
        self,
        cfg: Optional[dict] = None,
        fi_db: Path = FI_DB,
        universe_db: Path = UNIVERSE_DB,
    ):
        if cfg is None:
            with open(CONFIG_PATH) as f:
                cfg = yaml.safe_load(f)
        self.cfg = cfg
        self.fi_db = fi_db
        self.universe_db = universe_db
        self.validator = CandidateValidator(db_path=universe_db)
        self.scorer = ConvictionScorer(cfg)

        sig = cfg.get("signal", {})
        pos = cfg.get("position", {})
        ex = cfg.get("exit", {})
        uni = cfg.get("universe", {})

        self.insider_window = sig.get("insider_window_days", 30)
        self.min_distinct = sig.get("min_distinct_insiders", 2)
        self.max_positions = pos.get("max_open_positions", 6)
        self.max_portfolio_pct = pos.get("max_portfolio_pct", 7.0)
        self.max_pct_adv = pos.get("max_pct_of_adv", 2.0)
        self.half_kelly = pos.get("use_half_kelly", True)
        self.stop_loss = ex.get("stop_loss_pct", -10.0)
        self.profit_target = ex.get("profit_target_pct", 15.0)
        self.time_stop = ex.get("time_stop_trading_days", 30)
        self.min_adv = uni.get("min_adv_sek", 500_000)

    def run(
        self,
        start_date: date,
        end_date: date,
        initial_capital_sek: float = 50_000.0,
    ) -> BacktestResult:
        """
        Run the backtest over [start_date, end_date].
        Returns a BacktestResult with all trades and daily portfolio values.

        Call this ONCE per period. Never re-run with adjusted parameters.
        """
        logger.info("Backtest: %s → %s | capital=%.0f SEK", start_date, end_date, initial_capital_sek)

        cash = initial_capital_sek
        open_positions: dict[str, Position] = {}
        closed_trades: list[ClosedTrade] = []
        daily_values: dict[date, float] = {}

        price_conn = sqlite3.connect(self.universe_db)
        trading_days = self._trading_days(start_date, end_date)

        for d in trading_days:
            portfolio_value = self._portfolio_value(cash, open_positions, d, price_conn)

            # ── 1. Exit checks ────────────────────────────────────────────────
            for isin in list(open_positions.keys()):
                pos = open_positions[isin]
                price = _price_in_window(pos.ticker, d, price_conn)

                if price:
                    pos.trading_days_held += 1

                thesis_broken = _has_new_insider_sell(
                    isin, pos.entry_date, d, pos.buyers_at_entry, self.fi_db
                )
                decision = evaluate_exit(
                    entry_date=pos.entry_date,
                    entry_price=pos.entry_price,
                    current_date=d,
                    current_price=price or 0.0,
                    trading_days_held=pos.trading_days_held,
                    new_insider_sell_detected=thesis_broken,
                    stop_loss_pct=self.stop_loss,
                    profit_target_pct=self.profit_target,
                    time_stop_trading_days=self.time_stop,
                )

                if decision.should_exit and price:
                    cash += price * pos.shares
                    sek_return = (price - pos.entry_price) * pos.shares
                    closed_trades.append(ClosedTrade(
                        isin=isin,
                        ticker=pos.ticker,
                        entry_date=pos.entry_date,
                        exit_date=d,
                        entry_price=pos.entry_price,
                        exit_price=price,
                        shares=pos.shares,
                        conviction_score=pos.conviction_score,
                        exit_reason=decision.reason,
                        pct_return=decision.pct_change,
                        sek_return=sek_return,
                        trading_days_held=pos.trading_days_held,
                    ))
                    logger.info(
                        "EXIT %s @ %.2f | %s | %+.1f%% | %dd",
                        isin, price, decision.reason, decision.pct_change,
                        pos.trading_days_held,
                    )
                    del open_positions[isin]

            # ── 2. New signal detection ───────────────────────────────────────
            if len(open_positions) < self.max_positions:
                new_isins = _get_new_fi_isins(d, self.fi_db)
                candidates = [i for i in new_isins if i not in open_positions]

                for isin in candidates:
                    if len(open_positions) >= self.max_positions:
                        break

                    # Cluster check
                    buyers = _get_buying_insiders(
                        isin, d, self.insider_window, self.fi_db
                    )
                    if len(buyers) < self.min_distinct:
                        continue

                    # Universe validation
                    val = self.validator.validate(isin, d)
                    if not val.valid or not val.ticker or not val.adv_sek:
                        logger.debug("SKIP %s: %s", isin, val.reason)
                        continue

                    # Conviction scoring
                    breakdown = self.scorer.score(isin, d)
                    if not breakdown.tradeable:
                        logger.debug("SKIP %s: score=%d veto=%s",
                                     isin, breakdown.final_score, breakdown.veto_active)
                        continue

                    # Fill price: next available close (d+1 preferred)
                    fill_price = None
                    for delta in range(1, 4):
                        fill_price = _get_price(val.ticker, d + timedelta(days=delta), price_conn)
                        if fill_price:
                            break
                    if not fill_price:
                        logger.debug("SKIP %s: no fill price available after %s", isin, d)
                        continue

                    # Position sizing
                    sizing = size_position(
                        conviction_score=breakdown.final_score,
                        portfolio_value_sek=portfolio_value,
                        adv_sek=val.adv_sek,
                        price_sek=fill_price,
                        max_portfolio_pct=self.max_portfolio_pct,
                        max_pct_of_adv=self.max_pct_adv,
                        use_half_kelly=self.half_kelly,
                    )
                    if not sizing:
                        logger.debug("SKIP %s: position too small to size", isin)
                        continue

                    if sizing.position_value_sek > cash:
                        logger.debug("SKIP %s: insufficient cash (need %.0f, have %.0f)",
                                     isin, sizing.position_value_sek, cash)
                        continue

                    # Enter position
                    cash -= sizing.position_value_sek
                    open_positions[isin] = Position(
                        isin=isin,
                        ticker=val.ticker,
                        entry_date=d,
                        entry_price=fill_price,
                        shares=sizing.shares,
                        conviction_score=breakdown.final_score,
                        sizing=sizing,
                        buyers_at_entry=buyers,
                    )
                    logger.info(
                        "ENTER %s (%s) @ %.2f | %d shares | score=%d | %s",
                        isin, val.ticker, fill_price, sizing.shares,
                        breakdown.final_score, sizing.binding_constraint,
                    )

            # ── 3. Record daily portfolio value ───────────────────────────────
            daily_values[d] = self._portfolio_value(cash, open_positions, d, price_conn)

        # Force-close any positions still open at end date (at last known price)
        for isin, pos in open_positions.items():
            price = _price_in_window(pos.ticker, end_date, price_conn, window=5)
            if price:
                cash += price * pos.shares
                sek_return = (price - pos.entry_price) * pos.shares
                closed_trades.append(ClosedTrade(
                    isin=isin, ticker=pos.ticker,
                    entry_date=pos.entry_date, exit_date=end_date,
                    entry_price=pos.entry_price, exit_price=price,
                    shares=pos.shares, conviction_score=pos.conviction_score,
                    exit_reason="end_of_period",
                    pct_return=(price - pos.entry_price) / pos.entry_price * 100,
                    sek_return=sek_return,
                    trading_days_held=pos.trading_days_held,
                ))

        price_conn.close()

        series = pd.Series(daily_values).sort_index()
        result = BacktestResult(
            trades=closed_trades,
            daily_portfolio_value=series,
            initial_capital=initial_capital_sek,
            start_date=start_date,
            end_date=end_date,
        )
        logger.info("Backtest complete.\n%s", result.summary())
        return result

    def _portfolio_value(
        self,
        cash: float,
        positions: dict[str, Position],
        d: date,
        conn: sqlite3.Connection,
    ) -> float:
        position_value = 0.0
        for pos in positions.values():
            price = _price_in_window(pos.ticker, d, conn)
            if price:
                position_value += price * pos.shares
            else:
                position_value += pos.entry_price * pos.shares  # carry at cost if no price
        return cash + position_value

    @staticmethod
    def _trading_days(start: date, end: date) -> list[date]:
        """
        Approximate Swedish trading days using pandas business day calendar.
        Does not account for Swedish public holidays — close enough for backtesting.
        """
        idx = pd.bdate_range(start=start, end=end, freq="B")
        return [d.date() for d in idx]


# ── CLI runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    bt = cfg.get("backtest", {})
    engine = BacktestEngine()

    # ── Validate period (run this ONCE, never adjust parameters after) ────────
    validate_start = date.fromisoformat(bt.get("validate_start", "2023-01-01"))
    validate_end   = date.fromisoformat(bt.get("validate_end",   "2024-12-31"))

    if len(sys.argv) >= 3:
        validate_start = date.fromisoformat(sys.argv[1])
        validate_end   = date.fromisoformat(sys.argv[2])

    capital = float(sys.argv[3]) if len(sys.argv) >= 4 else 50_000.0

    print(f"\nRunning validate period: {validate_start} → {validate_end}")
    print(f"Initial capital: {capital:,.0f} SEK\n")

    result = engine.run(validate_start, validate_end, capital)

    print("\n" + "=" * 50)
    print(result.summary())
    print("=" * 50)

    if result.trades:
        print(f"\nAll trades ({len(result.trades)}):")
        for t in result.trades:
            print(
                f"  {t.entry_date} → {t.exit_date} | {t.isin} | "
                f"{t.pct_return:+.1f}% | {t.exit_reason}"
            )
