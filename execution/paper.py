"""
execution/paper.py — Paper trading tracker.

Records virtual trades (no real money) in a local SQLite database so you
can run the live scanner every day and track whether signals are profitable
before risking real capital.

Transaction cost model:
    Commission = 10 SEK + 0.1% of trade value (per leg, applied on both entry and exit).

Usage:
    from execution.paper import PaperPortfolio
    port = PaperPortfolio()
    port.enter("SE0008613731", "BIOVIC-B.ST", date(2026,4,25), 8.37, 100, score=77)
    port.update_prices()   # fetch latest prices and mark positions to market
    port.report()          # print open positions + closed trade history
"""

import sqlite3
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import yfinance as yf

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "paper_trades.db"

COMMISSION_FIXED = 10.0    # SEK per leg
COMMISSION_PCT   = 0.001   # 0.1% of trade value per leg

# Exit rules (must match backtest for like-for-like comparison)
STOP_LOSS_PCT   = -0.08
TAKE_PROFIT_PCT =  0.25
MAX_HOLD_DAYS   = 90   # Wahlström (2003) Swedish market optimum, confirmed by GU thesis 2014


def _commission(trade_value_sek: float) -> float:
    return COMMISSION_FIXED + COMMISSION_PCT * trade_value_sek


def _get_conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS open_positions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            isin          TEXT NOT NULL UNIQUE,
            ticker        TEXT NOT NULL,
            entry_date    TEXT NOT NULL,
            entry_price   REAL NOT NULL,
            shares        INTEGER NOT NULL,
            score         INTEGER NOT NULL,
            stop_loss     REAL NOT NULL,
            entry_cost    REAL NOT NULL,
            entered_at    TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS closed_trades (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            isin          TEXT NOT NULL,
            ticker        TEXT NOT NULL,
            entry_date    TEXT NOT NULL,
            exit_date     TEXT NOT NULL,
            entry_price   REAL NOT NULL,
            exit_price    REAL NOT NULL,
            shares        INTEGER NOT NULL,
            score         INTEGER NOT NULL,
            gross_pnl     REAL NOT NULL,
            total_costs   REAL NOT NULL,
            net_pnl       REAL NOT NULL,
            pct_return    REAL NOT NULL,
            exit_reason   TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


class PaperPortfolio:

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path

    def _conn(self) -> sqlite3.Connection:
        return _get_conn(self.db_path)

    def enter(
        self,
        isin: str,
        ticker: str,
        entry_date: date,
        entry_price: float,
        shares: int,
        score: int,
    ) -> bool:
        """
        Open a new paper position. Returns True if entered, False if skipped.
        Skips if ISIN is already open (Fix 3: duplicate check).
        """
        trade_value = entry_price * shares
        cost = _commission(trade_value)
        stop_loss = round(entry_price * (1 + STOP_LOSS_PCT), 4)

        conn = self._conn()
        existing = conn.execute(
            "SELECT id FROM open_positions WHERE isin = ?", (isin,)
        ).fetchone()
        if existing:
            conn.close()
            logger.debug("Paper: %s already open — duplicate skipped", isin)
            return False

        conn.execute("""
            INSERT INTO open_positions
                (isin, ticker, entry_date, entry_price, shares, score,
                 stop_loss, entry_cost, entered_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            isin, ticker, entry_date.isoformat(), entry_price, shares, score,
            stop_loss, round(cost, 2),
            datetime.now(timezone.utc).isoformat(),
        ))
        conn.commit()
        conn.close()

        net_cost = trade_value + cost
        print(f"  PAPER ENTER  {ticker:<14} @ {entry_price:.2f} SEK  "
              f"x{shares}  cost={net_cost:,.0f} SEK (incl. {cost:.0f} commission)  "
              f"score={score}")
        return True

    def update_prices(self) -> None:
        """Fetch latest prices and apply exit rules to all open positions."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT id, isin, ticker, entry_date, entry_price, shares, score, stop_loss, entry_cost "
            "FROM open_positions"
        ).fetchall()
        conn.close()

        if not rows:
            return

        tickers = list({r[2] for r in rows})
        try:
            raw = yf.download(tickers, period="5d", progress=False, auto_adjust=True)
        except Exception as e:
            logger.error("Paper price update failed: %s", e)
            return

        today = date.today()

        for row in rows:
            rid, isin, ticker, entry_date_str, entry_price, shares, score, stop_loss, entry_cost = row
            entry_dt = date.fromisoformat(entry_date_str)
            days_held = (today - entry_dt).days

            try:
                if hasattr(raw.columns, "levels"):
                    lvl = 0 if ticker in raw.columns.get_level_values(0) else 1
                    df = raw.xs(ticker, axis=1, level=lvl)
                else:
                    df = raw
                current_price = float(df["Close"].dropna().iloc[-1])
            except Exception:
                logger.debug("No price for %s — skipping", ticker)
                continue

            pct = (current_price - entry_price) / entry_price

            exit_reason = None
            if pct <= STOP_LOSS_PCT:
                exit_reason = "stop_loss"
            elif pct >= TAKE_PROFIT_PCT:
                exit_reason = "take_profit"
            elif days_held >= MAX_HOLD_DAYS:
                exit_reason = "time_exit"

            if exit_reason:
                self._close(
                    rid, isin, ticker, entry_dt, entry_price,
                    shares, score, entry_cost, current_price, today, exit_reason,
                )
            else:
                gross_pnl = (current_price - entry_price) * shares
                net_pnl   = gross_pnl - entry_cost   # exit cost not charged yet (still open)
                print(f"  OPEN  {ticker:<14}  "
                      f"entry={entry_price:.2f}  now={current_price:.2f}  "
                      f"gross P&L={gross_pnl:+,.0f}  net≈{net_pnl:+,.0f} SEK "
                      f"({pct:+.1%})  {days_held}d")

    def _close(
        self,
        row_id: int,
        isin: str,
        ticker: str,
        entry_date: date,
        entry_price: float,
        shares: int,
        score: int,
        entry_cost: float,
        exit_price: float,
        exit_date: date,
        exit_reason: str,
    ) -> None:
        exit_value   = exit_price * shares
        exit_cost    = _commission(exit_value)
        total_costs  = entry_cost + exit_cost
        gross_pnl    = (exit_price - entry_price) * shares
        net_pnl      = gross_pnl - total_costs
        pct_return   = net_pnl / (entry_price * shares)   # return on capital deployed

        conn = self._conn()
        conn.execute("DELETE FROM open_positions WHERE id = ?", (row_id,))
        conn.execute("""
            INSERT INTO closed_trades
                (isin, ticker, entry_date, exit_date, entry_price, exit_price,
                 shares, score, gross_pnl, total_costs, net_pnl, pct_return, exit_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            isin, ticker, entry_date.isoformat(), exit_date.isoformat(),
            entry_price, exit_price, shares, score,
            round(gross_pnl, 2), round(total_costs, 2),
            round(net_pnl, 2), round(pct_return, 6), exit_reason,
        ))
        conn.commit()
        conn.close()

        print(f"  CLOSE {ticker:<14}  "
              f"entry={entry_price:.2f}  exit={exit_price:.2f}  "
              f"gross={gross_pnl:+,.0f}  costs={total_costs:.0f}  "
              f"net={net_pnl:+,.0f} SEK ({pct_return:+.1%})  [{exit_reason}]")

    def close_manually(self, ticker: str, exit_price: float, reason: str = "manual") -> None:
        conn = self._conn()
        row = conn.execute(
            "SELECT id, isin, ticker, entry_date, entry_price, shares, score, entry_cost "
            "FROM open_positions WHERE ticker = ?", (ticker,)
        ).fetchone()
        conn.close()
        if not row:
            print(f"No open paper position for {ticker}")
            return
        rid, isin, t, entry_date_str, entry_price, shares, score, entry_cost = row
        self._close(
            rid, isin, t, date.fromisoformat(entry_date_str),
            entry_price, shares, score, entry_cost,
            exit_price, date.today(), reason,
        )

    def report(self) -> None:
        conn = self._conn()
        open_rows = conn.execute(
            "SELECT isin, ticker, entry_date, entry_price, shares, score, stop_loss, entry_cost "
            "FROM open_positions ORDER BY entry_date"
        ).fetchall()
        closed_rows = conn.execute(
            "SELECT ticker, entry_date, exit_date, entry_price, exit_price, "
            "shares, gross_pnl, total_costs, net_pnl, pct_return, exit_reason "
            "FROM closed_trades ORDER BY exit_date"
        ).fetchall()
        conn.close()

        print("\n" + "=" * 64)
        print("PAPER PORTFOLIO REPORT")
        print("=" * 64)

        print(f"\nOpen positions: {len(open_rows)}")
        if open_rows:
            print(f"  {'Ticker':<14} {'Entry':<12} {'Price':>8} {'Shares':>6} "
                  f"{'Score':>6} {'Stop':>8} {'Comm':>6}")
            for isin, ticker, ed, ep, shares, score, stop, cost in open_rows:
                print(f"  {ticker:<14} {ed:<12} {ep:>8.2f} {shares:>6} "
                      f"{score:>6} {stop:>8.2f} {cost:>6.0f}")

        print(f"\nClosed trades: {len(closed_rows)}")
        if closed_rows:
            wins     = [r for r in closed_rows if r[8] > 0]
            total_net = sum(r[8] for r in closed_rows)
            win_rate  = len(wins) / len(closed_rows) * 100
            print(f"  Win rate: {win_rate:.0f}%  |  Total net P&L: {total_net:+,.0f} SEK")
            print(f"\n  {'Ticker':<14} {'Entry':<12} {'Exit':<12} "
                  f"{'Gross':>8} {'Costs':>6} {'Net':>8} {'Ret%':>7}  Reason")
            for t, ed, xd, ep, xp, sh, gross, costs, net, pct, reason in closed_rows:
                print(f"  {t:<14} {ed:<12} {xd:<12} "
                      f"{gross:>+8,.0f} {costs:>6,.0f} {net:>+8,.0f} "
                      f"{pct:>+7.1%}  {reason}")
