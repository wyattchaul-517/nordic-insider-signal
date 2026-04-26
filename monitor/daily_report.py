"""
monitor/daily_report.py — Morning signal scanner.

Scoring (paper-trading phase, simplified):
  Only insider metrics are used — no macro/sector/fundamental layers.
  Signal qualifies if ALL of:
    1. >= MIN_BUYERS distinct insiders bought the same stock (within LOOKBACK_DAYS)
    2. At least one buyer is CEO or CFO (VD, Ekonomichef, etc.)
    3. ADV >= MIN_ADV_SEK (liquidity gate)

Pricing:
  Enters at today's OPEN price (post-09:00 Stockholm time).
  If run before market open, prints a warning and skips entry.

Run each morning at ~09:15 Stockholm time:
    python monitor/daily_report.py

Schedule via Windows Task Scheduler for fully automated operation.
"""

import logging
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yfinance as yf

from data.fi_scraper import DB_PATH as FI_DB, SENIOR_ROLES
from data.price_data import get_adv_sek
from backtest.universe import CandidateValidator, DB_PATH as UNIVERSE_DB
from execution.paper import PaperPortfolio, DB_PATH as PAPER_DB
from agents.analyst import confirm_signal
from agents.risk_gate import check_entry

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

# ── Parameters ────────────────────────────────────────────────────────────────
LOOKBACK_DAYS   = 3          # scan buys published within last N days
MIN_BUYERS         = 4          # minimum distinct insiders buying same stock
REQUIRE_EXEC       = True       # require at least one CEO or CFO among buyers
MIN_ADV_SEK        = 200_000    # liquidity gate: average daily volume in SEK
POSITION_SEK       = 5_000      # target SEK per paper position
MIN_PURCHASE_SEK   = 50_000     # minimum per-transaction value — filters stock grants/options at 0
MIN_PRICE_SEK      = 2.0        # minimum stock price — filters penny stocks with wide spreads
MAX_TRANSACTION_LAG = 7         # max days between transaction_date and report_date

STOCKHOLM_TZ    = ZoneInfo("Europe/Stockholm")
MARKET_OPEN     = datetime.strptime("09:00", "%H:%M").time()

# CEO/CFO role keywords (Swedish + English)
EXEC_ROLES = {
    "vd", "ceo", "verkställande direktör",
    "cfo", "finanschef", "ekonomichef",
    "chief financial officer", "chief executive officer",
}


# ── Insider signal check ──────────────────────────────────────────────────────

def _is_exec(role: str) -> bool:
    return any(kw in role.lower() for kw in EXEC_ROLES)


def _get_signals(since: date) -> list[dict]:
    """
    Query FI DB for stocks with cluster insider buying since `since`.
    Returns list of dicts with keys: isin, buyers, has_exec, roles, company.
    """
    conn = sqlite3.connect(FI_DB)
    # Sum per insider per ISIN so split transactions don't get filtered individually
    rows = conn.execute("""
        SELECT isin, company_name, insider_name, role,
               SUM(COALESCE(volume, 0) * COALESCE(unit_price, 0)) AS total_purchase_sek
        FROM insider_transactions
        WHERE DATE(report_date) >= ?
          AND (
              LOWER(transaction_type) LIKE '%förvärv%'
              OR LOWER(transaction_type) LIKE '%acquisition%'
          )
          AND instrument_type = 'Aktie'
          AND UPPER(isin) LIKE 'SE%'
          AND UPPER(isin) NOT IN ('', 'NONE', 'NAN', 'N/A')
          AND (
              transaction_date IS NULL
              OR julianday(report_date) - julianday(transaction_date) <= ?
          )
        GROUP BY isin, insider_name, role
        HAVING total_purchase_sek >= ?
    """, (since.isoformat(), MAX_TRANSACTION_LAG, MIN_PURCHASE_SEK)).fetchall()
    conn.close()

    # Group by ISIN
    by_isin: dict[str, dict] = {}
    for isin, company, name, role, purchase_value in rows:
        if isin not in by_isin:
            by_isin[isin] = {
                "isin": isin, "company": company or isin,
                "buyers": set(), "has_exec": False, "roles": [],
                "exec_roles": [], "total_purchase_sek": 0.0,
            }
        by_isin[isin]["buyers"].add(name)
        by_isin[isin]["roles"].append(role)
        by_isin[isin]["total_purchase_sek"] += purchase_value or 0.0
        if _is_exec(role or ""):
            by_isin[isin]["has_exec"] = True
            by_isin[isin]["exec_roles"].append(role)

    # Apply filters
    signals = []
    for s in by_isin.values():
        buyer_count = len(s["buyers"])
        if buyer_count < MIN_BUYERS:
            continue
        if REQUIRE_EXEC and not s["has_exec"]:
            continue
        signals.append({
            "isin":               s["isin"],
            "company":            s["company"],
            "buyers":             buyer_count,
            "has_exec":           s["has_exec"],
            "exec_roles":         list(set(s["exec_roles"])),
            "total_purchase_sek": s["total_purchase_sek"],
        })

    return sorted(signals, key=lambda x: -x["buyers"])


# ── Price helpers ─────────────────────────────────────────────────────────────

def _market_is_open() -> bool:
    """True if Stockholm market has already opened today."""
    now_sthlm = datetime.now(STOCKHOLM_TZ).time()
    return now_sthlm >= MARKET_OPEN


def _get_open_price(ticker: str) -> tuple[float | None, str]:
    """
    Returns (open_price, source_label).
    Uses today's open if available, else previous close with a warning label.
    """
    try:
        raw = yf.download(ticker, period="3d", progress=False, auto_adjust=True)
        if raw.empty:
            return None, "no_data"

        # Flatten MultiIndex if present (newer yfinance)
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw.xs(ticker, axis=1, level=1) if ticker in raw.columns.get_level_values(1) \
                  else raw.xs(ticker, axis=1, level=0)

        today = date.today()
        today_rows = raw[raw.index.date == today]

        if not today_rows.empty:
            open_px = float(today_rows["Open"].iloc[0])
            if open_px > 0:
                return open_px, "today_open"

        # Market hasn't opened yet or no today row — use previous close
        prev_close = float(raw["Close"].dropna().iloc[-1])
        return prev_close, "prev_close"
    except Exception as e:
        logger.debug("Price fetch failed for %s: %s", ticker, e)
        return None, "error"


# ── Main report ───────────────────────────────────────────────────────────────

def run_daily_report(broker=None) -> None:
    """
    broker: optional NordnetBroker instance for live execution.
            When None (default), uses PaperPortfolio.
    """
    today = date.today()
    since = today - timedelta(days=LOOKBACK_DAYS)
    now_sthlm = datetime.now(STOCKHOLM_TZ)

    print("=" * 64)
    print(f"DAZED Daily Signal Report  |  {today}  |  {now_sthlm.strftime('%H:%M')} CEST")
    print("=" * 64)

    market_open = _market_is_open()
    if not market_open:
        print(f"\n  WARNING: Market not yet open ({now_sthlm.strftime('%H:%M')} < 09:00).")
        print(f"  Prices will use previous close. Re-run after 09:00 for open prices.\n")

    # ── Step 1: Refresh FI data ───────────────────────────────
    print(f"Refreshing FI data (last 7 days)...")
    from data.fi_scraper import refresh as fi_refresh
    new_rows = fi_refresh(days_back=7)
    print(f"  {new_rows} new FI records\n")

    # ── Step 2: Find cluster signals ──────────────────────────
    signals = _get_signals(since)
    print(f"Insider clusters (>= {MIN_BUYERS} buyers, exec present, since {since}): {len(signals)}")

    if not signals:
        print("  No qualifying clusters — nothing to trade today.")
    else:
        validator  = CandidateValidator(db_path=UNIVERSE_DB)
        paper      = broker if broker is not None else PaperPortfolio()
        trade_list = []

        print(f"\n  {'Company':<28} {'Buyers':>6}  {'Exec?':>5}  {'Ticker':<14}  {'ADV SEK':>10}  Action")
        print("  " + "-" * 80)

        for sig in signals:
            isin               = sig["isin"]
            company            = sig["company"][:27]
            buyers             = sig["buyers"]
            has_exec           = sig["has_exec"]
            exec_roles         = sig["exec_roles"]
            total_purchase_sek = sig["total_purchase_sek"]

            val = validator.validate(isin, signal_date=today)
            if not val.valid or not val.ticker:
                print(f"  {company:<28} {buyers:>6}  {'YES' if has_exec else 'no':>5}  "
                      f"{'(no ticker)':<14}  {'—':>10}  skip")
                continue

            ticker = val.ticker
            adv = get_adv_sek(ticker, today)
            adv_str = f"{adv:,.0f}" if adv else "—"
            passes_liq = adv is not None and adv >= MIN_ADV_SEK

            action = "TRADE" if passes_liq else f"LOW LIQ (<{MIN_ADV_SEK:,})"

            print(f"  {company:<28} {buyers:>6}  {'YES' if has_exec else 'no':>5}  "
                  f"{ticker:<14}  {adv_str:>10}  {action}")

            if passes_liq:
                trade_list.append((isin, ticker, exec_roles, total_purchase_sek))

        # ── Step 3: LLM analyst + risk gate + enter trades ───────
        if trade_list:
            print(f"\n{'='*64}")
            print(f"EVALUATING {len(trade_list)} CANDIDATE(S)")
            print(f"{'='*64}")
            for isin, ticker, exec_roles, total_purchase_sek in trade_list:
                buyers_count = next(
                    s["buyers"] for s in signals if s["isin"] == isin
                )
                company_name = next(
                    s["company"] for s in signals if s["isin"] == isin
                )
                print(f"\n  {ticker} — {company_name}")

                # LLM analyst confirmation
                decision = confirm_signal(
                    ticker=ticker,
                    company=company_name,
                    buyers=buyers_count,
                    exec_roles=exec_roles,
                    total_purchase_sek=total_purchase_sek,
                    signal_date=today,
                )
                print(decision)
                if not decision.go:
                    print(f"  → Skipped by analyst")
                    continue

                # Risk gate
                risk = check_entry(ticker, PAPER_DB)
                print(risk)
                if not risk.approved:
                    print(f"  → Blocked by risk gate")
                    continue

                # Price check
                price, price_src = _get_open_price(ticker)
                if price is None:
                    print(f"  → Could not get price — skipped")
                    continue
                if price_src == "prev_close":
                    print(f"  → Using previous close ({price:.2f}) — re-run after 09:00 for open price")
                if price < MIN_PRICE_SEK:
                    print(f"  → Price {price:.2f} SEK below minimum {MIN_PRICE_SEK:.2f} — skipped")
                    continue

                shares = max(1, int(POSITION_SEK / price))
                paper.enter(isin, ticker, today, price, shares,
                            score=decision.confidence)

    # ── Step 4: Portfolio status ──────────────────────────────
    portfolio = broker if broker is not None else PaperPortfolio()
    print(f"\n{'='*64}")
    if hasattr(portfolio, "update_prices"):
        portfolio.update_prices()
    portfolio.report()
    print(f"\nDone. Run again tomorrow at 09:15.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dazed daily signal scanner")
    parser.add_argument(
        "--live", action="store_true",
        help="Submit real orders via NordnetBroker instead of paper-trading. "
             "Requires NORDNET_USERNAME / NORDNET_PASSWORD / NORDNET_ACCOUNT env vars. "
             "Passes dry_run=True unless --confirm-live is also set.",
    )
    parser.add_argument(
        "--confirm-live", action="store_true",
        help="Combined with --live: disables dry-run and submits real orders.",
    )
    args = parser.parse_args()

    if args.live:
        from execution.nordnet import NordnetBroker
        dry = not args.confirm_live
        broker = NordnetBroker(dry_run=dry)
        run_daily_report(broker=broker)
    else:
        run_daily_report()
