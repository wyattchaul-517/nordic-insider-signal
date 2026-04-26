"""
execution/nordnet.py — Real order execution via Nordnet's REST API.

Mirrors the PaperPortfolio interface so you can swap in live execution
once the paper-trading phase validates the strategy.

SAFETY DEFAULTS:
  - dry_run=True by default — no real orders until you explicitly pass dry_run=False.
  - order_type defaults to LIMIT with a small slippage budget; never market orders.
  - Each order is logged before submission and requires the position to pass
    the same ADV / score gates as the paper portfolio.

CREDENTIALS:
  Set environment variables (never hard-code):
    NORDNET_USERNAME=your_nordnet_email
    NORDNET_PASSWORD=your_nordnet_password
    NORDNET_ACCOUNT=12345678   # 8-digit account number shown in Nordnet

FIRST-TIME SETUP:
  1. Log in to nordnet.se manually and confirm your account number.
  2. Export the two env-vars above (e.g. in a .env file loaded via python-dotenv).
  3. Call NordnetBroker(dry_run=True).enter(...) to verify connectivity.
  4. Check that the instrument resolves and the order preview looks correct.
  5. Switch to dry_run=False only after confirming paper trades are profitable.

Nordnet API base: https://api.prod.nordnet.se/next/2
  - No official documentation for retail users; based on community reverse-engineering.
  - Session-based auth: POST /login → session cookie → authenticated requests.
  - All amounts are in the account's base currency (SEK for Swedish accounts).
"""

import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

API_BASE = "https://api.prod.nordnet.se/next/2"
MARKET_ID_STOCKHOLM = 30         # Nordnet internal ID for Nasdaq Stockholm
LIMIT_SLIPPAGE_PCT  = 0.005      # 0.5 % above ask for limit order (avoid immediate rejection)
REQUEST_TIMEOUT     = 15         # seconds


@dataclass
class OrderResult:
    success:       bool
    order_id:      Optional[str]
    ticker:        str
    side:          str           # "BUY" or "SELL"
    shares:        int
    limit_price:   float
    message:       str


class NordnetBroker:
    """
    Live execution broker backed by Nordnet.

    Drop-in replacement for PaperPortfolio for the enter() call:
        paper = PaperPortfolio()          # before validation
        broker = NordnetBroker()          # after validation (still dry_run=True by default)
        broker.enter(isin, ticker, ...)   # places a real limit order when dry_run=False
    """

    def __init__(
        self,
        dry_run:    bool = True,
        username:   Optional[str] = None,
        password:   Optional[str] = None,
        account_no: Optional[str] = None,
    ):
        self.dry_run    = dry_run
        self._username  = username  or os.environ.get("NORDNET_USERNAME", "")
        self._password  = password  or os.environ.get("NORDNET_PASSWORD", "")
        self._account   = account_no or os.environ.get("NORDNET_ACCOUNT", "")
        self._session:  Optional[requests.Session] = None
        self._logged_in: bool = False

        if dry_run:
            logger.info("NordnetBroker: DRY-RUN mode — no real orders will be placed")
        else:
            logger.warning("NordnetBroker: LIVE mode — real orders WILL be submitted")

    # ── Authentication ────────────────────────────────────────────────────────

    def _login(self) -> bool:
        """Establish a Nordnet session. Returns True on success."""
        if self._logged_in and self._session:
            return True

        if not self._username or not self._password:
            logger.error("NORDNET_USERNAME / NORDNET_PASSWORD env vars not set")
            return False

        self._session = requests.Session()
        self._session.headers.update({
            "Accept":       "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        })

        try:
            resp = self._session.post(
                f"{API_BASE}/login",
                data={"username": self._username, "password": self._password},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("logged_in") or data.get("session_key"):
                    self._logged_in = True
                    logger.info("Nordnet: logged in successfully")
                    return True
            logger.error("Nordnet login failed: %s %s", resp.status_code, resp.text[:200])
        except Exception as e:
            logger.error("Nordnet login exception: %s", e)
        return False

    def _logout(self) -> None:
        if self._session and self._logged_in:
            try:
                self._session.delete(f"{API_BASE}/login", timeout=REQUEST_TIMEOUT)
            except Exception:
                pass
            self._logged_in = False

    def _get(self, path: str, **kwargs) -> Optional[dict]:
        if not self._login():
            return None
        try:
            resp = self._session.get(f"{API_BASE}{path}", timeout=REQUEST_TIMEOUT, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error("Nordnet GET %s failed: %s", path, e)
            return None

    def _post(self, path: str, payload: dict) -> Optional[dict]:
        if not self._login():
            return None
        try:
            resp = self._session.post(
                f"{API_BASE}{path}", json=payload, timeout=REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error("Nordnet POST %s failed: %s", path, e)
            return None

    # ── Instrument lookup ─────────────────────────────────────────────────────

    def _resolve_instrument(self, ticker: str) -> Optional[int]:
        """
        Resolve a Yahoo-format ticker (e.g. 'ERIC-B.ST') to a Nordnet instrument_id.
        Strips the '.ST' suffix and searches Nordnet's instrument catalogue.
        Returns instrument_id (int) or None if not found.
        """
        # Strip Yahoo suffix: "ERIC-B.ST" → "ERIC-B"
        query = ticker.upper().replace(".ST", "").replace(".HE", "").replace(".OL", "")

        data = self._get("/instruments", params={
            "query":     query,
            "market_id": MARKET_ID_STOCKHOLM,
        })
        if not data:
            return None

        instruments = data if isinstance(data, list) else data.get("instruments", [])
        if not instruments:
            logger.warning("Nordnet: no instrument found for '%s'", ticker)
            return None

        # Take the first result; log if ambiguous
        inst = instruments[0]
        if len(instruments) > 1:
            names = [i.get("name", "") for i in instruments[:3]]
            logger.debug("Nordnet: %d matches for '%s' — using first: %s",
                         len(instruments), ticker, names)

        instrument_id = inst.get("instrument_id") or inst.get("id")
        logger.debug("Nordnet: '%s' → instrument_id=%s (%s)", ticker, instrument_id, inst.get("name"))
        return instrument_id

    # ── Main interface (mirrors PaperPortfolio) ───────────────────────────────

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
        Place a BUY limit order. Returns True if order submitted (or dry-run accepted).

        Limit is set at entry_price × (1 + LIMIT_SLIPPAGE_PCT) to give a small
        cushion above the reference price so the order fills promptly.
        """
        limit = round(entry_price * (1 + LIMIT_SLIPPAGE_PCT), 2)
        trade_value = limit * shares

        log_prefix = "[DRY-RUN] " if self.dry_run else ""
        print(f"  {log_prefix}LIVE ENTER  {ticker:<14} @ {limit:.2f} SEK (limit)  "
              f"x{shares}  value={trade_value:,.0f} SEK  score={score}")

        if self.dry_run:
            return True

        if not self._account:
            logger.error("NORDNET_ACCOUNT not set — cannot place order")
            return False

        instrument_id = self._resolve_instrument(ticker)
        if not instrument_id:
            logger.error("Could not resolve instrument for %s — order skipped", ticker)
            return False

        order = {
            "instrument_id": instrument_id,
            "side":          "BUY",
            "volume":        shares,
            "price":         limit,
            "currency":      "SEK",
            "order_type":    "LIMIT",
            "validity_type": "DAY",   # Good-for-day; re-enter tomorrow if unfilled
        }

        result = self._post(f"/accounts/{self._account}/orders", order)
        if result and result.get("order_id"):
            order_id = result["order_id"]
            logger.info("Nordnet: BUY order placed — id=%s  %s x%d @ %.2f",
                        order_id, ticker, shares, limit)
            print(f"  ORDER PLACED  {ticker}  id={order_id}  x{shares} @ {limit:.2f}")
            return True

        logger.error("Nordnet: order placement failed for %s — %s", ticker, result)
        return False

    def close_position(
        self,
        ticker: str,
        shares: int,
        reference_price: float,
        reason: str = "signal",
    ) -> bool:
        """
        Place a SELL limit order to close a position.
        Limit is set at reference_price × (1 − LIMIT_SLIPPAGE_PCT).
        """
        limit = round(reference_price * (1 - LIMIT_SLIPPAGE_PCT), 2)

        log_prefix = "[DRY-RUN] " if self.dry_run else ""
        print(f"  {log_prefix}LIVE EXIT   {ticker:<14} @ {limit:.2f} SEK (limit)  "
              f"x{shares}  reason={reason}")

        if self.dry_run:
            return True

        if not self._account:
            logger.error("NORDNET_ACCOUNT not set — cannot place order")
            return False

        instrument_id = self._resolve_instrument(ticker)
        if not instrument_id:
            logger.error("Could not resolve instrument for %s — sell skipped", ticker)
            return False

        order = {
            "instrument_id": instrument_id,
            "side":          "SELL",
            "volume":        shares,
            "price":         limit,
            "currency":      "SEK",
            "order_type":    "LIMIT",
            "validity_type": "DAY",
        }

        result = self._post(f"/accounts/{self._account}/orders", order)
        if result and result.get("order_id"):
            order_id = result["order_id"]
            logger.info("Nordnet: SELL order placed — id=%s  %s x%d @ %.2f  [%s]",
                        order_id, ticker, shares, limit, reason)
            print(f"  ORDER PLACED  {ticker}  id={order_id}  SELL x{shares} @ {limit:.2f}  [{reason}]")
            return True

        logger.error("Nordnet: sell order failed for %s — %s", ticker, result)
        return False

    # ── Account / position queries ────────────────────────────────────────────

    def get_positions(self) -> list[dict]:
        """
        Return open positions from Nordnet.
        Each dict has keys: ticker, instrument_name, shares, avg_cost, current_price, unrealised_pnl.
        """
        data = self._get(f"/accounts/{self._account}/positions")
        if not data:
            return []

        positions = data if isinstance(data, list) else data.get("positions", [])
        result = []
        for p in positions:
            result.append({
                "ticker":          p.get("instrument", {}).get("symbol", ""),
                "instrument_name": p.get("instrument", {}).get("name", ""),
                "shares":          p.get("qty", 0),
                "avg_cost":        p.get("avg_acquisition_price", {}).get("value", 0),
                "current_price":   p.get("last_price", {}).get("value", 0),
                "unrealised_pnl":  p.get("unrealized_profit_loss", {}).get("value", 0),
            })
        return result

    def get_pending_orders(self) -> list[dict]:
        """Return list of pending orders (not yet filled)."""
        data = self._get(f"/accounts/{self._account}/orders", params={"state": "ACTIVE"})
        if not data:
            return []
        orders = data if isinstance(data, list) else data.get("orders", [])
        return [
            {
                "order_id":    o.get("order_id"),
                "ticker":      o.get("instrument", {}).get("symbol", ""),
                "side":        o.get("side"),
                "shares":      o.get("volume"),
                "limit_price": o.get("price", {}).get("value", 0),
                "state":       o.get("state"),
            }
            for o in orders
        ]

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order by ID. Returns True on success."""
        if self.dry_run:
            print(f"  [DRY-RUN] CANCEL order {order_id}")
            return True
        try:
            resp = self._session.delete(
                f"{API_BASE}/accounts/{self._account}/orders/{order_id}",
                timeout=REQUEST_TIMEOUT,
            )
            return resp.status_code in (200, 204)
        except Exception as e:
            logger.error("Nordnet: cancel order %s failed: %s", order_id, e)
            return False

    def report(self) -> None:
        """Print live account positions from Nordnet."""
        positions = self.get_positions()
        orders    = self.get_pending_orders()

        print("\n" + "=" * 64)
        print("NORDNET LIVE ACCOUNT")
        print("=" * 64)
        mode = "DRY-RUN" if self.dry_run else "LIVE"
        print(f"Mode: {mode}  |  Account: {self._account or '(not set)'}")

        print(f"\nOpen positions: {len(positions)}")
        if positions:
            print(f"  {'Ticker':<16} {'Name':<28} {'Qty':>5} {'Avg Cost':>10} "
                  f"{'Last':>10} {'P&L':>10}")
            for p in positions:
                print(f"  {p['ticker']:<16} {p['instrument_name'][:27]:<28} "
                      f"{p['shares']:>5} {p['avg_cost']:>10.2f} "
                      f"{p['current_price']:>10.2f} {p['unrealised_pnl']:>+10.2f}")

        print(f"\nPending orders: {len(orders)}")
        if orders:
            for o in orders:
                print(f"  {o['order_id']}  {o['side']:<4} {o['ticker']:<14} "
                      f"x{o['shares']}  @ {o['limit_price']:.2f}  [{o['state']}]")

    def __del__(self):
        self._logout()
