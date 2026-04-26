"""
Conviction scorer for the Nordic Insider Clustering strategy.

Scores a candidate (ISIN + date) across four layers (0–100 pts total),
then subtracts risk deduction items. A trade fires only if:
  1. No hard veto is active
  2. Final score >= min_conviction_score (default 65)

All thresholds come from config/strategy.yaml and are FIXED — never tune
these to improve backtest results.
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import yaml

from data.fi_scraper import InsiderTransaction, query_buys, query_sells
from data.fundamental_data import FundamentalSnapshot, fetch as fetch_fundamentals
from data.price_data import get_adv_sek, resolve_ticker

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "strategy.yaml"


def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@dataclass
class ScoreBreakdown:
    isin: str
    as_of_date: date
    ticker: Optional[str]

    # Layer scores (before deductions)
    macro_score: int = 0
    sector_score: int = 0
    fundamental_score: int = 0
    insider_score: int = 0

    # Risk deductions (negative values)
    earnings_deduction: int = 0
    unlock_deduction: int = 0
    rights_issue_deduction: int = 0

    # Hard veto flags
    veto_active: bool = False
    veto_reasons: list[str] = field(default_factory=list)

    # Data quality flag
    data_incomplete: bool = False

    # Supporting detail
    distinct_buyers: list[str] = field(default_factory=list)
    buy_transactions: list[InsiderTransaction] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def raw_score(self) -> int:
        return self.macro_score + self.sector_score + self.fundamental_score + self.insider_score

    @property
    def total_deductions(self) -> int:
        return self.earnings_deduction + self.unlock_deduction + self.rights_issue_deduction

    @property
    def final_score(self) -> int:
        return max(0, self.raw_score + self.total_deductions)

    @property
    def tradeable(self) -> bool:
        return not self.veto_active and self.final_score >= 65

    def summary(self) -> str:
        status = "TRADE" if self.tradeable else ("VETO" if self.veto_active else "BELOW_THRESHOLD")
        return (
            f"{self.isin} [{self.as_of_date}] {status} | "
            f"score={self.final_score} (macro={self.macro_score} sector={self.sector_score} "
            f"fund={self.fundamental_score} insider={self.insider_score} "
            f"deductions={self.total_deductions}) | "
            f"buyers={len(self.distinct_buyers)}"
        )


class ConvictionScorer:
    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or _load_cfg()

    # ── Layer 0: Macro ───────────────────────────────────────────────────────

    def _score_macro(self, breakdown: ScoreBreakdown) -> None:
        """
        Macro environment check. Uses market breadth proxy.
        Full implementation in signals/macro_filter.py — this is the integration point.
        Defaults to a neutral 8/15 when live breadth data is unavailable.
        """
        try:
            from signals.macro_filter import get_breadth_score
            score, notes = get_breadth_score(breakdown.as_of_date, self.cfg)
            breakdown.macro_score = score
            breakdown.notes.extend(notes)
        except ImportError:
            breakdown.macro_score = 8
            breakdown.notes.append("macro_filter not yet available — neutral score applied")
        except Exception as e:
            breakdown.macro_score = 5
            breakdown.notes.append(f"macro_filter error ({e}) — conservative score applied")

    # ── Layer 1: Sector ──────────────────────────────────────────────────────

    def _score_sector(self, breakdown: ScoreBreakdown, isin: str) -> None:
        """
        Sector health check. Full implementation in signals/sector_filter.py.
        """
        try:
            from signals.sector_filter import get_sector_score
            score, veto, notes = get_sector_score(isin, breakdown.as_of_date, self.cfg)
            breakdown.sector_score = score
            if veto:
                breakdown.veto_active = True
                breakdown.veto_reasons.append("sector_systemic_risk")
            breakdown.notes.extend(notes)
        except ImportError:
            breakdown.sector_score = 10
            breakdown.notes.append("sector_filter not yet available — neutral score applied")
        except Exception as e:
            breakdown.sector_score = 5
            breakdown.notes.append(f"sector_filter error ({e}) — conservative score applied")

    # ── Layer 2: Fundamentals ────────────────────────────────────────────────

    def _score_fundamentals(
        self,
        breakdown: ScoreBreakdown,
        snap: FundamentalSnapshot,
    ) -> None:
        cfg_f = self.cfg.get("fundamentals", {})
        score = 0
        max_de = cfg_f.get("max_debt_equity", 3.0)
        min_runway = cfg_f.get("min_cash_runway_months", 12)

        # Hard veto
        if snap.has_fraud_investigation:
            breakdown.veto_active = True
            breakdown.veto_reasons.append("fraud_investigation")
            return

        if not snap.data_complete:
            breakdown.data_incomplete = True
            breakdown.notes.append("fundamental data incomplete — conservative score")
            breakdown.fundamental_score = 10  # conservative, not zero (data gap ≠ bad company)
            return

        # Revenue trend: +10
        if snap.revenue_trend_ok is True:
            score += 10
        elif snap.revenue_trend_ok is None:
            score += 5
            breakdown.notes.append("revenue trend unknown")

        # Debt/equity: +10
        if snap.debt_equity_ok is True:
            score += 10
        elif snap.debt_equity is not None and snap.debt_equity > max_de:
            breakdown.notes.append(f"high D/E={snap.debt_equity:.1f}")

        # Cash runway: +10
        runway = snap.cash_runway_months
        if runway is not None:
            if runway >= min_runway:
                score += 10
            else:
                breakdown.notes.append(f"short cash runway={runway:.0f}mo")
        else:
            score += 5
            breakdown.notes.append("cash runway unknown")

        # Data reliability bonus: +5
        if snap.data_complete:
            score += 5

        breakdown.fundamental_score = min(score, 35)

    # ── Layer 3: Insider Signal ──────────────────────────────────────────────

    def _score_insider(
        self,
        breakdown: ScoreBreakdown,
        buys: list[InsiderTransaction],
        sells: list[InsiderTransaction],
    ) -> None:
        cfg_i = self.cfg.get("insider", {})
        score = 0

        # Deduplicate by insider name (different people only)
        distinct = {}
        for tx in buys:
            name = tx.insider_name.lower().strip()
            if name not in distinct or tx.report_date > distinct[name].report_date:
                distinct[name] = tx
        distinct_buys = list(distinct.values())

        breakdown.distinct_buyers = list(distinct.keys())
        breakdown.buy_transactions = distinct_buys

        min_insiders = self.cfg.get("signal", {}).get("min_distinct_insiders", 2)
        if len(distinct_buys) < min_insiders:
            breakdown.notes.append(f"only {len(distinct_buys)} distinct buyer(s) — need {min_insiders}")
            breakdown.insider_score = 0
            return

        # Base: 2+ distinct insiders → +10
        score += 10

        # CEO/CFO present: +10
        has_senior = any(tx.is_senior for tx in distinct_buys)
        ceo_cfo_roles = {"vd", "ceo", "cfo", "finanschef", "ekonomichef", "verkställande"}
        has_ceo_cfo = any(
            any(r in tx.role_normalized for r in ceo_cfo_roles)
            for tx in distinct_buys
        )
        if has_ceo_cfo:
            score += cfg_i.get("ceo_cfo_bonus_pts", 10)
            breakdown.notes.append("CEO/CFO present")
        elif has_senior:
            score += cfg_i.get("board_member_pts", 5)

        # Purchase significance vs. salary estimate
        # We need a FundamentalSnapshot for salary estimates — use role-based defaults
        min_salary_pct = cfg_i.get("min_salary_pct", 10)
        salary_pts = cfg_i.get("salary_significance_pts", 5)
        for tx in distinct_buys:
            estimated_salary = _estimate_salary_sek(tx.role)
            threshold = estimated_salary * (min_salary_pct / 100.0)
            if tx.total_value_sek >= threshold:
                score += salary_pts
                breakdown.notes.append(
                    f"{tx.insider_name}: {tx.total_value_sek:,.0f} SEK "
                    f"(>{min_salary_pct}% est. salary)"
                )
                break  # Only award once

        # Prior selling penalty
        sell_window = cfg_i.get("no_prior_selling_window_days", 180)
        sell_penalty = cfg_i.get("prior_selling_penalty_pts", 5)
        selling_names = {tx.insider_name.lower().strip() for tx in sells}
        buying_names = set(distinct.keys())
        overlap = buying_names & selling_names
        if overlap:
            score -= sell_penalty
            breakdown.notes.append(f"prior selling detected: {overlap}")

        breakdown.insider_score = max(0, min(score, 30))

    # ── Risk Deductions ──────────────────────────────────────────────────────

    def _apply_risk_deductions(
        self,
        breakdown: ScoreBreakdown,
        earnings_days_out: Optional[int],
        unlock_days_out: Optional[int],
        rights_issue_days_out: Optional[int],
    ) -> None:
        cfg_r = self.cfg.get("risk_deductions", {})

        if earnings_days_out is not None:
            threshold = cfg_r.get("earnings_within_days", 5)
            if 0 <= earnings_days_out <= threshold:
                pts = cfg_r.get("earnings_deduction_pts", 20)
                breakdown.earnings_deduction = -pts
                breakdown.notes.append(
                    f"earnings in {earnings_days_out}d — deducting {pts}pts"
                )

        if unlock_days_out is not None:
            threshold = cfg_r.get("unlock_within_days", 10)
            if 0 <= unlock_days_out <= threshold:
                pts = cfg_r.get("unlock_deduction_pts", 15)
                breakdown.unlock_deduction = -pts
                breakdown.notes.append(
                    f"share unlock in {unlock_days_out}d — deducting {pts}pts"
                )

        if rights_issue_days_out is not None:
            threshold = cfg_r.get("rights_issue_within_days", 20)
            if 0 <= rights_issue_days_out <= threshold:
                pts = cfg_r.get("rights_issue_deduction_pts", 10)
                breakdown.rights_issue_deduction = -pts
                breakdown.notes.append(
                    f"rights issue in {rights_issue_days_out}d — deducting {pts}pts"
                )

    # ── Entry gate ───────────────────────────────────────────────────────────

    def _check_min_conviction(self, breakdown: ScoreBreakdown) -> None:
        min_score = self.cfg.get("entry", {}).get("min_conviction_score", 65)
        if not breakdown.veto_active and breakdown.final_score < min_score:
            breakdown.notes.append(
                f"final score {breakdown.final_score} < threshold {min_score}"
            )

    # ── Main entry point ─────────────────────────────────────────────────────

    def score(
        self,
        isin: str,
        as_of_date: date,
        fundamental_source: str = "yahoo_finance",
        earnings_days_out: Optional[int] = None,
        unlock_days_out: Optional[int] = None,
        rights_issue_days_out: Optional[int] = None,
    ) -> ScoreBreakdown:
        """
        Score a single ISIN as of a given date.

        earnings_days_out: days until next earnings release (None if unknown)
        unlock_days_out: days until next share unlock/lock-up expiry (None if unknown)
        rights_issue_days_out: days until rights issue (None if unknown)
        """
        ticker = resolve_ticker(isin)
        breakdown = ScoreBreakdown(isin=isin, as_of_date=as_of_date, ticker=ticker)

        cfg_s = self.cfg.get("signal", {})
        window = cfg_s.get("insider_window_days", 30)
        sell_window = self.cfg.get("insider", {}).get("no_prior_selling_window_days", 180)

        # ── Fetch inputs ─────────────────────────────────────────────────────
        buys = query_buys(isin, as_of_date, window)
        sells = query_sells(isin, as_of_date, sell_window)

        snap = None
        if ticker:
            snap = fetch_fundamentals(ticker, as_of_date, fundamental_source)

        # ── Score each layer ─────────────────────────────────────────────────
        self._score_macro(breakdown)
        if breakdown.veto_active:
            return breakdown

        self._score_sector(breakdown, isin)
        if breakdown.veto_active:
            return breakdown

        if snap:
            self._score_fundamentals(breakdown, snap)
        else:
            breakdown.fundamental_score = 5
            breakdown.notes.append("no ticker mapping — fundamentals skipped")
        if breakdown.veto_active:
            return breakdown

        self._score_insider(breakdown, buys, sells)

        # ── Risk deductions ──────────────────────────────────────────────────
        self._apply_risk_deductions(
            breakdown, earnings_days_out, unlock_days_out, rights_issue_days_out
        )

        # ── Final gate check ─────────────────────────────────────────────────
        self._check_min_conviction(breakdown)

        logger.info(breakdown.summary())
        return breakdown


# ── Helpers ──────────────────────────────────────────────────────────────────

def _estimate_salary_sek(role: str) -> float:
    role_lower = role.lower()
    if any(r in role_lower for r in ["vd", "ceo", "verkställande"]):
        return 3_000_000
    if any(r in role_lower for r in ["cfo", "finanschef", "ekonomichef"]):
        return 2_500_000
    if "ordförande" in role_lower or "chairman" in role_lower:
        return 1_500_000
    if "styrelseledamot" in role_lower or "board" in role_lower:
        return 500_000
    return 1_000_000


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    scorer = ConvictionScorer()
    result = scorer.score(
        isin="SE0000108656",   # Ericsson — just for smoke test
        as_of_date=date.today(),
        earnings_days_out=3,   # simulate upcoming earnings
    )
    print(result.summary())
    print("Notes:", result.notes)
