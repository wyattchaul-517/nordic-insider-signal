"""
agents/analyst.py — LLM confirmation layer for insider cluster signals.

Architecture:
  The insider cluster is the TRIGGER (hard rule-based, unchanged).
  This analyst is the GATE — a quick sanity check before capital is deployed.

  Flow:
    cluster passes all hard filters
        → fetch_company_context() pulls yfinance .info
        → confirm_signal() sends one LLM call
        → returns go/no-go + reason + red flags logged to console

Provider selection (ANALYST_PROVIDER env var):
  anthropic  — Claude Haiku (paid, ~$0.001/call). Default if key present.
  groq       — Llama 3.3 70B via Groq cloud (FREE, needs GROQ_API_KEY).
               Sign up: console.groq.com — no credit card required.
  ollama     — Any local model (FREE, offline). Needs Ollama running locally.
               Install: ollama.ai → `ollama pull llama3.2`

Cost: ~$0.001 per call (Anthropic Haiku) or free (Groq/Ollama).

Failure mode: if the API call fails or JSON parsing fails, defaults to go=True
              so a connectivity issue never silently blocks a good signal.
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import yfinance as yf

logger = logging.getLogger(__name__)

# Default models per provider
DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "groq":      "llama-3.3-70b-versatile",
    "ollama":    "llama3.2",
}


@dataclass
class AnalystDecision:
    go:         bool
    confidence: int          # 0–100
    reason:     str
    red_flags:  list[str]
    model_used: str
    skipped:    bool = False  # True if analyst was bypassed (no API key, error, etc.)

    def __str__(self) -> str:
        verdict = "GO" if self.go else "NO-GO"
        flags = f"  flags: {self.red_flags}" if self.red_flags else ""
        skip = " [skipped — defaulted to GO]" if self.skipped else ""
        return f"  ANALYST {verdict} ({self.confidence}%): {self.reason}{flags}{skip}"


def _fetch_company_context(ticker: str) -> dict:
    """
    Pull key fundamentals from yfinance for the LLM prompt.
    Returns an empty dict on failure — the analyst will note the gap.
    """
    try:
        info = yf.Ticker(ticker).info
        earnings_ts = info.get("earningsTimestamp") or info.get("earningsDate")
        earnings_str = "unknown"
        if earnings_ts:
            try:
                from datetime import datetime
                earnings_str = datetime.fromtimestamp(int(earnings_ts)).strftime("%Y-%m-%d")
            except Exception:
                earnings_str = str(earnings_ts)

        return {
            "sector":          info.get("sector", "unknown"),
            "industry":        info.get("industry", "unknown"),
            "market_cap":      info.get("marketCap"),
            "pe_ratio":        info.get("trailingPE"),
            "debt_equity":     info.get("debtToEquity"),
            "revenue_growth":  info.get("revenueGrowth"),
            "current_price":   info.get("currentPrice") or info.get("regularMarketPrice"),
            "week52_high":     info.get("fiftyTwoWeekHigh"),
            "week52_low":      info.get("fiftyTwoWeekLow"),
            "next_earnings":   earnings_str,
            "short_ratio":     info.get("shortRatio"),
            "description":     (info.get("longBusinessSummary") or "")[:300],
        }
    except Exception as e:
        logger.debug("yfinance context fetch failed for %s: %s", ticker, e)
        return {}


def _build_prompt(
    ticker: str,
    company: str,
    buyers: int,
    exec_roles: list[str],
    total_purchase_sek: float,
    signal_date: date,
    ctx: dict,
) -> str:
    def fmt(val, suffix="", pct=False):
        if val is None:
            return "n/a"
        if pct:
            return f"{val*100:.1f}%"
        return f"{val:,.2f}{suffix}"

    days_to_earnings = "unknown"
    try:
        if ctx.get("next_earnings") and ctx["next_earnings"] != "unknown":
            from datetime import datetime
            ed = datetime.strptime(ctx["next_earnings"], "%Y-%m-%d").date()
            days_to_earnings = str((ed - signal_date).days)
    except Exception:
        pass

    return f"""You are a risk analyst reviewing a Swedish insider trading signal.
Your job: identify red flags that would make this a BAD trade, or confirm it looks clean.

== COMPANY ==
Name: {company} ({ticker})
Sector: {ctx.get('sector', 'n/a')} | Industry: {ctx.get('industry', 'n/a')}
Market cap: {fmt(ctx.get('market_cap'))} SEK
P/E ratio: {fmt(ctx.get('pe_ratio'))}
Debt/Equity: {fmt(ctx.get('debt_equity'))}
Revenue growth YoY: {fmt(ctx.get('revenue_growth'), pct=True)}
Current price: {fmt(ctx.get('current_price'))} SEK
52-week range: {fmt(ctx.get('week52_low'))} – {fmt(ctx.get('week52_high'))} SEK
Days to next earnings: {days_to_earnings}
Short interest ratio: {fmt(ctx.get('short_ratio'))}
Business: {ctx.get('description', 'n/a')}

== INSIDER CLUSTER SIGNAL ==
Date: {signal_date}
Distinct buyers: {buyers}
Exec roles present: {', '.join(exec_roles) if exec_roles else 'none identified'}
Total SEK purchased: {total_purchase_sek:,.0f}

== DECISION RULES ==
Flag and REJECT if:
- Earnings within 5 days (front-running risk, not sustained conviction)
- Debt/Equity > 3.0 (overleveraged — insider buying may be stabilising optics)
- Revenue declining > 20% YoY (structural deterioration)
- Short interest ratio > 10 (heavily shorted — insiders may be wrong)
- Price within 3% of 52-week high with only 4 buyers (possible distribution)

CONFIRM if the signal is clean and no major flags exist.

Respond with ONLY valid JSON, no other text:
{{"go": true, "confidence": 75, "reason": "one concise sentence", "red_flags": []}}"""


def _call_anthropic(prompt: str, model: str, api_key: str) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def _call_openai_compat(prompt: str, model: str, api_key: str, base_url: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


def _parse_response(raw: str) -> dict:
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def confirm_signal(
    ticker:             str,
    company:            str,
    buyers:             int,
    exec_roles:         list[str],
    total_purchase_sek: float,
    signal_date:        date,
    model:              str = None,
) -> AnalystDecision:
    """
    Ask an LLM whether the insider cluster signal is worth trading.

    Provider is selected via ANALYST_PROVIDER env var (default: anthropic).
    Model is selected via ANALYST_MODEL env var, or provider default.

    Returns AnalystDecision with go=True to enter, go=False to skip.
    On any error, defaults to go=True so connectivity issues don't silently block trades.
    """
    provider = os.environ.get("ANALYST_PROVIDER", "anthropic").lower()

    if model is None:
        model = os.environ.get("ANALYST_MODEL", DEFAULT_MODELS.get(provider, "llama-3.3-70b-versatile"))

    ctx = _fetch_company_context(ticker)
    prompt = _build_prompt(ticker, company, buyers, exec_roles, total_purchase_sek, signal_date, ctx)

    raw = ""
    try:
        if provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY not set — analyst skipped, defaulting to GO")
                return AnalystDecision(
                    go=True, confidence=50, reason="API key not set",
                    red_flags=[], model_used=model, skipped=True,
                )
            raw = _call_anthropic(prompt, model, api_key)

        elif provider == "groq":
            api_key = os.environ.get("GROQ_API_KEY", "")
            if not api_key:
                logger.warning("GROQ_API_KEY not set — analyst skipped, defaulting to GO")
                return AnalystDecision(
                    go=True, confidence=50, reason="GROQ_API_KEY not set",
                    red_flags=[], model_used=model, skipped=True,
                )
            raw = _call_openai_compat(prompt, model, api_key, "https://api.groq.com/openai/v1")

        elif provider == "ollama":
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            raw = _call_openai_compat(prompt, model, "ollama", base_url)

        else:
            logger.warning("Unknown ANALYST_PROVIDER=%s — defaulting to GO", provider)
            return AnalystDecision(
                go=True, confidence=50, reason=f"unknown provider: {provider}",
                red_flags=[], model_used=model, skipped=True,
            )

        data = _parse_response(raw)
        return AnalystDecision(
            go=bool(data.get("go", True)),
            confidence=int(data.get("confidence", 50)),
            reason=str(data.get("reason", "")),
            red_flags=list(data.get("red_flags", [])),
            model_used=f"{provider}/{model}",
        )

    except json.JSONDecodeError as e:
        logger.warning("Analyst JSON parse error: %s — raw: %s", e, raw[:200])
        return AnalystDecision(
            go=True, confidence=50, reason=f"parse error: {e}",
            red_flags=[], model_used=f"{provider}/{model}", skipped=True,
        )
    except Exception as e:
        logger.warning("Analyst API call failed: %s — defaulting to GO", e)
        return AnalystDecision(
            go=True, confidence=50, reason=f"API error: {e}",
            red_flags=[], model_used=f"{provider}/{model}", skipped=True,
        )
