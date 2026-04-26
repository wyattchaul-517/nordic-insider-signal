# Nordic Insider Signal

Event-driven quantitative trading system that detects and trades clusters of insider buying on Swedish equities listed on Nasdaq Stockholm / First North.

## Strategy overview

Insider cluster detection is the **trigger** — a configurable minimum number of distinct insiders buying the same stock within a short lookback window. An LLM confirmation layer (Groq Llama / Anthropic Haiku / local Ollama) acts as a lightweight sanity gate before capital is deployed.

Signal scoring is split across four independent layers (0–100 pts), with hard vetoes for earnings windows, lock-up expirations, and recent rights issues:

| Layer | What it measures |
|---|---|
| Insider | Cluster size, role seniority, net SEK bought vs. recent sells |
| Fundamental | Revenue trend, profitability, balance sheet quality |
| Macro | SEK/USD momentum, OMX regime filter |
| Sector | Relative sector strength on Nasdaq Stockholm |

Minimum conviction score of 65 required to fire a trade. All thresholds are fixed in `config/strategy.yaml` and never adjusted post-backtest.

## Risk management

- **Half-Kelly sizing** — position value capped at `min(max_portfolio_pct, max_pct_of_ADV)`, then scaled by conviction tier × 0.5 Kelly fraction
- **Liquidity cap** — hard ceiling at 2% of 20-day average daily volume (SEK) to avoid market impact on small-cap names
- **Exits** — configurable trailing stop, time-based stop, and profit target; evaluated nightly

## Backtesting

- Event-driven walk-forward engine; trades enter at **day d+1 close** to model realistic lag from FI publication to order placement
- Point-in-time correct: no forward-looking data; price universe resolved via OpenFIGI before the validate window is opened
- Validate period: configurable via `config/strategy.yaml`

Known limitations documented in `backtest/engine.py`: no transaction costs in v1, survivorship bias in the downloaded universe, yfinance coverage gaps treated as held positions.

## Data sources

| Source | Data |
|---|---|
| [Finansinspektionen (FI)](https://www.fi.se) | Swedish insider transaction filings |
| [yfinance](https://github.com/ranaroussi/yfinance) | OHLCV price history |
| [OpenFIGI](https://www.openfigi.com) | ISIN → Yahoo ticker resolution |
| Groq / Anthropic / Ollama | LLM confirmation gate |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Configure LLM provider (Groq is free — no credit card)
cp .env.example .env   # then set ANALYST_PROVIDER and GROQ_API_KEY

# First run fetches ~2 years of FI data + price history (~5–15 min)
python run_backtest.py

# Paper trading loop
python run_steps45.py
```

## Project layout

```
agents/       LLM confirmation gate and risk gate
backtest/     Walk-forward engine and universe resolver
config/       strategy.yaml — all thresholds live here
data/         FI scraper, price cache, fundamental fetcher
execution/    Nordnet live connector and paper trader
monitor/      Daily P&L report
risk/         Half-Kelly sizer and exit evaluator
signals/      Conviction scorer, macro filter, sector filter
```

## Tech stack

Python 3.11 · pandas · yfinance · SQLite · PyYAML · OpenFIGI API · Groq / Anthropic / Ollama
