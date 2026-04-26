"""Run only steps 4 and 5 using already-cached ticker resolutions."""
import sys, sqlite3, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from run_backtest import step4_download_prices, step5_run_backtest
from backtest.universe import DB_PATH as UNIVERSE_DB

conn = sqlite3.connect(str(UNIVERSE_DB))
rows = conn.execute("SELECT DISTINCT ticker FROM ticker_cache WHERE ticker IS NOT NULL").fetchall()
conn.close()
tickers = [r[0] for r in rows]
print(f"Found {len(tickers)} resolved tickers in cache")

t0 = time.time()
step4_download_prices(tickers)
print(f"Step 4 done in {time.time()-t0:.0f}s")

step5_run_backtest()
print(f"Total: {(time.time()-t0)/60:.1f} min")
