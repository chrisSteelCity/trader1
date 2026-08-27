"""
Configuration for the paper-trading agent.
Edit these values to change how the agent behaves.
"""

# --- Portfolio setup ---
STARTING_CASH = 10_000.00          # the "given amount" you start with
PORTFOLIO_FILE = "portfolio.json"  # where portfolio state persists between runs
TRADE_LOG_FILE = "trade_log.csv"   # append-only log of every executed trade

# --- Universe the agent is allowed to trade ---
# The full universe is the S&P 500 (fetched/cached in universe.py), which
# covers most NYSE-listed large caps (plus some Nasdaq names). Every run,
# daily_run.py narrows this down to CANDIDATE_COUNT tickers (today's biggest
# movers, plus anything already held) before asking Claude to decide — the
# full list would be far too large to put in a single prompt.
import universe
WATCHLIST = universe.get_sp500_tickers()
CANDIDATE_COUNT = 40  # how many tickers get full fundamentals + history + a prompt slot each day

# How many tickers get pulled into the fundamentals-fetch pool before ranking
# down to CANDIDATE_COUNT. yfinance has no bulk endpoint for fundamentals
# (.info is one HTTP request per ticker, ~1s each), so fetching fundamentals
# for the full ~500-name WATCHLIST every run isn't practical — this bounds
# the pool to a price-movement pre-filter first (see market_data.pick_candidates),
# then prepare_data.py ranks that pool by fundamental quality and keeps the
# top CANDIDATE_COUNT (always including anything currently held).
PRESCREEN_COUNT = 80

# --- Benchmark for comparison ---
BENCHMARK_TICKER = "SPY"

# --- Risk controls (enforced in code, not just requested from Claude) ---
MAX_POSITION_PCT = 0.25   # no single position may exceed 25% of portfolio value
MIN_CASH_BUFFER = 0.02    # keep at least 2% of portfolio value in cash
MAX_TRADES_PER_DAY = 30   # cap on number of trades Claude can execute in one run

# --- Claude API ---
CLAUDE_MODEL = "claude-opus-4.8"   # swap to "claude-opus-4-8" for deeper reasoning
