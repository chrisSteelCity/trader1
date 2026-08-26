"""
Configuration for the paper-trading agent.
Edit these values to change how the agent behaves.
"""

# --- Portfolio setup ---
STARTING_CASH = 10_000.00          # the "given amount" you start with
PORTFOLIO_FILE = "portfolio.json"  # where portfolio state persists between runs
TRADE_LOG_FILE = "trade_log.csv"   # append-only log of every executed trade

# --- Universe the agent is allowed to trade ---
WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "TSLA", "META", "SPY", "QQQ",
]

# --- Benchmark for comparison ---
BENCHMARK_TICKER = "SPY"

# --- Risk controls (enforced in code, not just requested from Claude) ---
MAX_POSITION_PCT = 0.25   # no single position may exceed 25% of portfolio value
MIN_CASH_BUFFER = 0.02    # keep at least 2% of portfolio value in cash
MAX_TRADES_PER_DAY = 5    # cap on number of trades Claude can execute in one run

# --- Claude API ---
CLAUDE_MODEL = "claude-sonnet-5"   # swap to "claude-opus-4-8" for deeper reasoning
