"""
Run this to prepare a day's trading data for review.

Fetches current + recent prices for the tradeable universe, narrows it down
to today's candidates (see universe.py / config.CANDIDATE_COUNT), and writes
everything a trade decision needs — portfolio state, prices, history, rules —
to `today_data.json`.

This does NOT call any LLM and does NOT execute trades. It's step 1 of 3:

    1. python prepare_data.py         -> writes today_data.json
    2. (decide trades, see TRADING_INSTRUCTIONS.md) -> write decision.json
    3. python apply_trades.py decision.json  -> validates, executes, commits state
"""

import json
from datetime import date

import config
import portfolio as port
from market_data import fetch_prices, fetch_recent_history, pick_candidates

OUTPUT_FILE = "today_data.json"


def main():
    portfolio = port.load_portfolio()

    all_tickers = sorted(set(config.WATCHLIST + [config.BENCHMARK_TICKER]))
    universe_prices = fetch_prices(all_tickers)
    universe_history = fetch_recent_history(config.WATCHLIST)

    held_tickers = list(portfolio["positions"].keys())
    candidates = pick_candidates(
        config.WATCHLIST, universe_history, held_tickers, config.CANDIDATE_COUNT
    )

    prices = {t: universe_prices[t] for t in candidates + [config.BENCHMARK_TICKER] if t in universe_prices}
    history = {t: universe_history[t] for t in candidates if t in universe_history}

    port_value = port.total_value(portfolio, universe_prices)

    data = {
        "date": date.today().isoformat(),
        "cash": portfolio["cash"],
        "positions": portfolio["positions"],
        "portfolio_value": round(port_value, 2),
        "candidates": candidates,
        "prices": prices,
        "history": history,
        "rules": {
            "max_position_pct": config.MAX_POSITION_PCT,
            "min_cash_buffer": config.MIN_CASH_BUFFER,
            "max_trades_per_day": config.MAX_TRADES_PER_DAY,
        },
        "benchmark_ticker": config.BENCHMARK_TICKER,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote {OUTPUT_FILE}: {len(candidates)} candidates, "
          f"portfolio value ${port_value:,.2f}, cash ${portfolio['cash']:,.2f}")


if __name__ == "__main__":
    main()
