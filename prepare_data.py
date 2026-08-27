"""
Run this to prepare a day's trading data for review.

Fetches current + recent prices for the tradeable universe, narrows it down
to a price-movement prescreen pool (see universe.py / config.PRESCREEN_COUNT),
fetches fundamentals for that pool, then ranks it by fundamental quality down
to today's final candidates (config.CANDIDATE_COUNT) — see
market_data.rank_by_fundamentals for the scoring. Writes everything a trade
decision needs — portfolio state, prices, history, fundamentals, rules — to
`today_data.json`.

Fundamentals are now the primary lens for candidate selection (valuation,
profitability, growth, leverage, analyst sentiment); the 10-day price history
is kept as a secondary signal for timing entries/exits among names that pass
the fundamentals bar, not for picking which names to look at in the first
place. See TRADING_INSTRUCTIONS.md for how these should be weighed together.

This does NOT call any LLM and does NOT execute trades. It's step 1 of 3:

    1. python prepare_data.py         -> writes today_data.json
    2. (decide trades, see TRADING_INSTRUCTIONS.md) -> write decision.json
    3. python apply_trades.py decision.json  -> validates, executes, commits state
"""

import json
from datetime import date

import config
import portfolio as port
from market_data import (
    fetch_fundamentals,
    fetch_prices,
    fetch_recent_history,
    pick_candidates,
    rank_by_fundamentals,
)

OUTPUT_FILE = "today_data.json"


def main():
    portfolio = port.load_portfolio()

    all_tickers = sorted(set(config.WATCHLIST + [config.BENCHMARK_TICKER]))
    universe_prices = fetch_prices(all_tickers)
    universe_history = fetch_recent_history(config.WATCHLIST)

    held_tickers = list(portfolio["positions"].keys())

    # Stage 1: cheap price-movement prefilter (bounds how many tickers need
    # a fundamentals fetch, since that's ~1s/ticker with no bulk endpoint).
    prescreen = pick_candidates(
        config.WATCHLIST, universe_history, held_tickers, config.PRESCREEN_COUNT
    )

    # Stage 2: fetch fundamentals for the prescreen pool, then rank down to
    # the final candidate list by fundamental quality (always keeping held
    # tickers so they stay sellable regardless of score).
    fundamentals = fetch_fundamentals(prescreen)
    candidates = rank_by_fundamentals(
        prescreen, fundamentals, held_tickers, config.CANDIDATE_COUNT
    )

    prices = {t: universe_prices[t] for t in candidates + [config.BENCHMARK_TICKER] if t in universe_prices}
    history = {t: universe_history[t] for t in candidates if t in universe_history}
    candidate_fundamentals = {t: fundamentals[t] for t in candidates if t in fundamentals}

    port_value = port.total_value(portfolio, universe_prices)

    data = {
        "date": date.today().isoformat(),
        "cash": portfolio["cash"],
        "positions": portfolio["positions"],
        "portfolio_value": round(port_value, 2),
        "candidates": candidates,
        "prices": prices,
        "history": history,
        "fundamentals": candidate_fundamentals,
        "rules": {
            "max_position_pct": config.MAX_POSITION_PCT,
            "min_cash_buffer": config.MIN_CASH_BUFFER,
            "max_trades_per_day": config.MAX_TRADES_PER_DAY,
        },
        "benchmark_ticker": config.BENCHMARK_TICKER,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    missing_fundamentals = [t for t in candidates if t not in fundamentals]
    print(f"Wrote {OUTPUT_FILE}: {len(candidates)} candidates "
          f"(from a {len(prescreen)}-ticker prescreen pool), "
          f"portfolio value ${port_value:,.2f}, cash ${portfolio['cash']:,.2f}")
    if missing_fundamentals:
        print(f"  Note: no fundamentals data for {len(missing_fundamentals)} "
              f"candidate(s): {', '.join(missing_fundamentals)}")


if __name__ == "__main__":
    main()
