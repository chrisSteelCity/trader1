"""
Algorithmic (non-LLM) trade decision script.

TRADING_INSTRUCTIONS.md's step 2 ("read today_data.json and decide trades")
normally means a person or an LLM session reads the fundamentals and reasons
about them. This script is a deliberately simple, transparent, mechanical
stand-in for that step, for use when full unattended automation is wanted
(e.g. the local portal's "Run full pipeline" button) and no one is available
to read the file and reason about it.

It is NOT a replacement for that reasoning process -- it's a single scoring
rule with no judgment, context, or ability to notice things a real analyst
would (a one-off earnings distortion, a pending spinoff, sector concentration,
etc.). Treat trades it proposes as exactly that: a mechanical proposal, still
independently checked against every risk rule in apply_trades.py before
anything executes.

Rule, in short (using market_data.fundamentals_quality_score, the same
heuristic prepare_data.py already uses to rank candidates):
  - SELL any held candidate that's fundamentally red-flagged, or whose
    quality score has fallen at/below SELL_THRESHOLD.
  - BUY the top-scoring non-held candidates that clear BUY_THRESHOLD, sized
    to a fraction of portfolio value each, capped by available cash.

Usage (requires today_data.json -- run prepare_data.py first):

    python decide_trades.py

Writes decision.json in the shape apply_trades.py expects.
"""

import json
import math

import config
from market_data import fundamentals_quality_score, is_fundamental_red_flag

DATA_FILE = "today_data.json"
OUTPUT_FILE = "decision.json"

SELL_THRESHOLD = -0.05     # sell a held candidate scoring at/below this
BUY_THRESHOLD = 0.08       # only buy candidates scoring above this
MAX_NEW_BUYS = 5           # cap how many new positions to open in one run
BUY_ALLOCATION_PCT = 0.08  # target size per new buy, as a fraction of portfolio value


def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    positions = data["positions"]
    prices = data["prices"]
    fundamentals = data["fundamentals"]
    portfolio_value = data["portfolio_value"]
    cash = data["cash"]
    candidates = set(data["candidates"])

    trades = []
    reasons = []

    # --- Sells: drop held candidates with weak or red-flagged fundamentals ---
    for ticker, pos in positions.items():
        if ticker not in candidates:
            continue  # not in today's candidate pool -- leave it alone
        f = fundamentals.get(ticker)
        score = fundamentals_quality_score(f)
        red_flag = is_fundamental_red_flag(f)
        if red_flag or (score is not None and score <= SELL_THRESHOLD):
            shares = pos["shares"]
            why = "red-flagged fundamentals" if red_flag else f"quality score {score:.3f} <= {SELL_THRESHOLD}"
            trades.append({
                "ticker": ticker, "action": "sell", "shares": shares,
                "reason": f"Algorithmic sell: {why}.",
            })
            reasons.append(f"SELL {shares:g} {ticker} ({why})")
            # Rough running estimate so buy sizing below isn't starved by
            # cash this same run's sells are about to free up -- the real
            # sequencing/limits are enforced for real by apply_trades.py.
            price = prices.get(ticker)
            if price:
                cash += shares * price

    # --- Buys: top-scoring non-held candidates above threshold ---
    held = set(positions.keys())
    scored = []
    for t in candidates:
        if t in held:
            continue
        score = fundamentals_quality_score(fundamentals.get(t))
        if score is not None and score > BUY_THRESHOLD:
            scored.append((score, t))
    scored.sort(reverse=True)

    min_cash_buffer = config.MIN_CASH_BUFFER * portfolio_value
    available_cash = max(0.0, cash - min_cash_buffer)
    target_alloc = BUY_ALLOCATION_PCT * portfolio_value

    for score, ticker in scored[:MAX_NEW_BUYS]:
        price = prices.get(ticker)
        if not price or price <= 0:
            continue
        alloc = min(target_alloc, available_cash)
        shares = math.floor(alloc / price)
        if shares < 1:
            continue
        cost = shares * price
        available_cash -= cost
        trades.append({
            "ticker": ticker, "action": "buy", "shares": shares,
            "reason": f"Algorithmic buy: quality score {score:.3f} (top-ranked, above {BUY_THRESHOLD}).",
        })
        reasons.append(f"BUY {shares} {ticker} (score {score:.3f})")

    commentary = (
        "Automated run, no LLM in the loop: " + "; ".join(reasons)
        if reasons else
        "Automated run: no candidate cleared the buy threshold and no held "
        "position triggered a sell -- no trades."
    )

    decision = {"trades": trades, "commentary": commentary}
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)

    print(f"Wrote {OUTPUT_FILE}: {len(trades)} proposed trade(s)")
    print(commentary)


if __name__ == "__main__":
    main()
