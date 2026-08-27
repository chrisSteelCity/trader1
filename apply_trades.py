"""
Step 3: apply a trade decision to the portfolio.

Reads a decision JSON (written by whoever/whatever decided today's trades —
see TRADING_INSTRUCTIONS.md), validates every trade against the risk rules in
config.py (position size cap, cash buffer, daily trade cap, candidate
restriction — the same checks daily_run.py used to apply), executes what's
valid, and updates portfolio.json / trade_log.csv.

Usage:
    python apply_trades.py decision.json

decision.json shape:
    {
      "trades": [
        {"ticker": "AAPL", "action": "buy", "shares": 2, "reason": "..."}
      ],
      "commentary": "1-3 sentence summary of the reasoning"
    }

Requires today_data.json to already exist (run prepare_data.py first) —
that's where prices and the candidate list for today come from, so trades
can't be validated against stale or hand-typed prices.
"""

import csv
import json
import os
import sys
from datetime import date

import config
import portfolio as port

DATA_FILE = "today_data.json"


def append_trade_log_csv(results):
    file_exists = os.path.exists(config.TRADE_LOG_FILE)
    with open(config.TRADE_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["date", "ticker", "action", "shares", "price", "status", "why", "reason"]
        )
        if not file_exists:
            writer.writeheader()
        for r in results:
            writer.writerow({
                "date": date.today().isoformat(),
                "ticker": r.get("ticker"),
                "action": r.get("action"),
                "shares": r.get("shares"),
                "price": r.get("price", ""),
                "status": r.get("status"),
                "why": r.get("why", ""),
                "reason": r.get("reason", ""),
            })


def main():
    if len(sys.argv) != 2:
        print("Usage: python apply_trades.py decision.json")
        sys.exit(1)

    decision_file = sys.argv[1]
    with open(decision_file, "r", encoding="utf-8") as f:
        decision = json.load(f)

    if not os.path.exists(DATA_FILE):
        print(f"{DATA_FILE} not found — run prepare_data.py first.")
        sys.exit(1)
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        today_data = json.load(f)

    if today_data["date"] != date.today().isoformat():
        print(f"Warning: {DATA_FILE} was prepared for {today_data['date']}, "
              f"but today is {date.today().isoformat()}. Prices may be stale — "
              f"consider re-running prepare_data.py.")

    portfolio = port.load_portfolio()
    prices = today_data["prices"]
    candidates = today_data["candidates"]

    # Use current portfolio's own valuation for held tickers not covered by
    # today's candidate price subset, same fallback daily_run.py used.
    valuation_prices = dict(prices)
    for ticker in portfolio["positions"]:
        if ticker not in valuation_prices and ticker in today_data.get("prices", {}):
            valuation_prices[ticker] = today_data["prices"][ticker]

    results = port.execute_trades(
        portfolio, decision.get("trades", []), prices,
        allowed_tickers=candidates, valuation_prices=prices,
    )
    append_trade_log_csv(results)

    port_value_after = port.total_value(portfolio, prices)
    benchmark_price = prices.get(today_data["benchmark_ticker"])

    portfolio["daily_history"].append({
        "date": date.today().isoformat(),
        "total_value": round(port_value_after, 2),
        "cash": round(portfolio["cash"], 2),
        "positions_value": round(port.positions_value(portfolio, prices), 2),
        "benchmark_price": benchmark_price,
    })

    port.save_portfolio(portfolio)

    print(f"=== {date.today().isoformat()} ===")
    print(decision.get("commentary", ""))
    for r in results:
        print(f"  {r['status'].upper()}: {r.get('action')} {r.get('shares')} {r.get('ticker')} "
              f"{'@ $' + str(r['price']) if 'price' in r else ''} "
              f"{'(' + r['why'] + ')' if r.get('why') else ''}")
    print(f"Portfolio value: ${port_value_after:,.2f} (started at ${config.STARTING_CASH:,.2f})")


if __name__ == "__main__":
    main()
