"""
Run this once per trading day (via cron or GitHub Actions).

Flow:
  1. Load the paper portfolio.
  2. Pull current + recent prices for the watchlist and benchmark.
  3. Ask Claude for a trade decision, as strict JSON.
  4. Validate and execute trades against the portfolio (risk rules in config.py).
  5. Record today's total value + benchmark value for later comparison.
  6. Save portfolio.json and append to trade_log.csv.
"""

import csv
import json
import os
from datetime import date

import yfinance as yf
from anthropic import Anthropic

import config
import portfolio as port


def fetch_prices(tickers):
    """Latest close price for each ticker."""
    prices = {}
    data = yf.download(tickers, period="1d", progress=False)["Close"]
    if len(tickers) == 1:
        prices[tickers[0]] = float(data.iloc[-1])
    else:
        for t in tickers:
            try:
                prices[t] = float(data[t].dropna().iloc[-1])
            except Exception:
                pass
    return prices


def fetch_recent_history(tickers, days=10):
    """Last N days of closes, for Claude to see recent trend context."""
    hist = {}
    data = yf.download(tickers, period=f"{days}d", progress=False)["Close"]
    for t in tickers:
        try:
            series = data[t].dropna() if len(tickers) > 1 else data.dropna()
            hist[t] = [round(v, 2) for v in series.tolist()]
        except Exception:
            pass
    return hist


def build_prompt(portfolio, prices, history, port_value):
    positions_desc = json.dumps(portfolio["positions"], indent=2) or "{}"
    return f"""You are a paper-trading agent managing a simulated ${config.STARTING_CASH:,.2f} account.
Today is {date.today().isoformat()}. This is not real money — it's a tracked simulation.

Current state:
- Cash: ${portfolio['cash']:.2f}
- Total portfolio value: ${port_value:.2f}
- Current positions: {positions_desc}

Current prices: {json.dumps(prices, indent=2)}

Last {len(next(iter(history.values()), []))} trading days of closes per ticker:
{json.dumps(history, indent=2)}

Watchlist you may trade: {config.WATCHLIST}

Rules you must respect (also enforced in code, so don't bother trying to exceed them):
- No single position over {config.MAX_POSITION_PCT*100:.0f}% of portfolio value.
- Keep at least {config.MIN_CASH_BUFFER*100:.0f}% of portfolio value in cash.
- At most {config.MAX_TRADES_PER_DAY} trades today.

Decide whether to buy, sell, or hold. Respond with ONLY valid JSON, no markdown fences,
no commentary outside the JSON, in exactly this shape:

{{
  "trades": [
    {{"ticker": "AAPL", "action": "buy", "shares": 2, "reason": "short reason"}}
  ],
  "commentary": "1-3 sentence summary of your overall reasoning today"
}}

If you want to do nothing today, return an empty "trades" list.
"""


def call_claude(prompt):
    client = Anthropic()  # reads ANTHROPIC_API_KEY from env
    resp = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)


def append_trade_log_csv(results):
    file_exists = os.path.exists(config.TRADE_LOG_FILE)
    with open(config.TRADE_LOG_FILE, "a", newline="") as f:
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
    portfolio = port.load_portfolio()

    tickers = sorted(set(config.WATCHLIST + [config.BENCHMARK_TICKER]))
    prices = fetch_prices(tickers)
    history = fetch_recent_history(config.WATCHLIST)

    port_value_before = port.total_value(portfolio, prices)

    prompt = build_prompt(portfolio, prices, history, port_value_before)
    decision = call_claude(prompt)

    results = port.execute_trades(portfolio, decision.get("trades", []), prices)
    append_trade_log_csv(results)

    port_value_after = port.total_value(portfolio, prices)
    benchmark_price = prices.get(config.BENCHMARK_TICKER)

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
