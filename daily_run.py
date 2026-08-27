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


BATCH_SIZE = 50  # yfinance bulk downloads get slower/flakier past a few hundred tickers


def _batches(tickers, size=BATCH_SIZE):
    for i in range(0, len(tickers), size):
        yield tickers[i:i + size]


def fetch_prices(tickers):
    """Latest close price for each ticker, fetched in batches so one bad
    batch (timeout, rate limit) doesn't take out the whole universe."""
    prices = {}
    for batch in _batches(tickers):
        try:
            data = yf.download(batch, period="1d", progress=False)["Close"]
        except Exception:
            continue
        if len(batch) == 1:
            try:
                prices[batch[0]] = float(data.iloc[-1])
            except Exception:
                pass
        else:
            for t in batch:
                try:
                    prices[t] = float(data[t].dropna().iloc[-1])
                except Exception:
                    pass
    return prices


def fetch_recent_history(tickers, days=10):
    """Last N days of closes per ticker, fetched in batches (see fetch_prices)."""
    hist = {}
    for batch in _batches(tickers):
        try:
            data = yf.download(batch, period=f"{days}d", progress=False)["Close"]
        except Exception:
            continue
        for t in batch:
            try:
                series = data[t].dropna() if len(batch) > 1 else data.dropna()
                hist[t] = [round(v, 2) for v in series.tolist()]
            except Exception:
                pass
    return hist


def pick_candidates(tickers, history, held_tickers, count):
    """Narrows the full universe down to `count` tickers for today's prompt:
    the biggest movers by recent % change, plus anything already held (so
    the agent can still decide to sell a position that isn't moving)."""
    moves = []
    for t in tickers:
        closes = history.get(t)
        if not closes or len(closes) < 2 or closes[0] == 0:
            continue
        pct_change = abs(closes[-1] / closes[0] - 1)
        moves.append((pct_change, t))
    moves.sort(reverse=True)

    top_movers = [t for _, t in moves[:count]]
    candidates = list(dict.fromkeys(list(held_tickers) + top_movers))
    return candidates[:max(count, len(held_tickers))]


def build_prompt(portfolio, prices, history, port_value, candidates):
    positions_desc = json.dumps(portfolio["positions"], indent=2) or "{}"
    return f"""You are a paper-trading agent managing a simulated ${config.STARTING_CASH:,.2f} account.
Today is {date.today().isoformat()}. This is not real money — it's a tracked simulation.

Current state:
- Cash: ${portfolio['cash']:.2f}
- Total portfolio value: ${port_value:.2f}
- Current positions: {positions_desc}

Your full tradeable universe is the S&P 500 ({len(config.WATCHLIST)} tickers). To keep this
prompt a reasonable size, it's been narrowed to today's {len(candidates)} biggest movers
plus anything you currently hold — these are the only tickers you may act on today.

Current prices: {json.dumps(prices, indent=2)}

Last {len(next(iter(history.values()), []))} trading days of closes per ticker:
{json.dumps(history, indent=2)}

Tickers you may trade today: {candidates}

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

    all_tickers = sorted(set(config.WATCHLIST + [config.BENCHMARK_TICKER]))
    universe_prices = fetch_prices(all_tickers)
    universe_history = fetch_recent_history(config.WATCHLIST)

    held_tickers = list(portfolio["positions"].keys())
    candidates = pick_candidates(
        config.WATCHLIST, universe_history, held_tickers, config.CANDIDATE_COUNT
    )

    prices = {t: universe_prices[t] for t in candidates + [config.BENCHMARK_TICKER] if t in universe_prices}
    history = {t: universe_history[t] for t in candidates if t in universe_history}

    port_value_before = port.total_value(portfolio, universe_prices)

    prompt = build_prompt(portfolio, prices, history, port_value_before, candidates)
    decision = call_claude(prompt)

    results = port.execute_trades(
        portfolio, decision.get("trades", []), prices,
        allowed_tickers=candidates, valuation_prices=universe_prices,
    )
    append_trade_log_csv(results)

    port_value_after = port.total_value(portfolio, universe_prices)
    benchmark_price = universe_prices.get(config.BENCHMARK_TICKER)

    portfolio["daily_history"].append({
        "date": date.today().isoformat(),
        "total_value": round(port_value_after, 2),
        "cash": round(portfolio["cash"], 2),
        "positions_value": round(port.positions_value(portfolio, universe_prices), 2),
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
