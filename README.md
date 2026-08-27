# Claude paper-trading agent

Simulates a stock portfolio starting from a fixed cash amount. Every trading
run, Claude looks at current prices + recent trends and decides whether to
buy, sell, or hold from a watchlist. Trades are simulated only — no real
money or broker is involved. Everything is tracked in `portfolio.json` so you
can see performance over time vs. the S&P 500 (SPY).

The tradeable universe is the full S&P 500 (fetched from Wikipedia and cached
locally, `universe.py`), which covers most large-cap NYSE-listed stocks. Since
that's too large to fit in a single prompt, each run narrows it down to
the `CANDIDATE_COUNT` (default 40) biggest movers plus anything already held,
and only those are shown to Claude / eligible to trade that day — enforced in
`portfolio.py`, not just requested in the prompt.

**This project does not use the Anthropic API.** There's no `ANTHROPIC_API_KEY`
anywhere and no code that calls the Anthropic SDK. Instead, Claude makes the
trade decision directly inside a Claude.ai session that has a GitHub
connector to this repo (e.g. a scheduled Cowork-style session on a Pro plan),
running up to a few times per weekday. See `TRADING_INSTRUCTIONS.md` for the
exact procedure that session follows.

## Files
- `config.py` — starting cash, watchlist, risk limits
- `universe.py` — fetches/caches the S&P 500 ticker list used as the tradeable universe
- `market_data.py` — shared yfinance fetch/candidate-selection helpers
- `portfolio.py` — portfolio state + trade validation/execution logic
- `prepare_data.py` — step 1: fetches today's prices/history, writes `today_data.json`
- `apply_trades.py` — step 3: validates + executes a trade decision against the portfolio
- `TRADING_INSTRUCTIONS.md` — the procedure Claude follows each scheduled run
- `report.py` — prints performance so far, optional chart

## 1. Local setup

```bash
cd trading_agent
pip install -r requirements.txt
```

Edit `config.py` to set your starting cash and watchlist tickers.

## 2. Run it manually (to test, or as a substitute for the connector flow)

```bash
python prepare_data.py          # fetches prices/history -> today_data.json
# ... decide trades, write decision.json in the shape TRADING_INSTRUCTIONS.md describes ...
python apply_trades.py decision.json   # validates + executes -> updates portfolio.json
python report.py                # see performance so far
python report.py --chart        # also saves performance_chart.png
```

Do this a few times on different days to build up history before trusting it
fully — check that the trades and reasoning logged in `trade_log.csv` actually
make sense to you.

## 3. Automate it via Claude.ai's GitHub connector

1. Push this repo to GitHub and connect it to Claude.ai (Settings → Connectors
   → GitHub, or however your plan exposes repo connectors).
2. Set up a recurring/scheduled Claude.ai session (e.g. Cowork-style
   scheduled task) pointed at this repo, running on your desired cadence
   (e.g. market open / midday / before close on weekdays).
3. Give that session `TRADING_INSTRUCTIONS.md` as its procedure — it fetches
   data, decides trades, applies them through `apply_trades.py` (so risk
   limits are enforced in code, not just by the model's own restraint), and
   commits the updated `portfolio.json` / `trade_log.csv` back to the repo.

There is no GitHub Actions workflow in this repo — automation runs through
the Claude.ai connector/session itself, not CI.

## How performance tracking works

Each run appends a snapshot to `portfolio.json`'s `daily_history`: total
value, cash, positions value, and the benchmark's price that day. `report.py`
reads that history to compute your return vs. SPY's return since day one.
`trade_log.csv` has a row for every trade attempt (executed or rejected, with
why).

## Notes and honest caveats

- This is a simulation for learning/tracking purposes, not investment advice,
  and Claude's daily picks are not a validated trading strategy — treat
  the results as an experiment, not a signal to trade real money.
- Risk limits (position size cap, cash buffer, daily trade cap) are enforced
  in `portfolio.py`, not just requested from Claude, so they can't be talked
  around by the model's own reasoning.
- Price data comes from `yfinance` (Yahoo Finance), which is free but
  occasionally has outages or delisted-ticker quirks — worth skimming
  `prepare_data.py`'s output periodically for tickers that silently failed
  to fetch.
- The S&P 500 list is fetched from Wikipedia and cached in `sp500_cache.json`
  (gitignored, refreshed weekly) so a Wikipedia hiccup doesn't fail every
  run; if both the cache and the live fetch fail, it falls back to a small
  static list defined in `universe.py`.
