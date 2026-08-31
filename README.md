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
exact procedure that session follows. There's also a `decide_trades.py`
fallback (a fixed scoring rule, not an LLM) for fully unattended runs — see
the portal section below.

## Files
- `config.py` — starting cash, watchlist, risk limits
- `universe.py` — fetches/caches the S&P 500 ticker list used as the tradeable universe
- `market_data.py` — shared yfinance fetch/candidate-selection helpers
- `portfolio.py` — portfolio state + trade validation/execution logic
- `prepare_data.py` — step 1: fetches today's prices/history, writes `today_data.json`
- `decide_trades.py` — algorithmic (non-LLM) trade decision, for unattended runs
- `apply_trades.py` — step 3: validates + executes a trade decision against the portfolio
- `generate_report.py` — writes `REPORT.md` + `performance_chart.svg` (readable on GitHub)
- `TRADING_INSTRUCTIONS.md` — the procedure Claude follows each scheduled run
- `report.py` — prints performance so far, to the terminal
- `portal.py` / `portal_static/` — local browser dashboard (see below)
- `ml/` — proof-of-concept ML classifier, not yet used by any of the above (see below)

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
python generate_report.py       # writes REPORT.md + performance_chart.svg
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

## 4. Browse it and run it by hand from a local dashboard

```bash
pip install -r requirements.txt
python portal.py
```

Then open `http://localhost:5000`. It's a small local Flask app — nothing
here talks to GitHub or any external service, it just reads the files in
this folder and runs the existing scripts as subprocesses:

- **Refresh market data** — re-fetches prices/fundamentals and marks the
  portfolio to market (adds a no-trade snapshot to the performance chart).
  No trades are proposed or executed.
- **Run full pipeline** — does the same, then runs `decide_trades.py` (a
  fixed, transparent scoring rule — see that file's docstring) to propose
  trades, and executes whatever clears the risk checks in `apply_trades.py`.
  This is a mechanical stand-in for the "read `today_data.json` and decide"
  step in `TRADING_INSTRUCTIONS.md`, which normally takes a person or an LLM
  session reading it — treat its trades as a fixed-rule proposal, not
  investment reasoning.

## 5. ML proof-of-concept (not yet integrated)

`ml/` holds a first pass at replacing `decide_trades.py`'s hand-weighted
scoring rule with a genuine trained model: a 3-way (BUY/HOLD/SELL)
classifier trained on 10 years of historical price/volume data across a
sample of ~120 S&P 500 tickers (widened from an initial 3-year version so
training data spans multiple market regimes — the 2018 selloff, the 2020
COVID crash/recovery, and the 2022 bear market — not just one bull run).

```bash
python ml/build_dataset.py   # fetches history, engineers features, writes ml/dataset.csv (not committed — see below)
python ml/train_model.py     # chronological train/test split, trains, writes ml/REPORT.md + ml/model.joblib
```

Read `ml/REPORT.md` for the full write-up: setup, classification metrics,
feature importance, a simulated-trading comparison against buy-and-hold, and
an explicit caveats section. Short version: accuracy (36.5%) is still barely
above the "always guess HOLD" baseline (34.2%), but with the longer history
the model's BUY calls edged out SPY out of sample (+0.13 percentage points),
a reversal from the original 3-year version where BUY calls underperformed.
That's a small, likely noisy improvement, not a validated edge — the
held-out test window is still just one recent stretch (2024–2026, mostly a
bull run), so it doesn't show how the model would hold up in a downturn; see
the caveats section for what a more thorough validation would need (e.g.
walk-forward testing across several regimes). It is **not** wired into
`decide_trades.py` or the portal, and no fundamentals are used as features
(yfinance only exposes a ticker's *current* fundamentals, not what they were
on a past date, so using them for historical training rows would leak future
information — see the caveats section for detail).

`ml/dataset.csv` (~25MB) and `ml/model.joblib` are build artifacts, generated
by the two scripts above rather than committed to the repo — regenerate them
locally by running `build_dataset.py` then `train_model.py`.

## How performance tracking works

Each run appends a snapshot to `portfolio.json`'s `daily_history`: total
value, cash, positions value, and the benchmark's price that day. `report.py`
(and the portal) reads that history to compute your return vs. SPY's return
since day one. `trade_log.csv` has a row for every trade attempt (executed or
rejected, with why).

## Notes and honest caveats

- This is a simulation for learning/tracking purposes, not investment advice,
  and neither Claude's daily picks nor `decide_trades.py`'s fixed rule are a
  validated trading strategy — treat the results as an experiment, not a
  signal to trade real money. The `ml/` proof-of-concept is even earlier
  stage than that — see `ml/REPORT.md`'s caveats.
- Risk limits (position size cap, cash buffer, daily trade cap) are enforced
  in `portfolio.py`, not just requested from Claude, so they can't be talked
  around by the model's own reasoning, and apply the same way to
  `decide_trades.py`'s proposals.
- Price data comes from `yfinance` (Yahoo Finance), which is free but
  occasionally has outages or delisted-ticker quirks — worth skimming
  `prepare_data.py`'s output periodically for tickers that silently failed
  to fetch.
- The S&P 500 list is fetched from Wikipedia and cached in `sp500_cache.json`
  (gitignored, refreshed weekly) so a Wikipedia hiccup doesn't fail every
  run; if both the cache and the live fetch fail, it falls back to a small
  static list defined in `universe.py`.
