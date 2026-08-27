# Claude paper-trading agent

Simulates a stock portfolio starting from a fixed cash amount. Every trading
day, Claude looks at current prices + recent trends and decides whether to
buy, sell, or hold from a watchlist. Trades are simulated only — no real
money or broker is involved. Everything is tracked in `portfolio.json` so you
can see performance over time vs. the S&P 500 (SPY).

The tradeable universe is the full S&P 500 (fetched from Wikipedia and cached
locally, `universe.py`), which covers most large-cap NYSE-listed stocks. Since
that's too large to fit in a single prompt, each day's run narrows it down to
the `CANDIDATE_COUNT` (default 40) biggest movers plus anything already held,
and only those are shown to Claude / eligible to trade that day — enforced in
`portfolio.py`, not just requested in the prompt.

## Files
- `config.py` — starting cash, watchlist, risk limits, which Claude model to use
- `universe.py` — fetches/caches the S&P 500 ticker list used as the tradeable universe
- `portfolio.py` — portfolio state + trade validation/execution logic
- `daily_run.py` — the script that runs once per day
- `report.py` — prints performance so far, optional chart
- `daily-trade.yml` — GitHub Actions workflow for full automation (move into `.github/workflows/`)

## 1. Local setup

```bash
cd trading_agent
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...     # get one at console.anthropic.com
```

Edit `config.py` to set your starting cash and watchlist tickers.

## 2. Test it manually

```bash
python daily_run.py     # runs one simulated trading day
python report.py        # see performance so far
python report.py --chart  # also saves performance_chart.png
```

Run `daily_run.py` a few times on different days to build up history before
trusting it fully — check that the trades and reasoning it logs actually make
sense to you.

## 3. Make it fully automated (no manual runs)

**Recommended: GitHub Actions** (runs in the cloud, even if your computer is off)

1. Push this folder to a GitHub repo.
2. Move `daily-trade.yml` into `.github/workflows/daily-trade.yml`.
3. In the repo: **Settings → Secrets and variables → Actions** → add secret
   `ANTHROPIC_API_KEY` with your key.
4. **Settings → Actions → General → Workflow permissions** → set to
   "Read and write permissions" (so it can commit the updated portfolio back).
5. It will now run automatically on the schedule in the workflow file
   (default: weekdays at market close). You can also trigger it manually from
   the repo's **Actions** tab.

**Alternative: local cron** (only runs while your machine is on)

```bash
crontab -e
# add a line like:
30 16 * * 1-5 cd /path/to/trading_agent && /usr/bin/python3 daily_run.py >> agent.log 2>&1
```

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
  occasionally has outages or delisted-ticker quirks — worth glancing at
  `agent.log` / Action run logs periodically.
- The S&P 500 list is fetched from Wikipedia and cached in `sp500_cache.json`
  (gitignored, refreshed weekly) so a Wikipedia hiccup doesn't fail every
  run; if both the cache and the live fetch fail, it falls back to a small
  static list defined in `universe.py`.
