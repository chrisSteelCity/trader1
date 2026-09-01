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
scoring rule with a genuine trained model: a 3-way (BUY/HOLD/SELL) classifier
on price/volume-only technical features (deliberately no fundamentals —
yfinance only exposes a ticker's *current* fundamentals, not what they were
on a past date, so using them for historical training rows would leak future
information into the past).

```bash
python ml/build_dataset.py               # fetches each ticker's full available history, engineers features, writes ml/dataset.csv (not committed — see below)
python ml/train_model.py                 # single chronological train/test split, writes ml/REPORT.md + ml/model.joblib
python ml/train_walkforward.py           # expanding-window walk-forward across ~25 years, writes ml/WALKFORWARD_REPORT.md + ml/model_walkforward.joblib
python ml/build_dataset_pit.py           # point-in-time universe version, writes ml/dataset_pit.csv (not committed)
python ml/train_walkforward_pit.py       # same walk-forward, but on the point-in-time universe, writes ml/WALKFORWARD_PIT_REPORT.md + ml/model_walkforward_pit.joblib
python ml/train_walkforward_pit_models.py # same point-in-time walk-forward with XGBoost + LightGBM instead of HistGradientBoosting, writes ml/WALKFORWARD_PIT_MODEL_COMPARISON.md
python ml/fetch_earnings_pit.py           # fetches historical earnings-report data for the point-in-time universe, writes ml/earnings_pit.csv (not committed)
python ml/build_dataset_pit_earnings.py   # merges earnings features into the point-in-time dataset with no lookahead, writes ml/dataset_pit_earnings.csv (not committed)
python ml/train_walkforward_pit_earnings.py # same walk-forward with earnings features added, writes ml/WALKFORWARD_PIT_EARNINGS_REPORT.md
python ml/backtest_pit_strategy.py        # turns the model's BUY/SELL calls into an actual simulated portfolio, writes ml/BACKTEST_PIT_REPORT.md
python ml/backtest_pit_strategy_v2.py     # tries diversification + confidence filtering + vol-scaled sizing to fix volatility drag, writes ml/BACKTEST_PIT_V2_REPORT.md
```

There are seven evaluation reports, each answering a gap the previous one
left open:

- **`ml/REPORT.md`** — a single chronological train/test split on 10 years of
  history. Accuracy (36.5%) is barely above the "always guess HOLD" baseline
  (34.2%), and BUY calls edged out SPY by only +0.13 percentage points out of
  sample. The catch: the one held-out test window (2024–2026) was mostly a
  bull run, so this alone doesn't show how the model would do in a downturn.
- **`ml/WALKFORWARD_REPORT.md`** — the more thorough answer to that gap.
  `build_dataset.py` now pulls each ticker's *full* available price history
  (`period="max"`, reaching into the 1960s-80s for old names like KO or AAPL),
  and `train_walkforward.py` retrains the model once per calendar year in an
  expanding window (train on everything before year Y, test only on Y),
  refitting BUY/SELL label cutoffs from each fold's own training data so
  later years' data can't leak into earlier folds' labels. Across 27 yearly
  folds from 2000–2026, BUY calls beat SPY in 24/27 years — and in **all 6**
  years SPY itself was down (2000, 2001, 2002, 2008, 2018, 2022). That's a
  meaningfully stronger result than the single-split version, but it comes
  with real caveats specific to reaching that far back: SPY itself only
  launched in 1993 (so no labeled row exists before then, even for tickers
  with longer price history), and the ticker universe is sampled from
  *today's* S&P 500 list, so it can only ever see companies that survived
  long enough to still be in the index now — there's no free source of
  point-in-time historical index membership to fix that survivorship bias.
  Read the report's caveats section for the rest (market-structure changes
  across decades, no transaction costs/position sizing, hyperparameters
  fixed rather than tuned). That report also flags survivorship bias as an
  open problem: the universe is 120 tickers sampled from *today's* S&P 500
  list, so it can only ever see companies that survived long enough to
  still be in the index now.
- **`ml/WALKFORWARD_PIT_REPORT.md`** — a direct attempt at that survivorship-
  bias gap, using a free point-in-time membership dataset
  ([fja05680/sp500](https://github.com/fja05680/sp500) on GitHub, which
  tracks S&P 500 additions/removals since 1996 from Wikipedia's change log).
  `ml/sp500_membership.py` reads it; `ml/build_dataset_pit.py` samples 250
  tickers from every company that was EVER an S&P 500 member 2000–2026 —
  not just today's survivors — and restricts each ticker's rows to its
  actual documented membership window(s), so a company that got acquired or
  fell out of the index in, say, 2012 can still appear in training/test data
  for the years it really was a member. The result with that fixed: BUY
  calls beat SPY in 21/27 years (down from 24/27), beat the majority-class
  baseline in only 16/27 years (down from 21/27), and mean BUY excess return
  fell from +0.75pp to +0.37pp — a meaningfully weaker but still net-positive
  result, and a fairly direct read on how much the non-PIT version's
  strength was inflated by survivorship bias. It's still not a full fix: of
  the 250 sampled point-in-time-eligible tickers, 91 had no fetchable price
  data at all through yfinance (companies fully delisted after bankruptcy or
  an acquisition — Lehman Brothers, Enron, WorldCom-style outcomes — aren't
  available through any free source), so real survivorship bias remains,
  just narrower. See that report's caveats for a data-quality wrinkle found
  along the way (one "unavailable" ticker turned out to be a live company
  hitting a sandbox data-access limit, not a genuine delisting).
- **`ml/WALKFORWARD_PIT_MODEL_COMPARISON.md`** — having narrowed the
  survivorship-bias gap, the next open question was whether the model
  architecture itself was leaving performance on the table.
  `ml/train_walkforward_pit_models.py` reruns the identical point-in-time
  walk-forward (same dataset, same per-fold BUY/SELL thresholds) with
  XGBoost and LightGBM in place of `HistGradientBoostingClassifier`, on the
  same train/test splits, and compares all three. Result: no meaningful
  difference. Mean BUY excess return is +0.37pp (HistGB), +0.35pp (XGBoost),
  +0.36pp (LightGBM); years beating baseline accuracy are 16/27, 16/27,
  17/27 respectively; years BUY calls beat SPY are 21/27, 20/27, 20/27 — all
  three land in a statistical tie given only 27 one-year folds to compare
  on. The honest takeaway: switching gradient-boosting implementations
  isn't where this project's remaining upside is. Untuned hyperparameters,
  the feature set, and the residual survivorship bias are more promising
  places to look next than trying yet another model family.
- **`ml/WALKFORWARD_PIT_EARNINGS_REPORT.md`** — a first attempt at "feed the
  model more information." News and IPO data aren't available free at the
  scale/history this project needs (no free historical news archive covers
  hundreds of tickers over 25 years), but historical earnings-report data
  is: `ml/fetch_earnings_pit.py` pulls each ticker's real earnings-report
  dates and EPS surprise history from yfinance (free, up to 25 years back),
  and `ml/build_dataset_pit_earnings.py` attaches 4 new features
  (`days_since_earnings`, `last_surprise_pct`, `surprise_trend_4q`,
  `recent_earnings_flag`) to each row using `pandas.merge_asof` in backward
  mode — a hard, structural no-lookahead guarantee, since a row can only see
  reports that had already happened as of that date. The result is a mixed,
  honestly-reported non-win: permutation importance ranks the earnings
  features among the model's most-relied-on inputs (the top feature overall
  is `surprise_trend_4q`), yet overall walk-forward performance is flat to
  slightly worse than without them — mean BUY excess return +0.31pp vs
  +0.37pp, same 16/27 years beating baseline accuracy, same 21/27 years
  beating SPY. The report calls out that nuance directly: a model relying
  heavily on a feature isn't the same as that feature improving results.
- **`ml/BACKTEST_PIT_REPORT.md`** — every report above measures an abstract
  per-signal statistic ("average excess return of BUY calls"). This one asks
  a more concrete question: what happens if you actually turn those BUY/SELL
  calls into a real portfolio? `ml/backtest_pit_strategy.py` simulates a
  $100,000 account using the same point-in-time walk-forward (the plain
  11-feature version, since XGBoost/LightGBM/earnings all tied or
  underperformed it), with equal-weight position sizing (6.7% per position,
  15 concurrent slots max), a 5% cash buffer, and 10bps transaction costs on
  both legs of every trade. Result: 8,142 trades, 53.6% win rate, +0.42%
  average trade return net of costs — a real, positive edge — yet the
  portfolio ended at $441,488 (CAGR +5.74%), **underperforming** a
  $100,000 SPY buy-and-hold position over the same 26.6 years (CAGR
  +8.36%). The likely cause, diagnosed in the report: **volatility drag**,
  not cash drag (median cash utilization was only 5.7%) — individual
  trades swing by roughly 20x the average edge itself (std ~8.6% vs mean
  +0.42%), and compounding a return stream that noisy grows slower than the
  simple average suggests. It's a concrete illustration of why every
  earlier report's "no position sizing" caveat mattered more than it might
  have seemed.
- **`ml/BACKTEST_PIT_V2_REPORT.md`** — a direct attempt to fix that
  volatility-drag problem, without touching the model itself.
  `ml/backtest_pit_strategy_v2.py` tries three position-management changes
  at once: more concurrent position slots (15 → 40, to diversify away
  independent per-trade noise), confidence filtering (only trade the top
  50% of each year's BUY signals by the model's own predicted probability),
  and volatility-scaled sizing (0.5x-1.75x per position based on how calm
  or volatile that ticker has recently been, instead of equal-weighting).
  Result: **worse, not better** — CAGR fell to +4.15% (v1 was +5.74%, SPY
  +8.36%). The interesting part is *why*: confidence filtering worked
  exactly as intended in isolation (the per-trade edge more than doubled,
  +0.42% → +0.90%, and the edge-to-noise ratio nearly doubled too), but
  combining it with more position slots backfired — cutting BUY signals in
  half while nearly tripling the number of slots to fill left the book
  chronically under-invested (median only 16.9% invested vs. v1's ~94.3%),
  and that new cash drag outweighed the improved per-trade edge. A genuine
  lesson: two changes that each sound reasonable in isolation can trade off
  against each other in combination, and this project's remaining honest
  next step is testing these levers separately rather than all at once.

All seven are **not** wired into `decide_trades.py` or the portal — they're
validation exercises, not a strategy ready to trade with, even on paper.

`ml/dataset.csv`, `ml/dataset_pit.csv`, `ml/dataset_pit_earnings.csv`
(200-250MB+ with full history), `ml/earnings_pit.csv`, `ml/backtest_pit_trades.csv`,
`ml/backtest_pit_equity.csv`, `ml/backtest_pit_equity_chart.svg`,
`ml/backtest_pit_v2_trades.csv`, `ml/backtest_pit_v2_equity.csv`,
`ml/backtest_pit_v2_equity_chart.svg`, and the `.joblib` model files are
build artifacts, generated by the scripts above rather than committed to
the repo — regenerate them locally by running the matching
`build_dataset*.py`/`fetch_earnings_pit.py`/`backtest_pit_strategy*.py`
then training script.
`ml/walkforward_folds.csv`, `ml/walkforward_pit_folds.csv`,
`ml/walkforward_pit_xgboost_folds.csv`, `ml/walkforward_pit_lightgbm_folds.csv`,
`ml/walkforward_pit_earnings_folds.csv`, `ml/feature_importance_pit_earnings.csv`
(the small per-year numbers and feature rankings behind each walk-forward
table), and `ml/sp500_ticker_start_end.csv` (the point-in-time membership
data itself, pinned so results are reproducible without re-fetching it) are
committed.

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
