# Trading instructions (for Claude, via the GitHub connector)

This project runs without an Anthropic API key. Instead, a scheduled Claude.ai
session (with a GitHub connector to this repo) makes the trade decision
directly, then hands it to a script for validated execution. Follow this
procedure exactly, in order, on each scheduled run.

## 0. Install dependencies (if not already installed in this environment)

```bash
pip install -r requirements.txt
```

## 1. Prepare today's data

Run:

```bash
python prepare_data.py
```

This fetches current prices and 10-day history for the full watchlist,
narrows that down to a price-movement prescreen pool, fetches fundamentals
for that pool, and ranks it by fundamental quality down to today's final
candidates. It writes `today_data.json`. It does not decide or execute
anything.

## 2. Read `today_data.json` and decide trades

Read the file. It contains:
- `cash`, `positions`, `portfolio_value` — current account state
- `candidates` — the only tickers you may trade today (already narrowed by
  fundamental quality, not just price movement — see below)
- `prices` — current price for each candidate (+ the benchmark)
- `history` — last 10 trading days of closes per candidate
- `fundamentals` — per-candidate valuation, profitability, growth, leverage,
  and analyst-sentiment data pulled from Yahoo Finance (see field list below)
- `rules` — the risk limits (also enforced independently in step 3, so
  there's no point proposing trades that violate them — they'll just be
  rejected)

### How to weigh fundamentals vs. price history

Fundamentals are the primary lens now — use them to judge whether a business
is even worth holding, the way a real investment desk would, before looking
at recent price action at all:

- **Valuation** — `trailingPE`, `forwardPE`, `priceToBook`, `enterpriseToEbitda`,
  `pegRatio`. Is the price reasonable relative to earnings/growth, not just
  "is it going up"?
- **Profitability & quality** — `returnOnEquity`, `returnOnAssets`,
  `profitMargins`, `operatingMargins`. Efficient, consistently profitable
  businesses are the default preference.
- **Growth** — `revenueGrowth`, `earningsGrowth`.
- **Cash generation** — `freeCashflow`, `totalCash`. Cash is harder to
  manufacture with accounting choices than reported earnings.
- **Balance-sheet health / leverage** — `debtToEquity`, `currentRatio`,
  `quickRatio`. The closest free proxy to real creditworthiness (actual
  credit ratings from Moody's/S&P aren't available without a paid data
  vendor — treat this as a stand-in, not the real thing).
- **Analyst sentiment** — `recommendationKey`, `recommendationMean` (1 =
  strong buy … 5 = strong sell), `targetMeanPrice`, `numberOfAnalystOpinions`.
  Aggregates Street research you otherwise can't access directly.
- **Dividend/capital return** — `dividendYield`, `payoutRatio`.
- **Liquidity & risk sizing** — `marketCap`, `averageVolume`, `beta`.

Only *after* forming a view on the business itself should you use `history`
(the 10-day price trend) — and only for timing: among names that clear the
fundamentals bar, prefer entries that aren't chasing a one-day spike, and
treat a multi-day trend reversal in a held position as a reason to revisit it
sooner rather than waiting.

Any fundamentals field can be `null` — Yahoo's coverage has real gaps,
especially for smaller or newer listings. Reason with whatever is available
for a given ticker rather than requiring every field, and don't treat a
missing field as a red flag on its own.

Decide whether to buy, sell, or hold, the same way a disciplined desk would:
form a short thesis per trade grounded in the fundamentals (and price data
where relevant) actually given in `today_data.json` — don't invent numbers
not present in the file — size positions sensibly within `max_position_pct`,
and don't feel obligated to trade if nothing looks good — an empty trade list
is a valid, often correct, decision.

## 3. Write your decision to `decision.json`

In this exact shape:

```json
{
  "trades": [
    {"ticker": "AAPL", "action": "buy", "shares": 2, "reason": "short reason"}
  ],
  "commentary": "1-3 sentence summary of your overall reasoning today"
}
```

Use `"action": "sell"` to reduce/close a position. If you want to do nothing,
write `{"trades": [], "commentary": "..."}`.

## 4. Apply the decision

Run:

```bash
python apply_trades.py decision.json
```

This validates every trade against the risk rules (position size cap, cash
buffer, daily trade cap, candidate restriction) independently of your
reasoning — it will reject anything that breaks the rules rather than trust
the proposal — then executes what's valid, updates `portfolio.json` and
`trade_log.csv`, and prints a summary. Read the output; if trades were
rejected, note why (informational only — do not retry rejected trades this
run).

## 5. Regenerate the report

```bash
python generate_report.py
```

This writes `REPORT.md` — a plain-language summary (performance vs. the
benchmark, current holdings with gain/loss, recent trades) readable directly
on GitHub. Commit it alongside the state files.

## 6. Commit the updated state

```bash
git add portfolio.json trade_log.csv REPORT.md
git commit -m "Trade run: $(date +%Y-%m-%d) $(date +%H:%M) ET"
git push
```

`today_data.json` and `decision.json` are scratch files for this run only —
don't commit them (they're gitignored).

## Notes

- Run this procedure up to 3x per weekday (market open, midday, before
  close) if scheduling multiple runs, or once a day — whatever schedule
  you've set up for this session.
- If `prepare_data.py` reports stale prices or a fetch failure, re-run it
  before proceeding — don't decide trades on data you know is bad.
- Fundamentals fetches (`.info` per ticker) are slower and flakier than the
  bulk price/history fetches — if `today_data.json` notes missing
  fundamentals for a handful of candidates, that's expected Yahoo coverage
  gaps, not a failure; only stop and report if the run fails outright or
  candidates come back empty.
- See `README.md` for the overall project shape and `config.py` for the
  current risk limits, starting cash, and prescreen/candidate pool sizes.
