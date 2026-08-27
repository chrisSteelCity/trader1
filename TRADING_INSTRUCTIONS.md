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

This fetches current prices and 10-day history, narrows the S&P 500 down to
today's candidates (biggest movers + anything currently held), and writes
`today_data.json`. It does not decide or execute anything.

## 2. Read `today_data.json` and decide trades

Read the file. It contains:
- `cash`, `positions`, `portfolio_value` — current account state
- `candidates` — the only tickers you may trade today
- `prices` — current price for each candidate (+ the benchmark)
- `history` — last 10 trading days of closes per candidate
- `rules` — the risk limits (also enforced independently in step 3, so
  there's no point proposing trades that violate them — they'll just be
  rejected)

Decide whether to buy, sell, or hold, the same way a disciplined paper-trading
desk would: form a short thesis per trade from the price/trend data given
(don't invent information not present in `today_data.json`), size positions
sensibly within `max_position_pct`, and don't feel obligated to trade if
nothing looks good — an empty trade list is a valid, often correct, decision.

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
- See `README.md` for the overall project shape and `config.py` for the
  current risk limits and starting cash.
