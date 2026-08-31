# Simulating the point-in-time model as an actual strategy

**Status: research exercise, not a validated trading strategy, and NOT wired into decide_trades.py or the portal.** Every earlier report in `ml/` measured "average excess return of BUY calls" -- an abstract, per-signal statistic. This report asks a different, more concrete question: if you actually turned those BUY/SELL calls into a real portfolio with position-sizing limits, a maximum number of concurrent holdings, and transaction costs, what would the resulting account balance actually look like next to just buying and holding SPY?

## Setup

- Same point-in-time universe and walk-forward as `ml/WALKFORWARD_PIT_REPORT.md` (the plain 11-feature version -- XGBoost, LightGBM, and the earnings-features version all tied or underperformed it in earlier testing, so this uses the best-tested configuration).
- Starting capital: $100,000.
- **Position sizing**: equal-weight target of 6.7% of current portfolio value per position (1 / 15 concurrent slots), capped by available cash and a 5% cash buffer kept in reserve at all times.
- **Position limit**: at most 15 positions held at once -- a BUY signal is skipped if the book is already full, same as a real account would have to skip opportunities once fully invested.
- **Holding period**: every position is held for exactly the 10-trading-day horizon the model was trained to predict, then closed using that row's actual (point-in-time-safe) forward return.
- **Transaction costs**: 10 basis points (0.10%) charged on both the entry and exit leg of every trade -- a round-trip cost of 0.20%, a reasonable stand-in for commission + bid/ask spread + slippage at retail scale.

## Results

- Backtest period: 2000-01-03 to 2026-08-17 (26.6 years).
- 8,142 total trades executed, average 12.4 concurrent positions held (peak 15 of 15 slots).
- Win rate (net of costs): 53.6% of trades were profitable.
- Average trade return (net of costs): +0.42% per trade.
- **Ending portfolio value: $441,488** (started at $99,905), total return +341.5%, CAGR +5.74%.
- **SPY buy-and-hold over the same period**: total return +747.9%, CAGR +8.36%.
- The simulated strategy **underperformed** SPY buy-and-hold over this period on a CAGR basis.

*(An equity-curve chart comparing the two lines above is generated locally as `ml/backtest_pit_equity_chart.svg` when you run `backtest_pit_strategy.py` -- it isn't committed to the repo, same policy as the other regenerable build artifacts in `ml/`, since a chart with ~320 plotted points is a large text blob relative to the value of storing it in git.)*

## Why did it underperform despite a positive average edge?

The portfolio was usually close to fully invested (median cash balance was only 5.7% of portfolio value, and it held all 15 of its position slots most of the time), so idle cash sitting on the sidelines isn't the explanation. The more likely one is **volatility drag**: the average per-trade return was positive (+0.42%) but individual trades swung widely around that average (standard deviation of roughly 8.6% per trade, about 20x the average return itself). Compounding a return stream that noisy grows slower than the simple average suggests -- the same reason a coin flip that's +60% half the time and -50% the other half has a positive average outcome per flip but loses money over many flips compounded. A statistically real but thin per-trade edge, spread across a volatile 10-day return, is a fragile foundation for compounded growth -- and this is a direct, concrete illustration of why every earlier report's caveat about "no position sizing" mattered more than it might have seemed at the time.

## Honest caveats -- read before drawing conclusions

- **No interim mark-to-market.** This simulation only realizes profit/loss when a position closes at its 10-day horizon -- the portfolio-value line does not reflect what the account would actually be worth if you checked it mid-holding-period. This means the max-drawdown figure computed on this curve (-57.8%) is understated: a real account would see day-to-day swings this simulation can't capture without full daily price paths for every open position.
- **SELL signals don't trigger early exits.** Every BUY position runs its full 10-day course regardless of what the model says about that ticker on days in between -- a real trader watching a SELL signal fire on a held position might act on it sooner. Modeling that needs daily price paths for open positions, which this version doesn't use.
- **No fractional-share rounding.** Position sizes are treated as continuously divisible dollar amounts, not whole shares -- `apply_trades.py` in the live paper-trading system does enforce integer shares; this simulation doesn't.
- **Same underlying survivorship bias as `ml/WALKFORWARD_PIT_REPORT.md`.** 91 of 250 sampled point-in-time-eligible tickers still have no free price data, so this simulated portfolio can never buy into a name that was later delisted after bankruptcy or acquisition, the same gap discussed there.
- **Approximate exit dates for tickers with multiple membership stints.** Exit date is computed as 10 rows ahead within each ticker's already-filtered date sequence, which can drift slightly from a true 10-trading-day gap around a rejoin -- affects a small number of tickers, not separately corrected for here.
- **Transaction cost and position-sizing parameters are assumptions, not fitted or validated** -- a different cost rate or concurrency limit would change the result, sometimes substantially given how thin the underlying per-trade edge is (see the walk-forward reports).
- **Not integrated.** `decide_trades.py` and the portal are unchanged.
