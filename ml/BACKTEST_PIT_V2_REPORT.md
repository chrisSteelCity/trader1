# Trying to fix the volatility-drag problem

**Status: research exercise, not a validated trading strategy, and NOT wired into decide_trades.py or the portal.** `ml/BACKTEST_PIT_REPORT.md` found that the point-in-time model, turned into an actual portfolio, underperformed SPY buy-and-hold despite a real positive average per-trade edge -- diagnosed there as volatility drag: each trade's return swung by roughly 20x the average edge itself. This report tries three changes aimed directly at that ratio, without touching the underlying model:
1. **More concurrent positions** (15 -> 40) to diversify away independent per-trade noise.
2. **Confidence filtering** -- only trade the top 50% of each year's BUY signals by the model's own predicted probability.
3. **Volatility-scaled sizing** -- size each position by 0.5x-1.75x based on how calm or volatile that ticker has recently been, instead of equal-weighting everything.

## Results: v2 vs. v1 vs. SPY

| | v2 (this report) | v1 (original) | SPY buy-and-hold |
|---|---|---|---|
| Ending value (from $100,000) | $295,476 | $441,488 | $847,851 |
| CAGR | +4.15% | +5.74% | +8.36% |
| Total return | +195.5% | +341.5% | +747.9% |

v2 **did not beat** v1 on CAGR, and **still underperformed** SPY buy-and-hold.

## Trade-level stats

- 8,631 total trades executed, average 13.1 concurrent positions held (peak 40 of 40 slots).
- Win rate (net of costs): 55.0% of trades were profitable.
- Average trade return (net of costs): +0.90% per trade (v1 was +0.42%).
- Per-trade return std dev: 10.4% (v1 was ~8.6%) -- the edge-to-noise ratio is 0.087 (v1 was ~0.049).
- Median cash utilization: 83.1% (i.e. only 16.9% invested on a typical day, vs. v1's ~94.3%).

## Why did the combination get worse, not better?

Lever 2 (confidence filtering) worked exactly as intended in isolation: the per-trade edge more than doubled (+0.42% -> +0.90%) and the edge-to-noise ratio nearly doubled too (0.049 -> 0.087), confirming the model's confidence score does carry real information. The problem is what that filtering did in combination with lever 1: cutting the BUY signals in half while *simultaneously* nearly tripling the number of position slots (15 -> 40) left far too few signals to fill the book. The portfolio sat at a median of only 16.9% invested (v1 was ~94.3%), meaning most of the capital earned 0% sitting in cash most days. That cash drag outweighed the improved per-trade edge. This is a useful, concrete lesson: levers 1 and 2 both sounded reasonable on paper, but they trade off against each other -- more slots only helps if there's enough signal flow left to fill them, and confidence filtering directly reduces that flow.

*(An equity-curve chart comparing v2, v1, and SPY is generated locally as `ml/backtest_pit_v2_equity_chart.svg` when you run `backtest_pit_strategy_v2.py` -- not committed to the repo, same build-artifact policy as the rest of `ml/`.)*

## Honest caveats -- read before drawing conclusions

- **Same underlying signal.** Steps 1-3 only change how BUY/SELL calls get turned into trades -- the model itself, its features, and its walk-forward folds are unchanged from `ml/backtest_pit_strategy.py`. If the edge-to-noise ratio didn't meaningfully improve, none of the position-management changes can fully fix it.
- **Confidence filtering was tested on the same data the model was scored on.** The 50% cutoff wasn't independently validated on a separate holdout -- it's a reasonable first guess, not a tuned parameter.
- **Vol-scaled sizing uses `vol_20d`, a feature already in the model** -- this isn't new information, just a different use of data the model already sees, so it's a position-management change rather than a new signal.
- **More concurrent positions only helps if per-ticker noise is genuinely independent.** If many of the model's BUY calls cluster around the same macro moves (e.g. it tends to buy broadly during rallies), the diversification benefit is smaller than the position count alone would suggest.
- **Same caveats as `ml/BACKTEST_PIT_REPORT.md`**: no interim mark-to-market, SELL signals don't trigger early exits, no fractional-share rounding, same survivorship bias, approximate exit dates for multi-stint tickers, transaction-cost assumption unchanged, not integrated into `decide_trades.py` or the portal.
