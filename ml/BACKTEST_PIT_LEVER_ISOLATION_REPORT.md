# Isolating the two levers: which one actually helps?

**Status: research exercise, not a validated trading strategy, and NOT wired into decide_trades.py or the portal.** `ml/BACKTEST_PIT_V2_REPORT.md` stacked three position-management changes at once and got a worse result than the original -- but one of the three (confidence filtering) actually improved the per-trade edge on its own; it was the *combination* with more position slots that backfired via cash drag. This report re-runs the two biggest levers **one at a time**, holding everything else at `ml/backtest_pit_strategy.py`'s original settings, using the exact same underlying predictions as v2 (so results are directly comparable):
- **`confidence_only`**: v1's 15 position slots and equal-weight sizing, but only trading the top 50% of each year's BUY signals by the model's predicted confidence.
- **`diversify_only`**: 40 position slots and equal-weight sizing, but trading every BUY signal like v1 does (no confidence filter).

## Results

| Variant | Ending value | CAGR | Trades | Win rate | Avg trade | Median invested |
|---|---|---|---|---|---|---|
| v1 (original, 15 slots, no filter) | $441,488 | +5.74% | 8,142 | 53.6% | +0.42% | 94.3% |
| confidence_only (15 slots + filter) | $588,236 | +6.88% | 6,327 | 53.6% | +0.61% | 65.4% |
| diversify_only (40 slots, no filter) | $471,876 | +6.00% | 13,985 | 53.9% | +0.58% | 46.7% |
| v2 (all three levers combined) | $295,476 | +4.15% | 8,631 | 55.0% | +0.90% | 16.9% |
| SPY buy-and-hold | $847,851 | +8.36% | - | - | - | - |

**Confidence filtering alone beat v1** on CAGR (+6.88% vs +5.74%). It still traded less capital than v1 (median 65.4% invested vs v1's 94.3% -- filtering BUY signals in half necessarily means fewer positions get opened), but nowhere near as thin as v2's 16.9%, so there was enough signal flow left to make the improved per-trade edge (+0.61% vs v1's +0.42%) show up in the compounded result instead of being swamped by cash drag like it was in v2.

**Diversification alone beat v1** on CAGR (+6.00% vs +5.74%). With no confidence filter thinning out the signal supply, 40 slots stayed reasonably full (median 46.7% invested), so this isolates whether spreading capital across more simultaneous positions reduces volatility drag on its own.

*(An equity-curve chart comparing all four variants and SPY is generated locally as `ml/backtest_pit_lever_isolation_chart.svg` when you run `backtest_pit_lever_isolation.py` -- not committed to the repo, same build-artifact policy as the rest of `ml/`.)*

## Honest caveats -- read before drawing conclusions

- **Neither isolated variant was independently re-tuned.** The 50% confidence cutoff and the 40-slot count are the same untuned values tried in `ml/BACKTEST_PIT_V2_REPORT.md` -- a different cutoff or slot count could change either result, sometimes substantially.
- **Confidence filtering was scored on the same data the model was trained to predict** -- not independently validated on a separate holdout.
- **This still doesn't test every combination.** If confidence filtering alone helps, the next honest question is what slot count best complements it (rather than assuming 15 or 40 is optimal) -- not attempted here.
- **Same caveats as `ml/BACKTEST_PIT_REPORT.md`**: no interim mark-to-market, SELL signals don't trigger early exits, no fractional-share rounding, same survivorship bias, approximate exit dates for multi-stint tickers, transaction-cost assumption unchanged, not integrated into `decide_trades.py` or the portal.
