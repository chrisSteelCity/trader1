# ML proof-of-concept: buy/hold/sell classifier

**Status: proof-of-concept only. Not a validated trading strategy, and NOT wired into decide_trades.py or the portal.** This report is meant to honestly show what a first pass at a real model looks like, including where it falls short.

## Setup

- Universe: 120 S&P 500 tickers, randomly sampled (fixed seed 42) -- not the full index, kept small for a proof-of-concept.
- History: 10y of daily prices per ticker.
- Features: price/volume technicals only (returns, moving-average ratios, RSI, volatility, distance from 52-week high, volume ratio) -- **deliberately no fundamentals**, because yfinance only exposes a company's CURRENT fundamentals, not what they were as of a past date, so using them here would leak future information into historical rows.
- Label: 10-trading-day forward return vs. SPY (excess return), split into 3 classes by fixed quantile cutoffs computed once over the whole dataset (BUY >= 0.0209, SELL <= -0.0221, else HOLD).
- Model: scikit-learn `HistGradientBoostingClassifier` (xgboost/lightgbm aren't installed in this environment).
- Split: **chronological**, not random -- train on data before 2024-03-11, test strictly after. A random split would let the model see overlapping-window rows from dates adjacent to test rows, inflating accuracy in a way that wouldn't hold up on genuinely new data.
- Train rows: 212854. Test rows: 73318.

## Classification performance (test set, held out chronologically)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| SELL | 0.374 | 0.077 | 0.128 | 25422 |
| HOLD | 0.363 | 0.835 | 0.506 | 25091 |
| BUY | 0.369 | 0.168 | 0.231 | 22805 |
| **Overall accuracy** | | | **0.365** | 73318 |

**Baseline check:** always predicting the majority class from the training set (`HOLD`) would score 0.342 accuracy on the same test set. The model's accuracy above should be compared against this, not against 100% -- a 3-way classification problem where classes are roughly balanced has a much lower "do nothing" floor than a skewed one.

Confusion matrix (rows = actual, columns = predicted; order SELL / HOLD / BUY):

```
           pred:SELL  pred:HOLD  pred:BUY
actual:SELL       1965      19644       3813
actual:HOLD       1375      20950       2766
actual:BUY        1918      17045       3842
```

## Feature importance (permutation importance on test set)

Mean drop in accuracy when a feature's values are randomly shuffled -- bigger means the model relies on it more. Negative values mean shuffling that feature didn't hurt (or slightly helped) accuracy on this test set.

| Feature | Importance | Std dev |
|---|---:|---:|
| dist_from_high | +0.0104 | 0.0007 |
| vol_20d | +0.0073 | 0.0010 |
| ret_60d | +0.0035 | 0.0005 |
| rsi_14 | +0.0012 | 0.0007 |
| ret_20d | +0.0012 | 0.0007 |
| sma20_ratio | +0.0012 | 0.0007 |
| sma50_ratio | +0.0010 | 0.0010 |
| ret_10d | +0.0008 | 0.0004 |
| sma_cross | -0.0002 | 0.0004 |
| volume_ratio | -0.0005 | 0.0002 |
| ret_5d | -0.0007 | 0.0004 |

## Simulated trading comparison (NOT a backtest -- see caveats)

- Test-period BUY calls: 10421 (out of 73318 rows)
- Test-period SELL calls: 5258
- Avg 10-day forward return of BUY calls: +1.17%
- Avg 10-day forward return of SELL calls: +1.28%
- Avg 10-day forward return, whole test universe (naive equal-weight everything baseline): +0.59%
- Avg 10-day SPY return over the same windows: +0.73%
- Avg excess return (vs SPY) of BUY calls: +0.13 pp
- Avg excess return (vs SPY), whole test universe: -0.15 pp

## Honest caveats

This is a small proof-of-concept, not a strategy ready for real (even paper) money:

- **No fundamentals.** Excluded specifically to avoid lookahead bias (see Setup). A production version would need a point-in-time fundamentals data source, which isn't available for free.
- **No transaction costs, slippage, or position sizing** in the simulated trading comparison above -- it only checks whether BUY-labeled rows had better-than-average forward returns, not whether trading on them would be profitable after costs.
- **Overlapping labels.** Every ticker contributes a row for nearly every trading day, and consecutive days' forward-return windows overlap heavily, so test rows are not independent -- effective sample size is much smaller than the row count suggests.
- **Training now spans multiple regimes, but the held-out test window is still just one stretch.** Widening history to 10y put the 2018 selloff, the 2020 COVID crash/recovery, and the 2022 bear market into the *training* data, which is an improvement over the original 3y version. But the chronological split still means the *test* period is whatever the most recent ~25% of dates happens to be (here, 2024-03 to 2026-08 -- largely a bull run), so this still doesn't tell you how the model would have performed if evaluated during a downturn. A more thorough version would test on multiple disjoint historical windows (e.g. walk-forward validation across several regimes), not just the single most-recent slice.
- **Small, randomly-sampled universe** (see Setup) rather than the full S&P 500 or a broader market -- results may not generalize to tickers outside the sample.
- **Not integrated.** decide_trades.py and the portal are unchanged; this model is not used to make any real (paper) trades yet.
