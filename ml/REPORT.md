# ML proof-of-concept: buy/hold/sell classifier

**Status: proof-of-concept only. Not a validated trading strategy, and NOT wired into decide_trades.py or the portal.** This report is meant to honestly show what a first pass at a real model looks like, including where it falls short.

## Setup

- Universe: 120 S&P 500 tickers, randomly sampled (fixed seed 42) -- not the full index, kept small for a proof-of-concept.
- History: 3y of daily prices per ticker.
- Features: price/volume technicals only (returns, moving-average ratios, RSI, volatility, distance from 52-week high, volume ratio) -- **deliberately no fundamentals**, because yfinance only exposes a company's CURRENT fundamentals, not what they were as of a past date, so using them here would leak future information into historical rows.
- Label: 10-trading-day forward return vs. SPY (excess return), split into 3 classes by fixed quantile cutoffs computed once over the whole dataset (BUY >= 0.0221, SELL <= -0.0277, else HOLD).
- Model: scikit-learn `HistGradientBoostingClassifier` (xgboost/lightgbm aren't installed in this environment).
- Split: **chronological**, not random -- train on data before 2025-12-10, test strictly after. A random split would let the model see overlapping-window rows from dates adjacent to test rows, inflating accuracy in a way that wouldn't hold up on genuinely new data.
- Train rows: 61295. Test rows: 20519.

## Classification performance (test set, held out chronologically)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| SELL | 0.344 | 0.201 | 0.254 | 6610 |
| HOLD | 0.368 | 0.719 | 0.487 | 6870 |
| BUY | 0.361 | 0.166 | 0.227 | 7039 |
| **Overall accuracy** | | | **0.362** | 20519 |

**Baseline check:** always predicting the majority class from the training set (`HOLD`) would score 0.335 accuracy on the same test set. The model's accuracy above should be compared against this, not against 100% -- a 3-way classification problem where classes are roughly balanced has a much lower "do nothing" floor than a skewed one.

Confusion matrix (rows = actual, columns = predicted; order SELL / HOLD / BUY):

```
           pred:SELL  pred:HOLD  pred:BUY
actual:SELL       1327       4046       1237
actual:HOLD       1107       4942        821
actual:BUY        1423       4451       1165
```

## Feature importance (permutation importance on test set)

Mean drop in accuracy when a feature's values are randomly shuffled -- bigger means the model relies on it more. Negative values mean shuffling that feature didn't hurt (or slightly helped) accuracy on this test set.

| Feature | Importance | Std dev |
|---|---:|---:|
| dist_from_high | +0.0132 | 0.0020 |
| vol_20d | +0.0092 | 0.0011 |
| sma50_ratio | +0.0044 | 0.0022 |
| sma_cross | +0.0039 | 0.0020 |
| rsi_14 | +0.0032 | 0.0022 |
| ret_5d | +0.0023 | 0.0011 |
| sma20_ratio | +0.0023 | 0.0016 |
| ret_10d | +0.0009 | 0.0015 |
| volume_ratio | +0.0008 | 0.0014 |
| ret_60d | -0.0004 | 0.0019 |
| ret_20d | -0.0026 | 0.0014 |

## Simulated trading comparison (NOT a backtest -- see caveats)

- Test-period BUY calls: 3223 (out of 20519 rows)
- Test-period SELL calls: 3857
- Avg 10-day forward return of BUY calls: +0.90%
- Avg 10-day forward return of SELL calls: +1.01%
- Avg 10-day forward return, whole test universe (naive equal-weight everything baseline): +0.74%
- Avg 10-day SPY return over the same windows: +0.76%
- Avg excess return (vs SPY) of BUY calls: -0.10 pp
- Avg excess return (vs SPY), whole test universe: -0.02 pp

## Honest caveats

This is a small proof-of-concept, not a strategy ready for real (even paper) money:

- **No fundamentals.** Excluded specifically to avoid lookahead bias (see Setup). A production version would need a point-in-time fundamentals data source, which isn't available for free.
- **No transaction costs, slippage, or position sizing** in the simulated trading comparison above -- it only checks whether BUY-labeled rows had better-than-average forward returns, not whether trading on them would be profitable after costs.
- **Overlapping labels.** Every ticker contributes a row for nearly every trading day, and consecutive days' forward-return windows overlap heavily, so test rows are not independent -- effective sample size is much smaller than the row count suggests.
- **One time period.** The test window covers whatever the last ~25% of the fetched history happens to contain (one particular stretch of market conditions), not multiple market regimes (e.g. no 2022-style bear market in this sample window).
- **Small, randomly-sampled universe** (see Setup) rather than the full S&P 500 or a broader market -- results may not generalize to tickers outside the sample.
- **Not integrated.** decide_trades.py and the portal are unchanged; this model is not used to make any real (paper) trades yet.
