# Walk-forward validation: buy/hold/sell classifier across ~25 years

**Status: proof-of-concept validation exercise, not a validated trading strategy, and NOT wired into decide_trades.py or the portal.** This answers the question the single-split report (`ml/REPORT.md`) couldn't: does the model hold up across genuinely different market regimes, or did it just get lucky on one recent stretch?

## Setup

- Universe: 120 S&P 500 tickers, randomly sampled (fixed seed 42).
- History: each ticker's full available price history from yfinance (`period="max"`) -- for old names this reaches into the 1960s-80s, for others only back to their IPO. See `ml/build_dataset.py`'s docstring for why labeled rows still only start in 1993 (SPY's inception) and for the survivorship-bias caveat this approach can't avoid.
- Labeled dataset: 1993-01-29 through 2026, 10-day forward excess return vs. SPY, thresholded into BUY/HOLD/SELL by quantile (top 30% BUY, bottom 30% SELL -- same shape as `ml/REPORT.md`'s single-split version). **Unlike that version, the cutoffs are recomputed per fold from ONLY that fold's training data** (see the table's buy/sell cutoff columns) rather than fixed once from the full 1993-2026 distribution -- using a global cutoff would let the year-2000 fold's labels implicitly know about decades of future return data, which would be a real leak for a walk-forward test.
- Validation: **expanding-window walk-forward** -- for each calendar year Y starting 2000, train a fresh `HistGradientBoostingClassifier` on every row before Y, test only on year Y, then move to Y+1 and retrain with Y's data folded in. This is closer to how you'd actually deploy and periodically retrain a model than a single train/test split, and it surfaces whether performance holds up across different market conditions instead of just the most recent one.

## Results by year

Buy/sell cutoffs are each fold's own training-only quantile thresholds (see Setup) -- they drift over time as more history (and more volatile periods) enters the training window.

| Year | SPY return | Train rows | BUY cutoff | SELL cutoff | Accuracy | Baseline acc. | BUY calls | BUY excess return | SELL calls | SELL excess return |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2000 | -8.8% | 134863 | +2.32% | -2.63% | 0.364 | 0.239 | 7887 | +1.56 pp | 8274 | +1.27 pp |
| 2001 | -10.1% | 157232 | +2.65% | -2.66% | 0.383 | 0.337 | 5554 | +1.15 pp | 5654 | +0.99 pp |
| 2002 | -22.4% | 179827 | +2.83% | -2.62% | 0.406 | 0.377 | 5835 | +0.68 pp | 4636 | -0.22 pp |
| 2003 | +24.2% | 203319 | +2.94% | -2.55% | 0.501 | 0.507 | 2767 | +1.79 pp | 2942 | +1.32 pp |
| 2004 | +10.7% | 227213 | +2.86% | -2.45% | 0.552 | 0.547 | 2110 | +2.12 pp | 1574 | +0.90 pp |
| 2005 | +5.3% | 251309 | +2.78% | -2.33% | 0.537 | 0.526 | 1961 | +1.82 pp | 1605 | +1.11 pp |
| 2006 | +13.8% | 276043 | +2.72% | -2.23% | 0.526 | 0.521 | 2588 | +0.51 pp | 2006 | +0.19 pp |
| 2007 | +5.3% | 301100 | +2.64% | -2.17% | 0.477 | 0.472 | 3019 | -0.17 pp | 2654 | -0.92 pp |
| 2008 | -36.2% | 326248 | +2.57% | -2.16% | 0.409 | 0.314 | 10242 | +0.74 pp | 7707 | -0.05 pp |
| 2009 | +22.7% | 352040 | +2.62% | -2.20% | 0.403 | 0.355 | 8268 | +1.62 pp | 8596 | +0.79 pp |
| 2010 | +13.1% | 377785 | +2.64% | -2.22% | 0.480 | 0.494 | 3395 | +0.78 pp | 2792 | +0.43 pp |
| 2011 | +0.9% | 404104 | +2.59% | -2.19% | 0.455 | 0.469 | 4949 | +0.32 pp | 4193 | +0.13 pp |
| 2012 | +14.2% | 430888 | +2.55% | -2.16% | 0.512 | 0.509 | 2530 | +0.84 pp | 1994 | +0.69 pp |
| 2013 | +29.0% | 458103 | +2.51% | -2.12% | 0.544 | 0.547 | 1296 | +0.85 pp | 1253 | +0.99 pp |
| 2014 | +14.6% | 486057 | +2.44% | -2.07% | 0.548 | 0.547 | 1365 | +0.29 pp | 1127 | +0.28 pp |
| 2015 | +1.3% | 514502 | +2.39% | -2.02% | 0.503 | 0.500 | 2383 | +0.05 pp | 1794 | -0.45 pp |
| 2016 | +13.6% | 542974 | +2.34% | -2.00% | 0.465 | 0.461 | 2715 | +1.19 pp | 2607 | +1.06 pp |
| 2017 | +20.8% | 571450 | +2.32% | -1.98% | 0.529 | 0.530 | 1089 | +0.42 pp | 805 | +0.64 pp |
| 2018 | -5.2% | 600046 | +2.27% | -1.95% | 0.444 | 0.442 | 3017 | +0.79 pp | 1922 | +0.28 pp |
| 2019 | +31.1% | 628654 | +2.26% | -1.94% | 0.463 | 0.465 | 3167 | +0.43 pp | 2494 | +0.34 pp |
| 2020 | +17.2% | 657666 | +2.24% | -1.93% | 0.377 | 0.305 | 8756 | +0.69 pp | 7899 | +0.11 pp |
| 2021 | +30.5% | 687010 | +2.25% | -1.96% | 0.410 | 0.392 | 3274 | +0.97 pp | 2990 | +0.90 pp |
| 2022 | -18.6% | 716615 | +2.25% | -1.97% | 0.360 | 0.355 | 10524 | +0.58 pp | 5168 | +0.19 pp |
| 2023 | +26.7% | 746233 | +2.27% | -1.97% | 0.410 | 0.405 | 4456 | +0.68 pp | 2608 | -0.39 pp |
| 2024 | +25.6% | 775839 | +2.25% | -1.99% | 0.379 | 0.372 | 3059 | -0.29 pp | 1880 | -0.15 pp |
| 2025 | +18.0% | 806079 | +2.24% | -2.01% | 0.368 | 0.354 | 5375 | +0.03 pp | 3374 | +0.29 pp |
| 2026 | +12.7% | 836078 | +2.23% | -2.04% | 0.334 | 0.274 | 4419 | -0.26 pp | 2737 | +0.79 pp |

## Summary

- 27 yearly folds tested (2000-2026; the final year may be partial, through whenever the dataset was built).
- Beat the majority-class baseline accuracy in 21/27 years.
- BUY calls' excess return vs SPY was positive in 24/27 years with at least one BUY call (mean +0.75 pp, median +0.69 pp).
- **In years SPY itself was down** (2000, 2001, 2002, 2008, 2018, 2022), BUY calls' excess return was positive in 6/6 of them -- the case that actually matters for whether this generalizes beyond one bull-market stretch.

## Honest caveats

- **Survivorship bias.** The universe is sampled from TODAY's S&P 500 list, so this can only ever look at companies that survived long enough to still be in the index now -- it can't see delistings, bankruptcies, or acquisitions along the way. Free data sources don't offer point-in-time historical index membership, so this bias is structural to this whole approach, not a bug to fix.
- **SPY only exists from 1993.** Several tickers' own price history reaches earlier (their features can use it for lookback), but no labeled row exists before 1993 -- there's no benchmark to compute an excess return against.
- **Different market structure across the decades.** Pre-2001 markets traded in eighths/sixteenths rather than decimals, had far less algorithmic trading, and a very different sector mix (much less large-cap tech). Whether patterns learned from the 1990s and 2000s transfer to today's market is an open question this validation doesn't resolve, even though it's a more honest test than a single recent split.
- **No fundamentals, no transaction costs, no position sizing** -- same limits as `ml/REPORT.md`'s single-split version; this is a classification/excess-return sanity check across time, not a backtest of a tradeable strategy.
- **Fixed hyperparameters, not tuned per fold or at all** -- refitting per year keeps this a fair walk-forward test rather than implicitly hand-fitting to any one period, but it also means no fold got a chance to use better settings for its particular market conditions.
- **The BUY/SELL quantile thresholds are still an in-sample choice, just no longer a future-leaking one.** Each fold recomputes its cutoffs from only its own training data (fixing the more serious global-threshold leak an earlier version of this script had -- see Setup), but *which* quantiles to use (top/bottom 30%) was still picked once, by inspection, rather than validated. A more rigorous version would treat that as a hyperparameter tuned on a held-out slice of training data, not fixed by the person building the pipeline.
- **Not integrated.** `decide_trades.py` and the portal are unchanged; this is not used to make any real (paper) trades.
