# Walk-forward validation on a point-in-time universe (2000-2026)

**Status: proof-of-concept validation exercise, not a validated trading strategy, and NOT wired into decide_trades.py or the portal.** This is the same expanding-window walk-forward as `ml/WALKFORWARD_REPORT.md`, but on a universe sampled from every company that was EVER an S&P 500 member during 2000-2026 (via the free [fja05680/sp500](https://github.com/fja05680/sp500) membership history), not just today's 120-ticker survivor sample. It answers a narrower version of the question raised when reaching back to the 1980s came up: does the result hold if the universe includes companies that got removed from the index along the way, not only ones that are still in it today?

## Setup

- Universe: 1094 tickers were EVER an S&P 500 member between 2000 and 2026 (per the membership file); 250 were randomly sampled from that full set (fixed seed 42).
- **Residual survivorship bias, measured directly:** of those 250 sampled tickers, only 159 had usable price data from yfinance (after an individual retry pass to rule out ordinary batch-fetch flakiness) -- 91 did not. That gap is largely genuine: companies fully delisted after bankruptcy or an acquisition (e.g. Lehman-, Enron-, WorldCom-style outcomes) have no price history available through any free source, so this pipeline still can't see them, even though the membership list itself now includes them. **One caveat on that number**: spot-checking one "failed" ticker (AVB -- AvalonBay Communities, which is very much still trading) found this sandbox's yfinance access returned only ~27 rows of genuinely thin data for it rather than a clean failure, so a small unknown fraction of the 91 figure may be ordinary data-access limitations in this environment rather than true unavailability -- the true delisted-company count is likely close to but somewhat below 91. This wasn't fully diagnosed further; a paid vendor with dedicated delisted-security coverage (Norgate, EODHD, Sharadar) would remove this ambiguity entirely.
- Each surviving ticker's rows are further restricted to its documented S&P 500 membership window(s) -- a name that left the index in 2012 stops contributing test rows in 2013 onward, the same way it would stop being eligible for an index-tracking strategy in real life.
- Labeled dataset: 10-trading-day forward excess return vs. SPY, thresholded into BUY/HOLD/SELL by quantile (top 30% BUY, bottom 30% SELL), recomputed per fold from only that fold's training data (same leak-avoidance as `ml/WALKFORWARD_REPORT.md`).
- Validation: expanding-window walk-forward, one fresh `HistGradientBoostingClassifier` per calendar year from 2000 onward, trained on everything before that year and tested only on it.

## Results by year

"Tickers in play" is how many distinct point-in-time-eligible tickers actually had a row in that year's test set -- it shrinks in later years as more of the sampled universe has already been acquired, gone bankrupt, or dropped from the index.

| Year | SPY return | Tickers in play | Train rows | Accuracy | Baseline acc. | BUY calls | BUY excess return | SELL calls | SELL excess return |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2000 | -8.8% | 55 | 48055 | 0.361 | 0.234 | 4729 | +0.85 pp | 5432 | +0.65 pp |
| 2001 | -10.1% | 56 | 61153 | 0.365 | 0.348 | 2868 | +1.11 pp | 3540 | +1.00 pp |
| 2002 | -22.4% | 60 | 74737 | 0.398 | 0.398 | 3847 | +0.25 pp | 3217 | +0.02 pp |
| 2003 | +24.2% | 61 | 89496 | 0.507 | 0.527 | 1550 | +1.61 pp | 2421 | +0.41 pp |
| 2004 | +10.7% | 63 | 104812 | 0.557 | 0.564 | 789 | +0.30 pp | 839 | +0.18 pp |
| 2005 | +5.3% | 65 | 120353 | 0.559 | 0.567 | 784 | +0.36 pp | 875 | -0.35 pp |
| 2006 | +13.8% | 70 | 136311 | 0.544 | 0.547 | 1006 | -0.21 pp | 1282 | +0.19 pp |
| 2007 | +5.3% | 74 | 153201 | 0.484 | 0.481 | 1952 | -0.50 pp | 1695 | -0.81 pp |
| 2008 | -36.2% | 81 | 170891 | 0.387 | 0.306 | 7173 | +0.19 pp | 6054 | -0.10 pp |
| 2009 | +22.7% | 81 | 189974 | 0.417 | 0.344 | 5311 | +1.50 pp | 7304 | +0.23 pp |
| 2010 | +13.1% | 81 | 210014 | 0.458 | 0.473 | 2546 | +0.33 pp | 2641 | -0.05 pp |
| 2011 | +0.9% | 82 | 230406 | 0.439 | 0.456 | 3245 | -0.21 pp | 3106 | +0.02 pp |
| 2012 | +14.2% | 86 | 250817 | 0.499 | 0.497 | 1637 | +0.11 pp | 2135 | +0.28 pp |
| 2013 | +29.0% | 89 | 271726 | 0.537 | 0.540 | 556 | +1.70 pp | 608 | +1.40 pp |
| 2014 | +14.6% | 88 | 293491 | 0.545 | 0.546 | 582 | +0.03 pp | 535 | -0.37 pp |
| 2015 | +1.3% | 92 | 315182 | 0.511 | 0.504 | 1464 | -1.05 pp | 1911 | -1.34 pp |
| 2016 | +13.6% | 93 | 337774 | 0.456 | 0.454 | 1966 | +1.38 pp | 2699 | +0.57 pp |
| 2017 | +20.8% | 95 | 360432 | 0.482 | 0.484 | 1040 | +0.13 pp | 1177 | -0.06 pp |
| 2018 | -5.2% | 97 | 383543 | 0.406 | 0.408 | 2434 | -0.10 pp | 3165 | +0.15 pp |
| 2019 | +31.1% | 105 | 407174 | 0.423 | 0.427 | 2085 | +0.42 pp | 3183 | +0.38 pp |
| 2020 | +17.2% | 106 | 432121 | 0.375 | 0.303 | 6293 | +0.70 pp | 6963 | -0.05 pp |
| 2021 | +30.5% | 106 | 457958 | 0.387 | 0.370 | 2027 | +0.61 pp | 1807 | +0.19 pp |
| 2022 | -18.6% | 108 | 484095 | 0.353 | 0.343 | 8002 | +0.21 pp | 6010 | +0.21 pp |
| 2023 | +26.7% | 113 | 510921 | 0.399 | 0.377 | 3478 | +0.14 pp | 3366 | -0.56 pp |
| 2024 | +25.6% | 113 | 537905 | 0.368 | 0.359 | 2677 | -0.53 pp | 2343 | -0.06 pp |
| 2025 | +18.0% | 115 | 565447 | 0.367 | 0.332 | 4689 | +0.42 pp | 3925 | +0.08 pp |
| 2026 | +12.7% | 115 | 593213 | 0.334 | 0.250 | 4112 | +0.22 pp | 3460 | +0.11 pp |

## Summary

- 27 yearly folds tested (2000-2026).
- Beat the majority-class baseline accuracy in 16/27 years.
- BUY calls' excess return vs SPY was positive in 21/27 years with at least one BUY call (mean +0.37 pp, median +0.25 pp).
- **In years SPY itself was down** (2000, 2001, 2002, 2008, 2018, 2022), BUY calls' excess return was positive in 5/6 of them.

## Compared to the non-point-in-time version

See `ml/WALKFORWARD_REPORT.md` for the 120-ticker, today's-S&P-500-only version of this same walk-forward test. Comparing the two: if this point-in-time version's numbers are similar, that's modest evidence the earlier result wasn't purely a survivorship-bias artifact. If they diverge meaningfully, the difference is itself informative about how much the original sampling choice was driving the result -- read both reports side by side rather than trusting either one alone.

## Honest caveats

- **Bias narrowed, not eliminated.** This still can't see companies fully delisted after bankruptcy or an acquisition -- see the Setup section's residual-bias number. A paid vendor with dedicated delisted-security coverage (Norgate, EODHD, Sharadar, or CRSP/WRDS for academics) is the only way to close that gap completely; there's no free source for it.
- **The membership data itself has known limits.** It's a community-maintained export of Wikipedia's crowd-edited S&P 500 change log -- solid for retail backtesting, not audit-grade, and with acknowledged gaps in the pre-2000s portion (not relevant here since this test starts at 2000, but worth knowing if extended further back).
- **A handful of "unavailable" tickers may be ordinary data-access limitations in this sandbox, not real delistings** -- see the AVB example in Setup. Not separately diagnosed further.
- **No fundamentals, no transaction costs, no position sizing, untuned hyperparameters and quantile choice** -- same limits as both earlier reports.
- **Not integrated.** `decide_trades.py` and the portal are unchanged.
