# Model comparison on the point-in-time walk-forward (2000-2026)

**Status: proof-of-concept validation exercise, not a validated trading strategy, and NOT wired into decide_trades.py or the portal.** Same dataset, same expanding-window walk-forward, same per-fold BUY/SELL quantile thresholds as `ml/WALKFORWARD_PIT_REPORT.md` -- only the model changes. That report used scikit-learn's `HistGradientBoostingClassifier` without ever checking whether a different model family would do better; this compares it against XGBoost and LightGBM on identical train/test splits.

## Summary comparison

| Model | Folds | Beat baseline acc. | Mean accuracy | BUY beat SPY | Mean BUY excess | Median BUY excess |
|---|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting (original) | 27 | 16/27 | 0.441 | 21/27 | +0.37 pp | +0.25 pp |
| XGBoost | 27 | 16/27 | 0.442 | 20/27 | +0.35 pp | +0.24 pp |
| LightGBM | 27 | 17/27 | 0.442 | 20/27 | +0.36 pp | +0.30 pp |

By mean BUY excess return, **HistGradientBoosting (original)** comes out ahead. By fraction of years BUY calls beat SPY, **HistGradientBoosting (original)** comes out ahead (the same model).

## Year-by-year BUY excess return, all models

Positive = that model's BUY calls beat SPY that year; blank = no BUY calls issued or model not run for that year.

| Year | SPY return | HistGradientBoosting (original) | XGBoost | LightGBM |
|---|---:|---:|---:|---:|
| 2000 | -8.8% | +0.85 pp | +1.01 pp | +0.87 pp |
| 2001 | -10.1% | +1.11 pp | +1.39 pp | +1.05 pp |
| 2002 | -22.4% | +0.25 pp | +0.37 pp | +0.34 pp |
| 2003 | +24.2% | +1.61 pp | +1.73 pp | +1.54 pp |
| 2004 | +10.7% | +0.30 pp | +0.32 pp | +0.37 pp |
| 2005 | +5.3% | +0.36 pp | +0.10 pp | +0.13 pp |
| 2006 | +13.8% | -0.21 pp | -0.38 pp | -0.35 pp |
| 2007 | +5.3% | -0.50 pp | -0.51 pp | -0.56 pp |
| 2008 | -36.2% | +0.19 pp | +0.14 pp | +0.18 pp |
| 2009 | +22.7% | +1.50 pp | +1.28 pp | +1.18 pp |
| 2010 | +13.1% | +0.33 pp | +0.25 pp | +0.30 pp |
| 2011 | +0.9% | -0.21 pp | -0.13 pp | -0.10 pp |
| 2012 | +14.2% | +0.11 pp | +0.24 pp | +0.24 pp |
| 2013 | +29.0% | +1.70 pp | +1.38 pp | +1.77 pp |
| 2014 | +14.6% | +0.03 pp | -0.03 pp | -0.09 pp |
| 2015 | +1.3% | -1.05 pp | -1.25 pp | -1.34 pp |
| 2016 | +13.6% | +1.38 pp | +1.47 pp | +1.28 pp |
| 2017 | +20.8% | +0.13 pp | +0.22 pp | +0.32 pp |
| 2018 | -5.2% | -0.10 pp | -0.19 pp | -0.08 pp |
| 2019 | +31.1% | +0.42 pp | +0.46 pp | +0.65 pp |
| 2020 | +17.2% | +0.70 pp | +0.73 pp | +0.85 pp |
| 2021 | +30.5% | +0.61 pp | +0.61 pp | +0.61 pp |
| 2022 | -18.6% | +0.21 pp | +0.19 pp | +0.15 pp |
| 2023 | +26.7% | +0.14 pp | +0.05 pp | +0.21 pp |
| 2024 | +25.6% | -0.53 pp | -0.58 pp | -0.47 pp |
| 2025 | +18.0% | +0.42 pp | +0.45 pp | +0.41 pp |
| 2026 | +12.7% | +0.22 pp | +0.11 pp | +0.26 pp |

## Honest caveats

- **Same underlying data limits as `ml/WALKFORWARD_PIT_REPORT.md`.** Survivorship bias is narrowed, not eliminated (see that report's Setup section); nothing about changing the model architecture touches that.
- **Hyperparameters were matched by intent, not tuned.** Each model uses roughly comparable settings (depth 6, learning rate 0.08, ~150-200 trees) rather than each being separately tuned to its own best configuration -- a fairer 'best case' comparison would tune each model independently (e.g. via time-series cross-validation within each fold's training window), which is real future work, not done here.
- **No ensembling tried.** Averaging or stacking predictions across the three models might do better than any single one -- not tested in this pass.
- **Small, noisy per-year sample.** 27 one-year folds is not a lot of independent data points for comparing model families -- differences of a few tenths of a percentage point between models are well within what could be noise rather than a real architectural advantage.
- **No fundamentals, no transaction costs, no position sizing.** Same limits as every other report in `ml/`.
- **Not integrated.** `decide_trades.py` and the portal are unchanged.
