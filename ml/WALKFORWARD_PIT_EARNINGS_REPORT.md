# Adding earnings-report features to the point-in-time walk-forward

**Status: proof-of-concept validation exercise, not a validated trading strategy, and NOT wired into decide_trades.py or the portal.** Same point-in-time universe, same expanding-window walk-forward, same HistGradientBoostingClassifier and per-fold BUY/SELL quantile thresholds as `ml/WALKFORWARD_PIT_REPORT.md` -- the only change is 4 new features derived from each ticker's real historical earnings reports (via `ml/fetch_earnings_pit.py` and `ml/build_dataset_pit_earnings.py`), on top of the original 11 price/volume technicals:

- `days_since_earnings` -- calendar days since the most recent earnings report (999 if none yet known for that ticker)
- `last_surprise_pct` -- the % by which that report beat or missed analyst EPS estimates
- `surprise_trend_4q` -- mean surprise % over that company's last (up to) 4 reports, i.e. is it on a beat streak or a miss streak
- `recent_earnings_flag` -- 1 if a report happened in the last 5 calendar days (the sharp post-earnings price-reaction window)

**No-lookahead guarantee**: every feature is computed via `pandas.merge_asof(..., direction="backward", allow_exact_matches=False)`, which by construction can only see earnings reports strictly before each row's date -- a row can never see the report that will happen later that same day or in the future. Earnings data itself comes from yfinance's `get_earnings_dates()` (free, capped at 100 quarters/25 years per ticker by Yahoo), fetched for the same 143 tickers already in the point-in-time dataset -- 142 of them had usable earnings history.

## Results by year

| Year | Accuracy | Baseline acc. | BUY calls | BUY excess return | *(no-earnings BUY excess)* |
|---|---:|---:|---:|---:|---:|
| 2000 | 0.363 | 0.234 | 4744 | +0.97 pp | +0.85 pp |
| 2001 | 0.363 | 0.348 | 2927 | +0.92 pp | +1.11 pp |
| 2002 | 0.364 | 0.398 | 2825 | +0.37 pp | +0.25 pp |
| 2003 | 0.474 | 0.527 | 3100 | +0.98 pp | +1.61 pp |
| 2004 | 0.540 | 0.564 | 1968 | +0.32 pp | +0.30 pp |
| 2005 | 0.557 | 0.567 | 1598 | +0.52 pp | +0.36 pp |
| 2006 | 0.547 | 0.547 | 1455 | -0.12 pp | -0.21 pp |
| 2007 | 0.480 | 0.481 | 2389 | -0.84 pp | -0.50 pp |
| 2008 | 0.395 | 0.306 | 7238 | +0.16 pp | +0.19 pp |
| 2009 | 0.418 | 0.344 | 5506 | +1.28 pp | +1.50 pp |
| 2010 | 0.460 | 0.473 | 2721 | +0.19 pp | +0.33 pp |
| 2011 | 0.442 | 0.456 | 3432 | -0.51 pp | -0.21 pp |
| 2012 | 0.504 | 0.497 | 2127 | +0.59 pp | +0.11 pp |
| 2013 | 0.534 | 0.540 | 931 | +1.21 pp | +1.70 pp |
| 2014 | 0.540 | 0.546 | 1526 | +0.29 pp | +0.03 pp |
| 2015 | 0.514 | 0.504 | 2327 | -0.86 pp | -1.05 pp |
| 2016 | 0.469 | 0.454 | 3032 | +0.82 pp | +1.38 pp |
| 2017 | 0.484 | 0.484 | 1740 | -0.27 pp | +0.13 pp |
| 2018 | 0.422 | 0.408 | 3129 | +0.26 pp | -0.10 pp |
| 2019 | 0.429 | 0.427 | 3043 | +0.25 pp | +0.42 pp |
| 2020 | 0.373 | 0.303 | 6169 | +0.61 pp | +0.70 pp |
| 2021 | 0.395 | 0.370 | 3595 | +0.16 pp | +0.61 pp |
| 2022 | 0.367 | 0.343 | 8718 | +0.26 pp | +0.21 pp |
| 2023 | 0.403 | 0.377 | 4564 | -0.22 pp | +0.14 pp |
| 2024 | 0.387 | 0.359 | 3616 | +0.29 pp | -0.53 pp |
| 2025 | 0.377 | 0.332 | 5224 | +0.13 pp | +0.42 pp |
| 2026 | 0.343 | 0.250 | 4406 | +0.52 pp | +0.22 pp |

## Summary

- 27 yearly folds tested (2000-2026).
- Beat the majority-class baseline accuracy in 16/27 years (mean accuracy 0.442).
- BUY calls' excess return vs SPY was positive in 21/27 years with at least one BUY call (mean +0.31 pp, median +0.29 pp).
- **Compared to the no-earnings baseline** (`ml/WALKFORWARD_PIT_REPORT.md`): 16/27 → 16/27 years beating the majority-class baseline; 21/27 → 21/27 years BUY calls beat SPY; mean BUY excess return +0.37 pp → +0.31 pp.

## Did the model actually use the earnings features?

Permutation importance on the final fold's trained model (how much accuracy drops when a feature's values are randomly shuffled -- higher means the model relies on it more):

| Feature | Importance (mean) | Importance (std) |
|---|---:|---:|
| `surprise_trend_4q` | 0.03157 | 0.00051 |
| `dist_from_high` | 0.03067 | 0.00080 |
| `days_since_earnings` | 0.02862 | 0.00083 |
| `vol_20d` | 0.02602 | 0.00044 |
| `last_surprise_pct` | 0.02527 | 0.00087 |
| `ret_60d` | 0.01412 | 0.00066 |
| `sma_cross` | 0.01330 | 0.00073 |
| `rsi_14` | 0.00825 | 0.00093 |
| `sma50_ratio` | 0.00692 | 0.00059 |
| `volume_ratio` | 0.00668 | 0.00081 |
| `ret_20d` | 0.00552 | 0.00075 |
| `sma20_ratio` | 0.00439 | 0.00016 |
| `ret_10d` | 0.00276 | 0.00021 |
| `ret_5d` | 0.00239 | 0.00024 |
| `recent_earnings_flag` | 0.00000 | 0.00000 |

The best-ranked earnings feature came in at position 1 out of 15 features by this measure -- read directly off the table above alongside the price/volume technicals rather than summarized further here, since "more/less important than expected" is a judgment call worth making with the actual numbers in view.

**Important nuance on this specific result**: the earnings features rank as some of the model's most-relied-on inputs by this measure, which sounds like a win -- but the Summary section above shows mean BUY excess return was slightly *lower* with them added (+0.31pp vs +0.37pp without). That combination -- the model leans on a feature heavily, yet overall results don't improve -- is what it looks like when a feature is genuinely informative about *something* in the training data but doesn't add real out-of-sample predictive edge, or when it's crowding out attention the model would otherwise have given to features that generalize better. It's also worth noting permutation importance here was measured on the *training* data (as is standard for this diagnostic), not held-out test years, so it shows what the model leaned on to fit history, not proof that reliance translated into better predictions on unseen years.

## Honest caveats

- **Same underlying survivorship-bias and data limits as `ml/WALKFORWARD_PIT_REPORT.md`.** Adding earnings features doesn't touch that problem.
- **Earnings history is capped at 100 quarters (25 years) per ticker by Yahoo**, and only covers tickers that still have fetchable price data in the point-in-time dataset -- the same 91/250 sampled tickers missing from `dataset_pit.csv` are missing here too, plus one more (GENZ) with price data but no earnings history found.
- **Only EPS surprises, not full earnings-report content.** Real earnings reports contain guidance, management commentary, segment detail, and conference-call tone that a professional analyst would weigh heavily -- none of that is captured here, only the numeric beat/miss percentage.
- **No news, no IPO data.** Those aren't available free at the scale and history length this project needs -- see the discussion that led to building this earnings-only version instead.
- **Same model, untuned hyperparameters, no transaction costs or position sizing** -- same limits as every other report in `ml/`.
- **Not integrated.** `decide_trades.py` and the portal are unchanged.
