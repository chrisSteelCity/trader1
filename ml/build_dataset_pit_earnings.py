"""
Augments dataset_pit.csv with earnings-report features, from earnings_pit.csv
(built by fetch_earnings_pit.py). This is the "feed it more information"
step: on top of the existing price/volume technicals, each row also gets to
see this ticker's own earnings-report history -- but ONLY reports that had
already happened as of that row's date. No row can see an earnings report
that hasn't been released yet; that would be lookahead bias, letting the
model "know" about news from the future.

New features added per row (all computed only from earnings reports strictly
before that row's date):
  - days_since_earnings: trading-day-ish gap (calendar days) since the most
    recent earnings report. Large value (999) if no prior report exists yet
    for that ticker in the data.
  - last_surprise_pct: the % EPS surprise from the most recent report
    (positive = beat estimates, negative = missed).
  - surprise_trend_4q: mean of the last up-to-4 reports' surprise %, i.e. is
    this company on a streak of beating or missing estimates lately.
  - recent_earnings_flag: 1 if a report happened in the last 5 calendar days
    (captures the sharp post-earnings price reaction window), else 0.

Uses pandas.merge_asof (backward direction) per ticker, which is exactly the
right tool for "as of this date, what's the most recent known report" --
it's inherently look-back-only by construction.

Writes ml/dataset_pit_earnings.csv + ml/dataset_pit_earnings_meta.json (same
shape as dataset_pit_meta.json, with the earnings feature names appended to
feature_cols).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parent

NO_PRIOR_EARNINGS_DAYS = 999
SURPRISE_TREND_WINDOW = 4
RECENT_EARNINGS_WINDOW_DAYS = 5

EARNINGS_FEATURE_COLS = [
    "days_since_earnings",
    "last_surprise_pct",
    "surprise_trend_4q",
    "recent_earnings_flag",
]


def load_inputs():
    dataset = pd.read_csv(ML_DIR / "dataset_pit.csv", parse_dates=["date"])
    with open(ML_DIR / "dataset_pit_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    earnings = pd.read_csv(ML_DIR / "earnings_pit.csv", parse_dates=["earnings_date"])
    return dataset, meta, earnings


def build_earnings_features_for_ticker(dates, earnings_df):
    """dates: sorted DatetimeIndex/Series of price-row dates for one ticker.
    earnings_df: that ticker's earnings rows, sorted by earnings_date.
    Returns a DataFrame indexed like `dates` with the earnings feature cols.
    """
    earnings_df = earnings_df.sort_values("earnings_date").reset_index(drop=True)
    # Pre-compute a trailing rolling mean of surprise_pct, evaluated AT each
    # earnings report using only that and prior reports (so when we look up
    # "as of this report" via merge_asof, the trend itself has no lookahead
    # either).
    earnings_df["surprise_trend_4q"] = (
        earnings_df["surprise_pct"].rolling(SURPRISE_TREND_WINDOW, min_periods=1).mean()
    )

    left = pd.DataFrame({"date": pd.to_datetime(dates)}).sort_values("date")
    right = earnings_df[["earnings_date", "surprise_pct", "surprise_trend_4q"]].rename(
        columns={"surprise_pct": "last_surprise_pct"}
    )

    # merge_asof keeps the matched earnings_date column too (needed for the
    # gap calc below), since it's on the right-hand side under its own name.
    merged = pd.merge_asof(
        left, right, left_on="date", right_on="earnings_date",
        direction="backward",
        allow_exact_matches=False,  # a report released ON this date isn't
                                     # known yet during that trading session
    )
    gap_days = (merged["date"] - merged["earnings_date"]).dt.days
    merged["days_since_earnings"] = gap_days.fillna(NO_PRIOR_EARNINGS_DAYS)
    merged["recent_earnings_flag"] = (
        merged["days_since_earnings"] <= RECENT_EARNINGS_WINDOW_DAYS
    ).astype(int)
    merged = merged.drop(columns=["earnings_date"])
    return merged.set_index("date")[EARNINGS_FEATURE_COLS]


def build():
    dataset, meta, earnings = load_inputs()

    print(f"Base point-in-time dataset: {len(dataset)} rows, "
          f"{dataset['ticker'].nunique()} tickers.")
    print(f"Earnings history: {len(earnings)} reports across "
          f"{earnings['ticker'].nunique()} tickers.")

    pieces = []
    tickers_with_earnings = 0
    for ticker, group in dataset.groupby("ticker", sort=False):
        group = group.sort_values("date")
        tk_earnings = earnings[earnings["ticker"] == ticker]
        if tk_earnings.empty:
            feat = pd.DataFrame(
                {c: (NO_PRIOR_EARNINGS_DAYS if c == "days_since_earnings" else np.nan)
                 for c in EARNINGS_FEATURE_COLS},
                index=pd.to_datetime(group["date"]),
            )
            feat["recent_earnings_flag"] = 0
        else:
            tickers_with_earnings += 1
            feat = build_earnings_features_for_ticker(group["date"], tk_earnings)
        feat = feat.reset_index().rename(columns={"index": "date"})
        feat["ticker"] = ticker
        merged_group = group.merge(feat, on=["date", "ticker"], how="left")
        pieces.append(merged_group)

    augmented = pd.concat(pieces, ignore_index=True)
    augmented = augmented.sort_values(["date", "ticker"]).reset_index(drop=True)

    print(f"\n{tickers_with_earnings}/{dataset['ticker'].nunique()} tickers had "
          f"usable earnings history to attach.")
    print("Earnings feature coverage (non-null / non-default):")
    print(f"  last_surprise_pct known for "
          f"{augmented['last_surprise_pct'].notna().mean()*100:.1f}% of rows "
          f"(rest are before any ticker's first known earnings report)")

    out_path = ML_DIR / "dataset_pit_earnings.csv"
    augmented.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}: {len(augmented)} rows, "
          f"{len(meta['feature_cols']) + len(EARNINGS_FEATURE_COLS)} features "
          f"({len(meta['feature_cols'])} original + {len(EARNINGS_FEATURE_COLS)} earnings)")

    new_meta = dict(meta)
    new_meta["feature_cols"] = list(meta["feature_cols"]) + EARNINGS_FEATURE_COLS
    new_meta["earnings_feature_cols"] = EARNINGS_FEATURE_COLS
    new_meta["tickers_with_earnings_history"] = tickers_with_earnings
    with open(ML_DIR / "dataset_pit_earnings_meta.json", "w", encoding="utf-8") as f:
        json.dump(new_meta, f, indent=2, default=str)


if __name__ == "__main__":
    build()
