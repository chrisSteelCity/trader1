"""
Point-in-time S&P 500 membership, from a free community-maintained dataset:
https://github.com/fja05680/sp500 -- specifically its
sp500_ticker_start_end.csv, which lists every (ticker, start_date, end_date)
membership "stint" a company had in the index (a ticker can appear more than
once if it left and later rejoined). An empty end_date means still a member
as of when that file was last updated.

WHY THIS MATTERS: ml/build_dataset.py's original universe was 120 tickers
sampled from TODAY's S&P 500 list, which can only ever include companies
that survived long enough to still be in the index now -- classic
survivorship bias. This file lets ml/build_dataset_pit.py instead sample
from every ticker that was EVER an S&P 500 member during the test window
(2000-2026), including names later removed for being acquired, going
bankrupt, or just falling out of the top 500 by market cap -- and then
restrict each ticker's rows to the actual date range(s) it was really a
member, so a 2005 row for a name removed in 2010 doesn't quietly borrow
2020s price behavior it never had as an index member.

WHAT THIS DOESN'T FIX: yfinance still only serves currently-tradeable
tickers. A company that was fully delisted after bankruptcy or an
acquisition (Lehman Brothers, WorldCom, Enron, ...) has no price history
available here at all, free or otherwise -- see
ml/build_dataset_pit.py's docstring and the residual-bias note it prints
when it builds the dataset.
"""

from pathlib import Path

import pandas as pd

ML_DIR = Path(__file__).resolve().parent
MEMBERSHIP_FILE = ML_DIR / "sp500_ticker_start_end.csv"
MEMBERSHIP_SOURCE_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/main/"
    "sp500_ticker_start_end.csv"
)


def load_membership():
    df = pd.read_csv(MEMBERSHIP_FILE, parse_dates=["start_date", "end_date"])
    # No end_date means "still a member as of when this file was compiled" --
    # treat that as open-ended (today) for containment checks.
    df["end_date"] = df["end_date"].fillna(pd.Timestamp.today().normalize())
    return df


def relevant_universe(membership_df, window_start, window_end):
    """Every ticker with at least one membership stint overlapping
    [window_start, window_end]."""
    mask = (membership_df["end_date"] >= window_start) & (membership_df["start_date"] <= window_end)
    return sorted(membership_df.loc[mask, "ticker"].unique().tolist())


def member_mask(dates, stints):
    """dates: a DatetimeIndex/Series. stints: list of (start, end) Timestamps
    for one ticker. Returns a boolean array, True where `dates` falls inside
    ANY of that ticker's membership stints."""
    mask = pd.Series(False, index=range(len(dates)))
    dates = pd.DatetimeIndex(dates)
    for start, end in stints:
        mask |= (dates >= start) & (dates <= end)
    return mask.to_numpy()


def stints_by_ticker(membership_df):
    """{ticker: [(start, end), ...]} for quick per-ticker lookup."""
    out = {}
    for ticker, group in membership_df.groupby("ticker"):
        out[ticker] = list(zip(group["start_date"], group["end_date"]))
    return out
