"""
Point-in-time (PIT) version of build_dataset.py: samples the ticker universe
from every company that was EVER an S&P 500 member during 2000-2026 (using
the free membership history in sp500_ticker_start_end.csv -- see
sp500_membership.py's docstring), not just the 120 survivors sampled from
today's constituent list that ml/build_dataset.py used.

This fixes the "selection into the universe" half of the survivorship-bias
problem: a company that was in the index in 2005 but got removed by 2010
(acquired, went bankrupt, or just fell out of the top 500) now has a real
chance to appear in training/test data for the years it was actually a
member -- each ticker's rows are filtered to its documented membership
stint(s), so a 2005 row for a name removed in 2010 can't quietly use price
behavior from years it was never actually in the index for.

WHAT THIS DOESN'T FIX (residual bias, printed and reported at build time):
yfinance only serves currently-tradeable tickers. A company that was fully
delisted after bankruptcy or an acquisition -- Lehman Brothers (LEH),
WorldCom (WCOM), Enron (ENRNQ), and plenty of others from this membership
list -- has NO price history available here, at any price, free or paid,
through this pipeline. A proper fix needs a paid vendor that specifically
retains delisted-security prices (Norgate, EODHD, Sharadar, or CRSP/WRDS for
academics) -- there's no free source for that half. So this script narrows
the bias, it doesn't eliminate it: every ticker below that got dropped for
having no fetchable price data is itself evidence of the bias that remains.
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from curl_cffi import requests as _curl_requests

from sp500_membership import load_membership, relevant_universe, stints_by_ticker, member_mask

ML_DIR = Path(__file__).resolve().parent

WINDOW_START = pd.Timestamp("2000-01-01")
WINDOW_END = pd.Timestamp("2026-12-31")
TICKER_COUNT = 250   # sampled from the full point-in-time-relevant universe
                     # (bigger than build_dataset.py's 120 since a chunk of
                     # delisted names will fail to fetch -- see docstring)
HISTORY_PERIOD = "max"
LABEL_HORIZON_DAYS = 10
BUY_QUANTILE = 0.70
SELL_QUANTILE = 0.30
BENCHMARK = "SPY"
SEED = 42

BATCH_SIZE = 40


def _new_yf_session():
    return _curl_requests.Session(impersonate="chrome110")


def pick_universe(membership_df):
    universe = relevant_universe(membership_df, WINDOW_START, WINDOW_END)
    rng = random.Random(SEED)
    sample = rng.sample(universe, min(TICKER_COUNT, len(universe)))
    return sorted(sample), len(universe)


def _batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def fetch_history(tickers):
    out = {}
    failed = []
    all_tickers = list(tickers) + [BENCHMARK]
    for batch in _batches(all_tickers, BATCH_SIZE):
        for attempt in range(3):
            try:
                data = yf.download(
                    batch, period=HISTORY_PERIOD, interval="1d",
                    progress=False, session=_new_yf_session(), group_by="ticker",
                    auto_adjust=True,
                )
                break
            except Exception as e:
                print(f"  batch fetch attempt {attempt + 1} failed: {e}")
                time.sleep(2)
        else:
            print(f"  giving up on batch: {batch}")
            failed.extend(batch)
            continue

        for t in batch:
            try:
                df = data[t]
                df = df.dropna(how="all")
                if df.empty or len(df) < 120:
                    failed.append(t)
                    continue
                out[t] = df[["Close", "Volume"]].copy()
            except Exception:
                failed.append(t)
        print(f"  fetched {len(out)}/{len(tickers) + 1} so far "
              f"({len(failed)} unavailable/delisted so far)...")

    # Individual retry pass: a batch download can drop a perfectly-live
    # ticker for reasons that have nothing to do with delisting (rate
    # limiting, one bad symbol in the batch tripping up the whole group,
    # thin/partial data from contention). Retrying failures one at a time
    # with a fresh session, sequentially, separates genuine "no data exists"
    # cases (real delistings) from ordinary fetch flakiness -- without this,
    # the residual-bias number below would overstate survivorship bias by
    # counting live tickers that just had a bad batch.
    if failed:
        print(f"\nRetrying {len(failed)} failed tickers individually "
              f"(separating real delistings from batch-fetch flakiness)...")
        still_failed = []
        for t in failed:
            recovered = False
            for attempt in range(2):
                try:
                    df = yf.download(t, period=HISTORY_PERIOD, interval="1d",
                                      progress=False, session=_new_yf_session(),
                                      auto_adjust=True)
                    df = df.dropna(how="all")
                    if not df.empty and len(df) >= 120:
                        out[t] = df[["Close", "Volume"]].copy()
                        recovered = True
                        break
                except Exception:
                    pass
                time.sleep(1)
            if not recovered:
                still_failed.append(t)
        n_recovered = len(failed) - len(still_failed)
        print(f"  recovered {n_recovered}/{len(failed)} on individual retry "
              f"({len(still_failed)} confirmed unavailable)")
        failed = still_failed

    return out, failed


def compute_features(df):
    close = df["Close"]
    vol = df["Volume"]
    feat = pd.DataFrame(index=df.index)

    feat["ret_5d"] = close.pct_change(5)
    feat["ret_10d"] = close.pct_change(10)
    feat["ret_20d"] = close.pct_change(20)
    feat["ret_60d"] = close.pct_change(60)

    feat["vol_20d"] = close.pct_change().rolling(20).std()

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    feat["sma20_ratio"] = close / sma20 - 1
    feat["sma50_ratio"] = close / sma50 - 1
    feat["sma_cross"] = sma20 / sma50 - 1

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    feat["rsi_14"] = 100 - (100 / (1 + rs))

    high_252 = close.rolling(252, min_periods=60).max()
    feat["dist_from_high"] = close / high_252 - 1

    vol_avg5 = vol.rolling(5).mean()
    vol_avg20 = vol.rolling(20).mean()
    feat["volume_ratio"] = vol_avg5 / vol_avg20.replace(0, np.nan)

    return feat


def compute_forward_return(close, horizon):
    return close.shift(-horizon) / close - 1


def build():
    membership_df = load_membership()
    universe, total_relevant = pick_universe(membership_df)
    stints = stints_by_ticker(membership_df)
    print(f"Point-in-time universe: {total_relevant} tickers were ever S&P 500 "
          f"members {WINDOW_START.date()}-{WINDOW_END.date()}; sampled "
          f"{len(universe)} of them (+ {BENCHMARK} benchmark).")

    hist, failed_fetch = fetch_history(universe)
    if BENCHMARK not in hist:
        raise RuntimeError(f"Failed to fetch benchmark {BENCHMARK} history -- aborting.")

    n_fetched = len([t for t in hist if t != BENCHMARK])
    n_sampled = len(universe)
    print(f"\nResidual survivorship bias check: of {n_sampled} point-in-time-eligible "
          f"tickers sampled, {n_fetched} had fetchable price data via yfinance and "
          f"{len(failed_fetch)} did not (fully delisted/bankrupt names yfinance "
          f"doesn't serve -- e.g. {failed_fetch[:8]}{'...' if len(failed_fetch) > 8 else ''}). "
          f"That {len(failed_fetch)}/{n_sampled} gap IS the residual bias this dataset "
          f"still carries -- see this script's docstring.")

    bench_close = hist[BENCHMARK]["Close"]
    bench_fwd = compute_forward_return(bench_close, LABEL_HORIZON_DAYS)

    rows = []
    feature_cols = None
    dropped_for_membership = 0
    for ticker, df in hist.items():
        if ticker == BENCHMARK:
            continue
        feat = compute_features(df)
        if feature_cols is None:
            feature_cols = list(feat.columns)
        fwd = compute_forward_return(df["Close"], LABEL_HORIZON_DAYS)
        aligned_bench_fwd = bench_fwd.reindex(df.index)
        excess = fwd - aligned_bench_fwd

        block = feat.copy()
        block["ticker"] = ticker
        block["date"] = block.index
        block["fwd_return"] = fwd
        block["bench_fwd_return"] = aligned_bench_fwd
        block["excess_return"] = excess

        # Point-in-time filter: keep only rows falling inside this ticker's
        # documented S&P 500 membership stint(s) -- this is the actual fix,
        # not just a bigger ticker sample. A row for a name during years it
        # wasn't an index member gets dropped here.
        ticker_stints = stints.get(ticker)
        if ticker_stints:
            mask = member_mask(block.index, ticker_stints)
            dropped_for_membership += int((~mask).sum())
            block = block[mask]

        if not block.empty:
            rows.append(block)

    dataset = pd.concat(rows, ignore_index=True)
    before = len(dataset)
    dataset = dataset.dropna(subset=feature_cols + ["excess_return"])
    print(f"\nDropped {dropped_for_membership} rows outside each ticker's documented "
          f"membership window, then {before - len(dataset)} more rows with NaN "
          f"features/labels ({len(dataset)} remain)")

    buy_cut = dataset["excess_return"].quantile(BUY_QUANTILE)
    sell_cut = dataset["excess_return"].quantile(SELL_QUANTILE)
    print(f"Label thresholds: SELL if excess_return <= {sell_cut:.4f}, "
          f"BUY if excess_return >= {buy_cut:.4f}, else HOLD")

    def label_row(x):
        if x >= buy_cut:
            return "BUY"
        if x <= sell_cut:
            return "SELL"
        return "HOLD"

    dataset["label"] = dataset["excess_return"].apply(label_row)
    dataset = dataset.sort_values(["date", "ticker"]).reset_index(drop=True)

    out_path = ML_DIR / "dataset_pit.csv"
    dataset.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}: {len(dataset)} rows, {dataset['ticker'].nunique()} tickers, "
          f"date range {dataset['date'].min()} to {dataset['date'].max()}")
    print(dataset["label"].value_counts())

    meta = {
        "feature_cols": feature_cols,
        "label_horizon_days": LABEL_HORIZON_DAYS,
        "buy_cut": buy_cut,
        "sell_cut": sell_cut,
        "benchmark": BENCHMARK,
        "universe_size_sampled": n_sampled,
        "universe_size_total_relevant": total_relevant,
        "universe_size_fetched": n_fetched,
        "universe_size_fetch_failed": len(failed_fetch),
        "fetch_failed_examples": failed_fetch[:25],
        "history_period": HISTORY_PERIOD,
        "seed": SEED,
    }
    with open(ML_DIR / "dataset_pit_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


if __name__ == "__main__":
    build()
