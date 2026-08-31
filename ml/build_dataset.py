"""
Builds a historical, labeled dataset for the buy/hold/sell classifier
proof-of-concept.

WHY PRICE-ONLY FEATURES: yfinance's Ticker.info (used elsewhere in this repo
for fundamentals) only exposes a company's CURRENT fundamentals -- there's no
point-in-time history of "what the P/E was on 2023-06-01" available for free.
Using today's fundamentals as a feature for a 2023 training row would leak
information from the future into the past (the model would implicitly "know"
things like a subsequent earnings collapse that hadn't happened yet). Rather
than build something with hidden lookahead bias, this proof-of-concept uses
ONLY price/volume history, which genuinely was known as of each date.

Universe: a fixed, modest sample of S&P 500 tickers (see TICKER_COUNT below),
not the full index, to keep runtime and data volume manageable for a
proof-of-concept. Sampled deterministically (fixed seed) from sp500_cache.json
so the run is reproducible.

Output: ml/dataset.csv with one row per (ticker, date) containing features
computed using only data up to and including that date, plus a label derived
from the forward return over the following LABEL_HORIZON_DAYS trading days
relative to SPY over the same window.
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from curl_cffi import requests as _curl_requests

BASE_DIR = Path(__file__).resolve().parent.parent
ML_DIR = Path(__file__).resolve().parent

TICKER_COUNT = 120
HISTORY_PERIOD = "3y"
LABEL_HORIZON_DAYS = 10   # forward-looking window for the label
BUY_QUANTILE = 0.70       # top 30% of excess returns -> BUY
SELL_QUANTILE = 0.30      # bottom 30% of excess returns -> SELL
BENCHMARK = "SPY"
SEED = 42

BATCH_SIZE = 40


def _new_yf_session():
    # Same impersonation profile verified working elsewhere in this repo
    # (market_data.py) -- this sandbox's egress resets connections under
    # yfinance's default curl_cffi profile.
    return _curl_requests.Session(impersonate="chrome110")


def pick_universe():
    with open(BASE_DIR / "sp500_cache.json", "r", encoding="utf-8") as f:
        tickers = json.load(f)["tickers"]
    rng = random.Random(SEED)
    sample = rng.sample(tickers, min(TICKER_COUNT, len(tickers)))
    return sorted(sample)


def _batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def fetch_history(tickers):
    """Downloads HISTORY_PERIOD of daily OHLCV for each ticker, returned as
    {ticker: DataFrame[Open, High, Low, Close, Volume]} indexed by date."""
    out = {}
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
            continue

        for t in batch:
            try:
                # group_by="ticker" always yields a MultiIndex (Ticker, Price)
                # regardless of batch size, so always index by ticker first.
                df = data[t]
                df = df.dropna(how="all")
                if df.empty or len(df) < 120:
                    continue
                out[t] = df[["Close", "Volume"]].copy()
            except Exception:
                continue
        print(f"  fetched {len(out)}/{len(tickers) + 1} so far...")
    return out


def compute_features(df):
    """Point-in-time-safe technical features -- every value at row i uses
    only data from rows <= i, so there's no lookahead."""
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

    # RSI-14
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
    universe = pick_universe()
    print(f"Universe: {len(universe)} tickers (+ {BENCHMARK} benchmark)")
    hist = fetch_history(universe)
    if BENCHMARK not in hist:
        raise RuntimeError(f"Failed to fetch benchmark {BENCHMARK} history -- aborting.")

    bench_close = hist[BENCHMARK]["Close"]
    bench_fwd = compute_forward_return(bench_close, LABEL_HORIZON_DAYS)

    rows = []
    feature_cols = None
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
        rows.append(block)

    dataset = pd.concat(rows, ignore_index=True)
    before = len(dataset)
    dataset = dataset.dropna(subset=feature_cols + ["excess_return"])
    print(f"Dropped {before - len(dataset)} rows with NaN features/labels ({len(dataset)} remain)")

    # Label thresholds from the GLOBAL distribution of excess returns across
    # the whole dataset (not per-date), computed once so it doesn't leak
    # future-date information into earlier rows' labels in a data-snooping
    # way -- it's a fixed rule applied uniformly, same as a strategy designer
    # would set thresholds in advance.
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

    out_path = ML_DIR / "dataset.csv"
    dataset.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {len(dataset)} rows, {dataset['ticker'].nunique()} tickers, "
          f"date range {dataset['date'].min()} to {dataset['date'].max()}")
    print(dataset["label"].value_counts())

    meta = {
        "feature_cols": feature_cols,
        "label_horizon_days": LABEL_HORIZON_DAYS,
        "buy_cut": buy_cut,
        "sell_cut": sell_cut,
        "benchmark": BENCHMARK,
        "universe_size": len(universe),
        "history_period": HISTORY_PERIOD,
        "seed": SEED,
    }
    with open(ML_DIR / "dataset_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


if __name__ == "__main__":
    build()
