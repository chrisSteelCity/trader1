"""
Fetches historical earnings-report data (report date, estimated EPS, reported
EPS, surprise %) for every ticker already present in dataset_pit.csv, via
yfinance's get_earnings_dates(). This is a genuinely free, backtestable
source of "more information" beyond price/volume: unlike Ticker.info (which
only exposes a company's CURRENT fundamentals, unusable for historical rows
without leaking the future into the past), earnings history comes with a
real report date attached to each entry, so it can be aligned to historical
price rows without lookahead -- a row for 2015-03-01 can only see earnings
reports whose date is before 2015-03-01.

Yahoo caps get_earnings_dates at limit=100 (100 quarters = 25 years), which
comfortably covers this project's 2000-2026 walk-forward window.

Writes ml/earnings_pit.csv: ticker, earnings_date, eps_estimate,
reported_eps, surprise_pct -- one row per historical earnings report.
"""

import time
from pathlib import Path

import pandas as pd
import yfinance as yf
from curl_cffi import requests as _curl_requests

ML_DIR = Path(__file__).resolve().parent
DATASET_PIT = ML_DIR / "dataset_pit.csv"
OUT_FILE = ML_DIR / "earnings_pit.csv"


def _new_yf_session():
    return _curl_requests.Session(impersonate="chrome110")


def get_universe():
    df = pd.read_csv(DATASET_PIT, usecols=["ticker"])
    return sorted(df["ticker"].unique())


def fetch_one(ticker, attempts=2):
    for attempt in range(attempts):
        try:
            t = yf.Ticker(ticker, session=_new_yf_session())
            ed = t.get_earnings_dates(limit=100)
            if ed is None or ed.empty:
                return None
            ed = ed.reset_index()
            ed.columns = ["earnings_date", "eps_estimate", "reported_eps", "surprise_pct"]
            ed["ticker"] = ticker
            # Normalize to timezone-naive dates (report dates come back
            # tz-aware, e.g. "16:00:00-04:00" after-market-close).
            ed["earnings_date"] = pd.to_datetime(ed["earnings_date"]).dt.tz_localize(None)
            return ed
        except Exception as e:
            if attempt == attempts - 1:
                print(f"  {ticker}: failed after {attempts} attempts ({e})")
                return None
            time.sleep(1)
    return None


def main():
    tickers = get_universe()
    print(f"Fetching historical earnings-report data for {len(tickers)} tickers "
          f"(from dataset_pit.csv's universe)...")

    frames = []
    failed = []
    for i, t in enumerate(tickers, 1):
        result = fetch_one(t)
        if result is not None and not result.empty:
            frames.append(result)
        else:
            failed.append(t)
        if i % 20 == 0 or i == len(tickers):
            print(f"  {i}/{len(tickers)} done ({len(failed)} failed so far)")
        time.sleep(0.3)  # be polite to Yahoo's endpoint across 143 sequential calls

    if not frames:
        raise RuntimeError("No earnings data fetched for any ticker -- aborting.")

    earnings = pd.concat(frames, ignore_index=True)
    earnings = earnings.sort_values(["ticker", "earnings_date"]).reset_index(drop=True)
    earnings.to_csv(OUT_FILE, index=False)

    print(f"\nWrote {OUT_FILE}: {len(earnings)} historical earnings reports across "
          f"{earnings['ticker'].nunique()} tickers.")
    print(f"Date range: {earnings['earnings_date'].min()} to {earnings['earnings_date'].max()}")
    if failed:
        print(f"\n{len(failed)}/{len(tickers)} tickers had no earnings data available: "
              f"{failed[:15]}{'...' if len(failed) > 15 else ''}")


if __name__ == "__main__":
    main()
