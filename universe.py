"""
Fetches the current S&P 500 ticker list (used as the tradeable universe
instead of a small fixed watchlist) and caches it to disk so the agent
doesn't need network access to Wikipedia on every run.
"""

import json
import os
import time

import pandas as pd

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CACHE_FILE = "sp500_cache.json"
CACHE_MAX_AGE_DAYS = 7

# Fallback list used only if both the cache and the live fetch fail, so a
# Wikipedia outage doesn't take the whole daily run down. Not kept in sync
# with index changes — it's a safety net, not a source of truth.
FALLBACK_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "SPY", "QQQ",
    "BRK-B", "JPM", "V", "UNH", "XOM", "JNJ", "WMT", "PG", "MA", "HD", "CVX",
]


def _normalize_ticker(symbol):
    # yfinance expects '-' for the share-class separator (e.g. BRK.B -> BRK-B).
    return symbol.strip().replace(".", "-")


def _fetch_live():
    tables = pd.read_html(SP500_WIKI_URL)
    df = tables[0]
    return sorted({_normalize_ticker(s) for s in df["Symbol"].tolist()})


def _load_cache():
    if not os.path.exists(CACHE_FILE):
        return None
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
    age_days = (time.time() - cache["fetched_at"]) / 86400
    if age_days > CACHE_MAX_AGE_DAYS:
        return None
    return cache["tickers"]


def _save_cache(tickers):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump({"fetched_at": time.time(), "tickers": tickers}, f, indent=2)


def get_sp500_tickers():
    """Returns the S&P 500 ticker list, using a 7-day-old-or-fresher cache
    when available and falling back to a live Wikipedia fetch, then to a
    small static list if both fail."""
    cached = _load_cache()
    if cached:
        return cached

    try:
        tickers = _fetch_live()
        _save_cache(tickers)
        return tickers
    except Exception as e:
        print(f"Warning: failed to fetch S&P 500 list ({e}); using fallback list")
        return sorted(FALLBACK_TICKERS)
