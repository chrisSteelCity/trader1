"""
Shared yfinance fetch helpers used by prepare_data.py.
"""

import yfinance as yf
from curl_cffi import requests as _curl_requests

# yfinance's default curl_cffi impersonation profile ("chrome", which maps to
# the newest Chrome fingerprint) gets TLS connection resets in this sandbox's
# egress path. chrome110 was verified working via direct probing, so build a
# session pinned to it and pass it through explicitly rather than relying on
# yfinance's default.
def _new_yf_session():
    return _curl_requests.Session(impersonate="chrome110")

BATCH_SIZE = 50  # yfinance bulk downloads get slower/flakier past a few hundred tickers


def _batches(tickers, size=BATCH_SIZE):
    for i in range(0, len(tickers), size):
        yield tickers[i:i + size]


def fetch_prices(tickers):
    """Latest close price for each ticker, fetched in batches so one bad
    batch (timeout, rate limit) doesn't take out the whole universe."""
    prices = {}
    for batch in _batches(tickers):
        try:
            data = yf.download(batch, period="1d", progress=False, session=_new_yf_session())["Close"]
        except Exception:
            continue
        if len(batch) == 1:
            try:
                prices[batch[0]] = float(data.iloc[-1])
            except Exception:
                pass
        else:
            for t in batch:
                try:
                    prices[t] = float(data[t].dropna().iloc[-1])
                except Exception:
                    pass
    return prices


def fetch_recent_history(tickers, days=10):
    """Last N days of closes per ticker, fetched in batches (see fetch_prices)."""
    hist = {}
    for batch in _batches(tickers):
        try:
            data = yf.download(batch, period=f"{days}d", progress=False, session=_new_yf_session())["Close"]
        except Exception:
            continue
        for t in batch:
            try:
                series = data[t].dropna() if len(batch) > 1 else data.dropna()
                hist[t] = [round(v, 2) for v in series.tolist()]
            except Exception:
                pass
    return hist


def pick_candidates(tickers, history, held_tickers, count):
    """Narrows the full universe down to `count` tickers for today's prompt:
    the biggest movers by recent % change, plus anything already held (so
    the agent can still decide to sell a position that isn't moving)."""
    moves = []
    for t in tickers:
        closes = history.get(t)
        if not closes or len(closes) < 2 or closes[0] == 0:
            continue
        pct_change = abs(closes[-1] / closes[0] - 1)
        moves.append((pct_change, t))
    moves.sort(reverse=True)

    top_movers = [t for _, t in moves[:count]]
    candidates = list(dict.fromkeys(list(held_tickers) + top_movers))
    return candidates[:max(count, len(held_tickers))]
