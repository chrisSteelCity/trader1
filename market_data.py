"""
Shared yfinance fetch helpers used by prepare_data.py.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

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
    """Narrows the full universe down to `count` tickers: the biggest movers
    by recent % change, plus anything already held (so the agent can still
    decide to sell a position that isn't moving).

    This is a cheap price-based PRE-filter, not the final candidate list —
    see config.PRESCREEN_COUNT and rank_by_fundamentals() below, which narrow
    this pool further by fundamental quality before it reaches the prompt."""
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


# Fundamentals fields pulled from yfinance's Ticker.info — a free proxy for
# the kind of valuation/quality/balance-sheet/analyst-sentiment data a real
# investment desk would weigh, though not a substitute for actual credit
# ratings or proprietary research (those aren't available without a paid
# data vendor). Any field can be None — yfinance's coverage is inconsistent,
# especially for smaller or recently-listed names.
FUNDAMENTALS_FIELDS = [
    "trailingPE", "forwardPE", "priceToBook", "enterpriseToEbitda", "pegRatio",
    "returnOnEquity", "returnOnAssets", "profitMargins", "operatingMargins",
    "revenueGrowth", "earningsGrowth", "freeCashflow", "totalCash", "totalDebt",
    "debtToEquity", "currentRatio", "quickRatio", "dividendYield", "payoutRatio",
    "beta", "marketCap", "averageVolume", "recommendationKey", "recommendationMean",
    "targetMeanPrice", "numberOfAnalystOpinions",
]


def _fetch_one_fundamentals(ticker, session):
    try:
        info = yf.Ticker(ticker, session=session).info
        return ticker, {k: info.get(k) for k in FUNDAMENTALS_FIELDS}
    except Exception:
        return ticker, None


# A genuinely successful fetch populates nearly every field (~24/25 in
# practice; a handful of fields like pegRatio or dividendYield are
# legitimately None for some businesses). A result below this looks like a
# partial fetch failure from thread contention on the crumb, not real Yahoo
# coverage gaps — verified by re-fetching known "thin" tickers solo, which
# always came back full. Set well above the field count any single missing
# module could plausibly explain.
MIN_POPULATED_FIELDS = 15


def _non_null_count(data):
    return sum(1 for v in data.values() if v is not None) if data else 0


def fetch_fundamentals(tickers, max_workers=10, max_retries=4):
    """Per-ticker fundamentals via yfinance's Ticker.info. There's no bulk
    endpoint for this (unlike fetch_prices/fetch_recent_history), so this is
    one HTTP request per ticker (~1s each) — parallelized with a small thread
    pool to keep runtime reasonable, but still expensive enough that callers
    should only pass a bounded pool (see config.PRESCREEN_COUNT), not the
    full watchlist.

    Deliberately uses ONE shared session (with one negotiated crumb/cookie)
    per attempt rather than a fresh session per call — Yahoo's crumb
    negotiation gets hammered and starts failing ("Invalid Crumb" 401s) when
    many threads each spin up their own session concurrently, which silently
    degrades fetches to partial/mostly-null data instead of raising.

    Thin results (see MIN_POPULATED_FIELDS) are treated as failed fetches and
    retried with fresh sessions at successively lower concurrency (falling
    to fully sequential on the last attempt), since the contention that
    causes partial results scales with how many threads hit the crumb at
    once — a handful of stragglers get cleaned up fastest by just not
    racing them against each other."""
    fundamentals = {}
    pending = list(tickers)

    for attempt in range(max_retries):
        if not pending:
            break
        workers = max(1, max_workers // (2 ** attempt))  # 10 -> 5 -> 2 -> 1
        session = _new_yf_session()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_fetch_one_fundamentals, t, session) for t in pending]
            for future in as_completed(futures):
                ticker, data = future.result()
                if data is not None:
                    fundamentals[ticker] = data

        pending = [t for t in pending if _non_null_count(fundamentals.get(t)) < MIN_POPULATED_FIELDS]

    return fundamentals


def _avg(components):
    components = [c for c in components if c is not None]
    return sum(components) / len(components) if components else None


def fundamentals_quality_score(f):
    """A simple, transparent heuristic blending profitability, growth,
    leverage, and analyst sentiment into one score for ranking candidates.
    Not a rigorous factor model — just an average of whatever signals are
    available for a given ticker (missing fields are skipped, not penalized,
    since yfinance's coverage gaps aren't a signal about quality)."""
    if f is None:
        return None
    earnings_growth = f.get("earningsGrowth")
    if earnings_growth is not None:
        earnings_growth = max(-1.0, min(2.0, earnings_growth))  # clip outliers
    debt_to_equity = f.get("debtToEquity")
    leverage_penalty = -debt_to_equity / 1000 if debt_to_equity is not None else None
    rec_mean = f.get("recommendationMean")
    analyst_score = (3 - rec_mean) / 2 if rec_mean is not None else None  # >0 bullish, <0 bearish

    return _avg([
        f.get("returnOnEquity"),
        f.get("profitMargins"),
        f.get("revenueGrowth"),
        earnings_growth,
        leverage_penalty,
        analyst_score,
    ])


def is_fundamental_red_flag(f):
    """A narrow hard-exclusion for names that are both clearly unprofitable
    and carry a bearish Street consensus — meant to keep a pure momentum
    spike (e.g. a name up sharply on hype alone) from crowding out
    fundamentally sound candidates, without being so strict that ordinary
    missing-data gaps exclude reasonable names."""
    if f is None:
        return False
    profit_margins = f.get("profitMargins")
    rec_key = f.get("recommendationKey")
    return (
        profit_margins is not None and profit_margins < -0.20
        and rec_key in ("sell", "strong_sell")
    )


def rank_by_fundamentals(prescreen_tickers, fundamentals, held_tickers, count):
    """Narrows a price-based prescreen pool down to `count` tickers by
    fundamental quality score, always keeping anything currently held
    (regardless of score) so it stays sellable. Red-flag names are dropped
    unless held."""
    held = set(held_tickers)
    scored = []
    for t in prescreen_tickers:
        f = fundamentals.get(t)
        if t not in held and is_fundamental_red_flag(f):
            continue
        score = fundamentals_quality_score(f)
        scored.append((score if score is not None else float("-inf"), t))

    scored.sort(reverse=True)
    ranked = [t for _, t in scored]

    non_held_slots = max(count - len(held), 0)
    top_non_held = [t for t in ranked if t not in held][:non_held_slots]
    return list(dict.fromkeys(list(held_tickers) + top_non_held))
