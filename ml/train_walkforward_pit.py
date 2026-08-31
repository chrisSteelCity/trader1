"""
Same expanding-window walk-forward methodology as train_walkforward.py, but
on the point-in-time (PIT) dataset from build_dataset_pit.py -- sampled from
every ticker that was EVER an S&P 500 member 2000-2026 (using the free
sp500_ticker_start_end.csv membership history), with each ticker's rows
restricted to its documented membership window(s), rather than 120 survivors
sampled from today's constituent list.

This is the more honest version of the walk-forward test: ml/REPORT.md and
ml/WALKFORWARD_REPORT.md both used a universe that could only ever contain
companies that survived long enough to still be in the S&P 500 today. This
script's dataset can also include, say, a 2007 row for a company that got
acquired in 2012 and dropped out of the index -- as long as yfinance still
has SOME price history for it (see build_dataset_pit.py's docstring for what
that filters out: fully-delisted-after-bankruptcy names like Lehman Brothers
or Enron have no price history anywhere free, so they're still invisible
here -- narrower bias, not zero bias).

Still a validation exercise, not a trading strategy. See the "Honest
caveats" section this script writes to ml/WALKFORWARD_PIT_REPORT.md.
"""

import json
from pathlib import Path

import pandas as pd
import yfinance as yf
from curl_cffi import requests as _curl_requests
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

from build_dataset_pit import BUY_QUANTILE, SELL_QUANTILE

ML_DIR = Path(__file__).resolve().parent

FIRST_TEST_YEAR = 2000
MIN_TRAIN_ROWS = 5000


def _new_yf_session():
    return _curl_requests.Session(impersonate="chrome110")


def load_data():
    df = pd.read_csv(ML_DIR / "dataset_pit.csv", parse_dates=["date"])
    with open(ML_DIR / "dataset_pit_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return df, meta


def fetch_spy_yearly_returns(years):
    try:
        session = _new_yf_session()
        spy = yf.download("SPY", period="max", interval="1d", progress=False,
                           session=session, auto_adjust=True)["Close"].squeeze()
    except Exception as e:
        print(f"Could not fetch SPY yearly returns for context: {e}")
        return {y: None for y in years}
    spy.index = pd.to_datetime(spy.index)
    out = {}
    for y in years:
        year_prices = spy[spy.index.year == y]
        if len(year_prices) < 2:
            out[y] = None
            continue
        out[y] = float(year_prices.iloc[-1] / year_prices.iloc[0] - 1)
    return out


def run_walkforward(df, feature_cols):
    df = df.sort_values("date")
    last_year = int(df["date"].dt.year.max())
    years = list(range(FIRST_TEST_YEAR, last_year + 1))

    fold_rows = []
    final_model = None

    for year in years:
        train = df[df["date"].dt.year < year]
        test = df[df["date"].dt.year == year]
        if len(train) < MIN_TRAIN_ROWS or test.empty:
            print(f"  {year}: skipped (not enough training data yet)")
            continue

        buy_cut = train["excess_return"].quantile(BUY_QUANTILE)
        sell_cut = train["excess_return"].quantile(SELL_QUANTILE)

        def _label(x, buy_cut=buy_cut, sell_cut=sell_cut):
            if x >= buy_cut:
                return "BUY"
            if x <= sell_cut:
                return "SELL"
            return "HOLD"

        y_train = train["excess_return"].apply(_label)
        y_test = test["excess_return"].apply(_label)
        X_train, X_test = train[feature_cols], test[feature_cols]

        # A ticker can drop out of the index mid-training-window (acquired,
        # bankrupt, demoted) -- that's fine and expected; it just stops
        # contributing rows to later years the same way a real portfolio
        # manager would stop being able to buy it.
        n_active_tickers_test = test["ticker"].nunique()

        clf = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_depth=6, random_state=42,
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        acc = accuracy_score(y_test, preds)
        majority_class = y_train.value_counts().idxmax()
        baseline_acc = float((y_test == majority_class).mean())

        test = test.copy()
        test["pred"] = preds
        buy_rows = test[test["pred"] == "BUY"]
        sell_rows = test[test["pred"] == "SELL"]

        row = {
            "year": year,
            "train_rows": len(train),
            "test_rows": len(test),
            "test_tickers": n_active_tickers_test,
            "buy_cut": float(buy_cut),
            "sell_cut": float(sell_cut),
            "accuracy": acc,
            "baseline_accuracy": baseline_acc,
            "n_buy_calls": len(buy_rows),
            "n_sell_calls": len(sell_rows),
            "buy_excess_return": buy_rows["excess_return"].mean() if len(buy_rows) else None,
            "sell_excess_return": sell_rows["excess_return"].mean() if len(sell_rows) else None,
            "universe_excess_return": test["excess_return"].mean(),
        }
        fold_rows.append(row)
        print(f"  {year}: acc={acc:.3f} (baseline {baseline_acc:.3f}), "
              f"{n_active_tickers_test} tickers in play, {len(buy_rows)} BUY calls, "
              f"excess={row['buy_excess_return']*100 if row['buy_excess_return'] is not None else float('nan'):+.2f}pp")
        final_model = clf

    return pd.DataFrame(fold_rows), final_model


def main():
    df, meta = load_data()
    feature_cols = meta["feature_cols"]

    print("Running expanding-window walk-forward validation on the point-in-time dataset...")
    folds, final_model = run_walkforward(df, feature_cols)
    folds.to_csv(ML_DIR / "walkforward_pit_folds.csv", index=False)

    print("\nFetching SPY calendar-year returns for context...")
    spy_returns = fetch_spy_yearly_returns(folds["year"].tolist())
    folds["spy_calendar_year_return"] = folds["year"].map(spy_returns)
    folds.to_csv(ML_DIR / "walkforward_pit_folds.csv", index=False)

    write_report(folds, meta)

    import joblib
    joblib.dump({"model": final_model, "feature_cols": feature_cols, "meta": meta},
                ML_DIR / "model_walkforward_pit.joblib")
    print(f"\nSaved final (most-data) fold's model to {ML_DIR / 'model_walkforward_pit.joblib'}")


def write_report(folds, meta):
    lines = []
    lines.append("# Walk-forward validation on a point-in-time universe (2000-2026)\n")
    lines.append(
        "**Status: proof-of-concept validation exercise, not a validated trading "
        "strategy, and NOT wired into decide_trades.py or the portal.** This is "
        "the same expanding-window walk-forward as `ml/WALKFORWARD_REPORT.md`, "
        "but on a universe sampled from every company that was EVER an S&P 500 "
        "member during 2000-2026 (via the free "
        "[fja05680/sp500](https://github.com/fja05680/sp500) membership "
        "history), not just today's 120-ticker survivor sample. It answers a "
        "narrower version of the question raised when reaching back to the "
        "1980s came up: does the result hold if the universe includes "
        "companies that got removed from the index along the way, not only "
        "ones that are still in it today?\n"
    )

    lines.append("## Setup\n")
    lines.append(
        f"- Universe: {meta['universe_size_total_relevant']} tickers were EVER an "
        f"S&P 500 member between 2000 and 2026 (per the membership file); "
        f"{meta['universe_size_sampled']} were randomly sampled from that full set "
        f"(fixed seed {meta['seed']})."
    )
    lines.append(
        f"- **Residual survivorship bias, measured directly:** of those "
        f"{meta['universe_size_sampled']} sampled tickers, only "
        f"{meta['universe_size_fetched']} had usable price data from yfinance "
        f"(after an individual retry pass to rule out ordinary batch-fetch "
        f"flakiness) -- {meta['universe_size_fetch_failed']} did not. That gap is "
        f"largely genuine: companies fully delisted after bankruptcy or an "
        f"acquisition (e.g. Lehman-, Enron-, WorldCom-style outcomes) have no "
        f"price history available through any free source, so this pipeline still "
        f"can't see them, even though the membership list itself now includes "
        f"them. **One caveat on that number**: spot-checking one \"failed\" ticker "
        f"(AVB -- AvalonBay Communities, which is very much still trading) found "
        f"this sandbox's yfinance access returned only ~27 rows of genuinely thin "
        f"data for it rather than a clean failure, so a small unknown fraction of "
        f"the {meta['universe_size_fetch_failed']} figure may be ordinary data-access "
        f"limitations in this environment rather than true unavailability -- the "
        f"true delisted-company count is likely close to but somewhat below "
        f"{meta['universe_size_fetch_failed']}. This wasn't fully diagnosed further; "
        f"a paid vendor with dedicated delisted-security coverage (Norgate, EODHD, "
        f"Sharadar) would remove this ambiguity entirely."
    )
    lines.append(
        "- Each surviving ticker's rows are further restricted to its documented "
        "S&P 500 membership window(s) -- a name that left the index in 2012 stops "
        "contributing test rows in 2013 onward, the same way it would stop being "
        "eligible for an index-tracking strategy in real life."
    )
    lines.append(
        f"- Labeled dataset: 10-trading-day forward excess return vs. SPY, "
        f"thresholded into BUY/HOLD/SELL by quantile (top 30% BUY, bottom 30% "
        f"SELL), recomputed per fold from only that fold's training data (same "
        f"leak-avoidance as `ml/WALKFORWARD_REPORT.md`)."
    )
    lines.append(
        "- Validation: expanding-window walk-forward, one fresh "
        "`HistGradientBoostingClassifier` per calendar year from "
        f"{int(folds['year'].min())} onward, trained on everything before that "
        "year and tested only on it.\n"
    )

    lines.append("## Results by year\n")
    lines.append("\"Tickers in play\" is how many distinct point-in-time-eligible "
                 "tickers actually had a row in that year's test set -- it shrinks "
                 "in later years as more of the sampled universe has already been "
                 "acquired, gone bankrupt, or dropped from the index.\n")
    lines.append("| Year | SPY return | Tickers in play | Train rows | Accuracy | "
                 "Baseline acc. | BUY calls | BUY excess return | SELL calls | SELL excess return |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in folds.iterrows():
        spy_ret = (f"{r['spy_calendar_year_return']*100:+.1f}%"
                   if pd.notna(r["spy_calendar_year_return"]) else "n/a")
        buy_ex = (f"{r['buy_excess_return']*100:+.2f} pp"
                  if pd.notna(r["buy_excess_return"]) else "n/a")
        sell_ex = (f"{r['sell_excess_return']*100:+.2f} pp"
                   if pd.notna(r["sell_excess_return"]) else "n/a")
        lines.append(f"| {int(r['year'])} | {spy_ret} | {int(r['test_tickers'])} | "
                     f"{int(r['train_rows'])} | {r['accuracy']:.3f} | {r['baseline_accuracy']:.3f} | "
                     f"{int(r['n_buy_calls'])} | {buy_ex} | {int(r['n_sell_calls'])} | {sell_ex} |")
    lines.append("")

    n_folds = len(folds)
    beat_baseline = int((folds["accuracy"] > folds["baseline_accuracy"]).sum())
    buy_valid = folds["buy_excess_return"].notna()
    buy_beat_spy = int((folds.loc[buy_valid, "buy_excess_return"] > 0).sum())
    n_buy_valid = int(buy_valid.sum())
    mean_buy_excess = folds.loc[buy_valid, "buy_excess_return"].mean()
    median_buy_excess = folds.loc[buy_valid, "buy_excess_return"].median()

    down_years = folds[folds["spy_calendar_year_return"] < 0]

    lines.append("## Summary\n")
    lines.append(f"- {n_folds} yearly folds tested "
                 f"({int(folds['year'].min())}-{int(folds['year'].max())}).")
    lines.append(f"- Beat the majority-class baseline accuracy in {beat_baseline}/{n_folds} years.")
    lines.append(f"- BUY calls' excess return vs SPY was positive in {buy_beat_spy}/{n_buy_valid} "
                 f"years with at least one BUY call (mean {mean_buy_excess*100:+.2f} pp, "
                 f"median {median_buy_excess*100:+.2f} pp).")
    if not down_years.empty:
        down_valid = down_years["buy_excess_return"].notna()
        down_buy_beat = int((down_years.loc[down_valid, "buy_excess_return"] > 0).sum())
        n_down_valid = int(down_valid.sum())
        lines.append(f"- **In years SPY itself was down** "
                     f"({', '.join(str(int(y)) for y in down_years['year'])}), BUY calls' "
                     f"excess return was positive in {down_buy_beat}/{n_down_valid} of them.")
    lines.append("")

    lines.append("## Compared to the non-point-in-time version\n")
    lines.append(
        "See `ml/WALKFORWARD_REPORT.md` for the 120-ticker, today's-S&P-500-only "
        "version of this same walk-forward test. Comparing the two: if this "
        "point-in-time version's numbers are similar, that's modest evidence the "
        "earlier result wasn't purely a survivorship-bias artifact. If they "
        "diverge meaningfully, the difference is itself informative about how "
        "much the original sampling choice was driving the result -- read both "
        "reports side by side rather than trusting either one alone.\n"
    )

    lines.append("## Honest caveats\n")
    lines.append(
        "- **Bias narrowed, not eliminated.** This still can't see companies "
        "fully delisted after bankruptcy or an acquisition -- see the Setup "
        "section's residual-bias number. A paid vendor with dedicated delisted-"
        "security coverage (Norgate, EODHD, Sharadar, or CRSP/WRDS for "
        "academics) is the only way to close that gap completely; there's no "
        "free source for it.\n"
        "- **The membership data itself has known limits.** It's a "
        "community-maintained export of Wikipedia's crowd-edited S&P 500 "
        "change log -- solid for retail backtesting, not audit-grade, and with "
        "acknowledged gaps in the pre-2000s portion (not relevant here since "
        "this test starts at 2000, but worth knowing if extended further back).\n"
        "- **A handful of \"unavailable\" tickers may be ordinary data-access "
        "limitations in this sandbox, not real delistings** -- see the AVB "
        "example in Setup. Not separately diagnosed further.\n"
        "- **No fundamentals, no transaction costs, no position sizing, "
        "untuned hyperparameters and quantile choice** -- same limits as both "
        "earlier reports.\n"
        "- **Not integrated.** `decide_trades.py` and the portal are unchanged.\n"
    )

    with open(ML_DIR / "WALKFORWARD_PIT_REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {ML_DIR / 'WALKFORWARD_PIT_REPORT.md'}")


if __name__ == "__main__":
    main()
