"""
Walk-forward validation of the buy/hold/sell classifier across ~25 years of
history, made possible by widening build_dataset.py to fetch each ticker's
full available price history (some reaching into the 1960s-80s for old
names) rather than a fixed 3y/10y window. SPY itself only launches in 1993,
so labeled rows still start there -- see build_dataset.py's docstring for
that and the survivorship-bias caveat this approach can't avoid.

Unlike train_model.py's single chronological train/test split, this refits
the model once per calendar year in an EXPANDING window (train on everything
before year Y, test on year Y only, then move to Y+1 with Y's data folded
into training), so you can see whether performance holds up consistently
across genuinely different market regimes -- the 2000-02 dot-com bust, the
2008 financial crisis, the 2020 COVID crash, the 2022 bear market -- rather
than just the single most-recent stretch that train_model.py's split
happened to land on.

This is a validation exercise, not a trading strategy. See the "Honest
caveats" section this script writes to ml/WALKFORWARD_REPORT.md.
"""

import json
from pathlib import Path

import pandas as pd
import yfinance as yf
from curl_cffi import requests as _curl_requests
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

from build_dataset import BUY_QUANTILE, SELL_QUANTILE

ML_DIR = Path(__file__).resolve().parent

FIRST_TEST_YEAR = 2000   # everything before this is the initial training window
MIN_TRAIN_ROWS = 5000    # skip a fold if there isn't enough training data yet

# dataset.csv's "label" column was thresholded using quantiles computed over
# the WHOLE 1993-2026 dataset -- which means even the year-2000 fold would be
# using a label definition that "knows" about the full-period return
# distribution, decades into that fold's future. That's a real leak for a
# walk-forward test, not just a data-snooping technicality: it could inflate
# how consistent the model looks across regimes. So each fold below
# recomputes its own BUY/SELL cutoffs from ONLY that fold's training data
# (same quantiles, same idea as build_dataset.py, just refit fold-by-fold)
# and re-labels both train and test with those training-only cutoffs.


def _new_yf_session():
    return _curl_requests.Session(impersonate="chrome110")


def load_data():
    df = pd.read_csv(ML_DIR / "dataset.csv", parse_dates=["date"])
    with open(ML_DIR / "dataset_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return df, meta


def fetch_spy_yearly_returns(years):
    """Actual calendar-year SPY total return for each tested year -- purely
    for regime context in the report table, not used by the model itself."""
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

        # Fold-specific, training-only label thresholds -- see the module
        # docstring note above on why this matters for a walk-forward test.
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
              f"{len(buy_rows)} BUY calls, "
              f"excess={row['buy_excess_return']*100 if row['buy_excess_return'] is not None else float('nan'):+.2f}pp")
        final_model = clf  # the last (most-data) fit stands in as "the current model"

    return pd.DataFrame(fold_rows), final_model


def main():
    df, meta = load_data()
    feature_cols = meta["feature_cols"]

    print("Running expanding-window walk-forward validation...")
    folds, final_model = run_walkforward(df, feature_cols)
    folds.to_csv(ML_DIR / "walkforward_folds.csv", index=False)  # checkpoint before the SPY fetch

    print("\nFetching SPY calendar-year returns for context...")
    spy_returns = fetch_spy_yearly_returns(folds["year"].tolist())
    folds["spy_calendar_year_return"] = folds["year"].map(spy_returns)

    folds.to_csv(ML_DIR / "walkforward_folds.csv", index=False)
    write_report(folds, meta)

    import joblib
    joblib.dump({"model": final_model, "feature_cols": feature_cols, "meta": meta},
                ML_DIR / "model_walkforward.joblib")
    print(f"\nSaved final (most-data) fold's model to {ML_DIR / 'model_walkforward.joblib'}")


def write_report(folds, meta):
    lines = []
    lines.append("# Walk-forward validation: buy/hold/sell classifier across ~25 years\n")
    lines.append(
        "**Status: proof-of-concept validation exercise, not a validated trading "
        "strategy, and NOT wired into decide_trades.py or the portal.** This "
        "answers the question the single-split report (`ml/REPORT.md`) couldn't: "
        "does the model hold up across genuinely different market regimes, or "
        "did it just get lucky on one recent stretch?\n"
    )

    lines.append("## Setup\n")
    lines.append(f"- Universe: {meta['universe_size']} S&P 500 tickers, randomly sampled "
                 f"(fixed seed {meta['seed']}).")
    lines.append(
        "- History: each ticker's full available price history from yfinance "
        "(`period=\"max\"`) -- for old names this reaches into the 1960s-80s, for "
        "others only back to their IPO. See `ml/build_dataset.py`'s docstring for "
        "why labeled rows still only start in 1993 (SPY's inception) and for the "
        "survivorship-bias caveat this approach can't avoid."
    )
    lines.append(f"- Labeled dataset: 1993-01-29 through {int(folds['year'].max())}, "
                 f"{meta['label_horizon_days']}-day forward excess return vs. SPY, thresholded "
                 f"into BUY/HOLD/SELL by quantile (top 30% BUY, bottom 30% SELL -- same shape "
                 f"as `ml/REPORT.md`'s single-split version). **Unlike that version, "
                 f"the cutoffs are recomputed per fold from ONLY that fold's training data** "
                 f"(see the table's buy/sell cutoff columns) rather than fixed once from the "
                 f"full 1993-2026 distribution -- using a global cutoff would let the year-2000 "
                 f"fold's labels implicitly know about decades of future return data, which "
                 f"would be a real leak for a walk-forward test.")
    lines.append(
        "- Validation: **expanding-window walk-forward** -- for each calendar year Y "
        f"starting {int(folds['year'].min())}, train a fresh "
        "`HistGradientBoostingClassifier` on every row before Y, test only on year "
        "Y, then move to Y+1 and retrain with Y's data folded in. This is closer to "
        "how you'd actually deploy and periodically retrain a model than a single "
        "train/test split, and it surfaces whether performance holds up across "
        "different market conditions instead of just the most recent one.\n"
    )

    lines.append("## Results by year\n")
    lines.append("Buy/sell cutoffs are each fold's own training-only quantile thresholds "
                 "(see Setup) -- they drift over time as more history (and more volatile "
                 "periods) enters the training window.\n")
    lines.append("| Year | SPY return | Train rows | BUY cutoff | SELL cutoff | Accuracy | "
                 "Baseline acc. | BUY calls | BUY excess return | SELL calls | SELL excess return |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in folds.iterrows():
        spy_ret = (f"{r['spy_calendar_year_return']*100:+.1f}%"
                   if pd.notna(r["spy_calendar_year_return"]) else "n/a")
        buy_ex = (f"{r['buy_excess_return']*100:+.2f} pp"
                  if pd.notna(r["buy_excess_return"]) else "n/a")
        sell_ex = (f"{r['sell_excess_return']*100:+.2f} pp"
                   if pd.notna(r["sell_excess_return"]) else "n/a")
        lines.append(f"| {int(r['year'])} | {spy_ret} | {int(r['train_rows'])} | "
                     f"{r['buy_cut']*100:+.2f}% | {r['sell_cut']*100:+.2f}% | "
                     f"{r['accuracy']:.3f} | {r['baseline_accuracy']:.3f} | "
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
                 f"({int(folds['year'].min())}-{int(folds['year'].max())}; the final year "
                 f"may be partial, through whenever the dataset was built).")
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
                     f"excess return was positive in {down_buy_beat}/{n_down_valid} of them "
                     f"-- the case that actually matters for whether this generalizes beyond "
                     f"one bull-market stretch.")
    lines.append("")

    lines.append("## Honest caveats\n")
    lines.append(
        "- **Survivorship bias.** The universe is sampled from TODAY's S&P 500 list, so "
        "this can only ever look at companies that survived long enough to still be in "
        "the index now -- it can't see delistings, bankruptcies, or acquisitions along "
        "the way. Free data sources don't offer point-in-time historical index "
        "membership, so this bias is structural to this whole approach, not a bug to "
        "fix.\n"
        "- **SPY only exists from 1993.** Several tickers' own price history reaches "
        "earlier (their features can use it for lookback), but no labeled row exists "
        "before 1993 -- there's no benchmark to compute an excess return against.\n"
        "- **Different market structure across the decades.** Pre-2001 markets traded "
        "in eighths/sixteenths rather than decimals, had far less algorithmic trading, "
        "and a very different sector mix (much less large-cap tech). Whether patterns "
        "learned from the 1990s and 2000s transfer to today's market is an open "
        "question this validation doesn't resolve, even though it's a more honest test "
        "than a single recent split.\n"
        "- **No fundamentals, no transaction costs, no position sizing** -- same limits "
        "as `ml/REPORT.md`'s single-split version; this is a classification/"
        "excess-return sanity check across time, not a backtest of a tradeable "
        "strategy.\n"
        "- **Fixed hyperparameters, not tuned per fold or at all** -- refitting per "
        "year keeps this a fair walk-forward test rather than implicitly hand-fitting "
        "to any one period, but it also means no fold got a chance to use better "
        "settings for its particular market conditions.\n"
        "- **The BUY/SELL quantile thresholds are still an in-sample choice, just no "
        "longer a future-leaking one.** Each fold recomputes its cutoffs from only its "
        "own training data (fixing the more serious global-threshold leak an earlier "
        "version of this script had -- see Setup), but *which* quantiles to use (top/"
        "bottom 30%) was still picked once, by inspection, rather than validated. A "
        "more rigorous version would treat that as a hyperparameter tuned on a held-out "
        "slice of training data, not fixed by the person building the pipeline.\n"
        "- **Not integrated.** `decide_trades.py` and the portal are unchanged; this "
        "is not used to make any real (paper) trades.\n"
    )

    with open(ML_DIR / "WALKFORWARD_REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {ML_DIR / 'WALKFORWARD_REPORT.md'}")


if __name__ == "__main__":
    main()
