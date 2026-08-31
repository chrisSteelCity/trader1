"""
Trains the buy/hold/sell proof-of-concept classifier on ml/dataset.csv and
produces an honest evaluation: classification metrics on a held-out
CHRONOLOGICAL test period (not a random split -- random splits leak future
information into training via near-duplicate overlapping-window rows), a
permutation feature-importance ranking, and a simple simulated-trading
comparison against buy-and-hold baselines.

This is a proof-of-concept, not a validated trading strategy. See
ml/REPORT.md (written by this script) for the honest caveats section.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix

ML_DIR = Path(__file__).resolve().parent

TEST_FRACTION = 0.25  # most recent 25% of dates held out, chronologically


def load_data():
    df = pd.read_csv(ML_DIR / "dataset.csv", parse_dates=["date"])
    with open(ML_DIR / "dataset_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return df, meta


def chronological_split(df, test_fraction):
    """Splits by DATE, not by row, so no ticker has a train-period date after
    a test-period date and no single date straddles the boundary."""
    dates = sorted(df["date"].unique())
    cutoff_idx = int(len(dates) * (1 - test_fraction))
    cutoff_date = dates[cutoff_idx]
    train = df[df["date"] < cutoff_date].copy()
    test = df[df["date"] >= cutoff_date].copy()
    return train, test, cutoff_date


def simulate_trading(test_df, preds):
    """Very simple simulated strategy for comparison purposes only:
    each date, take an equal-weighted position in every ticker the model
    predicts BUY, held for the label horizon, and compare the average
    realized forward return of those picks against (a) holding SPY over
    the same dates and (b) equal-weighting the WHOLE test-set candidate
    pool every date (naive diversification baseline).

    This is not a realistic backtest (no transaction costs, no position
    sizing/risk limits, overlapping holding windows are treated
    independently) -- it's a sanity check on whether the model's BUY calls
    actually outperformed on average, not a tradeable strategy result."""
    test_df = test_df.copy()
    test_df["pred"] = preds

    buy_rows = test_df[test_df["pred"] == "BUY"]
    sell_rows = test_df[test_df["pred"] == "SELL"]

    model_buy_avg_fwd = buy_rows["fwd_return"].mean() if len(buy_rows) else float("nan")
    model_sell_avg_fwd = sell_rows["fwd_return"].mean() if len(sell_rows) else float("nan")
    universe_avg_fwd = test_df["fwd_return"].mean()
    bench_avg_fwd = test_df["bench_fwd_return"].mean()

    model_buy_avg_excess = buy_rows["excess_return"].mean() if len(buy_rows) else float("nan")
    universe_avg_excess = test_df["excess_return"].mean()

    return {
        "n_test_rows": len(test_df),
        "n_buy_calls": len(buy_rows),
        "n_sell_calls": len(sell_rows),
        "avg_fwd_return_buy_calls": model_buy_avg_fwd,
        "avg_fwd_return_sell_calls": model_sell_avg_fwd,
        "avg_fwd_return_whole_test_universe": universe_avg_fwd,
        "avg_fwd_return_benchmark_spy": bench_avg_fwd,
        "avg_excess_return_buy_calls": model_buy_avg_excess,
        "avg_excess_return_whole_test_universe": universe_avg_excess,
    }


def main():
    df, meta = load_data()
    feature_cols = meta["feature_cols"]

    train, test, cutoff_date = chronological_split(df, TEST_FRACTION)
    print(f"Chronological split at {cutoff_date}: {len(train)} train rows, {len(test)} test rows")
    print(f"Train dates: {train['date'].min()} to {train['date'].max()}")
    print(f"Test dates:  {test['date'].min()} to {test['date'].max()}")

    X_train, y_train = train[feature_cols], train["label"]
    X_test, y_test = test[feature_cols], test["label"]

    print("\nTraining HistGradientBoostingClassifier...")
    clf = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=6, random_state=42,
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    report_text = classification_report(y_test, preds)
    print("\nClassification report (test set):")
    print(report_text)

    labels_order = ["SELL", "HOLD", "BUY"]
    cm = confusion_matrix(y_test, preds, labels=labels_order)
    print("Confusion matrix (rows=true, cols=predicted), order SELL/HOLD/BUY:")
    print(cm)

    print("\nComputing permutation feature importance (this takes a bit)...")
    perm = permutation_importance(
        clf, X_test, y_test, n_repeats=8, random_state=42, n_jobs=-1,
    )
    importances = sorted(
        zip(feature_cols, perm.importances_mean, perm.importances_std),
        key=lambda x: x[1], reverse=True,
    )
    print("Permutation importance (mean drop in accuracy when shuffled):")
    for name, mean_imp, std_imp in importances:
        print(f"  {name:16s} {mean_imp:+.4f} (+/- {std_imp:.4f})")

    sim = simulate_trading(test, preds)
    print("\nSimulated trading comparison (see caveats in ml/REPORT.md):")
    for k, v in sim.items():
        print(f"  {k}: {v}")

    # A naive "always predict the majority class" baseline, to sanity-check
    # that the model is doing better than doing nothing.
    majority_class = y_train.value_counts().idxmax()
    baseline_preds = pd.Series([majority_class] * len(y_test))
    baseline_report = classification_report(y_test, baseline_preds, output_dict=True, zero_division=0)

    write_report(meta, cutoff_date, train, test, report, report_text, cm, labels_order,
                 importances, sim, majority_class, baseline_report)

    import joblib
    joblib.dump({"model": clf, "feature_cols": feature_cols, "meta": meta}, ML_DIR / "model.joblib")
    print(f"\nSaved model to {ML_DIR / 'model.joblib'}")


def write_report(meta, cutoff_date, train, test, report, report_text, cm, labels_order,
                  importances, sim, majority_class, baseline_report):
    lines = []
    lines.append("# ML proof-of-concept: buy/hold/sell classifier\n")
    lines.append(
        "**Status: proof-of-concept only. Not a validated trading strategy, "
        "and NOT wired into decide_trades.py or the portal.** This report is "
        "meant to honestly show what a first pass at a real model looks like, "
        "including where it falls short.\n"
    )

    lines.append("## Setup\n")
    lines.append(f"- Universe: {meta['universe_size']} S&P 500 tickers, randomly sampled "
                  f"(fixed seed {meta['seed']}) -- not the full index, kept small for a proof-of-concept.")
    lines.append(f"- History: {meta['history_period']} of daily prices per ticker.")
    lines.append(f"- Features: price/volume technicals only (returns, moving-average ratios, "
                  f"RSI, volatility, distance from 52-week high, volume ratio) -- "
                  f"**deliberately no fundamentals**, because yfinance only exposes a "
                  f"company's CURRENT fundamentals, not what they were as of a past date, "
                  f"so using them here would leak future information into historical rows.")
    lines.append(f"- Label: {meta['label_horizon_days']}-trading-day forward return vs. "
                  f"{meta['benchmark']} (excess return), split into 3 classes by fixed "
                  f"quantile cutoffs computed once over the whole dataset "
                  f"(BUY >= {meta['buy_cut']:.4f}, SELL <= {meta['sell_cut']:.4f}, else HOLD).")
    lines.append(f"- Model: scikit-learn `HistGradientBoostingClassifier` "
                  f"(xgboost/lightgbm aren't installed in this environment).")
    lines.append(f"- Split: **chronological**, not random -- train on data before "
                  f"{cutoff_date.date()}, test strictly after. A random split would let the "
                  f"model see overlapping-window rows from dates adjacent to test rows, "
                  f"inflating accuracy in a way that wouldn't hold up on genuinely new data.")
    lines.append(f"- Train rows: {len(train)}. Test rows: {len(test)}.\n")

    lines.append("## Classification performance (test set, held out chronologically)\n")
    lines.append("| Class | Precision | Recall | F1 | Support |")
    lines.append("|---|---:|---:|---:|---:|")
    for label in labels_order:
        r = report.get(label, {})
        lines.append(f"| {label} | {r.get('precision', 0):.3f} | {r.get('recall', 0):.3f} | "
                      f"{r.get('f1-score', 0):.3f} | {int(r.get('support', 0))} |")
    lines.append(f"| **Overall accuracy** | | | **{report['accuracy']:.3f}** | {int(report['macro avg']['support'])} |\n")

    baseline_acc = baseline_report.get("accuracy", 0)
    lines.append(f"**Baseline check:** always predicting the majority class from the training "
                 f"set (`{majority_class}`) would score {baseline_acc:.3f} accuracy on the same "
                 f"test set. The model's accuracy above should be compared against this, not "
                 f"against 100% -- a 3-way classification problem where classes are roughly "
                 f"balanced has a much lower \"do nothing\" floor than a skewed one.\n")

    lines.append("Confusion matrix (rows = actual, columns = predicted; order SELL / HOLD / BUY):\n")
    lines.append("```")
    lines.append("           pred:SELL  pred:HOLD  pred:BUY")
    for i, label in enumerate(labels_order):
        lines.append(f"actual:{label:<5} " + "  ".join(f"{cm[i][j]:9d}" for j in range(3)))
    lines.append("```\n")

    lines.append("## Feature importance (permutation importance on test set)\n")
    lines.append("Mean drop in accuracy when a feature's values are randomly shuffled -- "
                  "bigger means the model relies on it more. Negative values mean shuffling "
                  "that feature didn't hurt (or slightly helped) accuracy on this test set.\n")
    lines.append("| Feature | Importance | Std dev |")
    lines.append("|---|---:|---:|")
    for name, mean_imp, std_imp in importances:
        lines.append(f"| {name} | {mean_imp:+.4f} | {std_imp:.4f} |")
    lines.append("")

    lines.append("## Simulated trading comparison (NOT a backtest -- see caveats)\n")
    lines.append(f"- Test-period BUY calls: {sim['n_buy_calls']} (out of {sim['n_test_rows']} rows)")
    lines.append(f"- Test-period SELL calls: {sim['n_sell_calls']}")
    lines.append(f"- Avg {meta['label_horizon_days']}-day forward return of BUY calls: "
                 f"{sim['avg_fwd_return_buy_calls']*100:+.2f}%")
    lines.append(f"- Avg {meta['label_horizon_days']}-day forward return of SELL calls: "
                 f"{sim['avg_fwd_return_sell_calls']*100:+.2f}%")
    lines.append(f"- Avg {meta['label_horizon_days']}-day forward return, whole test universe "
                 f"(naive equal-weight everything baseline): {sim['avg_fwd_return_whole_test_universe']*100:+.2f}%")
    lines.append(f"- Avg {meta['label_horizon_days']}-day {meta['benchmark']} return over the same "
                 f"windows: {sim['avg_fwd_return_benchmark_spy']*100:+.2f}%")
    lines.append(f"- Avg excess return (vs {meta['benchmark']}) of BUY calls: "
                 f"{sim['avg_excess_return_buy_calls']*100:+.2f} pp")
    lines.append(f"- Avg excess return (vs {meta['benchmark']}), whole test universe: "
                 f"{sim['avg_excess_return_whole_test_universe']*100:+.2f} pp\n")

    lines.append("## Honest caveats\n")
    lines.append(
        "This is a small proof-of-concept, not a strategy ready for real (even paper) money:\n\n"
        "- **No fundamentals.** Excluded specifically to avoid lookahead bias (see Setup). "
        "A production version would need a point-in-time fundamentals data source, which "
        "isn't available for free.\n"
        "- **No transaction costs, slippage, or position sizing** in the simulated trading "
        "comparison above -- it only checks whether BUY-labeled rows had better-than-average "
        "forward returns, not whether trading on them would be profitable after costs.\n"
        "- **Overlapping labels.** Every ticker contributes a row for nearly every trading day, "
        "and consecutive days' forward-return windows overlap heavily, so test rows are not "
        "independent -- effective sample size is much smaller than the row count suggests.\n"
        "- **One time period.** The test window covers whatever the last ~25% of the fetched "
        "history happens to contain (one particular stretch of market conditions), not "
        "multiple market regimes (e.g. no 2022-style bear market in this sample window).\n"
        "- **Small, randomly-sampled universe** (see Setup) rather than the full S&P 500 or a "
        "broader market -- results may not generalize to tickers outside the sample.\n"
        "- **Not integrated.** decide_trades.py and the portal are unchanged; this model is "
        "not used to make any real (paper) trades yet.\n"
    )

    with open(ML_DIR / "REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {ML_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
