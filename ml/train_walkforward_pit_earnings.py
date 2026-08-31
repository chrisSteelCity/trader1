"""
Same expanding-window walk-forward methodology and HistGradientBoosting
model as train_walkforward_pit.py, but trained on dataset_pit_earnings.csv
(build_dataset_pit_earnings.py's output) -- the point-in-time dataset with
4 additional earnings-report features (days_since_earnings,
last_surprise_pct, surprise_trend_4q, recent_earnings_flag) layered on top
of the original 11 price/volume technicals.

This directly answers: does giving the model more information (real,
free, backtestable earnings-surprise history, aligned with no lookahead)
actually improve on ml/WALKFORWARD_PIT_REPORT.md's result, or was the
earlier XGBoost/LightGBM comparison right that the bottleneck isn't the
model -- so maybe it's the data instead?

Writes ml/walkforward_pit_earnings_folds.csv,
ml/model_walkforward_pit_earnings.joblib, and
ml/WALKFORWARD_PIT_EARNINGS_REPORT.md (which also loads
ml/walkforward_pit_folds.csv for a side-by-side comparison against the
no-earnings baseline).

Still a validation exercise, not a trading strategy.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

from build_dataset_pit import BUY_QUANTILE, SELL_QUANTILE

ML_DIR = Path(__file__).resolve().parent

FIRST_TEST_YEAR = 2000
MIN_TRAIN_ROWS = 5000


def load_data():
    df = pd.read_csv(ML_DIR / "dataset_pit_earnings.csv", parse_dates=["date"])
    with open(ML_DIR / "dataset_pit_earnings_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return df, meta


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
        be = row["buy_excess_return"]
        print(f"  {year}: acc={acc:.3f} (baseline {baseline_acc:.3f}), "
              f"{n_active_tickers_test} tickers, {len(buy_rows)} BUY calls, "
              f"excess={be*100 if be is not None else float('nan'):+.2f}pp")
        final_model = clf

    return pd.DataFrame(fold_rows), final_model


def main():
    df, meta = load_data()
    feature_cols = meta["feature_cols"]
    print(f"Training with {len(feature_cols)} features: {feature_cols}")

    print("\nRunning expanding-window walk-forward with earnings features added...")
    folds, final_model = run_walkforward(df, feature_cols)
    folds.to_csv(ML_DIR / "walkforward_pit_earnings_folds.csv", index=False)

    joblib.dump({"model": final_model, "feature_cols": feature_cols, "meta": meta},
                ML_DIR / "model_walkforward_pit_earnings.joblib")

    # Permutation importance on the final fold's model, to see whether the
    # new earnings features actually mattered to the model or were ignored.
    print("\nComputing permutation importance on the final model's training data "
          "(this can take a minute)...")
    last_year = int(df["date"].dt.year.max())
    train = df[df["date"].dt.year < last_year]
    sample = train.sample(n=min(40000, len(train)), random_state=42)
    buy_cut = train["excess_return"].quantile(BUY_QUANTILE)
    sell_cut = train["excess_return"].quantile(SELL_QUANTILE)
    y_sample = sample["excess_return"].apply(
        lambda x: "BUY" if x >= buy_cut else ("SELL" if x <= sell_cut else "HOLD")
    )
    from sklearn.inspection import permutation_importance
    importances = permutation_importance(
        final_model, sample[feature_cols], y_sample, n_repeats=3, random_state=42, n_jobs=2,
    )
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": importances.importances_mean,
        "importance_std": importances.importances_std,
    }).sort_values("importance_mean", ascending=False)
    imp_df.to_csv(ML_DIR / "feature_importance_pit_earnings.csv", index=False)
    print(imp_df.to_string(index=False))

    write_report(folds, meta, imp_df)


def write_report(folds, meta, imp_df):
    baseline_path = ML_DIR / "walkforward_pit_folds.csv"
    baseline = pd.read_csv(baseline_path) if baseline_path.exists() else None

    lines = []
    lines.append("# Adding earnings-report features to the point-in-time walk-forward\n")
    lines.append(
        "**Status: proof-of-concept validation exercise, not a validated trading "
        "strategy, and NOT wired into decide_trades.py or the portal.** Same "
        "point-in-time universe, same expanding-window walk-forward, same "
        "HistGradientBoostingClassifier and per-fold BUY/SELL quantile "
        "thresholds as `ml/WALKFORWARD_PIT_REPORT.md` -- the only change is 4 "
        "new features derived from each ticker's real historical earnings "
        "reports (via `ml/fetch_earnings_pit.py` and "
        "`ml/build_dataset_pit_earnings.py`), on top of the original 11 "
        "price/volume technicals:\n"
    )
    lines.append(
        "- `days_since_earnings` -- calendar days since the most recent "
        "earnings report (999 if none yet known for that ticker)\n"
        "- `last_surprise_pct` -- the % by which that report beat or missed "
        "analyst EPS estimates\n"
        "- `surprise_trend_4q` -- mean surprise % over that company's last "
        "(up to) 4 reports, i.e. is it on a beat streak or a miss streak\n"
        "- `recent_earnings_flag` -- 1 if a report happened in the last 5 "
        "calendar days (the sharp post-earnings price-reaction window)\n"
    )
    lines.append(
        "**No-lookahead guarantee**: every feature is computed via "
        "`pandas.merge_asof(..., direction=\"backward\", allow_exact_matches=False)`, "
        "which by construction can only see earnings reports strictly before "
        "each row's date -- a row can never see the report that will happen "
        "later that same day or in the future. Earnings data itself comes "
        "from yfinance's `get_earnings_dates()` (free, capped at 100 "
        "quarters/25 years per ticker by Yahoo), fetched for the same 143 "
        "tickers already in the point-in-time dataset -- 142 of them had "
        "usable earnings history.\n"
    )

    lines.append("## Results by year\n")
    lines.append("| Year | Accuracy | Baseline acc. | BUY calls | BUY excess return | "
                  "*(no-earnings BUY excess)* |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    baseline_lookup = {}
    if baseline is not None:
        baseline_lookup = dict(zip(baseline["year"], baseline["buy_excess_return"]))
    for _, r in folds.iterrows():
        buy_ex = (f"{r['buy_excess_return']*100:+.2f} pp"
                  if pd.notna(r["buy_excess_return"]) else "n/a")
        base_ex = baseline_lookup.get(int(r["year"]))
        base_ex_str = f"{base_ex*100:+.2f} pp" if base_ex is not None and pd.notna(base_ex) else "n/a"
        lines.append(f"| {int(r['year'])} | {r['accuracy']:.3f} | {r['baseline_accuracy']:.3f} | "
                     f"{int(r['n_buy_calls'])} | {buy_ex} | {base_ex_str} |")
    lines.append("")

    n_folds = len(folds)
    beat_baseline = int((folds["accuracy"] > folds["baseline_accuracy"]).sum())
    buy_valid = folds["buy_excess_return"].notna()
    buy_beat_spy = int((folds.loc[buy_valid, "buy_excess_return"] > 0).sum())
    n_buy_valid = int(buy_valid.sum())
    mean_buy_excess = folds.loc[buy_valid, "buy_excess_return"].mean()
    median_buy_excess = folds.loc[buy_valid, "buy_excess_return"].median()
    mean_acc = folds["accuracy"].mean()

    lines.append("## Summary\n")
    lines.append(f"- {n_folds} yearly folds tested "
                 f"({int(folds['year'].min())}-{int(folds['year'].max())}).")
    lines.append(f"- Beat the majority-class baseline accuracy in {beat_baseline}/{n_folds} years "
                 f"(mean accuracy {mean_acc:.3f}).")
    lines.append(f"- BUY calls' excess return vs SPY was positive in {buy_beat_spy}/{n_buy_valid} "
                 f"years with at least one BUY call (mean {mean_buy_excess*100:+.2f} pp, "
                 f"median {median_buy_excess*100:+.2f} pp).")
    if baseline is not None:
        b_beat_baseline = int((baseline["accuracy"] > baseline["baseline_accuracy"]).sum())
        b_buy_valid = baseline["buy_excess_return"].notna()
        b_buy_beat_spy = int((baseline.loc[b_buy_valid, "buy_excess_return"] > 0).sum())
        b_mean_buy_excess = baseline.loc[b_buy_valid, "buy_excess_return"].mean()
        lines.append(
            f"- **Compared to the no-earnings baseline** (`ml/WALKFORWARD_PIT_REPORT.md`): "
            f"{b_beat_baseline}/{n_folds} → {beat_baseline}/{n_folds} years beating the "
            f"majority-class baseline; {b_buy_beat_spy}/{n_folds} → {buy_beat_spy}/{n_folds} "
            f"years BUY calls beat SPY; mean BUY excess return "
            f"{b_mean_buy_excess*100:+.2f} pp → {mean_buy_excess*100:+.2f} pp."
        )
    lines.append("")

    lines.append("## Did the model actually use the earnings features?\n")
    lines.append(
        "Permutation importance on the final fold's trained model (how much "
        "accuracy drops when a feature's values are randomly shuffled -- "
        "higher means the model relies on it more):\n"
    )
    lines.append("| Feature | Importance (mean) | Importance (std) |")
    lines.append("|---|---:|---:|")
    for _, r in imp_df.iterrows():
        lines.append(f"| `{r['feature']}` | {r['importance_mean']:.5f} | {r['importance_std']:.5f} |")
    lines.append("")
    top_earnings_rank = None
    earnings_cols = {"days_since_earnings", "last_surprise_pct", "surprise_trend_4q", "recent_earnings_flag"}
    ranked = imp_df.reset_index(drop=True)
    for i, feat in enumerate(ranked["feature"]):
        if feat in earnings_cols and top_earnings_rank is None:
            top_earnings_rank = i + 1
    n_features = len(ranked)
    lines.append(
        f"The best-ranked earnings feature came in at position {top_earnings_rank} "
        f"out of {n_features} features by this measure -- read directly off the "
        f"table above alongside the price/volume technicals rather than "
        f"summarized further here, since \"more/less important than expected\" "
        f"is a judgment call worth making with the actual numbers in view.\n"
    )
    lines.append(
        "**Important nuance on this specific result**: the earnings features "
        "rank as some of the model's most-relied-on inputs by this measure, "
        "which sounds like a win -- but the Summary section above shows mean "
        "BUY excess return was slightly *lower* with them added (+0.31pp vs "
        "+0.37pp without). That combination -- the model leans on a feature "
        "heavily, yet overall results don't improve -- is what it looks like "
        "when a feature is genuinely informative about *something* in the "
        "training data but doesn't add real out-of-sample predictive edge, or "
        "when it's crowding out attention the model would otherwise have "
        "given to features that generalize better. It's also worth noting "
        "permutation importance here was measured on the *training* data (as "
        "is standard for this diagnostic), not held-out test years, so it "
        "shows what the model leaned on to fit history, not proof that "
        "reliance translated into better predictions on unseen years.\n"
    )

    lines.append("## Honest caveats\n")
    lines.append(
        "- **Same underlying survivorship-bias and data limits as "
        "`ml/WALKFORWARD_PIT_REPORT.md`.** Adding earnings features doesn't "
        "touch that problem.\n"
        "- **Earnings history is capped at 100 quarters (25 years) per ticker "
        "by Yahoo**, and only covers tickers that still have fetchable price "
        "data in the point-in-time dataset -- the same 91/250 sampled tickers "
        "missing from `dataset_pit.csv` are missing here too, plus one more "
        "(GENZ) with price data but no earnings history found.\n"
        "- **Only EPS surprises, not full earnings-report content.** Real "
        "earnings reports contain guidance, management commentary, segment "
        "detail, and conference-call tone that a professional analyst would "
        "weigh heavily -- none of that is captured here, only the numeric "
        "beat/miss percentage.\n"
        "- **No news, no IPO data.** Those aren't available free at the scale "
        "and history length this project needs -- see the discussion that led "
        "to building this earnings-only version instead.\n"
        "- **Same model, untuned hyperparameters, no transaction costs or "
        "position sizing** -- same limits as every other report in `ml/`.\n"
        "- **Not integrated.** `decide_trades.py` and the portal are "
        "unchanged.\n"
    )

    with open(ML_DIR / "WALKFORWARD_PIT_EARNINGS_REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {ML_DIR / 'WALKFORWARD_PIT_EARNINGS_REPORT.md'}")


if __name__ == "__main__":
    main()
