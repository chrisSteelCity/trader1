"""
Model-comparison version of train_walkforward_pit.py: same expanding-window
walk-forward methodology, same point-in-time dataset (dataset_pit.csv), same
fold-specific BUY/SELL label thresholds -- but instead of only
HistGradientBoostingClassifier, this also trains XGBoost and LightGBM
classifiers on every fold, so the three can be compared head-to-head on
identical train/test splits.

Why: ml/WALKFORWARD_PIT_REPORT.md's HistGradientBoostingClassifier result
(21/27 years beating SPY, mean BUY excess +0.37pp) was never compared against
a different model family -- "is this the best a model can do, or just the
first thing that was tried?" was an open question. This answers it directly
on the same data/methodology, changing only the model.

Writes:
  - ml/walkforward_pit_xgb_folds.csv, ml/walkforward_pit_lgbm_folds.csv
  - ml/model_walkforward_pit_xgb.joblib, ml/model_walkforward_pit_lgbm.joblib
  - ml/WALKFORWARD_PIT_MODEL_COMPARISON.md (reads the existing
    ml/walkforward_pit_folds.csv as the HistGB baseline -- run
    train_walkforward_pit.py first if that file doesn't exist yet)

Still a validation exercise, not a trading strategy -- see the "Honest
caveats" sections of the reports this builds on.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from build_dataset_pit import BUY_QUANTILE, SELL_QUANTILE

ML_DIR = Path(__file__).resolve().parent

FIRST_TEST_YEAR = 2000
MIN_TRAIN_ROWS = 5000
LABEL_ORDER = ["HOLD", "BUY", "SELL"]  # fixed encoding for LabelEncoder consistency


def load_data():
    df = pd.read_csv(ML_DIR / "dataset_pit.csv", parse_dates=["date"])
    with open(ML_DIR / "dataset_pit_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return df, meta


def make_xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", n_jobs=2, random_state=42,
        eval_metric="mlogloss",
    )


def make_lgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=2, random_state=42, verbosity=-1,
    )


def make_histgb():
    return HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.08, max_depth=6, random_state=42,
    )


MODEL_FACTORIES = {
    "xgboost": make_xgb,
    "lightgbm": make_lgbm,
}


def run_walkforward(df, feature_cols, model_name, factory, needs_label_encoding):
    df = df.sort_values("date")
    last_year = int(df["date"].dt.year.max())
    years = list(range(FIRST_TEST_YEAR, last_year + 1))

    le = LabelEncoder()
    le.fit(LABEL_ORDER)

    fold_rows = []
    final_model = None
    t_start = time.time()

    for year in years:
        train = df[df["date"].dt.year < year]
        test = df[df["date"].dt.year == year]
        if len(train) < MIN_TRAIN_ROWS or test.empty:
            print(f"  [{model_name}] {year}: skipped (not enough training data yet)")
            continue

        buy_cut = train["excess_return"].quantile(BUY_QUANTILE)
        sell_cut = train["excess_return"].quantile(SELL_QUANTILE)

        def _label(x, buy_cut=buy_cut, sell_cut=sell_cut):
            if x >= buy_cut:
                return "BUY"
            if x <= sell_cut:
                return "SELL"
            return "HOLD"

        y_train_str = train["excess_return"].apply(_label)
        y_test_str = test["excess_return"].apply(_label)
        X_train, X_test = train[feature_cols], test[feature_cols]

        n_active_tickers_test = test["ticker"].nunique()

        clf = factory()
        t0 = time.time()
        if needs_label_encoding:
            clf.fit(X_train, le.transform(y_train_str))
            preds_enc = clf.predict(X_test)
            preds = le.inverse_transform(preds_enc)
        else:
            clf.fit(X_train, y_train_str)
            preds = clf.predict(X_test)
        fit_secs = time.time() - t0

        acc = accuracy_score(y_test_str, preds)
        majority_class = y_train_str.value_counts().idxmax()
        baseline_acc = float((y_test_str == majority_class).mean())

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
        print(f"  [{model_name}] {year}: acc={acc:.3f} (baseline {baseline_acc:.3f}), "
              f"{n_active_tickers_test} tickers, {len(buy_rows)} BUY calls, "
              f"excess={be*100 if be is not None else float('nan'):+.2f}pp, "
              f"fit={fit_secs:.1f}s")
        final_model = clf

    print(f"  [{model_name}] total walk-forward time: {time.time()-t_start:.0f}s")
    return pd.DataFrame(fold_rows), final_model, le


def summarize(folds):
    n_folds = len(folds)
    beat_baseline = int((folds["accuracy"] > folds["baseline_accuracy"]).sum())
    buy_valid = folds["buy_excess_return"].notna()
    buy_beat_spy = int((folds.loc[buy_valid, "buy_excess_return"] > 0).sum())
    n_buy_valid = int(buy_valid.sum())
    mean_buy_excess = folds.loc[buy_valid, "buy_excess_return"].mean()
    median_buy_excess = folds.loc[buy_valid, "buy_excess_return"].median()
    mean_acc = folds["accuracy"].mean()
    return {
        "n_folds": n_folds,
        "beat_baseline": beat_baseline,
        "buy_beat_spy": buy_beat_spy,
        "n_buy_valid": n_buy_valid,
        "mean_buy_excess": mean_buy_excess,
        "median_buy_excess": median_buy_excess,
        "mean_acc": mean_acc,
    }


def main():
    df, meta = load_data()
    feature_cols = meta["feature_cols"]

    results = {}

    for model_name, factory in MODEL_FACTORIES.items():
        print(f"\nRunning expanding-window walk-forward with {model_name}...")
        folds, final_model, le = run_walkforward(
            df, feature_cols, model_name, factory, needs_label_encoding=True,
        )
        out_csv = ML_DIR / f"walkforward_pit_{model_name}_folds.csv"
        folds.to_csv(out_csv, index=False)
        joblib.dump(
            {"model": final_model, "label_encoder": le, "feature_cols": feature_cols, "meta": meta},
            ML_DIR / f"model_walkforward_pit_{model_name}.joblib",
        )
        results[model_name] = folds
        print(f"  Wrote {out_csv}")

    # Load the existing HistGradientBoostingClassifier results as the baseline
    # for comparison (from train_walkforward_pit.py's run) rather than
    # retraining it here.
    histgb_path = ML_DIR / "walkforward_pit_folds.csv"
    if histgb_path.exists():
        results["histgb"] = pd.read_csv(histgb_path)
    else:
        print(f"\nWarning: {histgb_path} not found -- run train_walkforward_pit.py "
              f"first for the HistGB baseline. Comparison report will omit it.")

    write_comparison_report(results, meta)


def write_comparison_report(results, meta):
    lines = []
    lines.append("# Model comparison on the point-in-time walk-forward (2000-2026)\n")
    lines.append(
        "**Status: proof-of-concept validation exercise, not a validated trading "
        "strategy, and NOT wired into decide_trades.py or the portal.** Same "
        "dataset, same expanding-window walk-forward, same per-fold BUY/SELL "
        "quantile thresholds as `ml/WALKFORWARD_PIT_REPORT.md` -- only the model "
        "changes. That report used scikit-learn's `HistGradientBoostingClassifier` "
        "without ever checking whether a different model family would do "
        "better; this compares it against XGBoost and LightGBM on identical "
        "train/test splits.\n"
    )

    lines.append("## Summary comparison\n")
    lines.append("| Model | Folds | Beat baseline acc. | Mean accuracy | BUY beat SPY | "
                  "Mean BUY excess | Median BUY excess |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    display_names = {"histgb": "HistGradientBoosting (original)", "xgboost": "XGBoost",
                      "lightgbm": "LightGBM"}
    order = [m for m in ["histgb", "xgboost", "lightgbm"] if m in results]
    summaries = {}
    for m in order:
        s = summarize(results[m])
        summaries[m] = s
        lines.append(
            f"| {display_names[m]} | {s['n_folds']} | {s['beat_baseline']}/{s['n_folds']} | "
            f"{s['mean_acc']:.3f} | {s['buy_beat_spy']}/{s['n_buy_valid']} | "
            f"{s['mean_buy_excess']*100:+.2f} pp | {s['median_buy_excess']*100:+.2f} pp |"
        )
    lines.append("")

    if len(order) >= 2:
        best_by_excess = max(order, key=lambda m: summaries[m]["mean_buy_excess"])
        best_by_years = max(order, key=lambda m: summaries[m]["buy_beat_spy"] / max(1, summaries[m]["n_buy_valid"]))
        lines.append(
            f"By mean BUY excess return, **{display_names[best_by_excess]}** comes out "
            f"ahead. By fraction of years BUY calls beat SPY, "
            f"**{display_names[best_by_years]}** comes out ahead"
            + (" (the same model)" if best_by_excess == best_by_years else
               " -- these two rankings disagree, which is itself a sign the "
               "differences between models are within noise rather than a "
               "clear win for one architecture") + ".\n"
        )

    lines.append("## Year-by-year BUY excess return, all models\n")
    lines.append("Positive = that model's BUY calls beat SPY that year; blank = no BUY "
                  "calls issued or model not run for that year.\n")
    all_years = sorted(set().union(*[set(results[m]["year"]) for m in order]))
    header = "| Year | SPY return | " + " | ".join(display_names[m] for m in order) + " |"
    lines.append(header)
    lines.append("|---|---:|" + "---:|" * len(order))
    spy_lookup = {}
    if "histgb" in results and "spy_calendar_year_return" in results["histgb"].columns:
        spy_lookup = dict(zip(results["histgb"]["year"], results["histgb"]["spy_calendar_year_return"]))
    for y in all_years:
        spy_ret = spy_lookup.get(y)
        spy_str = f"{spy_ret*100:+.1f}%" if spy_ret is not None and pd.notna(spy_ret) else "n/a"
        cells = []
        for m in order:
            row = results[m][results[m]["year"] == y]
            if row.empty or pd.isna(row["buy_excess_return"].iloc[0]):
                cells.append("n/a")
            else:
                cells.append(f"{row['buy_excess_return'].iloc[0]*100:+.2f} pp")
        lines.append(f"| {int(y)} | {spy_str} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Honest caveats\n")
    lines.append(
        "- **Same underlying data limits as `ml/WALKFORWARD_PIT_REPORT.md`.** "
        "Survivorship bias is narrowed, not eliminated (see that report's Setup "
        "section); nothing about changing the model architecture touches that.\n"
        "- **Hyperparameters were matched by intent, not tuned.** Each model "
        "uses roughly comparable settings (depth 6, learning rate 0.08, ~150-200 "
        "trees) rather than each being separately tuned to its own best "
        "configuration -- a fairer 'best case' comparison would tune each model "
        "independently (e.g. via time-series cross-validation within each "
        "fold's training window), which is real future work, not done here.\n"
        "- **No ensembling tried.** Averaging or stacking predictions across "
        "the three models might do better than any single one -- not tested in "
        "this pass.\n"
        "- **Small, noisy per-year sample.** 27 one-year folds is not a lot of "
        "independent data points for comparing model families -- differences "
        "of a few tenths of a percentage point between models are well within "
        "what could be noise rather than a real architectural advantage.\n"
        "- **No fundamentals, no transaction costs, no position sizing.** Same "
        "limits as every other report in `ml/`.\n"
        "- **Not integrated.** `decide_trades.py` and the portal are unchanged.\n"
    )

    with open(ML_DIR / "WALKFORWARD_PIT_MODEL_COMPARISON.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {ML_DIR / 'WALKFORWARD_PIT_MODEL_COMPARISON.md'}")


if __name__ == "__main__":
    main()
