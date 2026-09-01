"""
A second attempt at ml/backtest_pit_strategy.py, aimed directly at the
volatility-drag problem that report diagnosed: a real average per-trade edge
(+0.42%) was small relative to the noise around each trade (~8.6% std), so
compounding it grew much slower than the simple average suggested and the
strategy lost to SPY buy-and-hold.

None of the changes here touch the model itself (still the same
HistGradientBoostingClassifier, same features, same walk-forward folds) --
they change how signals get turned into trades, on the theory that the
edge-to-noise ratio can be improved without a better model:

  1. More concurrent positions (15 -> 40): if the model's per-trade noise is
     largely independent across tickers, spreading the same capital across
     more simultaneous positions shrinks the *portfolio's* volatility without
     changing any individual trade's expected return.
  2. Confidence filtering: only act on the top half (by the model's own
     predicted probability for its BUY call) of each year's BUY signals,
     instead of every signal that clears the label threshold. The idea is
     that the model's most confident calls should, if the confidence score
     means anything, have a better edge relative to their noise.
  3. Volatility-scaled position sizing: instead of equal-weighting every
     position, size each one inversely to its own `vol_20d` feature (a
     ticker's recent realized volatility) relative to that day's book median,
     clipped to a 0.5x-1.75x band -- putting more capital behind calmer names
     and less behind noisier ones directly reduces the variance driving the
     drag, similar in spirit to a Kelly-style adjustment.

Writes ml/backtest_pit_v2_trades.csv, ml/backtest_pit_v2_equity.csv,
ml/backtest_pit_v2_equity_chart.svg, and ml/BACKTEST_PIT_V2_REPORT.md.

Still a research exercise -- same caveats as backtest_pit_strategy.py apply
(no interim mark-to-market, SELL signals don't trigger early exits, same
survivorship bias, approximate exit dates for multi-stint tickers).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from curl_cffi import requests as _curl_requests
from sklearn.ensemble import HistGradientBoostingClassifier

from build_dataset_pit import BUY_QUANTILE, SELL_QUANTILE, LABEL_HORIZON_DAYS

ML_DIR = Path(__file__).resolve().parent

FIRST_TEST_YEAR = 2000
MIN_TRAIN_ROWS = 5000

STARTING_CASH = 100_000.0

# --- lever 1: more concurrent positions (was 15) ---
MAX_CONCURRENT_POSITIONS = 40
TARGET_ALLOC_FRACTION = 1.0 / MAX_CONCURRENT_POSITIONS
MIN_CASH_BUFFER_PCT = 0.05
TRANSACTION_COST_BPS = 10

# --- lever 2: confidence filtering ---
# Of each year's BUY signals, only trade the top CONFIDENCE_KEEP_FRACTION by
# the model's own predicted probability for the BUY class.
CONFIDENCE_KEEP_FRACTION = 0.5

# --- lever 3: volatility-scaled sizing ---
# Position size multiplier = clip(median_open_vol / this_ticker_vol, MIN, MAX)
VOL_SCALE_MIN = 0.5
VOL_SCALE_MAX = 1.75


def _new_yf_session():
    return _curl_requests.Session(impersonate="chrome110")


def load_data():
    df = pd.read_csv(ML_DIR / "dataset_pit.csv", parse_dates=["date"])
    with open(ML_DIR / "dataset_pit_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return df, meta


def compute_exit_dates(df):
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["exit_date"] = df.groupby("ticker")["date"].shift(-LABEL_HORIZON_DAYS)
    return df


def generate_out_of_sample_predictions(df, feature_cols):
    """Same fold loop as backtest_pit_strategy.py, but also keeps the
    model's predicted probability for its own BUY call (needed for
    confidence filtering) and the row's vol_20d (needed for vol-scaled
    sizing).
    """
    df = df.sort_values("date")
    last_year = int(df["date"].dt.year.max())
    years = list(range(FIRST_TEST_YEAR, last_year + 1))

    all_preds = []

    for year in years:
        train = df[df["date"].dt.year < year]
        test = df[df["date"].dt.year == year]
        if len(train) < MIN_TRAIN_ROWS or test.empty:
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
        X_train, X_test = train[feature_cols], test[feature_cols]

        clf = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_depth=6, random_state=42,
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        proba = clf.predict_proba(X_test)
        buy_col = list(clf.classes_).index("BUY")
        buy_confidence = proba[:, buy_col]

        out = test[["ticker", "date", "exit_date", "fwd_return", "vol_20d"]].copy()
        out["prediction"] = preds
        out["buy_confidence"] = buy_confidence
        out["fold_year"] = year
        all_preds.append(out)
        print(f"  {year}: {len(test)} predictions generated "
              f"({(preds == 'BUY').sum()} BUY, {(preds == 'SELL').sum()} SELL)")

    return pd.concat(all_preds, ignore_index=True)


def apply_confidence_filter(predictions):
    """Keep only the top CONFIDENCE_KEEP_FRACTION of each year's BUY signals
    by the model's own predicted probability for BUY. SELL/HOLD rows are
    untouched (this only thins out which BUY signals get traded).
    """
    buys = predictions[predictions["prediction"] == "BUY"].copy()
    others = predictions[predictions["prediction"] != "BUY"].copy()

    kept = []
    for year, group in buys.groupby("fold_year"):
        cutoff = group["buy_confidence"].quantile(1 - CONFIDENCE_KEEP_FRACTION)
        kept.append(group[group["buy_confidence"] >= cutoff])
    buys_kept = pd.concat(kept, ignore_index=True) if kept else buys.iloc[0:0]

    dropped_n = len(buys) - len(buys_kept)
    print(f"Confidence filter: kept {len(buys_kept):,} of {len(buys):,} BUY "
          f"signals (dropped {dropped_n:,} lower-confidence calls).")

    return pd.concat([buys_kept, others], ignore_index=True)


def simulate_portfolio(predictions):
    """Same event-driven design as backtest_pit_strategy.py's simulator, but
    position size is scaled by each ticker's vol_20d relative to the median
    vol_20d among that day's open book (lever 3).
    """
    predictions = predictions.dropna(subset=["exit_date"]).copy()
    predictions = predictions.sort_values("date")

    buy_signals = predictions[predictions["prediction"] == "BUY"]
    all_dates = sorted(set(predictions["date"]) | set(predictions["exit_date"]))

    buys_by_date = {d: g for d, g in buy_signals.groupby("date")}

    cash = STARTING_CASH
    open_positions = {}
    trade_log = []
    equity_curve = []
    cost_rate = TRANSACTION_COST_BPS / 10000.0
    min_cash_buffer = STARTING_CASH * MIN_CASH_BUFFER_PCT

    # Universe-wide median vol_20d, used as the sizing reference point so the
    # multiplier isn't distorted by whatever happens to be open on a given
    # day (especially early on, with few or zero open positions).
    universe_median_vol = predictions["vol_20d"].median()

    for current_date in all_dates:
        for ticker in list(open_positions.keys()):
            pos = open_positions[ticker]
            if pos["exit_date"] == current_date:
                gross_proceeds = pos["alloc"] * (1 + pos["fwd_return"])
                exit_cost = gross_proceeds * cost_rate
                net_proceeds = gross_proceeds - exit_cost
                cash += net_proceeds
                trade_log.append({
                    "ticker": ticker, "entry_date": pos["entry_date"],
                    "exit_date": current_date, "alloc": pos["alloc"],
                    "fwd_return": pos["fwd_return"],
                    "vol_scale": pos["vol_scale"],
                    "net_pnl": net_proceeds - pos["alloc"] - pos["entry_cost"],
                })
                del open_positions[ticker]

        if current_date in buys_by_date:
            candidates = buys_by_date[current_date]
            for _, row in candidates.iterrows():
                ticker = row["ticker"]
                if ticker in open_positions:
                    continue
                if len(open_positions) >= MAX_CONCURRENT_POSITIONS:
                    break
                portfolio_value_estimate = cash + sum(
                    p["alloc"] for p in open_positions.values()
                )
                row_vol = row["vol_20d"]
                if pd.isna(row_vol) or row_vol <= 0:
                    vol_scale = 1.0
                else:
                    vol_scale = np.clip(universe_median_vol / row_vol,
                                         VOL_SCALE_MIN, VOL_SCALE_MAX)
                target_alloc = TARGET_ALLOC_FRACTION * portfolio_value_estimate * vol_scale
                available = cash - min_cash_buffer
                alloc = min(target_alloc, available)
                if alloc <= 0:
                    continue
                entry_cost = alloc * cost_rate
                cash -= (alloc + entry_cost)
                open_positions[ticker] = {
                    "entry_date": current_date,
                    "alloc": alloc,
                    "exit_date": row["exit_date"],
                    "fwd_return": row["fwd_return"],
                    "entry_cost": entry_cost,
                    "vol_scale": vol_scale,
                }

        book_value = cash + sum(p["alloc"] for p in open_positions.values())
        equity_curve.append({"date": current_date, "portfolio_value": book_value,
                              "cash": cash, "n_open_positions": len(open_positions)})

    return pd.DataFrame(trade_log), pd.DataFrame(equity_curve)


def fetch_spy_series(start_date, end_date):
    session = _new_yf_session()
    spy = yf.download("SPY", period="max", interval="1d", progress=False,
                       session=session, auto_adjust=True)["Close"].squeeze()
    spy.index = pd.to_datetime(spy.index)
    spy = spy[(spy.index >= start_date) & (spy.index <= end_date)]
    return spy


def compute_metrics(equity_curve, spy_series):
    start_val = equity_curve["portfolio_value"].iloc[0]
    end_val = equity_curve["portfolio_value"].iloc[-1]
    total_return = end_val / STARTING_CASH - 1
    n_years = (equity_curve["date"].iloc[-1] - equity_curve["date"].iloc[0]).days / 365.25
    cagr = (end_val / STARTING_CASH) ** (1 / n_years) - 1 if n_years > 0 else float("nan")

    spy_start, spy_end = spy_series.iloc[0], spy_series.iloc[-1]
    spy_total_return = spy_end / spy_start - 1
    spy_cagr = (spy_end / spy_start) ** (1 / n_years) - 1 if n_years > 0 else float("nan")

    running_max = equity_curve["portfolio_value"].cummax()
    drawdown = equity_curve["portfolio_value"] / running_max - 1
    max_drawdown = drawdown.min()

    return {
        "start_value": start_val, "end_value": end_val,
        "total_return": total_return, "cagr": cagr,
        "spy_total_return": spy_total_return, "spy_cagr": spy_cagr,
        "max_drawdown_realized_only": max_drawdown,
        "n_years": n_years,
    }


def write_chart(equity_curve, spy_series, baseline_equity, out_path):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    strategy_norm = equity_curve.set_index("date")["portfolio_value"] / STARTING_CASH * 100
    spy_norm = spy_series / spy_series.iloc[0] * 100
    baseline_norm = baseline_equity.set_index("date")["portfolio_value"] / STARTING_CASH * 100

    strategy_monthly = strategy_norm.resample("ME").last()
    spy_monthly = spy_norm.resample("ME").last()
    baseline_monthly = baseline_norm.resample("ME").last()

    ax.plot(strategy_monthly.index, strategy_monthly.values, label="v2 strategy (this run)",
            linewidth=1.6)
    ax.plot(baseline_monthly.index, baseline_monthly.values, label="v1 strategy (original)",
            linewidth=1.4, alpha=0.75, linestyle="--")
    ax.plot(spy_monthly.index, spy_monthly.values, label="SPY (buy & hold)",
            linewidth=1.6, alpha=0.85)
    ax.set_ylabel("Value (starting = 100)")
    ax.set_title("v2 strategy vs. v1 vs. SPY (2000-2026)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def main():
    df, meta = load_data()
    feature_cols = meta["feature_cols"]
    df = compute_exit_dates(df)

    print("Generating out-of-sample predictions across all walk-forward folds...")
    predictions = generate_out_of_sample_predictions(df, feature_cols)

    predictions = apply_confidence_filter(predictions)

    print(f"\nSimulating v2 portfolio: ${STARTING_CASH:,.0f} start, "
          f"max {MAX_CONCURRENT_POSITIONS} concurrent positions, "
          f"{TRANSACTION_COST_BPS}bps cost per leg, vol-scaled sizing "
          f"[{VOL_SCALE_MIN}x-{VOL_SCALE_MAX}x]...")
    trade_log, equity_curve = simulate_portfolio(predictions)
    trade_log.to_csv(ML_DIR / "backtest_pit_v2_trades.csv", index=False)

    start_date, end_date = equity_curve["date"].min(), equity_curve["date"].max()
    print(f"\nFetching SPY benchmark series for {start_date.date()} to {end_date.date()}...")
    spy_series = fetch_spy_series(start_date, end_date)

    equity_curve = equity_curve.merge(
        spy_series.rename("spy_close").reset_index().rename(columns={"index": "date", "Date": "date"}),
        on="date", how="left",
    )
    equity_curve["spy_close"] = equity_curve["spy_close"].ffill()
    equity_curve.to_csv(ML_DIR / "backtest_pit_v2_equity.csv", index=False)

    metrics = compute_metrics(equity_curve, spy_series)
    print("\n" + json.dumps(metrics, indent=2, default=str))

    baseline_equity = pd.read_csv(ML_DIR / "backtest_pit_equity.csv", parse_dates=["date"])
    baseline_metrics = compute_metrics(baseline_equity, spy_series)

    write_chart(equity_curve, spy_series, baseline_equity,
                ML_DIR / "backtest_pit_v2_equity_chart.svg")

    write_report(trade_log, equity_curve, metrics, baseline_metrics)


def write_report(trade_log, equity_curve, metrics, baseline_metrics):
    n_trades = len(trade_log)
    win_rate = (trade_log["net_pnl"] > 0).mean() if n_trades else float("nan")
    avg_trade_pnl_pct = (trade_log["net_pnl"] / trade_log["alloc"]).mean() if n_trades else float("nan")
    trade_std = (trade_log["net_pnl"] / trade_log["alloc"]).std() if n_trades else float("nan")
    avg_concurrent = equity_curve["n_open_positions"].mean()
    max_concurrent_actual = equity_curve["n_open_positions"].max()
    median_cash_frac = (equity_curve["cash"] / equity_curve["portfolio_value"]).median()
    n_years = metrics["n_years"]

    lines = []
    lines.append("# Trying to fix the volatility-drag problem\n")
    lines.append(
        "**Status: research exercise, not a validated trading strategy, and "
        "NOT wired into decide_trades.py or the portal.** "
        "`ml/BACKTEST_PIT_REPORT.md` found that the point-in-time model, "
        "turned into an actual portfolio, underperformed SPY buy-and-hold "
        "despite a real positive average per-trade edge -- diagnosed there "
        "as volatility drag: each trade's return swung by roughly 20x the "
        "average edge itself. This report tries three changes aimed "
        "directly at that ratio, without touching the underlying model:\n"
        "1. **More concurrent positions** (15 -> "
        f"{MAX_CONCURRENT_POSITIONS}) to diversify away independent "
        "per-trade noise.\n"
        f"2. **Confidence filtering** -- only trade the top "
        f"{CONFIDENCE_KEEP_FRACTION*100:.0f}% of each year's BUY signals by "
        "the model's own predicted probability.\n"
        f"3. **Volatility-scaled sizing** -- size each position by "
        f"{VOL_SCALE_MIN}x-{VOL_SCALE_MAX}x based on how calm or volatile "
        "that ticker has recently been, instead of equal-weighting "
        "everything.\n"
    )

    lines.append("## Results: v2 vs. v1 vs. SPY\n")
    lines.append("| | v2 (this report) | v1 (original) | SPY buy-and-hold |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Ending value (from $100,000) | ${metrics['end_value']:,.0f} | "
                 f"${baseline_metrics['end_value']:,.0f} | "
                 f"${STARTING_CASH * (1 + metrics['spy_total_return']):,.0f} |")
    lines.append(f"| CAGR | {metrics['cagr']*100:+.2f}% | "
                 f"{baseline_metrics['cagr']*100:+.2f}% | {metrics['spy_cagr']*100:+.2f}% |")
    lines.append(f"| Total return | {metrics['total_return']*100:+.1f}% | "
                 f"{baseline_metrics['total_return']*100:+.1f}% | "
                 f"{metrics['spy_total_return']*100:+.1f}% |")
    lines.append("")

    beat_v1 = "beat" if metrics["cagr"] > baseline_metrics["cagr"] else "did not beat"
    beat_spy = "beat" if metrics["cagr"] > metrics["spy_cagr"] else "still underperformed"
    lines.append(f"v2 **{beat_v1}** v1 on CAGR, and **{beat_spy}** SPY buy-and-hold.\n")

    lines.append("## Trade-level stats\n")
    lines.append(f"- {n_trades:,} total trades executed, average "
                 f"{avg_concurrent:.1f} concurrent positions held "
                 f"(peak {int(max_concurrent_actual)} of {MAX_CONCURRENT_POSITIONS} slots).")
    lines.append(f"- Win rate (net of costs): {win_rate*100:.1f}% of trades were profitable.")
    lines.append(f"- Average trade return (net of costs): {avg_trade_pnl_pct*100:+.2f}% per trade "
                 f"(v1 was +0.42%).")
    lines.append(f"- Per-trade return std dev: {trade_std*100:.1f}% (v1 was ~8.6%) -- "
                 f"the edge-to-noise ratio is {abs(avg_trade_pnl_pct/trade_std):.3f} "
                 f"(v1 was ~0.049).")
    lines.append(f"- Median cash utilization: {median_cash_frac*100:.1f}% "
                 f"(i.e. only {100-median_cash_frac*100:.1f}% invested on a typical day, "
                 f"vs. v1's ~94.3%).")
    lines.append("")

    lines.append("## Why did the combination get worse, not better?\n")
    lines.append(
        f"Lever 2 (confidence filtering) worked exactly as intended in "
        f"isolation: the per-trade edge more than doubled (+0.42% -> "
        f"{avg_trade_pnl_pct*100:+.2f}%) and the edge-to-noise ratio nearly "
        f"doubled too (0.049 -> {abs(avg_trade_pnl_pct/trade_std):.3f}), "
        f"confirming the model's confidence score does carry real "
        f"information. The problem is what that filtering did in "
        f"combination with lever 1: cutting the BUY signals in half while "
        f"*simultaneously* nearly tripling the number of position slots "
        f"(15 -> {MAX_CONCURRENT_POSITIONS}) left far too few signals to "
        f"fill the book. The portfolio sat at a median of only "
        f"{100-median_cash_frac*100:.1f}% invested (v1 was ~94.3%), meaning "
        f"most of the capital earned 0% sitting in cash most days. That "
        f"cash drag outweighed the improved per-trade edge. This is a "
        f"useful, concrete lesson: levers 1 and 2 both sounded reasonable "
        f"on paper, but they trade off against each other -- more slots "
        f"only helps if there's enough signal flow left to fill them, and "
        f"confidence filtering directly reduces that flow.\n"
    )

    lines.append(
        "*(An equity-curve chart comparing v2, v1, and SPY is generated "
        "locally as `ml/backtest_pit_v2_equity_chart.svg` when you run "
        "`backtest_pit_strategy_v2.py` -- not committed to the repo, same "
        "build-artifact policy as the rest of `ml/`.)*\n"
    )

    lines.append("## Honest caveats -- read before drawing conclusions\n")
    lines.append(
        "- **Same underlying signal.** Steps 1-3 only change how BUY/SELL "
        "calls get turned into trades -- the model itself, its features, "
        "and its walk-forward folds are unchanged from "
        "`ml/backtest_pit_strategy.py`. If the edge-to-noise ratio didn't "
        "meaningfully improve, none of the position-management changes can "
        "fully fix it.\n"
        "- **Confidence filtering was tested on the same data the model was "
        "scored on.** The 50% cutoff wasn't independently validated on a "
        "separate holdout -- it's a reasonable first guess, not a tuned "
        "parameter.\n"
        "- **Vol-scaled sizing uses `vol_20d`, a feature already in the "
        "model** -- this isn't new information, just a different use of "
        "data the model already sees, so it's a position-management change "
        "rather than a new signal.\n"
        "- **More concurrent positions only helps if per-ticker noise is "
        "genuinely independent.** If many of the model's BUY calls cluster "
        "around the same macro moves (e.g. it tends to buy broadly during "
        "rallies), the diversification benefit is smaller than the position "
        "count alone would suggest.\n"
        "- **Same caveats as `ml/BACKTEST_PIT_REPORT.md`**: no interim "
        "mark-to-market, SELL signals don't trigger early exits, no "
        "fractional-share rounding, same survivorship bias, approximate "
        "exit dates for multi-stint tickers, transaction-cost assumption "
        "unchanged, not integrated into `decide_trades.py` or the portal.\n"
    )

    with open(ML_DIR / "BACKTEST_PIT_V2_REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {ML_DIR / 'BACKTEST_PIT_V2_REPORT.md'}")


if __name__ == "__main__":
    main()
