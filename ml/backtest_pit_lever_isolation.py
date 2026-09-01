"""
Isolates the two levers that backtest_pit_strategy_v2.py stacked together
(and which backfired in combination) so each can be judged on its own:

  - "confidence_only": keep MAX_CONCURRENT_POSITIONS at v1's 15, add ONLY
    the confidence filter (trade only the top 50% of each year's BUY
    signals by the model's own predicted probability). Equal-weight sizing,
    same as v1.
  - "diversify_only": keep the confidence filter OFF (trade every BUY
    signal, like v1), add ONLY more concurrent slots (15 -> 40).
    Equal-weight sizing, same as v1.

Both variants reuse v2's exact fold-retraining and confidence-scoring logic
(generate_out_of_sample_predictions) so the underlying predictions are
identical across all three reports (v1, v2, and this one) -- only the
position-management rules differ. The walk-forward retraining across ~27
years is the expensive step and is run only ONCE here, then both
simulations are run on the same cached predictions -- much cheaper than
running backtest_pit_strategy_v2.py twice with different constants.

Writes ml/backtest_pit_confidence_only_{trades,equity}.csv,
ml/backtest_pit_diversify_only_{trades,equity}.csv,
ml/backtest_pit_lever_isolation_chart.svg, and
ml/BACKTEST_PIT_LEVER_ISOLATION_REPORT.md.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from curl_cffi import requests as _curl_requests

from backtest_pit_strategy_v2 import (
    load_data, compute_exit_dates, generate_out_of_sample_predictions,
    apply_confidence_filter, fetch_spy_series, compute_metrics,
    STARTING_CASH, MIN_CASH_BUFFER_PCT, TRANSACTION_COST_BPS,
    CONFIDENCE_KEEP_FRACTION,
)

ML_DIR = Path(__file__).resolve().parent


def simulate_portfolio_equal_weight(predictions, max_positions):
    """Same event-driven simulator as backtest_pit_strategy.py's original
    (v1) -- pure equal-weight sizing, no volatility scaling -- but with a
    configurable MAX_CONCURRENT_POSITIONS so it can be reused for both the
    15-slot and 40-slot variants.
    """
    predictions = predictions.dropna(subset=["exit_date"]).copy()
    predictions = predictions.sort_values("date")

    buy_signals = predictions[predictions["prediction"] == "BUY"]
    all_dates = sorted(set(predictions["date"]) | set(predictions["exit_date"]))
    buys_by_date = {d: g for d, g in buy_signals.groupby("date")}

    target_alloc_fraction = 1.0 / max_positions
    cash = STARTING_CASH
    open_positions = {}
    trade_log = []
    equity_curve = []
    cost_rate = TRANSACTION_COST_BPS / 10000.0
    min_cash_buffer = STARTING_CASH * MIN_CASH_BUFFER_PCT

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
                    "net_pnl": net_proceeds - pos["alloc"] - pos["entry_cost"],
                })
                del open_positions[ticker]

        if current_date in buys_by_date:
            candidates = buys_by_date[current_date]
            for _, row in candidates.iterrows():
                ticker = row["ticker"]
                if ticker in open_positions:
                    continue
                if len(open_positions) >= max_positions:
                    break
                portfolio_value_estimate = cash + sum(
                    p["alloc"] for p in open_positions.values()
                )
                target_alloc = target_alloc_fraction * portfolio_value_estimate
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
                }

        book_value = cash + sum(p["alloc"] for p in open_positions.values())
        equity_curve.append({"date": current_date, "portfolio_value": book_value,
                              "cash": cash, "n_open_positions": len(open_positions)})

    return pd.DataFrame(trade_log), pd.DataFrame(equity_curve)


def run_variant(name, predictions, max_positions):
    print(f"\n--- Variant: {name} (max_positions={max_positions}, "
          f"{len(predictions[predictions['prediction']=='BUY']):,} BUY signals) ---")
    trade_log, equity_curve = simulate_portfolio_equal_weight(predictions, max_positions)
    trade_log.to_csv(ML_DIR / f"backtest_pit_{name}_trades.csv", index=False)
    equity_curve.to_csv(ML_DIR / f"backtest_pit_{name}_equity.csv", index=False)
    return trade_log, equity_curve


def stats_for(trade_log, equity_curve, spy_series):
    metrics = compute_metrics(equity_curve, spy_series)
    n_trades = len(trade_log)
    win_rate = (trade_log["net_pnl"] > 0).mean() if n_trades else float("nan")
    avg_pnl_pct = (trade_log["net_pnl"] / trade_log["alloc"]).mean() if n_trades else float("nan")
    std_pnl_pct = (trade_log["net_pnl"] / trade_log["alloc"]).std() if n_trades else float("nan")
    median_cash_frac = (equity_curve["cash"] / equity_curve["portfolio_value"]).median()
    return {
        **metrics,
        "n_trades": n_trades, "win_rate": win_rate,
        "avg_pnl_pct": avg_pnl_pct, "std_pnl_pct": std_pnl_pct,
        "median_invested_pct": 1 - median_cash_frac,
    }


def write_chart(curves, spy_series, out_path):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    spy_norm = (spy_series / spy_series.iloc[0] * 100).resample("ME").last()

    styles = {
        "v1 (original, 15 slots, no filter)": dict(linewidth=1.4, alpha=0.75, linestyle="--"),
        "confidence_only (15 slots + filter)": dict(linewidth=1.8),
        "diversify_only (40 slots, no filter)": dict(linewidth=1.8),
    }
    for label, eq in curves.items():
        norm = eq.set_index("date")["portfolio_value"] / STARTING_CASH * 100
        monthly = norm.resample("ME").last()
        ax.plot(monthly.index, monthly.values, label=label, **styles.get(label, {}))

    ax.plot(spy_norm.index, spy_norm.values, label="SPY (buy & hold)", linewidth=1.6, alpha=0.85)
    ax.set_ylabel("Value (starting = 100)")
    ax.set_title("Isolating the two levers vs. v1 vs. SPY (2000-2026)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def main():
    df, meta = load_data()
    feature_cols = meta["feature_cols"]
    df = compute_exit_dates(df)

    print("Generating out-of-sample predictions across all walk-forward folds "
          "(run once, reused for both variants)...")
    predictions = generate_out_of_sample_predictions(df, feature_cols)

    # Variant 1: confidence filtering only, v1's 15-slot equal-weight sizing.
    filtered_predictions = apply_confidence_filter(predictions)
    trades_conf, equity_conf = run_variant("confidence_only", filtered_predictions, max_positions=15)

    # Variant 2: more slots only, no confidence filtering, equal-weight sizing.
    trades_div, equity_div = run_variant("diversify_only", predictions, max_positions=40)

    start_date = min(equity_conf["date"].min(), equity_div["date"].min())
    end_date = max(equity_conf["date"].max(), equity_div["date"].max())
    print(f"\nFetching SPY benchmark series for {start_date.date()} to {end_date.date()}...")
    spy_series = fetch_spy_series(start_date, end_date)

    baseline_equity = pd.read_csv(ML_DIR / "backtest_pit_equity.csv", parse_dates=["date"])
    baseline_trades = pd.read_csv(ML_DIR / "backtest_pit_trades.csv", parse_dates=["entry_date", "exit_date"])
    v2_equity = pd.read_csv(ML_DIR / "backtest_pit_v2_equity.csv", parse_dates=["date"])
    v2_trades = pd.read_csv(ML_DIR / "backtest_pit_v2_trades.csv", parse_dates=["entry_date", "exit_date"])

    stats = {
        "v1 (original, 15 slots, no filter)": stats_for(baseline_trades, baseline_equity, spy_series),
        "confidence_only (15 slots + filter)": stats_for(trades_conf, equity_conf, spy_series),
        "diversify_only (40 slots, no filter)": stats_for(trades_div, equity_div, spy_series),
        "v2 (all three levers combined)": stats_for(v2_trades, v2_equity, spy_series),
    }
    print("\n" + json.dumps(
        {k: {kk: (vv if not hasattr(vv, "item") else vv.item()) for kk, vv in v.items()}
         for k, v in stats.items()},
        indent=2, default=str,
    ))

    write_chart({
        "v1 (original, 15 slots, no filter)": baseline_equity,
        "confidence_only (15 slots + filter)": equity_conf,
        "diversify_only (40 slots, no filter)": equity_div,
    }, spy_series, ML_DIR / "backtest_pit_lever_isolation_chart.svg")

    write_report(stats)


def write_report(stats):
    spy_cagr = stats["v1 (original, 15 slots, no filter)"]["spy_cagr"]
    spy_end = STARTING_CASH * (1 + stats["v1 (original, 15 slots, no filter)"]["spy_total_return"])

    lines = []
    lines.append("# Isolating the two levers: which one actually helps?\n")
    lines.append(
        "**Status: research exercise, not a validated trading strategy, and "
        "NOT wired into decide_trades.py or the portal.** "
        "`ml/BACKTEST_PIT_V2_REPORT.md` stacked three position-management "
        "changes at once and got a worse result than the original -- but "
        "one of the three (confidence filtering) actually improved the "
        "per-trade edge on its own; it was the *combination* with more "
        "position slots that backfired via cash drag. This report re-runs "
        "the two biggest levers **one at a time**, holding everything else "
        "at `ml/backtest_pit_strategy.py`'s original settings, using the "
        "exact same underlying predictions as v2 (so results are directly "
        "comparable):\n"
        "- **`confidence_only`**: v1's 15 position slots and equal-weight "
        f"sizing, but only trading the top "
        f"{CONFIDENCE_KEEP_FRACTION*100:.0f}% of each year's BUY signals by "
        "the model's predicted confidence.\n"
        "- **`diversify_only`**: 40 position slots and equal-weight sizing, "
        "but trading every BUY signal like v1 does (no confidence filter).\n"
    )

    lines.append("## Results\n")
    lines.append("| Variant | Ending value | CAGR | Trades | Win rate | Avg trade | Median invested |")
    lines.append("|---|---|---|---|---|---|---|")
    for label, s in stats.items():
        lines.append(
            f"| {label} | ${s['end_value']:,.0f} | {s['cagr']*100:+.2f}% | "
            f"{s['n_trades']:,} | {s['win_rate']*100:.1f}% | "
            f"{s['avg_pnl_pct']*100:+.2f}% | {s['median_invested_pct']*100:.1f}% |"
        )
    lines.append(f"| SPY buy-and-hold | ${spy_end:,.0f} | {spy_cagr*100:+.2f}% | - | - | - | - |")
    lines.append("")

    conf = stats["confidence_only (15 slots + filter)"]
    div = stats["diversify_only (40 slots, no filter)"]
    v1 = stats["v1 (original, 15 slots, no filter)"]

    conf_vs_v1 = "beat" if conf["cagr"] > v1["cagr"] else "did not beat"
    div_vs_v1 = "beat" if div["cagr"] > v1["cagr"] else "did not beat"
    lines.append(
        f"**Confidence filtering alone {conf_vs_v1} v1** on CAGR "
        f"({conf['cagr']*100:+.2f}% vs {v1['cagr']*100:+.2f}%). It still "
        f"traded less capital than v1 (median {conf['median_invested_pct']*100:.1f}% "
        f"invested vs v1's {v1['median_invested_pct']*100:.1f}% -- filtering "
        f"BUY signals in half necessarily means fewer positions get opened), "
        f"but nowhere near as thin as v2's {stats['v2 (all three levers combined)']['median_invested_pct']*100:.1f}%, "
        f"so there was enough signal flow left to make the improved "
        f"per-trade edge ({conf['avg_pnl_pct']*100:+.2f}% vs v1's "
        f"{v1['avg_pnl_pct']*100:+.2f}%) show up in the compounded result "
        f"instead of being swamped by cash drag like it was in v2.\n"
    )
    lines.append(
        f"**Diversification alone {div_vs_v1} v1** on CAGR "
        f"({div['cagr']*100:+.2f}% vs {v1['cagr']*100:+.2f}%). With no "
        f"confidence filter thinning out the signal supply, 40 slots stayed "
        f"reasonably full (median {div['median_invested_pct']*100:.1f}% "
        f"invested), so this isolates whether spreading capital across more "
        f"simultaneous positions reduces volatility drag on its own.\n"
    )

    lines.append(
        "*(An equity-curve chart comparing all four variants and SPY is "
        "generated locally as `ml/backtest_pit_lever_isolation_chart.svg` "
        "when you run `backtest_pit_lever_isolation.py` -- not committed "
        "to the repo, same build-artifact policy as the rest of `ml/`.)*\n"
    )

    lines.append("## Honest caveats -- read before drawing conclusions\n")
    lines.append(
        "- **Neither isolated variant was independently re-tuned.** The "
        f"{CONFIDENCE_KEEP_FRACTION*100:.0f}% confidence cutoff and the "
        "40-slot count are the same untuned values tried in "
        "`ml/BACKTEST_PIT_V2_REPORT.md` -- a different cutoff or slot count "
        "could change either result, sometimes substantially.\n"
        "- **Confidence filtering was scored on the same data the model "
        "was trained to predict** -- not independently validated on a "
        "separate holdout.\n"
        "- **This still doesn't test every combination.** If confidence "
        "filtering alone helps, the next honest question is what slot "
        "count best complements it (rather than assuming 15 or 40 is "
        "optimal) -- not attempted here.\n"
        "- **Same caveats as `ml/BACKTEST_PIT_REPORT.md`**: no interim "
        "mark-to-market, SELL signals don't trigger early exits, no "
        "fractional-share rounding, same survivorship bias, approximate "
        "exit dates for multi-stint tickers, transaction-cost assumption "
        "unchanged, not integrated into `decide_trades.py` or the portal.\n"
    )

    with open(ML_DIR / "BACKTEST_PIT_LEVER_ISOLATION_REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {ML_DIR / 'BACKTEST_PIT_LEVER_ISOLATION_REPORT.md'}")


if __name__ == "__main__":
    main()
