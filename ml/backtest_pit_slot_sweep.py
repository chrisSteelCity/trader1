"""
ml/BACKTEST_PIT_LEVER_ISOLATION_REPORT.md confirmed confidence filtering
helps on its own, using v1's untouched 15-slot position limit -- but that
slot count was never chosen with confidence filtering in mind (it was v1's
original number, back when every BUY signal above the label threshold got
traded). Since filtering to the top 50% by confidence roughly halves the
number of BUY signals, there's no reason to assume 15 is still the best
number of concurrent slots to hold. This sweeps a range of slot counts,
all with the confidence filter applied, to find the number that actually
suits the thinned-out signal flow best.

Reuses the exact same fold-retraining and confidence-filtered predictions
as v2 / the lever-isolation report (generated once, since retraining across
~27 years is the expensive step) and re-simulates the portfolio at each
candidate slot count using the same plain equal-weight simulator as v1.

Writes ml/backtest_pit_slot_sweep_summary.csv (one row per slot count),
ml/backtest_pit_slot_sweep_chart.svg, and
ml/BACKTEST_PIT_SLOT_SWEEP_REPORT.md.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from backtest_pit_strategy_v2 import (
    load_data, compute_exit_dates, generate_out_of_sample_predictions,
    apply_confidence_filter, fetch_spy_series, compute_metrics,
    STARTING_CASH, CONFIDENCE_KEEP_FRACTION,
)
from backtest_pit_lever_isolation import simulate_portfolio_equal_weight, stats_for

ML_DIR = Path(__file__).resolve().parent

# Slot counts to try, all with the confidence filter applied. Includes v1's
# original 15 and v2/diversify_only's 40 as reference points, plus values
# in between and a bit above, since it's not obvious in which direction the
# optimum lies.
SLOT_COUNTS_TO_TRY = [10, 15, 20, 25, 30, 35, 40, 50]


def write_chart(summary_df, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(summary_df["max_positions"], summary_df["cagr"] * 100,
              marker="o", linewidth=1.8, color="#1f77b4")
    ax1.axhline(summary_df.attrs["spy_cagr"] * 100, color="gray",
                linestyle="--", linewidth=1.2, label="SPY buy-and-hold")
    ax1.set_xlabel("Max concurrent positions")
    ax1.set_ylabel("CAGR (%)")
    ax1.set_title("CAGR vs. slot count (confidence filter on)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(summary_df["max_positions"], summary_df["median_invested_pct"] * 100,
              marker="o", linewidth=1.8, color="#ff7f0e")
    ax2.set_xlabel("Max concurrent positions")
    ax2.set_ylabel("Median % of portfolio invested")
    ax2.set_title("Capital utilization vs. slot count")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def main():
    df, meta = load_data()
    feature_cols = meta["feature_cols"]
    df = compute_exit_dates(df)

    print("Generating out-of-sample predictions across all walk-forward folds "
          "(run once, reused for every slot count)...")
    predictions = generate_out_of_sample_predictions(df, feature_cols)
    filtered_predictions = apply_confidence_filter(predictions)

    start_date = filtered_predictions["date"].min()
    end_date = filtered_predictions["exit_date"].max()
    print(f"\nFetching SPY benchmark series for {start_date.date()} to {end_date.date()}...")
    spy_series = fetch_spy_series(start_date, end_date)

    rows = []
    for max_positions in SLOT_COUNTS_TO_TRY:
        print(f"\n--- Simulating with max_positions={max_positions} "
              f"(confidence filter on) ---")
        trade_log, equity_curve = simulate_portfolio_equal_weight(
            filtered_predictions, max_positions
        )
        trade_log.to_csv(ML_DIR / f"backtest_pit_slots{max_positions}_trades.csv", index=False)
        equity_curve.to_csv(ML_DIR / f"backtest_pit_slots{max_positions}_equity.csv", index=False)
        s = stats_for(trade_log, equity_curve, spy_series)
        s["max_positions"] = max_positions
        rows.append(s)
        print(f"  CAGR: {s['cagr']*100:+.2f}%, ending value: ${s['end_value']:,.0f}, "
              f"median invested: {s['median_invested_pct']*100:.1f}%")

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df[["max_positions", "end_value", "cagr", "total_return",
                              "n_trades", "win_rate", "avg_pnl_pct", "std_pnl_pct",
                              "median_invested_pct", "max_drawdown_realized_only"]]
    summary_df.to_csv(ML_DIR / "backtest_pit_slot_sweep_summary.csv", index=False)
    summary_df.attrs["spy_cagr"] = rows[0]["spy_cagr"]

    print("\n" + summary_df.to_string(index=False))

    write_chart(summary_df, ML_DIR / "backtest_pit_slot_sweep_chart.svg")

    write_report(summary_df, rows[0]["spy_cagr"],
                 STARTING_CASH * (1 + rows[0]["spy_total_return"]))


def write_report(summary_df, spy_cagr, spy_end):
    best_row = summary_df.loc[summary_df["cagr"].idxmax()]
    v1_row = summary_df[summary_df["max_positions"] == 15].iloc[0] if 15 in summary_df["max_positions"].values else None

    lines = []
    lines.append("# Finding the slot count that fits the confidence-filtered signal\n")
    lines.append(
        "**Status: research exercise, not a validated trading strategy, and "
        "NOT wired into decide_trades.py or the portal.** "
        "`ml/BACKTEST_PIT_LEVER_ISOLATION_REPORT.md` confirmed confidence "
        "filtering improves results on its own, but tested it only at v1's "
        "original 15-slot limit -- a number chosen before filtering existed "
        "and never revisited. Since filtering to the model's top "
        f"{CONFIDENCE_KEEP_FRACTION*100:.0f}% most confident BUY calls "
        "roughly halves the signal supply, this sweeps a range of slot "
        "counts, always with the confidence filter on, to find the count "
        "that actually fits the thinned-out signal flow.\n"
    )

    lines.append("## Results\n")
    lines.append("| Max positions | Ending value | CAGR | Trades | Median invested | Max drawdown |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in summary_df.iterrows():
        marker = " **<- best**" if row["max_positions"] == best_row["max_positions"] else ""
        lines.append(
            f"| {int(row['max_positions'])} | ${row['end_value']:,.0f} | "
            f"{row['cagr']*100:+.2f}%{marker} | {int(row['n_trades']):,} | "
            f"{row['median_invested_pct']*100:.1f}% | "
            f"{row['max_drawdown_realized_only']*100:.1f}% |"
        )
    lines.append(f"| SPY buy-and-hold | ${spy_end:,.0f} | {spy_cagr*100:+.2f}% | - | - | - |")
    lines.append("")

    margin_pp = (best_row["cagr"] - spy_cagr) * 100
    beat_spy = "essentially ties" if abs(margin_pp) < 0.1 else ("beats" if margin_pp > 0 else "still trails")
    lines.append(
        f"The best slot count tested is **{int(best_row['max_positions'])}**, "
        f"reaching CAGR {best_row['cagr']*100:+.2f}% (ending value "
        f"${best_row['end_value']:,.0f}) -- this **{beat_spy}** SPY "
        f"buy-and-hold ({spy_cagr*100:+.2f}%), a margin of only "
        f"{margin_pp:+.2f} percentage points. That margin is small enough "
        f"that it should be read as \"finally in the same neighborhood as "
        f"buy-and-hold\" rather than \"a proven edge over it\" -- see the "
        f"in-sample-selection caveat below before treating 20 slots as the "
        f"answer. "
    )
    if v1_row is not None:
        lines.append(
            f"Compared to the 15-slot count used in every earlier "
            f"confidence-filtered test, this is "
            f"{(best_row['cagr']-v1_row['cagr'])*100:+.2f} percentage "
            f"points of CAGR different.\n"
        )
    else:
        lines.append("\n")

    lines.append(
        "*(A chart of CAGR and capital utilization across all slot counts "
        "tested is generated locally as `ml/backtest_pit_slot_sweep_chart.svg` "
        "when you run `backtest_pit_slot_sweep.py` -- not committed to the "
        "repo, same build-artifact policy as the rest of `ml/`.)*\n"
    )

    lines.append("## Honest caveats -- read before drawing conclusions\n")
    lines.append(
        "- **This slot count was chosen by looking at results on this exact "
        "backtest, then reported as \"best.\"** That's a form of in-sample "
        "selection -- picking the winner after seeing all the outcomes on "
        "the same 27 years of data being reported on. A slot count "
        "\"optimized\" this way is likely to look somewhat better here than "
        "it would on genuinely new, unseen years; treat the improvement "
        "over 15 slots as an optimistic upper bound, not a guaranteed gain.\n"
        "- **Only 8 slot counts were tried, on a coarse grid** "
        f"({', '.join(str(s) for s in SLOT_COUNTS_TO_TRY)}) -- the true "
        "optimum could sit between two tested values.\n"
        "- **The 50% confidence cutoff itself was never re-swept alongside "
        "slot count.** A different cutoff paired with a different slot "
        "count could do better still; this only searches one dimension at "
        "a time.\n"
        "- **Same caveats as `ml/BACKTEST_PIT_REPORT.md`**: no interim "
        "mark-to-market, SELL signals don't trigger early exits, no "
        "fractional-share rounding, same survivorship bias, approximate "
        "exit dates for multi-stint tickers, transaction-cost assumption "
        "unchanged, not integrated into `decide_trades.py` or the portal.\n"
    )

    with open(ML_DIR / "BACKTEST_PIT_SLOT_SWEEP_REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {ML_DIR / 'BACKTEST_PIT_SLOT_SWEEP_REPORT.md'}")


if __name__ == "__main__":
    main()
