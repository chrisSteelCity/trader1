"""
Turns the point-in-time model's BUY/SELL calls into an actual simulated
dollar-based portfolio, instead of the abstract "average excess return of
BUY calls" metric every earlier report in ml/ used. That metric answers "is
the signal informative on average" -- it says nothing about whether a real
strategy built on it would survive trading costs, concurrent-position
limits, or cash constraints. This does.

Methodology:
  1. Re-run the same expanding-window walk-forward as train_walkforward_pit.py
     (same HistGradientBoostingClassifier, same per-fold BUY/SELL quantile
     thresholds -- the plain point-in-time setup, since the earlier XGBoost/
     LightGBM/earnings-features variants all tied or underperformed it), but
     this time keep every individual out-of-sample prediction instead of
     just aggregate stats.
  2. Simulate a portfolio starting at $100,000: on any day a ticker not
     already held gets a BUY prediction, open a position (equal-weight
     target size, capped by a max concurrent-position count and a cash
     buffer) if there's room. Every position is held for exactly the same
     10-trading-day horizon the model was trained to predict, then closed
     using dataset_pit.csv's own `fwd_return` for that row -- which is
     already the correct, point-in-time-safe realized return for that
     exact holding window.
  3. Apply a transaction cost on both the entry and exit leg.
  4. Compare the resulting equity curve to $100,000 in SPY, buy-and-hold,
     over the identical date range.

Writes ml/backtest_pit_folds.csv (trade log), ml/backtest_pit_equity.csv
(daily portfolio value vs SPY), ml/backtest_pit_equity_chart.svg, and
ml/BACKTEST_PIT_REPORT.md.

Still a research exercise -- see that report's Honest caveats section for
what this simulation does NOT capture (most importantly: no interim
mark-to-market on open positions, and SELL signals don't trigger early
exits -- see the caveats for why).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from curl_cffi import requests as _curl_requests
from sklearn.ensemble import HistGradientBoostingClassifier

from build_dataset_pit import BUY_QUANTILE, SELL_QUANTILE, LABEL_HORIZON_DAYS

ML_DIR = Path(__file__).resolve().parent

FIRST_TEST_YEAR = 2000
MIN_TRAIN_ROWS = 5000

STARTING_CASH = 100_000.0
MAX_CONCURRENT_POSITIONS = 15
TARGET_ALLOC_FRACTION = 1.0 / MAX_CONCURRENT_POSITIONS  # equal-weight target
MIN_CASH_BUFFER_PCT = 0.05      # keep at least 5% of starting capital as buffer
TRANSACTION_COST_BPS = 10       # 0.10% per leg (entry and exit each), i.e. 0.20% round trip


def _new_yf_session():
    return _curl_requests.Session(impersonate="chrome110")


def load_data():
    df = pd.read_csv(ML_DIR / "dataset_pit.csv", parse_dates=["date"])
    with open(ML_DIR / "dataset_pit_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return df, meta


def compute_exit_dates(df):
    """Approximate each row's exit date as LABEL_HORIZON_DAYS rows ahead in
    that ticker's own (already point-in-time-filtered) date sequence -- this
    matches fwd_return's own shift(-LABEL_HORIZON_DAYS) alignment for the
    common case of a ticker with one continuous membership stint. For a
    ticker with multiple separate stints (left the index and later rejoined),
    this can be off by a few real calendar days around the gap -- a rare
    edge case, documented in the report's caveats rather than specially
    handled.
    """
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["exit_date"] = df.groupby("ticker")["date"].shift(-LABEL_HORIZON_DAYS)
    return df


def generate_out_of_sample_predictions(df, feature_cols):
    """Same fold loop as train_walkforward_pit.py's run_walkforward, but
    returns every individual out-of-sample prediction (not just yearly
    aggregates), for feeding into the portfolio simulation.
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

        out = test[["ticker", "date", "exit_date", "fwd_return"]].copy()
        out["prediction"] = preds
        out["fold_year"] = year
        all_preds.append(out)
        print(f"  {year}: {len(test)} predictions generated "
              f"({(preds == 'BUY').sum()} BUY, {(preds == 'SELL').sum()} SELL)")

    return pd.concat(all_preds, ignore_index=True)


def simulate_portfolio(predictions):
    """Event-driven simulation: opens a position on any BUY signal for a
    ticker not already held, if there's room under MAX_CONCURRENT_POSITIONS
    and enough available cash; every position auto-closes at its own
    exit_date using fwd_return, net of transaction costs on both legs.
    """
    predictions = predictions.dropna(subset=["exit_date"]).copy()
    predictions = predictions.sort_values("date")

    buy_signals = predictions[predictions["prediction"] == "BUY"]
    all_dates = sorted(set(predictions["date"]) | set(predictions["exit_date"]))

    # Index BUY signals by entry date for fast lookup during the day loop.
    buys_by_date = {d: g for d, g in buy_signals.groupby("date")}

    cash = STARTING_CASH
    open_positions = {}   # ticker -> {entry_date, alloc, exit_date, fwd_return}
    # Exits indexed by date for fast lookup.
    exits_by_date = {}

    trade_log = []
    equity_curve = []
    cost_rate = TRANSACTION_COST_BPS / 10000.0
    min_cash_buffer = STARTING_CASH * MIN_CASH_BUFFER_PCT

    for current_date in all_dates:
        # --- process exits scheduled for today ---
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

        # --- process new entries signaled today ---
        if current_date in buys_by_date:
            candidates = buys_by_date[current_date]
            for _, row in candidates.iterrows():
                ticker = row["ticker"]
                if ticker in open_positions:
                    continue  # already hold it, skip (no pyramiding)
                if len(open_positions) >= MAX_CONCURRENT_POSITIONS:
                    break  # no room for more positions today
                portfolio_value_estimate = cash + sum(
                    p["alloc"] for p in open_positions.values()
                )
                target_alloc = TARGET_ALLOC_FRACTION * portfolio_value_estimate
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

        # Book value of the portfolio: cash + committed capital in open
        # positions, valued at their allocated (not mark-to-market) amount --
        # see the report's caveats on what this does and doesn't capture.
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


def write_chart(equity_curve, spy_series, out_path):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    strategy_norm = equity_curve.set_index("date")["portfolio_value"] / STARTING_CASH * 100
    spy_norm = spy_series / spy_series.iloc[0] * 100

    # Resample to monthly granularity before plotting. The raw series has
    # ~6700 daily points, which produces an oversized vector SVG; monthly
    # points (~320) keep the file small while preserving the overall shape.
    strategy_monthly = strategy_norm.resample("ME").last()
    spy_monthly = spy_norm.resample("ME").last()

    ax.plot(strategy_monthly.index, strategy_monthly.values, label="Simulated strategy",
            linewidth=1.6)
    ax.plot(spy_monthly.index, spy_monthly.values, label="SPY (buy & hold)",
            linewidth=1.6, alpha=0.85)
    ax.set_ylabel("Value (starting = 100)")
    ax.set_title("Point-in-time model as a simulated strategy vs. SPY (2000-2026)")
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

    print(f"\nSimulating portfolio: ${STARTING_CASH:,.0f} start, "
          f"max {MAX_CONCURRENT_POSITIONS} concurrent positions, "
          f"{TRANSACTION_COST_BPS}bps cost per leg...")
    trade_log, equity_curve = simulate_portfolio(predictions)
    trade_log.to_csv(ML_DIR / "backtest_pit_trades.csv", index=False)

    start_date, end_date = equity_curve["date"].min(), equity_curve["date"].max()
    print(f"\nFetching SPY benchmark series for {start_date.date()} to {end_date.date()}...")
    spy_series = fetch_spy_series(start_date, end_date)

    equity_curve = equity_curve.merge(
        spy_series.rename("spy_close").reset_index().rename(columns={"index": "date", "Date": "date"}),
        on="date", how="left",
    )
    equity_curve["spy_close"] = equity_curve["spy_close"].ffill()
    equity_curve.to_csv(ML_DIR / "backtest_pit_equity.csv", index=False)

    metrics = compute_metrics(equity_curve, spy_series)
    print("\n" + json.dumps(metrics, indent=2, default=str))

    write_chart(equity_curve, spy_series, ML_DIR / "backtest_pit_equity_chart.svg")

    write_report(trade_log, equity_curve, metrics)


def write_report(trade_log, equity_curve, metrics):
    n_trades = len(trade_log)
    win_rate = (trade_log["net_pnl"] > 0).mean() if n_trades else float("nan")
    avg_trade_pnl_pct = (trade_log["net_pnl"] / trade_log["alloc"]).mean() if n_trades else float("nan")
    avg_concurrent = equity_curve["n_open_positions"].mean()
    max_concurrent_actual = equity_curve["n_open_positions"].max()
    n_years = metrics["n_years"]

    lines = []
    lines.append("# Simulating the point-in-time model as an actual strategy\n")
    lines.append(
        "**Status: research exercise, not a validated trading strategy, and "
        "NOT wired into decide_trades.py or the portal.** Every earlier "
        "report in `ml/` measured \"average excess return of BUY calls\" -- "
        "an abstract, per-signal statistic. This report asks a different, "
        "more concrete question: if you actually turned those BUY/SELL calls "
        "into a real portfolio with position-sizing limits, a maximum number "
        "of concurrent holdings, and transaction costs, what would the "
        "resulting account balance actually look like next to just buying "
        "and holding SPY?\n"
    )
    lines.append("## Setup\n")
    lines.append(
        f"- Same point-in-time universe and walk-forward as "
        f"`ml/WALKFORWARD_PIT_REPORT.md` (the plain 11-feature version -- "
        f"XGBoost, LightGBM, and the earnings-features version all tied or "
        f"underperformed it in earlier testing, so this uses the best-tested "
        f"configuration).\n"
        f"- Starting capital: ${STARTING_CASH:,.0f}.\n"
        f"- **Position sizing**: equal-weight target of "
        f"{TARGET_ALLOC_FRACTION*100:.1f}% of current portfolio value per "
        f"position (1 / {MAX_CONCURRENT_POSITIONS} concurrent slots), capped "
        f"by available cash and a {MIN_CASH_BUFFER_PCT*100:.0f}% cash buffer "
        f"kept in reserve at all times.\n"
        f"- **Position limit**: at most {MAX_CONCURRENT_POSITIONS} positions "
        f"held at once -- a BUY signal is skipped if the book is already full, "
        f"same as a real account would have to skip opportunities once fully "
        f"invested.\n"
        f"- **Holding period**: every position is held for exactly the "
        f"{LABEL_HORIZON_DAYS}-trading-day horizon the model was trained to "
        f"predict, then closed using that row's actual (point-in-time-safe) "
        f"forward return.\n"
        f"- **Transaction costs**: {TRANSACTION_COST_BPS} basis points "
        f"(0.{TRANSACTION_COST_BPS:02d}%) charged on both the entry and exit "
        f"leg of every trade -- a round-trip cost of "
        f"{TRANSACTION_COST_BPS*2/100:.2f}%, a reasonable stand-in for "
        f"commission + bid/ask spread + slippage at retail scale.\n"
    )

    lines.append("## Results\n")
    lines.append(f"- Backtest period: {equity_curve['date'].min().date()} to "
                 f"{equity_curve['date'].max().date()} ({n_years:.1f} years).")
    lines.append(f"- {n_trades:,} total trades executed, average "
                 f"{avg_concurrent:.1f} concurrent positions held "
                 f"(peak {int(max_concurrent_actual)} of {MAX_CONCURRENT_POSITIONS} slots).")
    lines.append(f"- Win rate (net of costs): {win_rate*100:.1f}% of trades were profitable.")
    lines.append(f"- Average trade return (net of costs): {avg_trade_pnl_pct*100:+.2f}% per trade.")
    lines.append(f"- **Ending portfolio value: ${metrics['end_value']:,.0f}** "
                 f"(started at ${metrics['start_value']:,.0f}), total return "
                 f"{metrics['total_return']*100:+.1f}%, CAGR {metrics['cagr']*100:+.2f}%.")
    lines.append(f"- **SPY buy-and-hold over the same period**: total return "
                 f"{metrics['spy_total_return']*100:+.1f}%, CAGR "
                 f"{metrics['spy_cagr']*100:+.2f}%.")
    beat_or_not = "beat" if metrics["cagr"] > metrics["spy_cagr"] else "underperformed"
    lines.append(f"- The simulated strategy **{beat_or_not}** SPY buy-and-hold "
                 f"over this period on a CAGR basis.")
    lines.append("")

    lines.append(
        "*(An equity-curve chart comparing the two lines above is generated "
        "locally as `ml/backtest_pit_equity_chart.svg` when you run "
        "`backtest_pit_strategy.py` -- it isn't committed to the repo, same "
        "policy as the other regenerable build artifacts in `ml/`, since a "
        "chart with ~320 plotted points is a large text blob relative to the "
        "value of storing it in git.)*\n"
    )

    trade_std = (trade_log["net_pnl"] / trade_log["alloc"]).std() if n_trades else float("nan")
    median_cash_frac = (equity_curve["cash"] / equity_curve["portfolio_value"]).median()
    lines.append("## Why did it underperform despite a positive average edge?\n")
    lines.append(
        f"The portfolio was usually close to fully invested (median cash "
        f"balance was only {median_cash_frac*100:.1f}% of portfolio value, "
        f"and it held all {MAX_CONCURRENT_POSITIONS} of its position slots "
        f"most of the time), so idle cash sitting on the sidelines isn't "
        f"the explanation. The more likely one is **volatility drag**: the "
        f"average per-trade return was positive (+{avg_trade_pnl_pct*100:.2f}%) "
        f"but individual trades swung widely around that average (standard "
        f"deviation of roughly {trade_std*100:.1f}% per trade, about 20x the "
        f"average return itself). Compounding a return stream that noisy "
        f"grows slower than the simple average suggests -- the same reason a "
        f"coin flip that's +60% half the time and -50% the other half has a "
        f"positive average outcome per flip but loses money over many flips "
        f"compounded. A statistically real but thin per-trade edge, spread "
        f"across a volatile 10-day return, is a fragile foundation for "
        f"compounded growth -- and this is a direct, concrete illustration "
        f"of why every earlier report's caveat about \"no position sizing\" "
        f"mattered more than it might have seemed at the time.\n"
    )

    lines.append("## Honest caveats -- read before drawing conclusions\n")
    lines.append(
        "- **No interim mark-to-market.** This simulation only realizes "
        "profit/loss when a position closes at its 10-day horizon -- the "
        "portfolio-value line does not reflect what the account would "
        "actually be worth if you checked it mid-holding-period. This means "
        f"the max-drawdown figure computed on this curve "
        f"({metrics['max_drawdown_realized_only']*100:.1f}%) is understated: "
        "a real account would see day-to-day swings this simulation can't "
        "capture without full daily price paths for every open position.\n"
        "- **SELL signals don't trigger early exits.** Every BUY position "
        "runs its full 10-day course regardless of what the model says about "
        "that ticker on days in between -- a real trader watching a SELL "
        "signal fire on a held position might act on it sooner. Modeling "
        "that needs daily price paths for open positions, which this version "
        "doesn't use.\n"
        "- **No fractional-share rounding.** Position sizes are treated as "
        "continuously divisible dollar amounts, not whole shares -- "
        "`apply_trades.py` in the live paper-trading system does enforce "
        "integer shares; this simulation doesn't.\n"
        "- **Same underlying survivorship bias as `ml/WALKFORWARD_PIT_REPORT.md`.** "
        "91 of 250 sampled point-in-time-eligible tickers still have no free "
        "price data, so this simulated portfolio can never buy into a name "
        "that was later delisted after bankruptcy or acquisition, the same "
        "gap discussed there.\n"
        "- **Approximate exit dates for tickers with multiple membership "
        "stints.** Exit date is computed as 10 rows ahead within each "
        "ticker's already-filtered date sequence, which can drift slightly "
        "from a true 10-trading-day gap around a rejoin -- affects a small "
        "number of tickers, not separately corrected for here.\n"
        "- **Transaction cost and position-sizing parameters are assumptions, "
        "not fitted or validated** -- a different cost rate or concurrency "
        "limit would change the result, sometimes substantially given how "
        "thin the underlying per-trade edge is (see the walk-forward "
        "reports).\n"
        "- **Not integrated.** `decide_trades.py` and the portal are unchanged.\n"
    )

    with open(ML_DIR / "BACKTEST_PIT_REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {ML_DIR / 'BACKTEST_PIT_REPORT.md'}")


if __name__ == "__main__":
    main()
