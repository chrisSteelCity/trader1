"""
Writes REPORT.md — a plain-language summary of portfolio performance, current
holdings, and recent trades, meant to be committed to the repo so it's
readable directly on GitHub without running anything.

    python generate_report.py

Run this after apply_trades.py, then commit REPORT.md along with
portfolio.json / trade_log.csv (see TRADING_INSTRUCTIONS.md).
"""

from reporting import build_report_data

OUTPUT_FILE = "REPORT.md"


def _fmt_money(x):
    return f"${x:,.2f}" if x is not None else "n/a"


def _fmt_pct(x):
    return f"{x:+.2f}%" if x is not None else "n/a"


def render(data):
    lines = []
    lines.append("# Paper trading report")
    lines.append("")
    lines.append(f"_Last updated: {data['as_of']}_")
    lines.append("")

    diff = data["total_return_pct"] - data["bench_return_pct"]
    verdict = "🟢 beating" if diff > 0 else ("🔴 trailing" if diff < 0 else "⚪ matching")
    lines.append("## Performance")
    lines.append("")
    lines.append(f"- **Current value:** {_fmt_money(data['current_value'])} "
                  f"(started at {_fmt_money(data['start_value'])})")
    lines.append(f"- **Total return:** {_fmt_pct(data['total_return_pct'])}")
    lines.append(f"- **{data['bench_ticker']} return over same period:** {_fmt_pct(data['bench_return_pct'])}")
    lines.append(f"- **Verdict:** {verdict} the benchmark by {abs(diff):.2f} percentage points")
    lines.append(f"- **Cash on hand:** {_fmt_money(data['cash'])}")
    lines.append("")

    lines.append("## Value over time")
    lines.append("")
    lines.append("| Date | Portfolio value | Return | " + data["bench_ticker"] + " return |")
    lines.append("|---|---:|---:|---:|")
    start_bench = data["history"][0]["benchmark_price"]
    for h in data["history"]:
        port_return = (h["total_value"] / data["start_value"] - 1) * 100
        bench_return = (h["benchmark_price"] / start_bench - 1) * 100 if h["benchmark_price"] and start_bench else 0
        lines.append(f"| {h['date']} | {_fmt_money(h['total_value'])} | "
                      f"{_fmt_pct(port_return)} | {_fmt_pct(bench_return)} |")
    lines.append("")

    lines.append("## Current holdings")
    lines.append("")
    if data["positions"]:
        lines.append("| Ticker | Shares | Avg cost | Current price | Market value | Gain/loss | % |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for p in data["positions"]:
            if p["gain_loss"] is None:
                arrow = "⚪"
            else:
                arrow = "🟢" if p["gain_loss"] >= 0 else "🔴"
            lines.append(
                f"| {p['ticker']} | {p['shares']:.2f} | {_fmt_money(p['avg_cost'])} | "
                f"{_fmt_money(p['current_price'])} | {_fmt_money(p['market_value'])} | "
                f"{arrow} {_fmt_money(p['gain_loss'])} | {_fmt_pct(p['gain_loss_pct'])} |"
            )
    else:
        lines.append("_No open positions — fully in cash._")
    lines.append("")

    lines.append(f"## Recent trades (of {data['total_trade_count']} total)")
    lines.append("")
    if data["recent_trades"]:
        lines.append("| Date | Action | Shares | Ticker | Price | Reason |")
        lines.append("|---|---|---:|---|---:|---|")
        for t in data["recent_trades"]:
            lines.append(
                f"| {t['date']} | {t['action'].upper()} | {t['shares']:.2f} | {t['ticker']} | "
                f"{_fmt_money(t.get('price'))} | {t.get('reason', '')} |"
            )
    else:
        lines.append("_No trades yet._")
    lines.append("")

    return "\n".join(lines)


def main():
    data = build_report_data()
    if data is None:
        print("No history yet — run prepare_data.py + apply_trades.py at least once first.")
        return

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(render(data))

    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
