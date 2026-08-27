"""
Run any time to see how the paper portfolio has performed so far.

    python report.py            # prints a table to the terminal
    python report.py --chart    # also saves performance_chart.png
"""

import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import config
from reporting import build_report_data


def _fmt_money(x):
    return f"${x:,.2f}" if x is not None else "n/a"


def _fmt_pct(x):
    return f"{x:+.2f}%" if x is not None else "n/a"


def main():
    data = build_report_data()
    if data is None:
        print("No history yet — run prepare_data.py + apply_trades.py at least once first.")
        return

    print(f"{'Date':<12}{'Value':>12}{'Return':>10}{'Benchmark Return':>20}")
    for h in data["history"]:
        port_return = (h["total_value"] / data["start_value"] - 1) * 100
        start_bench = data["history"][0]["benchmark_price"]
        bench_return = (h["benchmark_price"] / start_bench - 1) * 100 if h["benchmark_price"] and start_bench else 0
        print(f"{h['date']:<12}{h['total_value']:>12,.2f}{port_return:>9.2f}%{bench_return:>19.2f}%")

    print(f"\nStarting cash: {_fmt_money(data['start_value'])}")
    print(f"Current value: {_fmt_money(data['current_value'])}")
    diff = data["total_return_pct"] - data["bench_return_pct"]
    vs_word = "ahead of" if diff >= 0 else "behind"
    print(f"Total return:  {_fmt_pct(data['total_return_pct'])}  "
          f"(vs {data['bench_ticker']}: {_fmt_pct(data['bench_return_pct'])} - "
          f"{abs(diff):.2f}pp {vs_word} benchmark)")

    if data["positions"]:
        print(f"\nCurrent positions ({_fmt_money(data['cash'])} cash remaining):")
        print(f"{'Ticker':<8}{'Shares':>10}{'Avg Cost':>12}{'Price':>12}{'Value':>12}{'Gain/Loss':>14}{'%':>9}")
        for p in data["positions"]:
            print(f"{p['ticker']:<8}{p['shares']:>10.2f}{_fmt_money(p['avg_cost']):>12}"
                  f"{_fmt_money(p['current_price']):>12}{_fmt_money(p['market_value']):>12}"
                  f"{_fmt_money(p['gain_loss']):>14}{_fmt_pct(p['gain_loss_pct']):>9}")
    else:
        print("\nNo open positions — fully in cash.")

    if data["recent_trades"]:
        print(f"\nMost recent trades (of {data['total_trade_count']} total):")
        for t in data["recent_trades"][:10]:
            print(f"  {t['date']}  {t['action'].upper():<4} {t['shares']:>8.2f} {t['ticker']:<6} "
                  f"@ {_fmt_money(t['price'])}  — {t.get('reason', '')}")

    if "--chart" in sys.argv:
        import matplotlib.pyplot as plt
        history = data["history"]
        start_value = data["start_value"]
        start_benchmark = history[0]["benchmark_price"]
        dates = [h["date"] for h in history]
        port_vals = [(h["total_value"] / start_value - 1) * 100 for h in history]
        bench_vals = [
            (h["benchmark_price"] / start_benchmark - 1) * 100 if h["benchmark_price"] and start_benchmark else 0
            for h in history
        ]
        plt.figure(figsize=(9, 5))
        plt.plot(dates, port_vals, label="Agent portfolio", linewidth=2)
        plt.plot(dates, bench_vals, label=config.BENCHMARK_TICKER, linewidth=2, linestyle="--")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Return (%)")
        plt.title("Paper Trading Agent vs Benchmark")
        plt.legend()
        plt.tight_layout()
        plt.savefig("performance_chart.png", dpi=150)
        print("\nSaved performance_chart.png")


if __name__ == "__main__":
    main()
