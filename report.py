"""
Run any time to see how the paper portfolio has performed so far.

    python report.py            # prints a table to the terminal
    python report.py --chart    # also saves performance_chart.png
"""

import sys
import config
import portfolio as port


def main():
    portfolio = port.load_portfolio()
    history = portfolio["daily_history"]

    if not history:
        print("No history yet — run daily_run.py at least once first.")
        return

    start_value = config.STARTING_CASH
    start_benchmark = history[0]["benchmark_price"]

    print(f"{'Date':<12}{'Value':>12}{'Return':>10}{'Benchmark Return':>20}")
    for h in history:
        port_return = (h["total_value"] / start_value - 1) * 100
        bench_return = (h["benchmark_price"] / start_benchmark - 1) * 100 if h["benchmark_price"] else 0
        print(f"{h['date']:<12}{h['total_value']:>12,.2f}{port_return:>9.2f}%{bench_return:>19.2f}%")

    latest = history[-1]
    total_return = (latest["total_value"] / start_value - 1) * 100
    bench_return = (latest["benchmark_price"] / start_benchmark - 1) * 100 if latest["benchmark_price"] else 0
    print(f"\nStarting cash: ${start_value:,.2f}")
    print(f"Current value: ${latest['total_value']:,.2f}")
    print(f"Total return:  {total_return:+.2f}%  (vs {config.BENCHMARK_TICKER}: {bench_return:+.2f}%)")

    if "--chart" in sys.argv:
        import matplotlib.pyplot as plt
        dates = [h["date"] for h in history]
        port_vals = [(h["total_value"] / start_value - 1) * 100 for h in history]
        bench_vals = [
            (h["benchmark_price"] / start_benchmark - 1) * 100 if h["benchmark_price"] else 0
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
