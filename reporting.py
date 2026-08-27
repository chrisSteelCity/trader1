"""
Shared computation logic for report.py (terminal) and generate_report.py
(committed REPORT.md). Fetches current prices for held positions so both
reports can show unrealized gain/loss, not just total portfolio value.
"""

from datetime import date, datetime

import config
import portfolio as port
from market_data import fetch_prices


def build_report_data():
    """Returns a dict with everything needed to render a report, or None if
    there's no trade history yet."""
    portfolio = port.load_portfolio()
    history = portfolio["daily_history"]
    if not history:
        return None

    start_value = config.STARTING_CASH
    start_benchmark = history[0]["benchmark_price"]
    latest = history[-1]

    total_return_pct = (latest["total_value"] / start_value - 1) * 100
    bench_return_pct = (
        (latest["benchmark_price"] / start_benchmark - 1) * 100
        if latest["benchmark_price"] and start_benchmark else 0
    )

    held_tickers = list(portfolio["positions"].keys())
    current_prices = fetch_prices(held_tickers) if held_tickers else {}

    positions = []
    for ticker, pos in portfolio["positions"].items():
        price = current_prices.get(ticker)
        market_value = price * pos["shares"] if price else None
        cost_basis = pos["avg_cost"] * pos["shares"]
        gain_loss = market_value - cost_basis if market_value is not None else None
        gain_loss_pct = (gain_loss / cost_basis * 100) if gain_loss is not None and cost_basis else None
        positions.append({
            "ticker": ticker,
            "shares": pos["shares"],
            "avg_cost": pos["avg_cost"],
            "current_price": price,
            "market_value": market_value,
            "gain_loss": gain_loss,
            "gain_loss_pct": gain_loss_pct,
        })
    positions.sort(key=lambda p: p["market_value"] or 0, reverse=True)

    trade_log = portfolio["trade_log"]
    recent_trades = list(reversed(trade_log[-20:]))

    return {
        "as_of": datetime.now().isoformat(timespec="seconds"),
        "start_value": start_value,
        "current_value": latest["total_value"],
        "cash": latest["cash"],
        "total_return_pct": total_return_pct,
        "bench_ticker": config.BENCHMARK_TICKER,
        "bench_return_pct": bench_return_pct,
        "history": history,
        "positions": positions,
        "recent_trades": recent_trades,
        "total_trade_count": len(trade_log),
    }
