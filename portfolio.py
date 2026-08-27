"""
Portfolio state management for the paper-trading agent.

The portfolio is a JSON file that acts as the "brokerage account":
  {
    "cash": float,
    "positions": {"AAPL": {"shares": float, "avg_cost": float}, ...},
    "trade_log": [ {date, ticker, action, shares, price, reason}, ... ],
    "daily_history": [ {date, total_value, cash, positions_value, benchmark_value}, ... ]
  }
"""

import json
import os
from datetime import date

import config


def load_portfolio():
    if os.path.exists(config.PORTFOLIO_FILE):
        with open(config.PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    # First run: initialize a fresh paper account
    return {
        "cash": config.STARTING_CASH,
        "positions": {},
        "trade_log": [],
        "daily_history": [],
    }


def save_portfolio(portfolio):
    with open(config.PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2)


def positions_value(portfolio, prices):
    total = 0.0
    for ticker, pos in portfolio["positions"].items():
        price = prices.get(ticker)
        if price:
            total += pos["shares"] * price
    return total


def total_value(portfolio, prices):
    return portfolio["cash"] + positions_value(portfolio, prices)


def execute_trades(portfolio, decisions, prices, allowed_tickers=None, valuation_prices=None):
    """
    decisions: list of dicts like
        {"ticker": "AAPL", "action": "buy"|"sell", "shares": 3, "reason": "..."}
    Validates each against risk rules in config, executes what's valid,
    and returns a list of result dicts describing what happened.

    allowed_tickers: if given, restricts which tickers may be traded (e.g.
    today's candidate list), enforced here rather than just requested in the
    prompt, same as the other risk rules.

    valuation_prices: prices used to compute total portfolio value for the
    position-size/cash-buffer checks. Defaults to `prices`, but should be
    passed separately whenever `prices` doesn't cover every held position
    (e.g. a candidate subset of a larger universe) so those checks aren't
    skewed by silently-excluded holdings.
    """
    results = []
    today = str(date.today())
    port_value = total_value(portfolio, valuation_prices if valuation_prices is not None else prices)
    trades_done = 0
    allowed = set(allowed_tickers) if allowed_tickers is not None else None

    for d in decisions:
        if trades_done >= config.MAX_TRADES_PER_DAY:
            results.append({**d, "status": "rejected", "why": "daily trade cap reached"})
            continue

        ticker = d.get("ticker")
        action = d.get("action")
        shares = float(d.get("shares", 0))
        price = prices.get(ticker)

        if allowed is not None and ticker not in allowed:
            results.append({**d, "status": "rejected", "why": "ticker not in today's tradeable candidates"})
            continue
        if not price:
            results.append({**d, "status": "rejected", "why": "no price data for ticker"})
            continue
        if shares <= 0:
            results.append({**d, "status": "rejected", "why": "invalid share count"})
            continue

        if action == "buy":
            cost = shares * price
            cash_after = portfolio["cash"] - cost
            min_cash = config.MIN_CASH_BUFFER * port_value
            position_after_value = (
                portfolio["positions"].get(ticker, {}).get("shares", 0) * price + cost
            )

            if cash_after < min_cash:
                results.append({**d, "status": "rejected", "why": "would breach minimum cash buffer"})
                continue
            if position_after_value > config.MAX_POSITION_PCT * port_value:
                results.append({**d, "status": "rejected", "why": "would exceed max position size"})
                continue

            pos = portfolio["positions"].setdefault(ticker, {"shares": 0, "avg_cost": 0.0})
            new_shares = pos["shares"] + shares
            pos["avg_cost"] = ((pos["avg_cost"] * pos["shares"]) + cost) / new_shares
            pos["shares"] = new_shares
            portfolio["cash"] -= cost

            log_entry = {"date": today, "ticker": ticker, "action": "buy",
                         "shares": shares, "price": price, "reason": d.get("reason", "")}
            portfolio["trade_log"].append(log_entry)
            results.append({**d, "status": "executed", "price": price})
            trades_done += 1

        elif action == "sell":
            pos = portfolio["positions"].get(ticker)
            if not pos or pos["shares"] < shares:
                results.append({**d, "status": "rejected", "why": "insufficient shares held"})
                continue

            proceeds = shares * price
            pos["shares"] -= shares
            portfolio["cash"] += proceeds
            if pos["shares"] <= 1e-9:
                del portfolio["positions"][ticker]

            log_entry = {"date": today, "ticker": ticker, "action": "sell",
                         "shares": shares, "price": price, "reason": d.get("reason", "")}
            portfolio["trade_log"].append(log_entry)
            results.append({**d, "status": "executed", "price": price})
            trades_done += 1

        else:
            results.append({**d, "status": "rejected", "why": f"unknown action '{action}'"})

    return results
