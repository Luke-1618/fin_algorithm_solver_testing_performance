import numpy as np
import pandas as pd


def calculate_cvar(returns, confidence=0.95):
    cutoff = returns.quantile(1 - confidence)
    tail = returns[returns <= cutoff]

    if tail.empty:
        return np.nan

    return tail.mean()


def build_portfolio_risk_report(portfolio_value):
    equity = portfolio_value.dropna()
    returns = equity.pct_change().dropna()

    if equity.empty or returns.empty:
        return {}

    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    return {
        "Portfolio Volatility": returns.std() * np.sqrt(252),
        "Portfolio VaR 95": returns.quantile(0.05),
        "Portfolio CVaR 95": calculate_cvar(returns),
        "Portfolio Max Drawdown": drawdown.min(),
        "Current Drawdown": drawdown.iloc[-1],
    }


def build_universe_risk_table(signal_data):
    rows = []

    for ticker, df in signal_data.items():
        clean = df.dropna()

        if clean.empty:
            continue

        latest = clean.iloc[-1]

        rows.append({
            "Ticker": ticker,
            "Close": latest.get("Close"),
            "Trend": latest.get("trend"),
            "RSI": latest.get("rsi"),
            "Volatility 30D": latest.get("volatility_30d"),
            "Z-Score": latest.get("z_score"),
            "Return 1D": latest.get("return"),
        })

    return pd.DataFrame(rows)