import numpy as np


def calculate_metrics(backtest_df):
    equity = backtest_df["equity"].dropna()
    returns = equity.pct_change().dropna()

    total_return = equity.iloc[-1] / equity.iloc[0] - 1

    annualized_return = (
        (1 + total_return) ** (252 / len(returns))
    ) - 1

    annualized_volatility = returns.std() * np.sqrt(252)

    sharpe = (
        annualized_return / annualized_volatility
        if annualized_volatility != 0
        else np.nan
    )

    peak = equity.cummax()

    drawdown = (equity - peak) / peak

    max_drawdown = drawdown.min()

    return {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe": sharpe,
        "Max Drawdown": max_drawdown,
    }