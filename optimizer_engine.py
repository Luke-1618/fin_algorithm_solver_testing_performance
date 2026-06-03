import numpy as np
import pandas as pd


def calculate_markowitz_weights(
    close_prices,
    lookback_window=252,
    max_weight=0.35,
    min_weight=0.0,
):
    returns = close_prices.pct_change().dropna()

    if returns.empty:
        return pd.Series(1 / close_prices.shape[1], index=close_prices.columns)

    recent_returns = returns.tail(lookback_window)

    mean_returns = recent_returns.mean() * 252
    cov_matrix = recent_returns.cov() * 252

    try:
        inv_cov = np.linalg.pinv(cov_matrix.values)
        raw_weights = inv_cov @ mean_returns.values

        raw_weights = np.maximum(raw_weights, 0)

        if raw_weights.sum() == 0:
            weights = np.ones(len(mean_returns)) / len(mean_returns)
        else:
            weights = raw_weights / raw_weights.sum()

    except Exception:
        weights = np.ones(len(mean_returns)) / len(mean_returns)

    weights = np.clip(weights, min_weight, max_weight)

    if weights.sum() == 0:
        weights = np.ones(len(mean_returns)) / len(mean_returns)
    else:
        weights = weights / weights.sum()

    return pd.Series(weights, index=mean_returns.index)


def build_markowitz_weights(
    close_prices,
    rebalance_frequency="M",
    lookback_window=252,
    max_weight=0.35,
):
    rebalance_weights = pd.DataFrame(
        np.nan,
        index=close_prices.index,
        columns=close_prices.columns,
    )

    rebalance_dates = close_prices.resample(rebalance_frequency).last().index

    for date in rebalance_dates:
        historical_prices = close_prices.loc[:date]

        if len(historical_prices) < 60:
            continue

        weights = calculate_markowitz_weights(
            historical_prices,
            lookback_window=lookback_window,
            max_weight=max_weight,
        )

        nearest_date = close_prices.index[close_prices.index <= date][-1]

        row = pd.Series(
            0.0,
            index=close_prices.columns,
        )

        row.loc[weights.index] = weights

        if row.sum() > 0:
            row = row / row.sum()

        rebalance_weights.loc[nearest_date] = row

    weights_history = rebalance_weights.ffill().fillna(0)

    row_sums = weights_history.sum(axis=1)

    weights_history = weights_history.div(
        row_sums.replace(0, np.nan),
        axis=0
    ).fillna(0)

    return weights_history