import numpy as np
import pandas as pd
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType


def clean_close_prices(close_prices):
    close_prices = close_prices.copy()
    close_prices = close_prices.sort_index()
    close_prices = close_prices.loc[:, ~close_prices.columns.duplicated()]
    close_prices = close_prices.dropna(how="all")

    return close_prices


def clean_target_weights(target_weights, close_prices):
    target_weights = target_weights.copy()
    target_weights = target_weights.sort_index()
    target_weights = target_weights.loc[:, ~target_weights.columns.duplicated()]

    target_weights = target_weights.reindex(
        index=close_prices.index,
        columns=close_prices.columns,
    )

    target_weights = target_weights.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Normalize only rows that actually contain target weights.
    row_sums = target_weights.sum(axis=1, skipna=True)

    valid_rows = row_sums > 0

    target_weights.loc[valid_rows] = target_weights.loc[valid_rows].div(
        row_sums.loc[valid_rows],
        axis=0,
    )

    return target_weights


def make_sparse_rebalance_weights(target_weights, threshold=0.00):
    """
    Converts target weights into sparse rebalance orders.

    NaN rows mean: do nothing / keep current holdings.
    Non-NaN rows mean: submit a rebalance order.
    """

    sparse_weights = pd.DataFrame(
        np.nan,
        index=target_weights.index,
        columns=target_weights.columns,
        dtype="float64",
    )

    previous_weights = None

    for date in target_weights.index:
        row = target_weights.loc[date]

        # If the entire row is NaN, do nothing.
        if row.isna().all():
            continue

        current_weights = row.fillna(0)

        if previous_weights is None:
            if current_weights.sum() > 0:
                sparse_weights.loc[date] = current_weights
                previous_weights = current_weights
            continue

        total_weight_change = (
            current_weights - previous_weights
        ).abs().sum()

        if total_weight_change > threshold:
            sparse_weights.loc[date] = current_weights
            previous_weights = current_weights

    return sparse_weights


def backtest_weighted_portfolio(
    close_prices,
    target_weights,
    initial_capital=100_000,
    fees=0.001,
    slippage=0.001,
    shift_weights=True,
    rebalance_threshold=0.00,
):
    close_prices = clean_close_prices(close_prices)

    target_weights = clean_target_weights(
        target_weights=target_weights,
        close_prices=close_prices,
    )

    sparse_weights = make_sparse_rebalance_weights(
        target_weights=target_weights,
        threshold=rebalance_threshold,
    )

    if shift_weights:
        sparse_weights = sparse_weights.shift(1)

    portfolio = vbt.Portfolio.from_orders(
        close=close_prices,
        size=sparse_weights,
        size_type=SizeType.TargetPercent,
        init_cash=initial_capital,
        fees=fees,
        slippage=slippage,
        freq="1D",
        cash_sharing=True,
    )

    return portfolio


def backtest_equal_weight(
    close_prices,
    initial_capital=100_000,
    fees=0.001,
    slippage=0.001,
):
    """
    Buy-and-hold equal-weight benchmark.

    It buys once at the beginning and then holds.
    """

    close_prices = clean_close_prices(close_prices)

    n_assets = close_prices.shape[1]

    if n_assets == 0:
        raise ValueError("No assets available for equal-weight backtest.")

    weights = pd.DataFrame(
        np.nan,
        index=close_prices.index,
        columns=close_prices.columns,
        dtype="float64",
    )

    first_valid_date = close_prices.dropna(how="all").index[0]

    weights.loc[first_valid_date] = 1 / n_assets

    portfolio = backtest_weighted_portfolio(
        close_prices=close_prices,
        target_weights=weights,
        initial_capital=initial_capital,
        fees=fees,
        slippage=slippage,
        shift_weights=True,
        rebalance_threshold=0.00,
    )

    return portfolio