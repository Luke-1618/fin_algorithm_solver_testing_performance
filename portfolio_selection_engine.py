import numpy as np
import pandas as pd


def build_score_matrix(signal_data, short_window=30, long_window=200):
    score_parts = []

    ema_short = f"ema_{short_window}"
    ema_long = f"ema_{long_window}"

    for ticker, df in signal_data.items():
        out = df.copy()

        trend_score = np.where(out[ema_short] > out[ema_long], 1.0, 0.0)
        momentum_score = np.where(out["macd"] > out["macd_signal"], 1.0, 0.0)
        rsi_score = np.where(out["rsi"] < 70, 1.0, 0.0)

        vol_median = out["volatility_30d"].rolling(
            252,
            min_periods=30,
        ).median()

        risk_score = np.where(
            out["volatility_30d"] < vol_median,
            1.0,
            0.0,
        )

        score = (
            0.40 * trend_score +
            0.30 * momentum_score +
            0.15 * rsi_score +
            0.15 * risk_score
        )

        score_parts.append(
            pd.Series(score, index=out.index, name=ticker)
        )

    if not score_parts:
        return pd.DataFrame()

    return pd.concat(score_parts, axis=1).sort_index()


def calculate_markowitz_weights_for_selected(
    close_prices,
    selected_assets,
    lookback_window=252,
    max_weight=0.40,
):
    selected_prices = close_prices[selected_assets].dropna(how="all")
    returns = selected_prices.pct_change().dropna()

    if returns.empty:
        return pd.Series(
            1 / len(selected_assets),
            index=selected_assets,
        )

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
        weights = np.ones(len(selected_assets)) / len(selected_assets)

        return pd.Series(
            weights,
            index=selected_assets,
        )

    weights = np.clip(weights, 0, max_weight)

    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(mean_returns)) / len(mean_returns)

    return pd.Series(
        weights,
        index=mean_returns.index,
    )


def select_with_correlation_filter(
    scores,
    close_prices,
    max_assets=10,
    correlation_lookback=126,
    max_correlation=0.80,
):
    ranked = scores.sort_values(ascending=False)
    selected = []

    recent_returns = (
        close_prices
        .pct_change()
        .tail(correlation_lookback)
        .dropna(how="all")
    )

    corr_matrix = recent_returns.corr()

    for ticker in ranked.index:
        if ticker not in corr_matrix.columns:
            continue

        if not selected:
            selected.append(ticker)
        else:
            correlations = corr_matrix.loc[ticker, selected].abs()

            if correlations.max() <= max_correlation:
                selected.append(ticker)

        if len(selected) >= max_assets:
            break

    return selected


def calculate_daily_exposure_target(
    close_prices,
    benchmark="SPY",
    growth_benchmark="QQQ",
    trend_window=200,
    volatility_window=20,
):
    exposure = pd.Series(
        0.60,
        index=close_prices.index,
        dtype="float64",
    )

    if benchmark not in close_prices.columns:
        return exposure

    spy = close_prices[benchmark].reindex(close_prices.index).ffill()
    spy_ma = spy.rolling(trend_window).mean()

    spy_above_ma = (spy > spy_ma).reindex(close_prices.index).fillna(False)
    spy_distance = ((spy / spy_ma) - 1).reindex(close_prices.index)

    if growth_benchmark in close_prices.columns:
        qqq = close_prices[growth_benchmark].reindex(close_prices.index).ffill()
        qqq_ma = qqq.rolling(trend_window).mean()
        qqq_above_ma = (qqq > qqq_ma).reindex(close_prices.index).fillna(False)
    else:
        qqq_above_ma = spy_above_ma.copy()

    returns = spy.pct_change()
    realized_vol = returns.rolling(volatility_window).std() * np.sqrt(252)
    vol_median = realized_vol.rolling(252, min_periods=30).median()
    low_vol = (realized_vol < vol_median).reindex(close_prices.index).fillna(False)

    asset_ma = close_prices.rolling(trend_window).mean()
    asset_above_ma = close_prices > asset_ma
    breadth = asset_above_ma.mean(axis=1).reindex(close_prices.index).fillna(0)

    strong_breadth = (breadth > 0.60).reindex(close_prices.index).fillna(False)
    weak_breadth = (breadth < 0.35).reindex(close_prices.index).fillna(True)

    strong_bull = spy_above_ma & qqq_above_ma & low_vol & strong_breadth
    bull = spy_above_ma & qqq_above_ma & ~weak_breadth
    mixed_bull = spy_above_ma & ~qqq_above_ma
    weak_market = ~spy_above_ma & ~weak_breadth
    bear_market = ~spy_above_ma & weak_breadth

    exposure.loc[strong_bull] = 1.00
    exposure.loc[bull] = 0.90
    exposure.loc[mixed_bull] = 0.75
    exposure.loc[weak_market] = 0.50
    exposure.loc[bear_market] = 0.25
    exposure.loc[spy_distance > 0.10] = 1.00

    exposure = (
        exposure
        .reindex(close_prices.index)
        .ffill()
        .fillna(0.60)
        .clip(lower=0.25, upper=1.00)
    )

    return exposure


def apply_daily_regime_exposure_overlay(
    weights,
    close_prices,
):
    exposure = calculate_daily_exposure_target(close_prices)

    row_sums = weights.sum(axis=1)

    normalized_weights = weights.div(
        row_sums.replace(0, np.nan),
        axis=0,
    ).fillna(0)

    adjusted_weights = normalized_weights.mul(
        exposure,
        axis=0,
    )

    return adjusted_weights.clip(lower=0)


def build_dynamic_signal_markowitz_weights(
    signal_data,
    close_prices,
    top_n=10,
    rebalance_frequency="M",
    lookback_window=252,
    max_weight=0.40,
    entry_quantile=0.75,
    exit_quantile=0.50,
    min_holding_periods=2,
    max_correlation=0.80,
    correlation_lookback=126,
    turnover_buffer=0.05,
):
    max_assets = top_n

    score_matrix = build_score_matrix(signal_data)

    if score_matrix.empty:
        return pd.DataFrame()

    weights_history = pd.DataFrame(
        np.nan,
        index=close_prices.index,
        columns=close_prices.columns,
    )

    rebalance_dates = close_prices.resample(
        rebalance_frequency
    ).last().index

    current_holdings = []
    holding_age = {}
    previous_weights = pd.Series(
        0.0,
        index=close_prices.columns,
    )

    for date in rebalance_dates:
        available_scores = score_matrix.loc[:date]

        if available_scores.empty:
            continue

        current_scores = available_scores.iloc[-1].dropna()

        if current_scores.empty:
            continue

        valid_scores = current_scores[
            current_scores.index.isin(close_prices.columns)
        ]

        if valid_scores.empty:
            continue

        entry_threshold = valid_scores.quantile(entry_quantile)
        exit_threshold = valid_scores.quantile(exit_quantile)

        persistent_holdings = []

        for ticker in current_holdings:
            score = valid_scores.get(ticker, np.nan)
            age = holding_age.get(ticker, 0)

            if pd.isna(score):
                continue

            still_good = score >= exit_threshold
            still_locked = age < min_holding_periods

            if still_good or still_locked:
                persistent_holdings.append(ticker)

        candidate_scores = valid_scores[
            valid_scores >= entry_threshold
        ]

        candidate_scores = candidate_scores.drop(
            labels=persistent_holdings,
            errors="ignore",
        )

        remaining_slots = max(
            max_assets - len(persistent_holdings),
            0,
        )

        selected_new = []

        if remaining_slots > 0 and not candidate_scores.empty:
            selected_new = select_with_correlation_filter(
                scores=candidate_scores,
                close_prices=close_prices.loc[:date],
                max_assets=remaining_slots,
                correlation_lookback=correlation_lookback,
                max_correlation=max_correlation,
            )

        selected_assets = persistent_holdings + selected_new
        selected_assets = selected_assets[:max_assets]

        if not selected_assets:
            row = pd.Series(
                0.0,
                index=close_prices.columns,
            )
        else:
            historical_prices = close_prices.loc[
                :date,
                selected_assets,
            ]

            if len(historical_prices.dropna(how="all")) < 60:
                target_weights = pd.Series(
                    1 / len(selected_assets),
                    index=selected_assets,
                )
            else:
                target_weights = calculate_markowitz_weights_for_selected(
                    close_prices=historical_prices,
                    selected_assets=selected_assets,
                    lookback_window=lookback_window,
                    max_weight=max_weight,
                )

            row = pd.Series(
                0.0,
                index=close_prices.columns,
            )

            row.loc[target_weights.index] = target_weights

        total_change = (row - previous_weights).abs().sum()

        if total_change < turnover_buffer:
            row = previous_weights.copy()
        else:
            previous_weights = row.copy()

        nearest_date = close_prices.index[
            close_prices.index <= date
        ][-1]

        weights_history.loc[nearest_date] = row

        new_holdings = row[row > 0].index.tolist()

        updated_age = {}

        for ticker in new_holdings:
            updated_age[ticker] = (
                holding_age.get(ticker, 0) + 1
            )

        holding_age = updated_age
        current_holdings = new_holdings

    weights_history = weights_history.ffill().fillna(0)
    weights_history = weights_history.clip(lower=0)

    weights_history = apply_daily_regime_exposure_overlay(
        weights=weights_history,
        close_prices=close_prices,
    )

    return weights_history


def get_latest_selected_portfolio(weights):
    if weights.empty:
        return pd.Series(dtype=float)

    latest = weights.iloc[-1]
    selected = latest[latest > 0]

    return selected.sort_values(ascending=False)


def build_selection_diagnostics(weights, close_prices):
    diagnostics = pd.DataFrame(index=weights.index)

    diagnostics["invested_exposure"] = weights.sum(axis=1)
    diagnostics["cash_exposure"] = 1 - diagnostics["invested_exposure"]
    diagnostics["assets_held"] = (weights > 0).sum(axis=1)

    if "SPY" in close_prices.columns:
        spy = close_prices["SPY"]
        spy_ma_200 = spy.rolling(200).mean()

        diagnostics["spy_close"] = spy
        diagnostics["spy_ma_200"] = spy_ma_200
        diagnostics["spy_above_200d"] = spy > spy_ma_200
        diagnostics["spy_distance_to_200d"] = (spy / spy_ma_200) - 1

    diagnostics["daily_exposure_target"] = calculate_daily_exposure_target(
        close_prices
    ).reindex(weights.index)

    return diagnostics