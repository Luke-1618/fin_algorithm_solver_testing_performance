import numpy as np
import pandas as pd
import plotly.graph_objects as go

from signal_engine import compute_asset_signals
from portfolio_selection_engine import build_dynamic_signal_markowitz_weights
from backtest_engine import backtest_equal_weight, backtest_weighted_portfolio


# ============================================================
# BASIC METRICS
# ============================================================

def calculate_drawdown(equity_curve):
    equity_curve = np.asarray(equity_curve, dtype=float)

    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve / running_max) - 1.0

    return drawdown


def calculate_max_drawdown(equity_curve):
    drawdown = calculate_drawdown(equity_curve)
    return float(np.nanmin(drawdown))


def calculate_total_return(equity_curve):
    equity_curve = pd.Series(equity_curve).dropna()

    if equity_curve.empty:
        return np.nan

    return float((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0)


def calculate_cagr(equity_curve, trading_days=252):
    equity_curve = pd.Series(equity_curve).dropna()

    if len(equity_curve) < 2:
        return np.nan

    years = len(equity_curve) / trading_days

    if years <= 0:
        return np.nan

    return float((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1)


def calculate_sharpe(equity_curve, trading_days=252):
    equity_curve = pd.Series(equity_curve).dropna()
    returns = equity_curve.pct_change().dropna()

    if returns.empty or returns.std() == 0:
        return np.nan

    return float((returns.mean() / returns.std()) * np.sqrt(trading_days))


# ============================================================
# SYNTHETIC UNIVERSE GENERATION
# ============================================================

def build_return_matrix(close_prices):
    returns = close_prices.pct_change().dropna(how="all")
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(axis=1, how="all")

    return returns


def block_bootstrap_return_matrix(
    returns,
    total_days,
    block_size=20,
    random_seed=None,
):
    """
    Paired block bootstrap.

    This samples blocks of historical dates across ALL assets together.

    That preserves:
    - cross-asset correlations
    - sector relationships
    - volatility clusters
    - crash/recovery sequences
    """

    if returns.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(random_seed)

    returns = returns.dropna(how="all")

    if len(returns) < block_size + 1:
        return pd.DataFrame()

    max_start = len(returns) - block_size
    sampled_blocks = []

    while sum(len(block) for block in sampled_blocks) < total_days:
        start_idx = rng.integers(0, max_start + 1)
        block = returns.iloc[start_idx:start_idx + block_size].copy()
        sampled_blocks.append(block)

    synthetic_returns = pd.concat(sampled_blocks, axis=0)
    synthetic_returns = synthetic_returns.iloc[:total_days].copy()

    synthetic_returns.index = pd.bdate_range(
        start="2000-01-03",
        periods=len(synthetic_returns),
    )

    synthetic_returns = synthetic_returns.fillna(0.0)

    return synthetic_returns


def returns_to_price_matrix(
    synthetic_returns,
    start_price=100.0,
):
    synthetic_prices = start_price * (1.0 + synthetic_returns).cumprod()

    synthetic_prices = synthetic_prices.replace([np.inf, -np.inf], np.nan)
    synthetic_prices = synthetic_prices.ffill().bfill()

    return synthetic_prices


def price_matrix_to_price_data(synthetic_prices):
    """
    Converts synthetic close matrix into the dictionary format expected by signal_engine.

    Most signal engines only need Close, but this gives a simple OHLCV-like structure.
    """

    price_data = {}

    for ticker in synthetic_prices.columns:
        close = synthetic_prices[ticker].dropna()

        if close.empty:
            continue

        df = pd.DataFrame(index=close.index)
        df["Open"] = close
        df["High"] = close
        df["Low"] = close
        df["Close"] = close
        df["Adj Close"] = close
        df["Volume"] = 1_000_000

        price_data[ticker] = df

    return price_data


# ============================================================
# SINGLE FULL-ENGINE SIMULATION
# ============================================================

def run_single_universe_simulation(
    close_prices,
    initial_capital=100000,
    fees=0.001,
    slippage=0.001,
    max_assets=10,
    rebalance_frequency="M",
    rebalance_threshold=0.03,
    warmup_days=252,
    forecast_days=252,
    block_size=20,
    random_seed=None,
):
    """
    One full-engine universe simulation.

    Steps:
    1. Bootstrap synthetic universe returns.
    2. Convert returns into synthetic prices.
    3. Recompute signals.
    4. Rebuild Dynamic Signal-Markowitz weights.
    5. Backtest Dynamic strategy.
    6. Backtest synthetic SPY.
    7. Evaluate only the forecast window after warmup.
    """

    total_days = warmup_days + forecast_days

    real_returns = build_return_matrix(close_prices)

    synthetic_returns = block_bootstrap_return_matrix(
        returns=real_returns,
        total_days=total_days,
        block_size=block_size,
        random_seed=random_seed,
    )

    if synthetic_returns.empty:
        return None

    synthetic_prices = returns_to_price_matrix(
        synthetic_returns,
        start_price=100.0,
    )

    if "SPY" not in synthetic_prices.columns:
        return None

    synthetic_price_data = price_matrix_to_price_data(
        synthetic_prices
    )

    signal_data = compute_asset_signals(
        synthetic_price_data
    )

    dynamic_weights = build_dynamic_signal_markowitz_weights(
        signal_data=signal_data,
        close_prices=synthetic_prices,
        top_n=max_assets,
        rebalance_frequency=rebalance_frequency,
        lookback_window=252,
        max_weight=0.40,
        entry_quantile=0.75,
        exit_quantile=0.50,
    )

    dynamic_pf = backtest_weighted_portfolio(
        synthetic_prices,
        dynamic_weights,
        initial_capital=initial_capital,
        fees=fees,
        slippage=slippage,
        shift_weights=True,
        rebalance_threshold=rebalance_threshold,
    )

    spy_pf = backtest_equal_weight(
        synthetic_prices[["SPY"]],
        initial_capital=initial_capital,
        fees=0,
        slippage=0,
    )

    dynamic_value_full = dynamic_pf.value().dropna()
    spy_value_full = spy_pf.value().dropna()

    if len(dynamic_value_full) <= warmup_days or len(spy_value_full) <= warmup_days:
        return None

    dynamic_forecast_value = dynamic_value_full.iloc[warmup_days:].copy()
    spy_forecast_value = spy_value_full.iloc[warmup_days:].copy()

    # Rebase forecast window to initial capital.
    dynamic_forecast_value = (
        dynamic_forecast_value / dynamic_forecast_value.iloc[0]
    ) * initial_capital

    spy_forecast_value = (
        spy_forecast_value / spy_forecast_value.iloc[0]
    ) * initial_capital

    dynamic_final = float(dynamic_forecast_value.iloc[-1])
    spy_final = float(spy_forecast_value.iloc[-1])

    dynamic_drawdown = calculate_max_drawdown(dynamic_forecast_value.values)
    spy_drawdown = calculate_max_drawdown(spy_forecast_value.values)

    dynamic_total_return = calculate_total_return(dynamic_forecast_value.values)
    spy_total_return = calculate_total_return(spy_forecast_value.values)

    dynamic_cagr = calculate_cagr(dynamic_forecast_value.values)
    spy_cagr = calculate_cagr(spy_forecast_value.values)

    dynamic_sharpe = calculate_sharpe(dynamic_forecast_value.values)
    spy_sharpe = calculate_sharpe(spy_forecast_value.values)

    dynamic_exposure = dynamic_weights.iloc[warmup_days:].sum(axis=1)
    average_cash_exposure = float((1 - dynamic_exposure).clip(0, 1).mean())
    average_assets_held = float((dynamic_weights.iloc[warmup_days:] > 0).sum(axis=1).mean())

    return {
        "dynamic_curve": dynamic_forecast_value.reset_index(drop=True),
        "spy_curve": spy_forecast_value.reset_index(drop=True),
        "dynamic_final_value": dynamic_final,
        "spy_final_value": spy_final,
        "dynamic_total_return": dynamic_total_return,
        "spy_total_return": spy_total_return,
        "dynamic_cagr": dynamic_cagr,
        "spy_cagr": spy_cagr,
        "dynamic_sharpe": dynamic_sharpe,
        "spy_sharpe": spy_sharpe,
        "dynamic_max_drawdown": dynamic_drawdown,
        "spy_max_drawdown": spy_drawdown,
        "dynamic_beats_spy": dynamic_final > spy_final,
        "dynamic_loss": dynamic_final < initial_capital,
        "spy_loss": spy_final < initial_capital,
        "average_cash_exposure": average_cash_exposure,
        "average_assets_held": average_assets_held,
    }


# ============================================================
# FULL UNIVERSE MONTE CARLO
# ============================================================

def run_universe_monte_carlo(
    close_prices,
    initial_capital=100000,
    fees=0.001,
    slippage=0.001,
    max_assets=10,
    rebalance_frequency="M",
    rebalance_threshold=0.03,
    simulations=100,
    warmup_days=252,
    forecast_days=252,
    block_size=20,
    random_seed=42,
):
    """
    Full-engine universe Monte Carlo.

    This is slow compared to portfolio-return Monte Carlo.

    Recommended testing:
    - start with simulations=25 or 50
    - then increase to 100, 250, 500, 1000 later
    """

    simulation_rows = []
    dynamic_curves = []
    spy_curves = []

    for sim in range(simulations):
        result = run_single_universe_simulation(
            close_prices=close_prices,
            initial_capital=initial_capital,
            fees=fees,
            slippage=slippage,
            max_assets=max_assets,
            rebalance_frequency=rebalance_frequency,
            rebalance_threshold=rebalance_threshold,
            warmup_days=warmup_days,
            forecast_days=forecast_days,
            block_size=block_size,
            random_seed=random_seed + sim,
        )

        if result is None:
            continue

        simulation_rows.append(
            {
                "Simulation": sim + 1,
                "Dynamic Final Value": result["dynamic_final_value"],
                "SPY Final Value": result["spy_final_value"],
                "Dynamic Total Return": result["dynamic_total_return"],
                "SPY Total Return": result["spy_total_return"],
                "Dynamic CAGR": result["dynamic_cagr"],
                "SPY CAGR": result["spy_cagr"],
                "Dynamic Sharpe": result["dynamic_sharpe"],
                "SPY Sharpe": result["spy_sharpe"],
                "Dynamic Max Drawdown": result["dynamic_max_drawdown"],
                "SPY Max Drawdown": result["spy_max_drawdown"],
                "Dynamic Beats SPY": result["dynamic_beats_spy"],
                "Dynamic Loss": result["dynamic_loss"],
                "SPY Loss": result["spy_loss"],
                "Average Cash Exposure": result["average_cash_exposure"],
                "Average Assets Held": result["average_assets_held"],
            }
        )

        dynamic_curves.append(result["dynamic_curve"])
        spy_curves.append(result["spy_curve"])

    if not simulation_rows:
        return None

    results_df = pd.DataFrame(simulation_rows)

    dynamic_curves_df = pd.concat(dynamic_curves, axis=1)
    spy_curves_df = pd.concat(spy_curves, axis=1)

    dynamic_curves_df.columns = range(1, dynamic_curves_df.shape[1] + 1)
    spy_curves_df.columns = range(1, spy_curves_df.shape[1] + 1)

    summary = {
        "Simulations Completed": int(len(results_df)),
        "Median Final Value": float(results_df["Dynamic Final Value"].median()),
        "5th Percentile Final Value": float(results_df["Dynamic Final Value"].quantile(0.05)),
        "95th Percentile Final Value": float(results_df["Dynamic Final Value"].quantile(0.95)),
        "Probability of Loss": float(results_df["Dynamic Loss"].mean()),
        "Probability of Beating SPY": float(results_df["Dynamic Beats SPY"].mean()),
        "Worst Simulated Drawdown": float(results_df["Dynamic Max Drawdown"].min()),
        "Median Simulated Drawdown": float(results_df["Dynamic Max Drawdown"].median()),
        "Median CAGR": float(results_df["Dynamic CAGR"].median()),
        "Median Sharpe": float(results_df["Dynamic Sharpe"].median()),
        "Average Cash Exposure": float(results_df["Average Cash Exposure"].mean()),
        "Average Assets Held": float(results_df["Average Assets Held"].mean()),
    }

    return {
        "summary": summary,
        "simulation_results": results_df,
        "dynamic_curves": dynamic_curves_df,
        "spy_curves": spy_curves_df,
    }


# ============================================================
# PLOTS
# ============================================================

def plot_universe_monte_carlo_fan_chart(
    universe_mc_result,
    title="Universe Monte Carlo Fan Chart",
):
    curves = universe_mc_result["dynamic_curves"]

    median_curve = curves.median(axis=1)

    percentile_5 = curves.quantile(0.05, axis=1)
    percentile_25 = curves.quantile(0.25, axis=1)
    percentile_75 = curves.quantile(0.75, axis=1)
    percentile_95 = curves.quantile(0.95, axis=1)

    start_value = float(curves.iloc[0].median())

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=curves.index,
            y=percentile_95,
            mode="lines",
            name="95th Percentile",
            line=dict(width=0),
            hovertemplate="Day %{x}<br>95th: $%{y:,.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=curves.index,
            y=percentile_75,
            mode="lines",
            name="75th Percentile",
            fill="tonexty",
            fillcolor="rgba(0, 204, 150, 0.22)",
            line=dict(width=0),
            hovertemplate="Day %{x}<br>75th: $%{y:,.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=curves.index,
            y=percentile_25,
            mode="lines",
            name="25th Percentile",
            fill="tonexty",
            fillcolor="rgba(99, 110, 250, 0.30)",
            line=dict(width=0),
            hovertemplate="Day %{x}<br>25th: $%{y:,.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=curves.index,
            y=percentile_5,
            mode="lines",
            name="5th Percentile",
            fill="tonexty",
            fillcolor="rgba(239, 85, 59, 0.22)",
            line=dict(width=0),
            hovertemplate="Day %{x}<br>5th: $%{y:,.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=curves.index,
            y=median_curve,
            mode="lines",
            name="Median Outcome",
            line=dict(
                width=4,
                color="#FFFFFF",
            ),
            hovertemplate="Day %{x}<br>Median: $%{y:,.0f}<extra></extra>",
        )
    )

    fig.add_hline(
        y=start_value,
        line_width=1,
        line_dash="dash",
        line_color="gray",
        annotation_text="Starting Value",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Forecast Trading Days",
        yaxis_title="Portfolio Value",
        template="plotly_dark",
        height=600,
        hovermode="x unified",
    )

    fig.update_yaxes(
        tickprefix="$",
        separatethousands=True,
    )

    return fig


def plot_universe_monte_carlo_final_values(
    universe_mc_result,
    title="Universe Monte Carlo Final Value Distribution",
):
    results_df = universe_mc_result["simulation_results"]

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=results_df["Dynamic Final Value"],
            name="Dynamic Signal-Markowitz",
            opacity=0.45,
            nbinsx=50,
            histnorm="probability density",
        )
    )

    fig.add_trace(
        go.Histogram(
            x=results_df["SPY Final Value"],
            name="SPY Benchmark",
            opacity=0.45,
            nbinsx=50,
            histnorm="probability density",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Final Portfolio Value",
        yaxis_title="Density",
        barmode="overlay",
        template="plotly_dark",
        height=500,
    )

    return fig