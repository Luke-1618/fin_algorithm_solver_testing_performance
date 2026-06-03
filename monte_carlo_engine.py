import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import gaussian_kde


def calculate_drawdown(equity_curve):
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve / running_max) - 1
    return drawdown


def run_monte_carlo_bootstrap(
    portfolio_values,
    initial_capital=100000,
    simulations=1000,
    forecast_days=252,
    random_seed=42,
):
    values = portfolio_values.dropna()

    returns = values.pct_change().dropna()

    if returns.empty:
        return None

    rng = np.random.default_rng(random_seed)

    simulated_curves = []

    returns_array = returns.values

    for _ in range(simulations):
        sampled_returns = rng.choice(
            returns_array,
            size=forecast_days,
            replace=True,
        )

        equity_curve = initial_capital * np.cumprod(
            1 + sampled_returns
        )

        simulated_curves.append(equity_curve)

    simulated_curves = pd.DataFrame(simulated_curves).T

    simulated_curves.index = range(1, forecast_days + 1)

    final_values = simulated_curves.iloc[-1]

    drawdowns = simulated_curves.apply(
        lambda curve: calculate_drawdown(curve.values).min(),
        axis=0,
    )

    summary = {
        "Median Final Value": float(final_values.median()),
        "5th Percentile Final Value": float(final_values.quantile(0.05)),
        "95th Percentile Final Value": float(final_values.quantile(0.95)),
        "Probability of Loss": float((final_values < initial_capital).mean()),
        "Worst Simulated Drawdown": float(drawdowns.min()),
        "Median Simulated Drawdown": float(drawdowns.median()),
    }

    return {
        "simulated_curves": simulated_curves,
        "final_values": final_values,
        "drawdowns": drawdowns,
        "summary": summary,
    }


def compare_strategy_to_benchmark_monte_carlo(
    strategy_values,
    benchmark_values,
    initial_capital=100000,
    simulations=1000,
    forecast_days=252,
    random_seed=42,
):
    strategy_mc = run_monte_carlo_bootstrap(
        portfolio_values=strategy_values,
        initial_capital=initial_capital,
        simulations=simulations,
        forecast_days=forecast_days,
        random_seed=random_seed,
    )

    benchmark_mc = run_monte_carlo_bootstrap(
        portfolio_values=benchmark_values,
        initial_capital=initial_capital,
        simulations=simulations,
        forecast_days=forecast_days,
        random_seed=random_seed + 1,
    )

    if strategy_mc is None or benchmark_mc is None:
        return None

    strategy_final = strategy_mc["final_values"]
    benchmark_final = benchmark_mc["final_values"]

    probability_beating_benchmark = float(
        (strategy_final.values > benchmark_final.values).mean()
    )

    return {
        "strategy": strategy_mc,
        "benchmark": benchmark_mc,
        "Probability of Beating SPY": probability_beating_benchmark,
    }


def plot_monte_carlo_fan_chart(
    mc_result,
    title="Monte Carlo Simulation",
):
    curves = mc_result["simulated_curves"]

    median_curve = curves.median(axis=1)

    percentile_5 = curves.quantile(0.05, axis=1)
    percentile_25 = curves.quantile(0.25, axis=1)
    percentile_75 = curves.quantile(0.75, axis=1)
    percentile_95 = curves.quantile(0.95, axis=1)

    start_value = float(curves.iloc[0].median())

    fig = go.Figure()

    # Outer upside band: 75th to 95th percentile
    fig.add_trace(
        go.Scatter(
            x=curves.index,
            y=percentile_95,
            mode="lines",
            name="95th Percentile",
            line=dict(width=0),
            showlegend=True,
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
            showlegend=True,
            hovertemplate="Day %{x}<br>75th: $%{y:,.0f}<extra></extra>",
        )
    )

    # Inner middle band: 25th to 75th percentile
    fig.add_trace(
        go.Scatter(
            x=curves.index,
            y=percentile_25,
            mode="lines",
            name="25th Percentile",
            fill="tonexty",
            fillcolor="rgba(99, 110, 250, 0.30)",
            line=dict(width=0),
            showlegend=True,
            hovertemplate="Day %{x}<br>25th: $%{y:,.0f}<extra></extra>",
        )
    )

    # Outer downside band: 5th to 25th percentile
    fig.add_trace(
        go.Scatter(
            x=curves.index,
            y=percentile_5,
            mode="lines",
            name="5th Percentile",
            fill="tonexty",
            fillcolor="rgba(239, 85, 59, 0.22)",
            line=dict(width=0),
            showlegend=True,
            hovertemplate="Day %{x}<br>5th: $%{y:,.0f}<extra></extra>",
        )
    )

    # Median line
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

    # Starting value reference line
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
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
    )

    fig.update_yaxes(
        tickprefix="$",
        separatethousands=True,
    )

    return fig


def plot_final_value_distribution(
    mc_results,
    title="Monte Carlo Final Value Distribution",
):
    fig = go.Figure()

    colors = {
        "Dynamic Signal-Markowitz": "#636EFA",
        "SPY Benchmark": "#EF553B",
        "Markowitz": "#00CC96",
    }

    for name, result in mc_results.items():
        final_values = np.asarray(result["final_values"].dropna(), dtype=float)

        if len(final_values) < 2:
            continue

        kde = gaussian_kde(final_values)

        x_min = final_values.min() * 0.9
        x_max = final_values.max() * 1.1
        x_grid = np.linspace(x_min, x_max, 400)
        y_grid = kde(x_grid)

        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=y_grid,
                mode="lines",
                name=name,
                line=dict(width=2, color=colors.get(name)),
                fill="tozeroy",
                opacity=0.35,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Final Portfolio Value",
        yaxis_title="Density",
        template="plotly_dark",
        height=500,
        hovermode="x unified",
    )

    return fig