import pandas as pd
import plotly.graph_objects as go


def plot_equity_curves(portfolios):
    fig = go.Figure()

    for name, pf in portfolios.items():
        fig.add_trace(
            go.Scatter(
                x=pf.value().index,
                y=pf.value(),
                mode="lines",
                name=name,
            )
        )

    fig.update_layout(
        title="Equity Curve Comparison",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        template="plotly_white",
    )

    return fig


def plot_weight_matrix(weights, title):
    fig = go.Figure()

    for column in weights.columns:
        fig.add_trace(
            go.Scatter(
                x=weights.index,
                y=weights[column],
                mode="lines",
                stackgroup="one",
                name=str(column),
                hovertemplate=(
                    f"Asset: {column}<br>"
                    "Date: %{x}<br>"
                    "Weight: %{y:.2%}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Weight Inside Portfolio",
        yaxis_tickformat=".0%",
        template="plotly_white",
        height=600,
    )

    return fig


def plot_trade_markers(strategy_data, strategy_name):
    fig = go.Figure()

    for ticker, df in strategy_data.items():
        clean_df = df.dropna(subset=["Close"])

        if clean_df.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=clean_df.index,
                y=clean_df["Close"],
                mode="lines",
                name=f"{ticker} Close",
                opacity=0.45,
            )
        )

        long_trades = clean_df[clean_df["long_signal"]]
        exit_trades = clean_df[clean_df["exit_signal"]]

        fig.add_trace(
            go.Scatter(
                x=long_trades.index,
                y=long_trades["Close"],
                mode="markers",
                name=f"{ticker} LONG",
                marker=dict(symbol="triangle-up", size=10),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=exit_trades.index,
                y=exit_trades["Close"],
                mode="markers",
                name=f"{ticker} EXIT",
                marker=dict(symbol="x", size=10),
            )
        )

    fig.update_layout(
        title=f"{strategy_name} Trade Points by Asset",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        height=650,
    )

    return fig

def plot_cash_and_asset_count(weights):
    cash_exposure = 1 - weights.sum(axis=1)
    cash_exposure = cash_exposure.clip(lower=0, upper=1)

    asset_count = (weights > 0).sum(axis=1)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=cash_exposure.index,
            y=cash_exposure,
            mode="lines",
            name="Cash Exposure",
            yaxis="y1",
            hovertemplate="Date: %{x}<br>Cash: %{y:.2%}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=asset_count.index,
            y=asset_count,
            mode="lines",
            name="Number of Assets Held",
            yaxis="y2",
            hovertemplate="Date: %{x}<br>Assets Held: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Cash Exposure and Number of Assets Held Over Time",
        xaxis_title="Date",
        yaxis=dict(
            title="Cash Exposure",
            tickformat=".0%",
            side="left",
        ),
        yaxis2=dict(
            title="Assets Held",
            overlaying="y",
            side="right",
        ),
        template="plotly_white",
        height=550,
    )

    return fig