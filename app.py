import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from data_engine import fetch_data, build_close_matrix
from signal_engine import compute_asset_signals
from portfolio_selection_engine import (
    build_dynamic_signal_markowitz_weights,
    get_latest_selected_portfolio,
)
from optimizer_engine import build_markowitz_weights
from backtest_engine import (
    backtest_equal_weight,
    backtest_weighted_portfolio,
)
from risk_engine import (
    build_portfolio_risk_report,
    build_universe_risk_table,
)
from charts_engine import (
    plot_equity_curves,
    plot_weight_matrix,
    plot_cash_and_asset_count,
)

from monte_carlo_engine import (
    run_monte_carlo_bootstrap,
    compare_strategy_to_benchmark_monte_carlo,
    plot_monte_carlo_fan_chart,
    plot_final_value_distribution,
)

from universe_monte_carlo_engine import (
    run_universe_monte_carlo,
    plot_universe_monte_carlo_fan_chart,
    plot_universe_monte_carlo_final_values,
)


st.set_page_config(
    page_title="Market Intelligence MVP",
    layout="wide",
)

st.title("Market Intelligence MVP")


# =========================================================
# SESSION STATE
# =========================================================

if "multi_horizon_results" not in st.session_state:
    st.session_state.multi_horizon_results = None

if "summary_df" not in st.session_state:
    st.session_state.summary_df = None

if "horizons" not in st.session_state:
    st.session_state.horizons = None

if "monte_carlo_results" not in st.session_state:
    st.session_state.monte_carlo_results = None

if "monte_carlo_source_horizon" not in st.session_state:
    st.session_state.monte_carlo_source_horizon = None

if "universe_monte_carlo_results" not in st.session_state:
    st.session_state.universe_monte_carlo_results = None

if "close_prices_full" not in st.session_state:
    st.session_state.close_prices_full = None
# =========================================================
# SIDEBAR
# =========================================================

universe_input = st.sidebar.text_input(
    "Asset universe",
    (
        "AAPL,MSFT,NVDA,AMZN,META,GOOGL,TSLA,ORCL,ADBE,CRM,AVGO,AMD,INTC,"
        "CSCO,QCOM,TXN,IBM,SPY,QQQ,DIA,IWM,VTI,VOO,VEA,VWO,"
        "XLF,XLE,XLV,XLI,XLP,XLK,XLY,XLU,XLB,XLC,XLRE,"
        "JPM,BAC,GS,MS,C,BLK,SCHW,AXP,"
        "XOM,CVX,COP,SLB,EOG,MPC,"
        "UNH,JNJ,PFE,MRK,ABBV,LLY,TMO,ABT,"
        "KO,PEP,PG,COST,WMT,TGT,MCD,SBUX,CL,KMB,"
        "CAT,DE,BA,HON,GE,RTX,LMT,UPS,FDX,"
        "HD,LOW,NKE,CMG,DIS,BKNG,"
        "MU,ASML,TSM,NFLX,TMUS,VZ,T,CMCSA,CHTR,"
        "PLD,AMT,O,SPG,EQIX,DLR,"
        "LIN,APD,ECL,NEM,FCX,"
        "NEE,DUK,SO,AEP,"
        "DAL,UAL,LUV,AAL,UBER,"
        "F,GM,RIVN,"
        "BABA,TCEHY,NVO,TM,SNY,"
        "GLD,SLV,USO,DBA,CPER,"
        "TLT,IEF,SHY,HYG,LQD,BND,"
        "VIXY,UUP,"
        "MSTR,COIN,RIOT,MARA,"
        "IJH,IJR,VB,"
        "SCHD,VIG,DGRO,IVE,IWD,"
        "VUG,IWF,ARKK,"
        "MTUM,QUAL,"
        "EEM,FXI,EWJ,EWZ,INDA"
    ),
)

initial_capital = st.sidebar.number_input(
    "Initial capital",
    value=100000,
)

fees = st.sidebar.number_input(
    "Trading fee",
    value=0.001,
    format="%.4f",
)

slippage = st.sidebar.number_input(
    "Slippage",
    value=0.001,
    format="%.4f",
)

max_assets = st.sidebar.slider(
    "Maximum portfolio assets",
    1,
    25,
    10,
)

rebalance_frequency_label = st.sidebar.selectbox(
    "Rebalance frequency",
    ["Monthly", "Quarterly"],
    index=0,
)

rebalance_frequency_map = {
    "Monthly": "ME",
    "Quarterly": "QE",
}

rebalance_frequency = rebalance_frequency_map[rebalance_frequency_label]
rebalance_threshold = st.sidebar.slider(
    "Rebalance threshold",
    0.00,
    0.20,
    0.03,
    0.01,
)

run_backtest = st.sidebar.button(
    "Run Multi-Horizon Backtest"
)

st.sidebar.divider()

st.sidebar.subheader("Universe Monte Carlo Settings")

universe_mc_simulations = st.sidebar.number_input(
    "Universe Monte Carlo simulations",
    min_value=10,
    max_value=1000,
    value=50,
    step=10,
)

universe_mc_warmup_days = st.sidebar.number_input(
    "Universe MC warmup days",
    min_value=60,
    max_value=756,
    value=252,
    step=21,
)

universe_mc_forecast_days = st.sidebar.number_input(
    "Universe MC forecast days",
    min_value=21,
    max_value=1260,
    value=252,
    step=21,
)

universe_mc_block_size = st.sidebar.number_input(
    "Universe MC block size",
    min_value=5,
    max_value=63,
    value=20,
    step=5,
)

run_universe_mc = st.sidebar.button(
    "Run Universe Monte Carlo"
)
# =========================================================
# HELPERS
# =========================================================

def get_portfolio_summary(portfolio, initial_capital):
    value = portfolio.value().dropna()

    final_value = float(value.iloc[-1])
    total_profit = final_value - initial_capital

    days = (value.index[-1] - value.index[0]).days
    years = days / 365.25

    cagr = (
        ((final_value / initial_capital) ** (1 / years) - 1) * 100
        if years > 0
        else 0.0
    )

    trades = portfolio.trades.records_readable
    total_trades = len(trades)

    entry_fees = 0.0
    exit_fees = 0.0
    generic_fees = 0.0
    other_fees = 0.0
    total_fees = 0.0

    if not trades.empty:
        if "Entry Fees" in trades.columns:
            entry_fees = float(
                pd.to_numeric(
                    trades["Entry Fees"],
                    errors="coerce",
                ).fillna(0).sum()
            )

        if "Exit Fees" in trades.columns:
            exit_fees = float(
                pd.to_numeric(
                    trades["Exit Fees"],
                    errors="coerce",
                ).fillna(0).sum()
            )

        if "Fees" in trades.columns:
            generic_fees = float(
                pd.to_numeric(
                    trades["Fees"],
                    errors="coerce",
                ).fillna(0).sum()
            )

        known_fee_columns = {
            "Entry Fees",
            "Exit Fees",
            "Fees",
        }

        other_fee_columns = [
            col for col in trades.columns
            if "fee" in col.lower()
            and col not in known_fee_columns
        ]

        for col in other_fee_columns:
            other_fees += float(
                pd.to_numeric(
                    trades[col],
                    errors="coerce",
                ).fillna(0).sum()
            )

        if entry_fees > 0 or exit_fees > 0:
            total_fees = entry_fees + exit_fees + other_fees
        else:
            total_fees = generic_fees + other_fees

    return {
        "Final Value": final_value,
        "Total Profit": total_profit,
        "CAGR": cagr,
        "Total Fees": float(total_fees),
        "Entry Fees": float(entry_fees),
        "Exit Fees": float(exit_fees),
        "Generic Fees": float(generic_fees),
        "Other Fees": float(other_fees),
        "Total Trades": total_trades,
    }


def slice_by_months(close_prices, months):
    end_date = close_prices.index.max()
    start_date = end_date - pd.DateOffset(months=months)

    return close_prices.loc[
        close_prices.index >= start_date
    ].copy()


def slice_price_data(price_data, selected_index):
    sliced = {}

    for ticker, df in price_data.items():
        sliced_df = df.loc[
            df.index.isin(selected_index)
        ].copy()

        if not sliced_df.empty:
            sliced[ticker] = sliced_df

    return sliced


def run_single_horizon(
    horizon_name,
    close_prices,
    price_data,
    initial_capital,
    fees,
    slippage,
    max_assets,
    rebalance_frequency,
    rebalance_threshold,
):
    signal_data = compute_asset_signals(price_data)

    dynamic_weights = build_dynamic_signal_markowitz_weights(
        signal_data=signal_data,
        close_prices=close_prices,
        top_n=max_assets,
        rebalance_frequency=rebalance_frequency,
        lookback_window=252,
        max_weight=0.40,
        entry_quantile=0.75,
        exit_quantile=0.50,
    )

    markowitz_weights = build_markowitz_weights(
        close_prices,
        rebalance_frequency=rebalance_frequency,
    )

    spy_pf = backtest_equal_weight(
        close_prices[["SPY"]],
        initial_capital=initial_capital,
        fees=0,
        slippage=0,
    )

    dynamic_pf = backtest_weighted_portfolio(
        close_prices,
        dynamic_weights,
        initial_capital=initial_capital,
        fees=fees,
        slippage=slippage,
        shift_weights=True,
        rebalance_threshold=rebalance_threshold,
    )

    markowitz_pf = backtest_weighted_portfolio(
        close_prices,
        markowitz_weights,
        initial_capital=initial_capital,
        fees=fees,
        slippage=slippage,
        shift_weights=True,
        rebalance_threshold=rebalance_threshold,
    )

    portfolios = {
        "SPY Benchmark": spy_pf,
        "Dynamic Signal-Markowitz": dynamic_pf,
        "Markowitz": markowitz_pf,
    }

    rows = []

    for name, pf in portfolios.items():
        stat = pf.stats()

        summary = get_portfolio_summary(
            pf,
            initial_capital,
        )

        rows.append(
            {
                "Horizon": horizon_name,
                "Portfolio": name,
                "Total Return [%]": stat["Total Return [%]"],
                "CAGR / ARR [%]": summary["CAGR"],
                "Total Profit [$]": summary["Total Profit"],
                "Total Fees [$]": summary["Total Fees"],
                "Entry Fees [$]": summary["Entry Fees"],
                "Exit Fees [$]": summary["Exit Fees"],
                "Generic Fees [$]": summary["Generic Fees"],
                "Other Fees [$]": summary["Other Fees"],
                "Total Trades": summary["Total Trades"],
                "Sharpe": stat["Sharpe Ratio"],
                "Max Drawdown [%]": stat["Max Drawdown [%]"],
            }
        )

    diagnostics = pd.DataFrame(
        index=dynamic_weights.index
    )

    diagnostics["Invested Exposure"] = (
        dynamic_weights.sum(axis=1)
    )

    diagnostics["Cash Exposure"] = (
        1 - diagnostics["Invested Exposure"]
    )

    diagnostics["Assets Held"] = (
        dynamic_weights > 0
    ).sum(axis=1)

    return {
        "summary": rows,
        "portfolios": portfolios,
        "dynamic_weights": dynamic_weights,
        "markowitz_weights": markowitz_weights,
        "diagnostics": diagnostics,
        "signal_data": signal_data,
    }


def build_monte_carlo_results(
    source_result,
    initial_capital,
    simulations,
    forecast_days,
):
    selected_portfolios = source_result["portfolios"]

    dynamic_values = (
        selected_portfolios["Dynamic Signal-Markowitz"]
        .value()
    )

    spy_values = (
        selected_portfolios["SPY Benchmark"]
        .value()
    )

    markowitz_values = (
        selected_portfolios["Markowitz"]
        .value()
    )

    dynamic_mc = run_monte_carlo_bootstrap(
        portfolio_values=dynamic_values,
        initial_capital=initial_capital,
        simulations=simulations,
        forecast_days=forecast_days,
        random_seed=42,
    )

    spy_mc = run_monte_carlo_bootstrap(
        portfolio_values=spy_values,
        initial_capital=initial_capital,
        simulations=simulations,
        forecast_days=forecast_days,
        random_seed=43,
    )

    markowitz_mc = run_monte_carlo_bootstrap(
        portfolio_values=markowitz_values,
        initial_capital=initial_capital,
        simulations=simulations,
        forecast_days=forecast_days,
        random_seed=44,
    )

    comparison_mc = compare_strategy_to_benchmark_monte_carlo(
        strategy_values=dynamic_values,
        benchmark_values=spy_values,
        initial_capital=initial_capital,
        simulations=simulations,
        forecast_days=forecast_days,
        random_seed=42,
    )

    return {
        "dynamic_mc": dynamic_mc,
        "spy_mc": spy_mc,
        "markowitz_mc": markowitz_mc,
        "comparison_mc": comparison_mc,
    }


def format_monte_carlo_summary(
    dynamic_mc,
    spy_mc,
    markowitz_mc,
):
    mc_summary_rows = []

    for name, result in {
        "Dynamic Signal-Markowitz": dynamic_mc,
        "SPY Benchmark": spy_mc,
        "Markowitz": markowitz_mc,
    }.items():
        summary = result["summary"]

        mc_summary_rows.append(
            {
                "Portfolio": name,
                "Median Final Value": summary["Median Final Value"],
                "5th Percentile Final Value": summary["5th Percentile Final Value"],
                "95th Percentile Final Value": summary["95th Percentile Final Value"],
                "Probability of Loss": summary["Probability of Loss"],
                "Worst Simulated Drawdown": summary["Worst Simulated Drawdown"],
                "Median Simulated Drawdown": summary["Median Simulated Drawdown"],
            }
        )

    mc_summary_df = pd.DataFrame(mc_summary_rows)

    formatted = mc_summary_df.copy()

    money_cols = [
        "Median Final Value",
        "5th Percentile Final Value",
        "95th Percentile Final Value",
    ]

    percent_cols = [
        "Probability of Loss",
        "Worst Simulated Drawdown",
        "Median Simulated Drawdown",
    ]

    for col in money_cols:
        formatted[col] = formatted[col].map(
            lambda x: f"${x:,.0f}"
        )

    for col in percent_cols:
        formatted[col] = formatted[col].map(
            lambda x: f"{x:.2%}"
        )

    return formatted


# =========================================================
# RUN BACKTEST
# =========================================================

if run_backtest:
    universe_tickers = [
        t.strip().upper()
        for t in universe_input.split(",")
        if t.strip()
    ]

    if "SPY" not in universe_tickers:
        universe_tickers.append("SPY")

    price_data_full = fetch_data(
        universe_tickers,
        period="10y",
    )

    if not price_data_full:
        st.error("No data was downloaded. Check your tickers.")
        st.stop()

    close_prices_full = build_close_matrix(price_data_full)

    if "SPY" not in close_prices_full.columns:
        st.error("SPY data could not be downloaded.")
        st.stop()

    horizons = {
        "1M": 1,
        "3M": 3,
        "6M": 6,
        "1Y": 12,
        "2Y": 24,
        "5Y": 60,
        "10Y": 120,
    }

    results = {}
    summary_rows = []

    with st.spinner("Running multi-horizon backtests..."):
        for horizon_name, months in horizons.items():
            close_slice = slice_by_months(
                close_prices_full,
                months,
            )

            price_data_slice = slice_price_data(
                price_data_full,
                close_slice.index,
            )

            result = run_single_horizon(
                horizon_name=horizon_name,
                close_prices=close_slice,
                price_data=price_data_slice,
                initial_capital=initial_capital,
                fees=fees,
                slippage=slippage,
                max_assets=max_assets,
                rebalance_frequency=rebalance_frequency,
                rebalance_threshold=rebalance_threshold,
            )

            results[horizon_name] = result
            summary_rows.extend(result["summary"])

    summary_df = pd.DataFrame(summary_rows)

    st.session_state.multi_horizon_results = results
    st.session_state.summary_df = summary_df
    st.session_state.horizons = horizons

    st.session_state.monte_carlo_results = None
    st.session_state.monte_carlo_source_horizon = None
    st.session_state.close_prices_full = close_prices_full

# =========================================================
# RUN UNIVERSE MONTE CARLO
# =========================================================

if run_universe_mc:
    if st.session_state.close_prices_full is None:
        st.warning(
            "Run the multi-horizon backtest first before running Universe Monte Carlo."
        )
    else:
        with st.spinner("Running full-engine Universe Monte Carlo..."):
            universe_mc_result = run_universe_monte_carlo(
                close_prices=st.session_state.close_prices_full,
                initial_capital=initial_capital,
                fees=fees,
                slippage=slippage,
                max_assets=max_assets,
                rebalance_frequency=rebalance_frequency,
                rebalance_threshold=rebalance_threshold,
                simulations=universe_mc_simulations,
                warmup_days=universe_mc_warmup_days,
                forecast_days=universe_mc_forecast_days,
                block_size=universe_mc_block_size,
                random_seed=42,
            )

        st.session_state.universe_monte_carlo_results = universe_mc_result

# =========================================================
# DISPLAY
# =========================================================

if st.session_state.multi_horizon_results is not None:
    results = st.session_state.multi_horizon_results
    summary_df = st.session_state.summary_df

    horizon_order = [
        "1M",
        "3M",
        "6M",
        "1Y",
        "2Y",
        "5Y",
        "10Y",
    ]

    st.subheader("Performance Comparison")

    metric_choice = st.selectbox(
        "Metric to graph",
        [
            "Total Return [%]",
            "CAGR / ARR [%]",
            "Sharpe",
            "Max Drawdown [%]",
            "Total Profit [$]",
            "Total Fees [$]",
            "Total Trades",
        ],
        key="metric_select",
    )

    pivot = summary_df.pivot(
        index="Horizon",
        columns="Portfolio",
        values=metric_choice,
    )

    pivot = pivot.reindex(horizon_order)

    fig = go.Figure()

    for portfolio in pivot.columns:
        values = pivot[portfolio]

        if "%" in metric_choice:
            labels = [f"{v:.2f}%" for v in values]
        elif "$" in metric_choice:
            labels = [f"${v:,.0f}" for v in values]
        else:
            labels = [f"{v:.2f}" for v in values]

        fig.add_trace(
            go.Bar(
                x=pivot.index,
                y=values,
                name=portfolio,
                text=labels,
                textposition="outside",
            )
        )

    fig.update_traces(
        cliponaxis=False,
    )

    fig.update_layout(
        title=f"{metric_choice} Comparison Across Backtests",
        xaxis_title="Backtest Horizon",
        yaxis_title=metric_choice,
        barmode="group",
        height=600,
        template="plotly_dark",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # =====================================================
    # HORIZON SELECTOR
    # =====================================================

    selected_horizon = st.selectbox(
        "Detailed horizon view",
        horizon_order,
        index=4,
        key="horizon_select",
    )

    selected_result = results[selected_horizon]

    snapshot_df = summary_df[
        summary_df["Horizon"] == selected_horizon
    ].copy()

    # =====================================================
    # KPI DASHBOARD
    # =====================================================

    st.subheader(
        f"Performance Snapshot — {selected_horizon}"
    )

    snapshot_cols = st.columns(3)

    portfolio_order = [
        "Dynamic Signal-Markowitz",
        "Markowitz",
        "SPY Benchmark",
    ]

    for col, portfolio_name in zip(
        snapshot_cols,
        portfolio_order,
    ):
        row = snapshot_df[
            snapshot_df["Portfolio"] == portfolio_name
        ].iloc[0]

        with col:
            st.markdown(f"### {portfolio_name}")

            st.metric(
                "Total Return",
                f"{row['Total Return [%]']:.2f}%",
            )

            st.metric(
                "CAGR / ARR",
                f"{row['CAGR / ARR [%]']:.2f}%",
            )

            st.metric(
                "Sharpe",
                f"{row['Sharpe']:.2f}",
            )

            st.metric(
                "Max Drawdown",
                f"{row['Max Drawdown [%]']:.2f}%",
            )

            st.metric(
                "Total Profit",
                f"${row['Total Profit [$]']:,.0f}",
            )

            st.metric(
                "Total Fees",
                f"${row['Total Fees [$]']:,.0f}",
            )

            st.metric(
                "Total Trades",
                f"{int(row['Total Trades'])}",
            )

    # =====================================================
    # EQUITY CURVES
    # =====================================================

    st.subheader(
        f"Equity Curves — {selected_horizon}"
    )

    st.plotly_chart(
        plot_equity_curves(
            selected_result["portfolios"]
        ),
        use_container_width=True,
    )

    # =====================================================
    # MONTE CARLO DISPLAY
    # =====================================================

    if st.session_state.monte_carlo_results is not None:
        mc_results = st.session_state.monte_carlo_results
        mc_source = st.session_state.monte_carlo_source_horizon

        dynamic_mc = mc_results["dynamic_mc"]
        spy_mc = mc_results["spy_mc"]
        markowitz_mc = mc_results["markowitz_mc"]
        comparison_mc = mc_results["comparison_mc"]

        st.subheader(
            f"Monte Carlo Forward Simulation — Trained on {mc_source}"
        )

        if (
            dynamic_mc is not None
            and spy_mc is not None
            and markowitz_mc is not None
        ):
            if comparison_mc is not None:
                probability_beating_spy = comparison_mc[
                    "Probability of Beating SPY"
                ]

                st.metric(
                    "Dynamic Probability of Beating SPY",
                    f"{probability_beating_spy:.2%}",
                )

            formatted_mc_summary = format_monte_carlo_summary(
                dynamic_mc=dynamic_mc,
                spy_mc=spy_mc,
                markowitz_mc=markowitz_mc,
            )

            st.dataframe(
                formatted_mc_summary,
                use_container_width=True,
            )

            st.plotly_chart(
                plot_monte_carlo_fan_chart(
                    dynamic_mc,
                    title=(
                        f"Dynamic Signal-Markowitz Monte Carlo — "
                        f"Trained on {mc_source}"
                    ),
                ),
                use_container_width=True,
            )

            st.plotly_chart(
                plot_final_value_distribution(
                    {
                        "Dynamic Signal-Markowitz": dynamic_mc,
                        "SPY Benchmark": spy_mc,
                        "Markowitz": markowitz_mc,
                    },
                    title=(
                        f"Final Value Distribution — "
                        f"Trained on {mc_source}"
                    ),
                ),
                use_container_width=True,
            )
        else:
            st.warning(
                "Not enough return data to run Monte Carlo simulation."
            )

    # =====================================================
    # UNIVERSE MONTE CARLO DISPLAY
    # =====================================================

    if st.session_state.universe_monte_carlo_results is not None:
        universe_mc = st.session_state.universe_monte_carlo_results

        st.subheader("Universe Monte Carlo — Full Engine Simulation")

        if universe_mc is None:
            st.warning("Universe Monte Carlo could not complete.")
        else:
            summary = universe_mc["summary"]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Simulations Completed",
                    f"{summary['Simulations Completed']:,}",
                )
                st.metric(
                    "Median Final Value",
                    f"${summary['Median Final Value']:,.0f}",
                )
                st.metric(
                    "Probability of Loss",
                    f"{summary['Probability of Loss']:.2%}",
                )

            with col2:
                st.metric(
                    "5th Percentile Final Value",
                    f"${summary['5th Percentile Final Value']:,.0f}",
                )
                st.metric(
                    "95th Percentile Final Value",
                    f"${summary['95th Percentile Final Value']:,.0f}",
                )
                st.metric(
                    "Probability of Beating Synthetic SPY",
                    f"{summary['Probability of Beating SPY']:.2%}",
                )

            with col3:
                st.metric(
                    "Worst Simulated Drawdown",
                    f"{summary['Worst Simulated Drawdown']:.2%}",
                )
                st.metric(
                    "Median Simulated Drawdown",
                    f"{summary['Median Simulated Drawdown']:.2%}",
                )
                st.metric(
                    "Average Cash Exposure",
                    f"{summary['Average Cash Exposure']:.2%}",
                )

            st.plotly_chart(
                plot_universe_monte_carlo_fan_chart(
                    universe_mc,
                    title="Universe Monte Carlo Fan Chart — Dynamic Signal-Markowitz",
                ),
                use_container_width=True,
            )

            st.plotly_chart(
                plot_universe_monte_carlo_final_values(
                    universe_mc,
                    title="Universe Monte Carlo Final Value Distribution",
                ),
                use_container_width=True,
            )

            with st.expander("Universe Monte Carlo Raw Simulation Results"):
                st.dataframe(
                    universe_mc["simulation_results"],
                    use_container_width=True,
                )


    # =====================================================
    # CURRENT PORTFOLIO
    # =====================================================

    st.subheader(
        f"Latest Algorithm-Selected Portfolio — {selected_horizon}"
    )

    latest_selected = get_latest_selected_portfolio(
        selected_result["dynamic_weights"]
    )

    if latest_selected.empty:
        st.info(
            "No assets currently selected by the algorithm."
        )

    else:
        latest_selected = latest_selected.sort_values(
            ascending=False
        )

        col1, col2 = st.columns([1, 1.2])

        with col1:
            display_df = latest_selected.to_frame(
                "Current Weight"
            ).copy()

            display_df["Current Weight"] = (
                    display_df["Current Weight"] * 100
            ).round(2)

            display_df["Current Weight"] = (
                    display_df["Current Weight"].astype(str) + "%"
            )

            st.dataframe(
                display_df,
                use_container_width=True,
            )

        with col2:

            pie_fig = go.Figure()

            pie_fig.add_trace(
                go.Pie(
                    labels=latest_selected.index,
                    values=latest_selected.values,
                    hole=0.35,
                    textinfo="label+percent",
                    hovertemplate=(
                        "<b>%{label}</b><br>"
                        "Weight: %{percent}<br>"
                        "<extra></extra>"
                    ),
                )
            )

            pie_fig.update_layout(
                title="Current Portfolio Allocation",
                template="plotly_dark",
                height=500,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.02,
                ),
            )

            st.plotly_chart(
                pie_fig,
                use_container_width=True,
            )
    # =====================================================
    # CASH EXPOSURE
    # =====================================================

    st.subheader(
        f"Cash Exposure and Portfolio Breadth — {selected_horizon}"
    )

    dynamic_weights = selected_result[
        "dynamic_weights"
    ]

    cash_exposure = (
        1 - dynamic_weights.sum(axis=1)
    ).clip(0, 1)

    asset_count = (
        dynamic_weights > 0
    ).sum(axis=1)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Average Cash Exposure",
            f"{cash_exposure.mean():.2%}",
        )

    with col2:
        st.metric(
            "Average Assets Held",
            f"{asset_count.mean():.2f}",
        )

    st.plotly_chart(
        plot_cash_and_asset_count(
            dynamic_weights
        ),
        use_container_width=True,
    )

    # =====================================================
    # WEIGHTS
    # =====================================================

    st.subheader(
        f"Portfolio Weights — {selected_horizon}"
    )

    weight_tab1, weight_tab2 = st.tabs(
        [
            "Dynamic Signal-Markowitz Weights",
            "Markowitz Weights",
        ]
    )

    with weight_tab1:
        st.plotly_chart(
            plot_weight_matrix(
                selected_result["dynamic_weights"],
                (
                    "Dynamic Signal-Markowitz Portfolio Weights "
                    f"— {selected_horizon}"
                ),
            ),
            use_container_width=True,
        )

    with weight_tab2:
        st.plotly_chart(
            plot_weight_matrix(
                selected_result["markowitz_weights"],
                f"Markowitz Portfolio Weights — {selected_horizon}",
            ),
            use_container_width=True,
        )

    # =====================================================
    # RISK REPORTS
    # =====================================================

    st.subheader(
        f"Risk Reports — {selected_horizon}"
    )

    risk_rows = []

    for name, pf in selected_result["portfolios"].items():
        risk = build_portfolio_risk_report(
            pf.value()
        )

        risk["Portfolio"] = name
        risk_rows.append(risk)

    st.dataframe(
        risk_rows,
        use_container_width=True,
    )

    # =====================================================
    # UNIVERSE SNAPSHOT
    # =====================================================

    st.subheader(
        f"Universe Signal / Risk Snapshot — {selected_horizon}"
    )

    universe_table = build_universe_risk_table(
        selected_result["signal_data"]
    )

    st.dataframe(
        universe_table,
        use_container_width=True,
    )

    # =====================================================
    # FULL VECTORBT STATS
    # =====================================================

    st.subheader(
        f"Full Vectorbt Stats — {selected_horizon}"
    )

    stats = {
        name: pf.stats()
        for name, pf in selected_result[
            "portfolios"
        ].items()
    }

    tabs = st.tabs(list(stats.keys()))

    for tab, (name, stat) in zip(
        tabs,
        stats.items(),
    ):
        with tab:
            st.dataframe(
                stat.to_frame("Value"),
                use_container_width=True,
            )
