import numpy as np


def generate_strategy_signals(
    signal_data,
    strategy_type="trend",
    short_window=30,
    long_window=200,
    rsi_oversold=30,
    rsi_overbought=70,
    z_entry=2.0,
    z_exit=0.5,
):
    outputs = {}

    for ticker, df in signal_data.items():
        out = df.copy()

        ema_short = f"ema_{short_window}"
        ema_long = f"ema_{long_window}"
        sma_short = f"sma_{short_window}"
        sma_long = f"sma_{long_window}"

        out["long_signal"] = False
        out["short_signal"] = False
        out["exit_signal"] = False
        out["hold_signal"] = True

        if strategy_type == "trend":
            out["long_signal"] = (
                (out[ema_short] > out[ema_long]) &
                (out["macd"] > out["macd_signal"])
            )

            out["exit_signal"] = (
                (out[ema_short] < out[ema_long]) |
                (out["macd"] < out["macd_signal"])
            )

        elif strategy_type == "sma_trend":
            out["long_signal"] = (
                (out[sma_short] > out[sma_long]) &
                (out["macd"] > out["macd_signal"])
            )

            out["exit_signal"] = (
                (out[sma_short] < out[sma_long]) |
                (out["macd"] < out["macd_signal"])
            )

        elif strategy_type == "mean_reversion":
            out["long_signal"] = (
                (out["z_score"] < -z_entry) &
                (out["rsi"] < rsi_oversold)
            )

            out["exit_signal"] = (
                (out["z_score"].abs() < z_exit) |
                (out["rsi"] > 50)
            )

        elif strategy_type == "hybrid":
            trend_filter = out[ema_short] > out[ema_long]

            mean_reversion_entry = (
                (out["z_score"] < -z_entry) &
                (out["rsi"] < rsi_oversold)
            )

            momentum_confirmation = out["macd"] > out["macd_signal"]

            out["long_signal"] = (
                trend_filter &
                mean_reversion_entry &
                momentum_confirmation
            )

            out["exit_signal"] = (
                (out["z_score"].abs() < z_exit) |
                (out["rsi"] > rsi_overbought) |
                (out[ema_short] < out[ema_long])
            )

        else:
            raise ValueError(
                "strategy_type must be one of: "
                "'trend', 'sma_trend', 'mean_reversion', or 'hybrid'."
            )

        out["hold_signal"] = ~out["long_signal"] & ~out["exit_signal"]

        out["raw_position"] = np.nan
        out.loc[out["long_signal"], "raw_position"] = 1
        out.loc[out["exit_signal"], "raw_position"] = 0

        out["position"] = out["raw_position"].ffill().fillna(0)

        out["trade_action"] = "HOLD"
        out.loc[out["long_signal"], "trade_action"] = "LONG"
        out.loc[out["exit_signal"], "trade_action"] = "EXIT"

        outputs[ticker] = out

    return outputs