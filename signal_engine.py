import numpy as np
import pandas as pd


def calculate_rsi(close, period=14):
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def calculate_macd(close, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = close.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close.ewm(span=slow_period, adjust=False).mean()

    macd = fast_ema - slow_ema
    macd_signal = macd.ewm(span=signal_period, adjust=False).mean()

    return macd, macd_signal


def calculate_obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)

    return (direction * volume.fillna(0)).cumsum()


def compute_asset_signals(
    price_data,
    short_window=30,
    long_window=200,
    z_window=30,
):
    outputs = {}

    for ticker, df in price_data.items():
        out = df.copy()

        close = pd.to_numeric(out["Close"], errors="coerce")

        out["return"] = close.pct_change()
        out["log_return"] = np.log(close / close.shift(1))

        out[f"sma_{short_window}"] = close.rolling(short_window).mean()
        out[f"sma_{long_window}"] = close.rolling(long_window).mean()

        out[f"ema_{short_window}"] = close.ewm(span=short_window, adjust=False).mean()
        out[f"ema_{long_window}"] = close.ewm(span=long_window, adjust=False).mean()

        out["rsi"] = calculate_rsi(close)
        out["macd"], out["macd_signal"] = calculate_macd(close)
        out["obv"] = calculate_obv(close, out["Volume"])

        out["volatility_30d"] = out["return"].rolling(z_window).std() * np.sqrt(252)

        rolling_mean = close.rolling(z_window).mean()
        rolling_std = close.rolling(z_window).std()

        out["z_score"] = (close - rolling_mean) / rolling_std.replace(0, np.nan)

        out["trend"] = np.where(
            out[f"ema_{short_window}"] > out[f"ema_{long_window}"],
            "uptrend",
            "downtrend",
        )

        outputs[ticker] = out

    return outputs