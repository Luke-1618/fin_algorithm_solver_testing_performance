import numpy as np
import pandas as pd
import yfinance as yf


def normalize_tickers(tickers):
    if isinstance(tickers, str):
        tickers = tickers.split(",")

    cleaned = [
        str(ticker).strip().upper()
        for ticker in tickers
        if str(ticker).strip()
    ]

    cleaned = list(dict.fromkeys(cleaned))

    if not cleaned:
        raise ValueError("At least one valid ticker is required.")

    return cleaned


def _safe_series(df, column_name):
    if column_name not in df.columns:
        return pd.Series(np.nan, index=df.index)

    col = df[column_name]

    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]

    return col


def fetch_data(tickers, period="5y", interval="1d"):
    tickers = normalize_tickers(tickers)
    data = {}

    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            if df.empty:
                print(f"No data found for {ticker}. Skipping.")
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [
                    col[0] if isinstance(col, tuple) else col
                    for col in df.columns
                ]

            df = df.loc[:, ~df.columns.duplicated()].copy()

            df = df.reset_index()

            date_col = "Date" if "Date" in df.columns else df.columns[0]

            df[date_col] = (
                pd.to_datetime(df[date_col], errors="coerce")
                .dt.tz_localize(None)
                .dt.normalize()
            )

            df = df.rename(columns={date_col: "Date"})
            df = df.dropna(subset=["Date"])

            expected_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

            for col in expected_cols:
                if col not in df.columns:
                    df[col] = np.nan

            for col in expected_cols:
                series = _safe_series(df, col)
                df[col] = pd.to_numeric(series, errors="coerce")

            df = (
                df[["Date"] + expected_cols]
                .sort_values("Date")
                .drop_duplicates("Date")
                .set_index("Date")
            )

            data[ticker] = df

        except Exception as error:
            print(f"Failed to fetch {ticker}: {error}")
            continue

    return data


def build_close_matrix(price_data):
    close_series = []

    for ticker, df in price_data.items():
        if "Close" not in df.columns:
            continue

        close = df["Close"]

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        close_series.append(
            pd.to_numeric(close, errors="coerce").rename(ticker)
        )

    if not close_series:
        return pd.DataFrame()

    close_matrix = pd.concat(close_series, axis=1)
    close_matrix = close_matrix.loc[:, ~close_matrix.columns.duplicated()]
    close_matrix = close_matrix.sort_index().dropna(how="all")

    return close_matrix