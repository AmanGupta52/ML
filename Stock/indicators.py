import pandas as pd
import numpy as np

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add essential technical indicators safely.
    Ensures no NaN issues and preserves dataframe length.
    """
    df = df.copy()

    # -----------------------------
    # Simple Moving Averages
    # -----------------------------
    df["SMA20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    df["SMA50"] = df["Close"].rolling(window=50, min_periods=1).mean()

    # -----------------------------
    # RSI 14
    # -----------------------------
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50.0)

    # -----------------------------
    # EMAs + MACD
    # -----------------------------
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # -----------------------------
    # Price / Volume %
    # -----------------------------
    df["PriceChange%"] = df["Close"].pct_change().fillna(0)

    if "Volume" in df.columns:
        df["VolumeChange%"] = df["Volume"].pct_change().fillna(0)
    else:
        df["VolumeChange%"] = np.zeros(len(df))

    # Final safety fill
    df = df.fillna(0)

    return df
