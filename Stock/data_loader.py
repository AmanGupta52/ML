import yfinance as yf
import pandas as pd


def fetch_stock_data(ticker: str, period: str = "1y"):
    """
    Fetch historical stock data for a given ticker.
    """
    try:
        data = yf.download(
            ticker,
            period=period,
            progress=False,
            auto_adjust=True,
            threads=False
        )
        
        if data is None or data.empty:
            print(f"[ERROR] No data received for {ticker}")
            return None
        
        return data

    except Exception as e:
        print(f"[ERROR] Failed loading data: {e}")
        return None


def get_live_price(ticker: str):
    """
    Get the latest market price safely.
    """
    try:
        ticker_obj = yf.Ticker(ticker)

        # Try fast 1-minute history first
        hist = ticker_obj.history(period="1d", interval="1m")

        if isinstance(hist, pd.DataFrame) and not hist.empty:
            price = hist["Close"].iloc[-1]
            return round(float(price), 2)

        # Fallback: use fast_info safely
        fast_info = getattr(ticker_obj, "fast_info", None)
        if fast_info:
            info_price = fast_info.get("lastPrice") or fast_info.get("regularMarketPrice")
            if info_price:
                return round(float(info_price), 2)

        return 0.0

    except Exception as e:
        print(f"[ERROR] get_live_price failed: {e}")
        return 0.0
