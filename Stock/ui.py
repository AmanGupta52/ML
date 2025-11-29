# ui.py — FINAL PRODUCTION-READY VERSION (fixed)
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import datetime as dt
import logging

from data_loader import fetch_stock_data, get_live_price
from predictor import ModelTrainer
from news import fetch_news_rss as fetch_news            # <- use RSS function from news.py
from history import PredictionHistory
from indicators import add_indicators

logging.basicConfig(level=logging.INFO)


# ==================== THREADED DECORATOR ====================
def threaded(fn):
    """
    Threaded decorator that returns a (queue, thread) handle.
    The queue will receive tuples: ("success", result) or ("error", exception)
    """
    import queue
    import threading
    import traceback

    def wrapper(*args, **kwargs):
        q = queue.Queue()

        def target():
            try:
                result = fn(*args, **kwargs)
                q.put(("success", result))
            except Exception as e:
                logging.error(f"Thread {fn.__name__} failed:\n{traceback.format_exc()}")
                q.put(("error", e))

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        return q, thread

    return wrapper
# ============================================================


class StockPredictionUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Smart Stock Prediction System Pro")
        self.root.geometry("1600x900")
        try:
            self.root.state("zoomed")
        except Exception:
            pass

        self.data: pd.DataFrame | None = None
        self.history = PredictionHistory()

        # UI Vars
        self.ticker_var = tk.StringVar(value="RELIANCE.NS")
        self.period_var = tk.StringVar(value="1y")
        self.live_price_var = tk.StringVar(value="Fetching...")
        self.status_var = tk.StringVar(value="Ready")

        self.build_ui()
        self.root.after(100, self.update_live_price)  # Auto refresh price

    def build_ui(self):
        # Top bar
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=5)

        ttk.Label(top, text="Ticker:", font=("Arial", 10, "bold")).pack(side="left")
        ttk.Entry(top, textvariable=self.ticker_var, width=15).pack(side="left", padx=5)

        ttk.Label(top, text="Period:").pack(side="left", padx=(20, 5))
        ttk.Combobox(
            top,
            textvariable=self.period_var,
            values=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
            width=8,
            state="readonly"
        ).pack(side="left")

        ttk.Button(top, text="Fetch & Analyze", command=self.start_fetch).pack(side="left", padx=10)
        ttk.Button(top, text="Predict Now", command=self.start_predict).pack(side="left", padx=5)
        ttk.Button(top, text="News", command=self.start_news).pack(side="left", padx=5)

        ttk.Label(top, textvariable=self.live_price_var, font=("Arial", 14, "bold"), foreground="green").pack(side="right", padx=20)
        ttk.Label(top, text="Live Price:").pack(side="right")

        ttk.Label(top, textvariable=self.status_var, foreground="blue").pack(side="left", padx=20)

        # Tabs
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=10)

        self.chart_tab = ttk.Frame(self.tabs)
        self.pred_tab = ttk.Frame(self.tabs)
        self.news_tab = ttk.Frame(self.tabs)
        self.history_tab = ttk.Frame(self.tabs)

        self.tabs.add(self.chart_tab, text="Chart")
        self.tabs.add(self.pred_tab, text="AI Prediction")
        self.tabs.add(self.news_tab, text="News")
        self.tabs.add(self.history_tab, text="History")

        self.setup_chart()
        self.setup_prediction_tab()
        self.setup_news_tab()
        self.setup_history_tab()

    def setup_chart(self):
        self.fig = plt.Figure(figsize=(14, 8), dpi=100)
        gs = self.fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)

        self.ax_price = self.fig.add_subplot(gs[0])
        self.ax_vol = self.fig.add_subplot(gs[1], sharex=self.ax_price)
        self.ax_rsi = self.fig.add_subplot(gs[2], sharex=self.ax_price)
        self.ax_macd = self.fig.add_subplot(gs[3], sharex=self.ax_price)

        self.canvas = FigureCanvasTkAgg(self.fig, self.chart_tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def setup_prediction_tab(self):
        frame = ttk.Frame(self.pred_tab)
        frame.pack(expand=True, fill="both", padx=50, pady=50)

        self.result_label = ttk.Label(frame, text="Click 'Predict Now'", font=("Arial", 20))
        self.result_label.pack(pady=20)

        info = ttk.Frame(frame)
        info.pack(pady=20)

        self.conf_label = ttk.Label(info, text="Confidence: --", font=("Arial", 14))
        self.acc_label = ttk.Label(info, text="Backtest Accuracy: --", font=("Arial", 14))
        self.model_label = ttk.Label(info, text="Models: --", font=("Arial", 12))

        self.conf_label.grid(row=0, column=0, padx=30)
        self.acc_label.grid(row=0, column=1, padx=30)
        self.model_label.grid(row=1, column=0, columnspan=2, pady=10)

    def setup_news_tab(self):
        self.news_text = scrolledtext.ScrolledText(self.news_tab, wrap=tk.WORD, font=("Consolas", 10))
        self.news_text.pack(fill="both", expand=True, padx=10, pady=10)

    def setup_history_tab(self):
        cols = ("Date", "Ticker", "Signal", "Confidence")
        self.tree = ttk.Treeview(self.history_tab, columns=cols, show="headings", height=15)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=120, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=10, pady=10)
        self.refresh_history()

    # ======================= THREAD-SAFE UI =======================
    def safe_call(self, func, *args):
        def _run():
            try:
                func(*args)
            except Exception:
                logging.exception("UI callback error")
        self.root.after(0, _run)

    # ======================= ACTIONS =======================
    def start_fetch(self):
        self.status_var.set("Fetching data...")
        q, _ = self.fetch_data()
        self.root.after(100, lambda: self.check_task(q, self.on_data_fetched))

    @threaded
    def fetch_data(self):
        ticker = self.ticker_var.get().strip().upper()
        period = self.period_var.get()
        df = fetch_stock_data(ticker, period)

        if df is not None and not df.empty:
        # FIX: Rename lowercase yfinance columns to match predictor expectations
            df.rename(columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume"
            }, inplace=True)

            df = add_indicators(df)

        return df

    def on_data_fetched(self, df):
        if df is None or (hasattr(df, "empty") and df.empty):
            self.status_var.set("Failed to fetch data")
            messagebox.showerror("Error", "No data received")
            return
        self.data = df
        try:
            self.plot_chart()
        except Exception as e:
            logging.exception("Plotting failed")
            messagebox.showerror("Plot Error", str(e))
        self.status_var.set(f"Loaded {len(df)} rows for {self.ticker_var.get()}")
        # Immediately update price display
        try:
            price = get_live_price(self.ticker_var.get().strip().upper())
            self.live_price_var.set(f"₹{price:.2f}")
        except Exception:
            pass

    def plot_chart(self):
        if self.data is None:
            return

        df = self.data.tail(500).copy()
        self.ax_price.clear(); self.ax_vol.clear(); self.ax_rsi.clear(); self.ax_macd.clear()

        # Price + SMA
        self.ax_price.plot(df.index, df['Close'], label='Close', linewidth=1.5)
        if 'SMA20' in df.columns:
            self.ax_price.plot(df.index, df['SMA20'], label='SMA20', alpha=0.8)
        if 'SMA50' in df.columns:
            self.ax_price.plot(df.index, df['SMA50'], label='SMA50', alpha=0.8)
        self.ax_price.legend()
        self.ax_price.set_title(f"{self.ticker_var.get()} - Price & Indicators")
        self.ax_price.grid(alpha=0.3)

        # Volume (if available)
        if 'Volume' in df.columns:
            try:
                self.ax_vol.bar(df.index, df['Volume'], alpha=0.6)
                self.ax_vol.set_ylabel("Volume")
            except Exception:
                logging.exception("Volume plot failed")
        else:
            # No volume column — clear axis
            self.ax_vol.clear()

        # RSI
        if 'RSI' in df.columns:
            self.ax_rsi.plot(df.index, df['RSI'])
            self.ax_rsi.axhline(70, linestyle='--', alpha=0.7)
            self.ax_rsi.axhline(30, linestyle='--', alpha=0.7)
            self.ax_rsi.set_ylim(0, 100)
        else:
            self.ax_rsi.clear()

        # MACD - be tolerant of different naming
        macd_sig_col = None
        if 'MACD_Signal' in df.columns:
            macd_sig_col = 'MACD_Signal'
        elif 'MACD_signal' in df.columns:
            macd_sig_col = 'MACD_signal'

        if 'MACD' in df.columns:
            self.ax_macd.plot(df.index, df['MACD'], label='MACD')
            if macd_sig_col and macd_sig_col in df.columns:
                try:
                    self.ax_macd.plot(df.index, df[macd_sig_col], label='Signal')
                except Exception:
                    logging.exception("MACD signal plot failed")
            self.ax_macd.legend()
        else:
            self.ax_macd.clear()

        self.fig.autofmt_xdate()
        self.canvas.draw()

    def start_predict(self):
        if self.data is None:
            messagebox.showwarning("No Data", "Fetch data first")
            return
        self.status_var.set("AI Predicting...")
        q, _ = self.predict()
        self.root.after(100, lambda: self.check_task(q, self.on_prediction_done))


    @threaded
    def predict(self):
        if self.data is None or self.data.empty:
            raise ValueError("No stock data available for prediction")

        df = self.data.copy()

    # --- CRITICAL FIX ---
    # Ensure 'Close' exists (predictor needs this)
        if "Close" not in df.columns:
            raise ValueError("Data is missing 'Close' column. Cannot run prediction.")

    # Drop rows with missing Close
        df = df.dropna(subset=["Close"])

    # Run model
        trainer = ModelTrainer(df, verbose=False)
        return trainer.run()   # must return dict


    def on_prediction_done(self, result):
        if not isinstance(result, dict):
            messagebox.showerror("Prediction Error", "Invalid result from predictor")
            self.status_var.set("Prediction error")
            return

        signal = result.get("signal", "N/A")
        conf = result.get("confidence", 0)
        acc = result.get("accuracy", 0)
        models = ", ".join(result.get("models_used", []))

        color = "green" if signal == "BUY" else "red"
        self.result_label.config(text=f"AI SIGNAL: {signal}", font=("Arial", 28, "bold"), foreground=color)
        try:
            self.conf_label.config(text=f"Confidence: {conf:.1f}%")
            self.acc_label.config(text=f"Backtest Accuracy: {acc:.1f}%")
        except Exception:
            self.conf_label.config(text=f"Confidence: {conf}%")
            self.acc_label.config(text=f"Backtest Accuracy: {acc}%")

        self.model_label.config(text=f"Models: {models or 'Ensemble'}")

        # Save history (accept numeric or "xx.xx%" string)
        try:
            self.history.add(self.ticker_var.get().upper(), signal, conf)
        except Exception:
            logging.exception("Failed to save history")

        self.refresh_history()
        self.tabs.select(self.pred_tab)
        self.status_var.set(f"Prediction: {signal} ({conf:.1f}%)")

    def start_news(self):
        q, _ = self.load_news()
        self.root.after(100, lambda: self.check_task(q, self.on_news_loaded))

    @threaded
    def load_news(self):
        # fetch_news returns list[str]
        return fetch_news(self.ticker_var.get().strip().upper())

    def on_news_loaded(self, articles):
        self.news_text.delete(1.0, tk.END)
        for a in (articles or [])[:20]:
            self.news_text.insert(tk.END, f"• {a}\n\n")
        self.tabs.select(self.news_tab)

    def update_live_price(self):
        @threaded
        def get_price():
            return get_live_price(self.ticker_var.get().strip().upper())

        q, _ = get_price()
        self.root.after(100, lambda: self.check_task(q, lambda p: self.live_price_var.set(f"₹{p:.2f}"), default="--"))
        self.root.after(30_000, self.update_live_price)  # Every 30s

    def check_task(self, queue, on_success, default=None):
        """
        Non-blocking poll for the queue returned by threaded().
        If result not ready, re-schedule check after 100ms.
        """
        try:
            status, result = queue.get_nowait()
            if status == "success":
                self.safe_call(on_success, result)
            else:
                logging.exception("Background task error")
                self.safe_call(messagebox.showerror, "Error", str(result))
                self.status_var.set("Error occurred")
        except Exception:
            # not ready yet
            self.root.after(100, lambda: self.check_task(queue, on_success, default))

    def refresh_history(self):
        for i in self.tree.get_children():
            self.tree.delete(i)

        for entry in reversed(self.history.get_all()):
            # Accept two formats:
            # [ticker, signal, confidence]  OR  [date, ticker, signal, confidence]
            try:
                if len(entry) == 4:
                    date_str, ticker, signal, conf_val = entry
                elif len(entry) >= 3:
                    # no date stored
                    date_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
                    ticker, signal, conf_val = entry[:3]
                else:
                    continue

                # parse confidence whether "12.34%" or numeric
                if isinstance(conf_val, str) and conf_val.endswith("%"):
                    conf_num = float(conf_val.rstrip("%"))
                else:
                    conf_num = float(conf_val)

                self.tree.insert("", 0, values=(date_str, ticker, signal, f"{conf_num:.1f}%"))
            except Exception:
                logging.exception("Failed to render history entry")

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        self.root.mainloop()
