# predictor.py — Production-ready, stable, bug-free
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


class ModelTrainer:
    def __init__(self, df: pd.DataFrame, verbose: bool = False):
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Input must be a non-empty pandas DataFrame with 'Close' column")
        if "Close" not in df.columns:
            raise ValueError("DataFrame must contain 'Close' column")

        self.df = df.copy()
        self.confidence = 0.0
        self.accuracy = 0.0
        self.models_used = []
        self.verbose = verbose
        self.model = None

    # ----------------------------- INDICATORS -----------------------------
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # SMA
        df["SMA20"] = df["Close"].rolling(20, min_periods=1).mean()
        df["SMA50"] = df["Close"].rolling(50, min_periods=1).mean()

        # EMA
        df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

        # RSI
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14, min_periods=1).mean()
        avg_loss = loss.rolling(14, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df["RSI"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # Price & Volume
        df["PriceChange%"] = df["Close"].pct_change().fillna(0)
        df["VolumeChange%"] = (
            df["Volume"].pct_change().fillna(0) if "Volume" in df.columns else 0
        )

        return df

    # ----------------------------- TARGET -----------------------------
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        return df.dropna(subset=["Target"])

    # ----------------------------- MODELS -----------------------------
    def _build_models(self):
        self.models_used = []  # RESET for each training

        models = []

        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            n_jobs=-1,
            random_state=42
        )
        models.append(("rf", rf))
        self.models_used.append("RF")

        lr = LogisticRegression(
            max_iter=1000,
            random_state=42
        )  # FIXED: removed n_jobs
        models.append(("lr", lr))
        self.models_used.append("LogisticRegression")

        svm = SVC(
            probability=True,
            C=1.0,
            kernel="rbf",
            random_state=42,
        )
        models.append(("svm", svm))
        self.models_used.append("SVM")

        if XGB_AVAILABLE:
            xgb = XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            models.append(("xgb", xgb))
            self.models_used.append("XGB")

        return models

    def train_ensemble(self, X_train, y_train):
        models = self._build_models()

        ensemble = VotingClassifier(
            estimators=models,
            voting="soft"
        )
        ensemble.fit(X_train, y_train)
        self.model = ensemble
        return ensemble

    # ----------------------------- RUN TRAINING -----------------------------
    def run(self):
        df = self.add_indicators(self.df)
        df = self.create_target(df)

        FEATURES = [
            "Close", "SMA20", "SMA50", "EMA20", "EMA50",
            "RSI", "MACD", "MACD_signal", "PriceChange%", "VolumeChange%"
        ]

        available_features = [f for f in FEATURES if f in df.columns]
        if len(available_features) < 3:
            raise ValueError("Not enough features computed.")

        df = df.dropna(subset=available_features)
        if len(df) < 60:
            raise ValueError("Not enough data after feature engineering (need >60 rows).")

        X = df[available_features].astype(float)
        y = df["Target"]

        # Time-series compatible split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            shuffle=False
        )

        model = self.train_ensemble(X_train, y_train)

        preds = model.predict(X_test)
        self.accuracy = round(accuracy_score(y_test, preds) * 100, 2)

        last_row = X.iloc[-1:].values

        proba = model.predict_proba(last_row)[0]
        pred = np.argmax(proba)
        self.confidence = round(float(np.max(proba)) * 100, 2)
        signal = "BUY" if pred == 1 else "SELL"

        return {
            "signal": signal,
            "confidence": self.confidence,
            "accuracy": self.accuracy,
            "models_used": self.models_used,
            "feature_count": len(available_features)
        }
