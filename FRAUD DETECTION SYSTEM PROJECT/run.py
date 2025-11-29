# ============================================================
#               FRAUD DETECTION SYSTEM PROJECT
# ============================================================

# ====================== IMPORTS =============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("creditcard.csv")   # Change path if needed
print(df.head())
print(df.info())
print(df.describe())

# ============================================================
# 2. EDA (EXPLORATORY DATA ANALYSIS)
# ============================================================

# Fraud ratio
plt.figure(figsize=(6,4))
df['Class'].value_counts().plot(kind='bar')
plt.title("Fraud vs Normal Transaction Count")
plt.show()

fraud_ratio = df['Class'].value_counts(normalize=True)
print("Fraud Ratio:\n", fraud_ratio)

# Distribution of Amount
plt.figure(figsize=(6,4))
sns.histplot(df['Amount'], bins=50)
plt.title("Transaction Amount Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(15,8))
sns.heatmap(df.corr(), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.show()

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

df['Amount_log'] = np.log1p(df['Amount'])



# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ============================================================
# 5. HANDLE IMBALANCE (SMOTE)
# ============================================================

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_res.value_counts())

# ============================================================
# 6. NORMALIZATION
# ============================================================

scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

# ============================================================
# 7. MODEL 1 — RANDOM FOREST
# ============================================================

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42
)

rf.fit(X_train_res, y_train_res)
rf_pred = rf.predict(X_test)

print("\n===== RANDOM FOREST REPORT =====")
print(classification_report(y_test, rf_pred))
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("ROC-AUC:", roc_auc_score(y_test, rf_pred))

# ============================================================
# 8. MODEL 2 — XGBOOST
# ============================================================

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)

xgb.fit(X_train_res, y_train_res)
xgb_pred = xgb.predict(X_test)

print("\n===== XGBOOST REPORT =====")
print(classification_report(y_test, xgb_pred))
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print("ROC-AUC:", roc_auc_score(y_test, xgb_pred))

# ============================================================
# 9. CONFUSION MATRIX (BEST MODEL)
# ============================================================

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, xgb_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (XGBoost)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ============================================================
# 10. FINAL COMPARISON
# ============================================================

print("\n========== FINAL RESULTS ==========")
print("RandomForest Accuracy:", accuracy_score(y_test, rf_pred))
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))

print("RandomForest AUC:", roc_auc_score(y_test, rf_pred))
print("XGBoost AUC:", roc_auc_score(y_test, xgb_pred))
