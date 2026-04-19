# ============================================================
#  Naval Vessel Propulsion — XGBoost Failure Detection
#  Dataset : Naval Condition-Based Maintenance (11,934 records)
#  Strategy: Isolation Forest → anomaly labels → XGBoost
#  Parameters kept identical to the original 6-model comparison
# ============================================================

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data.csv")                          # ← your file

# Drop index column
df = df.loc[:, ~df.columns.str.contains('index', case=False)]

print("Dataset shape:", df.shape)
print(df.head())

# ============================================================
# 2. CLEAN COLUMN NAMES  (critical for XGBoost)
# ============================================================

df.columns = (
    df.columns
      .str.strip()
      .str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)   # replace bad chars
      .str.replace(r'_+', '_', regex=True)               # collapse repeated _
      .str.strip('_')                                     # trim leading/trailing _
)

print("\nCleaned columns:")
print(df.columns.tolist())

# ============================================================
# 3. GENERATE ANOMALY LABELS VIA ISOLATION FOREST
#    (dataset has NO machine-failure column)
#    
#    The two decay coefficients (kMc, kMt) are the strongest
#    health signals — values close to 1.0 = healthy,
#    lower values = component degradation / failure risk.
#    Isolation Forest captures multivariate anomalies across
#    ALL 18 sensor features simultaneously.
# ============================================================

print("\n[Step 3] Generating failure labels via Isolation Forest ...")

scaler  = StandardScaler()
X_scale = scaler.fit_transform(df)

iso_forest = IsolationForest(
    n_estimators  = 200,      # same ensemble depth as XGB below
    contamination = 0.08,     # ~8 % failure rate (domain-typical for naval props)
    random_state  = 42
)
iso_preds = iso_forest.fit_predict(X_scale)

# Isolation Forest: -1 = anomaly, 1 = normal  →  remap to 1/0
df['anomaly'] = (iso_preds == -1).astype(int)

print(f"   Normal  (0): {(df['anomaly']==0).sum():>6,}")
print(f"   Failure (1): {(df['anomaly']==1).sum():>6,}  "
      f"({df['anomaly'].mean()*100:.1f}%)")

# ============================================================
# 4. PREPROCESSING  (mirrors original code exactly)
# ============================================================

# Drop non-useful ID columns if present
drop_cols = [col for col in ['UDI', 'ProductID'] if col in df.columns]
df.drop(columns=drop_cols, inplace=True, errors='ignore')

# Features & Target
X = df.drop('anomaly', axis=1)
y = df['anomaly']

# Handle any remaining categorical features
X = pd.get_dummies(X, drop_first=True)

# Train-test split  (stratified to preserve failure ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")

# ============================================================
# 5. TRAIN XGBOOST  — exact parameters from original code
# ============================================================

model = XGBClassifier(
    n_estimators     = 200,
    max_depth        = 6,
    learning_rate    = 0.1,
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train),
    eval_metric      = 'logloss',
    random_state     = 42
)

model.fit(X_train, y_train)

# ============================================================
# 6. EVALUATION
# ============================================================

y_train_pred = model.predict(X_train)
y_test_pred  = model.predict(X_test)
y_test_prob  = model.predict_proba(X_test)[:, 1]

print("\n" + "="*50)
print("XGBOOST MODEL RESULTS — NAVAL VESSEL DATASET")
print("="*50)

print("\nTraining Set:")
print(f"  Accuracy:  {accuracy_score(y_train, y_train_pred):.4f}")
print(f"  Recall:    {recall_score(y_train, y_train_pred):.4f}")

print("\nTest Set:")
print(f"  Accuracy:  {accuracy_score(y_test, y_test_pred):.4f}")
print(f"  Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"  Recall:    {recall_score(y_test, y_test_pred):.4f}")
print(f"  F1 Score:  {f1_score(y_test, y_test_pred):.4f}")
print(f"  ROC-AUC:   {roc_auc_score(y_test, y_test_prob):.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_test_pred,
                             target_names=["Normal", "Failure"]))

# ============================================================
# 7. VISUALISATIONS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("XGBoost — Naval Vessel Failure Detection", fontsize=14, fontweight='bold')

# --- 7a. Confusion Matrix ---
cm   = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Failure"])
disp.plot(ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title("Confusion Matrix")

# --- 7b. ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
auc_val     = roc_auc_score(y_test, y_test_prob)
axes[1].plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {auc_val:.4f}')
axes[1].plot([0,1],[0,1], 'k--', lw=1)
axes[1].fill_between(fpr, tpr, alpha=0.1, color='steelblue')
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve")
axes[1].legend()

# --- 7c. Top 10 Feature Importance ---
importance = model.feature_importances_
features   = X.columns
feat_imp   = pd.Series(importance, index=features).sort_values(ascending=False)

feat_imp.head(10).plot(kind='bar', ax=axes[2], color='steelblue', edgecolor='white')
axes[2].set_title("Top 10 Feature Importance (XGBoost)")
axes[2].set_ylabel("Importance Score")
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("naval_xgboost_results.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n📊 Plot saved: naval_xgboost_results.png")

# ============================================================
# 8. SAVE MODEL
# ============================================================

with open("xgb_naval_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as xgb_naval_model.pkl")
