# ============================================================
#  Predictive Maintenance — Full Analysis Pipeline
#  Includes: EDA, Multi-model comparison, Hyperparameter tuning,
#            SHAP explainability, Failure type breakdown,
#            Model saving for Streamlit app
# ============================================================

import sys
import warnings
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve,
                             classification_report)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv(r"D:\New folder (2)\ai 2020.csv")
print("Dataset shape:", df.shape)
print(df.head())

# ============================================================
# 2. SEPARATE FAILURE SUBTYPES BEFORE DROPPING
#    We keep them to train individual failure-type models
# ============================================================
FAILURE_TYPES = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
FAILURE_LABELS = {
    'TWF': 'Tool Wear Failure',
    'HDF': 'Heat Dissipation Failure',
    'PWF': 'Power Failure',
    'OSF': 'Overstrain Failure',
    'RNF': 'Random Failure'
}

# Save subtype targets before dropping
y_subtypes = df[FAILURE_TYPES].copy()

# Drop leakage and ID columns
df.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], inplace=True)

# ============================================================
# 3. PREPROCESSING
# ============================================================
X = df.drop(['Machine failure'], axis=1)
y = df['Machine failure']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Split subtypes with same indices
_, _, ys_train, ys_test = train_test_split(
    X, y_subtypes, test_size=0.2, random_state=42
)

cat_features = X.select_dtypes(include="object").columns
preprocessor = ColumnTransformer([
    ("OneHotEncoder", OneHotEncoder(drop='first'), cat_features)
], remainder='passthrough')

X_train = preprocessor.fit_transform(X_train)
X_test  = preprocessor.transform(X_test)

# Feature names after encoding (needed for SHAP plots)
ohe_features  = list(preprocessor.named_transformers_['OneHotEncoder']
                     .get_feature_names_out(cat_features))
num_features   = list(X.select_dtypes(exclude="object").columns)
FEATURE_NAMES  = ohe_features + num_features

# ============================================================
# 4. TRAIN ALL 6 MODELS
# ============================================================
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(class_weight='balanced', random_state=42),
    "Random Forest":       RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
    "Gradient Boost":      GradientBoostingClassifier(n_estimators=100, random_state=42),
    "AdaBoost":            AdaBoostClassifier(n_estimators=100, random_state=42),
    "XGBoost":             XGBClassifier(scale_pos_weight=28.5, eval_metric='aucpr', random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)

    print(f"\n{'='*40}\n{name}")
    print('Training Set:')
    print(f"  Accuracy:  {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"  Recall:    {recall_score(y_train, y_train_pred):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_train, y_train_pred):.4f}")
    print('Test Set:')
    print(f"  Accuracy:  {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"  F1:        {f1_score(y_test, y_test_pred, average='weighted'):.4f}")
    print(f"  Precision: {precision_score(y_test, y_test_pred):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_test_pred):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_test_pred):.4f}")
    print(f'\nClassification Report:\n{classification_report(y_test, y_test_pred)}')

# ============================================================
# 5. HYPERPARAMETER TUNING (Top 3 models)
# ============================================================
rf_params = {
    "max_depth":        [5, 8, 15, None, 10],
    "max_features":     [5, 7, "sqrt", 8],
    "min_samples_split":[2, 8, 15, 20],
    "n_estimators":     [100, 200, 500, 1000],
    "class_weight":     ["balanced"]
}
gb_params = {
    "learning_rate":    [0.1, 0.01, 0.05],
    "max_depth":        [3, 5, 8, 10],
    "n_estimators":     [100, 200, 300],
    "subsample":        [0.6, 0.8, 1.0],
    "min_samples_split":[2, 5, 10]
}
xgboost_params = {
    "learning_rate":    [0.1, 0.01, 0.05],
    "max_depth":        [5, 8, 12, 20, 30],
    "n_estimators":     [100, 200, 300],
    "colsample_bytree": [0.5, 0.8, 1, 0.3, 0.4],
    "scale_pos_weight": [28.5]
}

randomcv_models = [
    ("RF",      RandomForestClassifier(),     rf_params),
    ("GB",      GradientBoostingClassifier(), gb_params),
    ("Xgboost", XGBClassifier(),              xgboost_params),
]

model_param = {}
for name, model, params in randomcv_models:
    random = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=50,
        cv=3,
        verbose=1,
        n_jobs=-1,
        scoring='recall'
    )
    random.fit(X_train, y_train)
    model_param[name] = random.best_params_
    print(f"Best params for {name}: {model_param[name]}")

final_models = {
    "Random Forest":  RandomForestClassifier(**model_param["RF"]),
    "Gradient Boost": GradientBoostingClassifier(**model_param["GB"]),
    "XGBoost":        XGBClassifier(**model_param["Xgboost"]),
}

for name, model in final_models.items():
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    print(f"\n{'='*40}\n{name} (Tuned)")
    print(f"  Recall:    {recall_score(y_test, y_test_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_test_pred):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_test_pred):.4f}")
    print(classification_report(y_test, y_test_pred))

# Use tuned XGBoost as the main model going forward
best_model = final_models["XGBoost"]

# ============================================================
# 6. ROC CURVE — ALL MODELS
# ============================================================
plt.figure(figsize=(10, 7))
for name, model in models.items():
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    auc_score   = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, label='%s (AUC = %0.3f)' % (name, auc_score))

plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier (AUC = 0.500)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
plt.title('ROC Curve Comparison — All Models', fontsize=14)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_auc_comparison.png", dpi=150)
plt.show()

# ============================================================
# 7. SHAP EXPLAINABILITY
# ============================================================
explainer  = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# --- 7a. Summary Plot — Global feature importance ---
plt.figure()
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=FEATURE_NAMES,
    show=False
)
plt.title("SHAP — Global Feature Importance")
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: shap_summary.png")

# --- 7b. Bar Plot — Mean absolute SHAP values ---
plt.figure()
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=FEATURE_NAMES,
    plot_type="bar",
    show=False
)
plt.title("SHAP — Mean Feature Impact")
plt.tight_layout()
plt.savefig("shap_bar.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: shap_bar.png")

# --- 7c. Force Plot — Single prediction explanation ---
# Pick the first actual failure in test set for a meaningful example
failure_idx = np.where(y_test.values == 1)[0][0]
shap.force_plot(
    explainer.expected_value,
    shap_values[failure_idx],
    X_test[failure_idx],
    feature_names=FEATURE_NAMES,
    matplotlib=True,
    show=False
)
plt.title("SHAP Force Plot — Single Failure Prediction")
plt.tight_layout()
plt.savefig("shap_force_plot.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: shap_force_plot.png")

# ============================================================
# 8. FAILURE TYPE BREAKDOWN
#    Train one XGBoost per failure subtype
# ============================================================
print("\n" + "="*50)
print("FAILURE TYPE BREAKDOWN MODELS")
print("="*50)

subtype_models = {}
for col in FAILURE_TYPES:
    pos = ys_train[col].sum()
    neg = (ys_train[col] == 0).sum()
    if pos < 5:
        print(f"  {FAILURE_LABELS[col]}: too few samples ({pos}), skipping")
        continue
    ratio = neg / pos
    m = XGBClassifier(scale_pos_weight=ratio, eval_metric='aucpr', random_state=42)
    m.fit(X_train, ys_train[col])
    subtype_models[col] = m

    y_pred_sub = m.predict(X_test)
    print(f"\n  {FAILURE_LABELS[col]}:")
    print(f"    Recall:    {recall_score(ys_test[col], y_pred_sub):.4f}")
    print(f"    Precision: {precision_score(ys_test[col], y_pred_sub, zero_division=0):.4f}")

# --- Failure Type Distribution Bar Chart ---
failure_counts = {FAILURE_LABELS[k]: ys_train[k].sum() for k in FAILURE_TYPES}
plt.figure(figsize=(8, 5))
plt.bar(failure_counts.keys(), failure_counts.values(), color='steelblue')
plt.title("Failure Type Distribution in Training Data")
plt.xlabel("Failure Type")
plt.ylabel("Count")
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig("failure_type_distribution.png", dpi=150)
plt.show()
print("Saved: failure_type_distribution.png")

# ============================================================
# 9. BUSINESS IMPACT CALCULATION
# ============================================================
avg_unplanned_cost  = 500000   # ₹ cost of unplanned breakdown
avg_maintenance_cost = 50000   # ₹ cost of planned maintenance

y_test_pred_final = best_model.predict(X_test)
failures_caught   = recall_score(y_test, y_test_pred_final)
total_failures    = y_test.sum()
caught_count      = int(failures_caught * total_failures)

money_saved = caught_count * (avg_unplanned_cost - avg_maintenance_cost)
print(f"\n{'='*50}")
print(f"BUSINESS IMPACT (Test Set Sample)")
print(f"{'='*50}")
print(f"  Total actual failures:      {total_failures}")
print(f"  Failures caught by model:   {caught_count} ({failures_caught*100:.1f}%)")
print(f"  Estimated savings:          ₹{money_saved:,.0f}")
print(f"  (assumes ₹{avg_unplanned_cost:,} unplanned vs ₹{avg_maintenance_cost:,} planned maintenance)")

# ============================================================
# 10. SAVE EVERYTHING FOR STREAMLIT APP
# ============================================================
artifacts = {
    "model":          best_model,
    "subtype_models": subtype_models,
    "preprocessor":   preprocessor,
    "feature_names":  FEATURE_NAMES,
    "failure_labels": FAILURE_LABELS,
    "explainer":      explainer,
}
with open("model_artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("\n✅ Saved model_artifacts.pkl — ready for Streamlit app!")
print("   Run the app with:  streamlit run app.py")