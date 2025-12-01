import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from src.preprocess import load_and_preprocess
from src.dl_models import EnsembleDL

# ---------------------------------------------------------
# Load preprocessed data
# ---------------------------------------------------------
(
    X_res, y_res,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test
) = load_and_preprocess("data/creditcard.csv")

results = []

# =========================================================
# 1️⃣ Evaluate Traditional ML Models
# =========================================================
ml_model_names = ["logistic_regression", "random_forest", "xgboost"]

for name in ml_model_names:
    print(f"Evaluating {name}...")

    model = joblib.load(f"models/{name}.pkl")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append([name, precision, recall, f1, auc])


# =========================================================
# 2️⃣ Evaluate Deep Learning Models
# =========================================================

dl_models = {
    "dl_class_weighted": "models/dl/model1_classweight.h5",
    "dl_threshold_adjusted": "models/dl/model2_threshold.h5",
    "dl_f1_optimized": "models/dl/model3_f1.h5",
}

for name, path in dl_models.items():
    print(f"Evaluating {name}...")

    model = tf.keras.models.load_model(path)

    y_proba = model.predict(X_test).flatten()
    y_pred = (y_proba >= 0.5).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append([name, precision, recall, f1, auc])


# =========================================================
# 3️⃣ Evaluate Deep Learning Ensemble
# =========================================================
print("Evaluating dl_ensemble...")

ensemble = joblib.load("models/dl/dl_ensemble.pkl")

y_pred = ensemble.predict(X_test).flatten()
y_proba = ensemble.predict_proba(X_test).flatten()

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

results.append(["dl_ensemble", precision, recall, f1, auc])


# =========================================================
# Save Results Table
# =========================================================
df = pd.DataFrame(results, columns=["Model", "Precision", "Recall", "F1-Score", "ROC-AUC"])
print("\nEvaluation Results:\n")
print(df)

df.to_csv("model_comparison copy.csv", index=False)
print("\nSaved → model_comparison.csv")
