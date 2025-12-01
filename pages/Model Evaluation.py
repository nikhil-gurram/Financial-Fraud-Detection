# evaluate_models_with_table.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    auc,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.preprocess import load_and_preprocess

st.set_page_config(layout="wide")
sns.set_style("darkgrid")

# -------------------------
# Load same splits used in training (no SMOTE)
# -------------------------
# load_and_preprocess should return consistent splits:
# X_res, y_res, X_train, y_train, X_val, y_val, X_test, y_test
(
    _,
    _,
    _,
    _,
    X_val,
    y_val,
    X_test,
    y_test
) = load_and_preprocess("data/creditcard - Copy.csv", use_smote=False)

# -------------------------
# Model registries (match training script)
# -------------------------
ML_WITH_SMOTE = {
    "LR (SMOTE)": "models/ml/with_smote/logistic_regression.pkl",
    "RF (SMOTE)": "models/ml/with_smote/random_forest.pkl",
    "XGB (SMOTE)": "models/ml/with_smote/xgboost.pkl",
}

ML_NO_SMOTE = {
    "LR (NO SMOTE)": "models/ml/without_smote/logistic_regression.pkl",
    "RF (NO SMOTE)": "models/ml/without_smote/random_forest.pkl",
    "XGB (NO SMOTE)": "models/ml/without_smote/xgboost.pkl",
}

DL_MODELS = {
    "DL - Class Weight": "models/dl/model1_classweight.h5",
    "DL - Threshold Tuned": "models/dl/model2_threshold.h5",
    "DL - F1 Optimized": "models/dl/model3_f1.h5",
}

ENSEMBLE_BASE = "models/dl/dl_ensemble"  # will load *_m1.h5 etc.

# -------------------------
# Ensemble loader
# -------------------------
class EnsembleDL:
    def __init__(self, base_path):
        self.m1 = tf.keras.models.load_model(base_path + "_m1.h5")
        self.m2 = tf.keras.models.load_model(base_path + "_m2.h5")
        self.m3 = tf.keras.models.load_model(base_path + "_m3.h5")
        with open(base_path + ".meta.pkl", "rb") as f:
            meta = pickle.load(f)
        self.threshold = float(meta.get("threshold_final", 0.5))

    def predict_proba(self, X):
        p1 = self.m1.predict(X, verbose=0).ravel()
        p2 = self.m2.predict(X, verbose=0).ravel()
        p3 = self.m3.predict(X, verbose=0).ravel()
        return (p1 + p2 + p3) / 3.0

    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(int)


# -------------------------
# Safe predict wrapper
# returns preds, probs (np arrays)
# -------------------------
def safe_predict(name, path, X):
    # Ensemble case
    if name == "DL - Ensemble":
        ens = EnsembleDL(ENSEMBLE_BASE)
        proba = ens.predict_proba(X)
        preds = ens.predict(X)
        return preds, proba

    # Keras model (.h5)
    if isinstance(path, str) and path.endswith(".h5"):
        model = tf.keras.models.load_model(path)
        proba = model.predict(X, verbose=0).ravel()
        preds = (proba >= 0.5).astype(int)
        return preds, proba

    # Pickle model (.pkl)
    if isinstance(path, str) and path.endswith(".pkl"):
        model = joblib.load(path)
        if hasattr(model, "predict_proba"):
            proba_all = model.predict_proba(X)
            # choose column 1 if available
            if proba_all.ndim == 1:
                proba = proba_all
            else:
                proba = proba_all[:, 1]
        else:
            # fallback: model only gives labels
            preds = model.predict(X)
            return preds.astype(int), preds.astype(float)
        preds = (proba >= 0.5).astype(int)
        return preds, proba

    raise ValueError(f"Unknown model type or path: {path}")


# -------------------------
# Metric helper: compute precision/recall/f1/roc/pr for positive class
# -------------------------
def compute_metrics(y_true, probs, thresh=0.5):
    # ensure 1d arrays
    probs = np.asarray(probs).ravel()
    y_true = np.asarray(y_true).ravel().astype(int)

    # preds from threshold
    preds = (probs >= thresh).astype(int)

    # safe compute ROC-AUC (requires >1 unique label in y_true)
    try:
        roc = roc_auc_score(y_true, probs)
    except Exception:
        roc = float("nan")

    # PR-AUC
    try:
        p, r, _ = precision_recall_curve(y_true, probs)
        pr_auc = auc(r, p)
    except Exception:
        pr_auc = float("nan")

    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)

    return {"precision": float(prec), "recall": float(rec), "f1": float(f1), "roc_auc": float(roc), "pr_auc": float(pr_auc)}


# -------------------------
# Evaluate and collect metrics for a registry
# returns dict[name] -> metric dict containing Val_* and Test_*
# -------------------------
def evaluate_registry(registry, registry_name):
    results = {}
    st.write(f"## {registry_name}")
    st.write("---")

    for name, path in registry.items():
        st.write(f"### {name}")
        if not os.path.exists(path):
            st.error(f"Missing file: {path}")
            # still include NaNs in results
            results[name] = {
                "Val_Precision": np.nan, "Val_Recall": np.nan, "Val_F1": np.nan, "Val_ROC_AUC": np.nan, "Val_PR_AUC": np.nan,
                "Test_Precision": np.nan, "Test_Recall": np.nan, "Test_F1": np.nan, "Test_ROC_AUC": np.nan, "Test_PR_AUC": np.nan,
            }
            continue

        try:
            # VAL
            preds_val, probs_val = safe_predict(name, path, X_val)
            m_val = compute_metrics(y_val, probs_val, thresh=0.5)

            # TEST
            preds_test, probs_test = safe_predict(name, path, X_test)
            m_test = compute_metrics(y_test, probs_test, thresh=0.5)

            # store
            results[name] = {
                "Val_Precision": m_val["precision"],
                "Val_Recall": m_val["recall"],
                "Val_F1": m_val["f1"],
                "Val_ROC_AUC": m_val["roc_auc"],
                "Val_PR_AUC": m_val["pr_auc"],
                "Test_Precision": m_test["precision"],
                "Test_Recall": m_test["recall"],
                "Test_F1": m_test["f1"],
                "Test_ROC_AUC": m_test["roc_auc"],
                "Test_PR_AUC": m_test["pr_auc"],
            }

            # quick detailed display: show classification report + confusion matrix + ROC curve (test)
            st.write("**Test classification report (class=1 metrics shown in table below)**")
            st.table(pd.DataFrame(classification_report(y_test, preds_test, output_dict=True)).T)

            cm = confusion_matrix(y_test, preds_test)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title("Test Confusion Matrix")
            st.pyplot(fig)

            # ROC curve (test)
            try:
                fpr, tpr, _ = roc_curve(y_test, probs_test)
                fig2, ax2 = plt.subplots(figsize=(5, 3))
                ax2.plot(fpr, tpr, label=f"ROC-AUC={m_test['roc_auc']:.4f}")
                ax2.plot([0, 1], [0, 1], "k--")
                ax2.set_xlabel("FPR")
                ax2.set_ylabel("TPR")
                ax2.legend()
                st.pyplot(fig2)
            except Exception:
                st.write("ROC curve unavailable (maybe only one class present in test labels).")

        except Exception as e:
            st.error(f"Error evaluating {name}: {e}")
            results[name] = {
                "Val_Precision": np.nan, "Val_Recall": np.nan, "Val_F1": np.nan, "Val_ROC_AUC": np.nan, "Val_PR_AUC": np.nan,
                "Test_Precision": np.nan, "Test_Recall": np.nan, "Test_F1": np.nan, "Test_ROC_AUC": np.nan, "Test_PR_AUC": np.nan,
            }

    return results


# -------------------------
# Run evaluation for registries
# -------------------------
st.title("📊 Combined Evaluation (VAL + TEST)")

# ML with SMOTE
res_with_smote = evaluate_registry(ML_WITH_SMOTE, "ML MODELS (WITH SMOTE)")

# ML without SMOTE
res_no_smote = evaluate_registry(ML_NO_SMOTE, "ML MODELS (NO SMOTE)")

# DL models
res_dl = evaluate_registry(DL_MODELS, "DEEP LEARNING MODELS")

# Ensemble (special)
# ensemble is stored as base path; we pass a tiny registry entry mapping so evaluate_registry can display
res_ens = {}
st.write("## DEEP LEARNING ENSEMBLE")
if os.path.exists(ENSEMBLE_BASE + "_m1.h5") and os.path.exists(ENSEMBLE_BASE + ".meta.pkl"):
    try:
        preds_val_e, probs_val_e = EnsembleDL(ENSEMBLE_BASE).predict(X_val), EnsembleDL(ENSEMBLE_BASE).predict_proba(X_val)
        preds_test_e, probs_test_e = EnsembleDL(ENSEMBLE_BASE).predict(X_test), EnsembleDL(ENSEMBLE_BASE).predict_proba(X_test)
        mval_e = compute_metrics(y_val, probs_val_e)
        mtest_e = compute_metrics(y_test, probs_test_e)
        res_ens["DL - Ensemble"] = {
            "Val_Precision": mval_e["precision"],
            "Val_Recall": mval_e["recall"],
            "Val_F1": mval_e["f1"],
            "Val_ROC_AUC": mval_e["roc_auc"],
            "Val_PR_AUC": mval_e["pr_auc"],
            "Test_Precision": mtest_e["precision"],
            "Test_Recall": mtest_e["recall"],
            "Test_F1": mtest_e["f1"],
            "Test_ROC_AUC": mtest_e["roc_auc"],
            "Test_PR_AUC": mtest_e["pr_auc"],
        }

        # show details
        st.table(pd.DataFrame(classification_report(y_test, preds_test_e, output_dict=True)).T)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(confusion_matrix(y_test, preds_test_e), annot=True, fmt="d", ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error evaluating ensemble: {e}")
else:
    st.error("Ensemble artifacts missing (check models/dl/dl_ensemble_m1.h5 and .meta.pkl)")

# -------------------------
# Combine all results into a single DataFrame for comparison
# -------------------------
combined = {**res_with_smote, **res_no_smote, **res_dl, **res_ens}

if len(combined) == 0:
    st.warning("No results collected — check model file paths and that models were saved.")
else:
    df = pd.DataFrame(combined).T
    # reorder columns for readability
    cols = [
        "Val_Precision", "Val_Recall", "Val_F1", "Val_ROC_AUC", "Val_PR_AUC",
        "Test_Precision", "Test_Recall", "Test_F1", "Test_ROC_AUC", "Test_PR_AUC"
    ]
    # keep only existing columns
    cols = [c for c in cols if c in df.columns]
    df = df[cols].round(4)

    st.write("## 📋 Final comparison table (validation + test)")
    st.dataframe(df, use_container_width=True)

    # Small bar chart comparison for Test_F1
    if "Test_F1" in df.columns:
        st.write("### Test F1 comparison")
        chart_df = df["Test_F1"].sort_values(ascending=False).reset_index()
        chart_df.columns = ["Model", "Test_F1"]
        st.bar_chart(chart_df.set_index("Model"))
