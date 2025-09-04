import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from src.preprocess import load_and_preprocess

st.set_page_config(page_title="Fraud Detection - Model Evaluation", layout="wide")

st.title("📊 Model Evaluation Dashboard")

st.markdown("""
This section compares **Logistic Regression, Random Forest, and XGBoost**  
on the test dataset using the following metrics:
- Precision, Recall, F1-score
- Confusion Matrix
- ROC Curve & AUC
""")

# Load dataset
X_res, y_res, X_test, y_test = load_and_preprocess("data/creditcard.csv")

models = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

selected_models = st.multiselect(
    "Select models to evaluate",
    list(models.keys()),
    default=list(models.keys())
)

cols_eval = st.columns(len(selected_models))

results = {}  # store metrics for comparison

for idx, model_name in enumerate(selected_models):
    model = joblib.load(f"models/{models[model_name]}")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    recall = report["1"]["recall"]
    precision = report["1"]["precision"]
    f1 = report["1"]["f1-score"]

    # Save metrics
    results[model_name] = {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": auc
    }

    with cols_eval[idx]:
        st.markdown(f"### {model_name}")

        # Classification Report
        st.dataframe(pd.DataFrame(report).transpose())

        # Confusion Matrix
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig, width="content")

        # ROC Curve
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        ax2.plot([0, 1], [0, 1], "k--")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.legend()
        ax2.set_title("ROC Curve")
        st.pyplot(fig2, width="content")

# ---------------------------------------------------
# 📊 Comparison Chart for All Models
# ---------------------------------------------------
if results:
    st.markdown("---")
    st.subheader("📊 Model Comparison")

    metrics_df = pd.DataFrame(results).T  # rows = models, cols = metrics
    st.dataframe(metrics_df.style.format("{:.2f}"))

    # Bar chart for Recall, Precision, F1, AUC
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    metrics_df.plot(kind="bar", ax=ax3)
    ax3.set_title("Model Performance Comparison")
    ax3.set_ylabel("Score")
    ax3.legend(loc="lower right")
    st.pyplot(fig3, width="stretch")

    # ---------------------------------------------------
    # 🏆 Best Model Recommendation
    # ---------------------------------------------------
    best_model = max(results, key=lambda m: results[m]["Recall"])  # pick by recall
    best_metrics = results[best_model]

    st.subheader("🏆 Best Model Recommendation")
    st.success(
        f"**Best Model:** {best_model}\n\n"
        f"- Precision: {best_metrics['Precision']:.2f}\n"
        f"- Recall: {best_metrics['Recall']:.2f}\n"
        f"- F1-score: {best_metrics['F1']:.2f}\n"
        f"- AUC: {best_metrics['AUC']:.2f}\n\n"
        f"👉 Recommended because it achieves the **highest Recall** "
        f"(catches maximum fraud cases)."
    )
