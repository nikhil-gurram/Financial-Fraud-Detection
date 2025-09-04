import streamlit as st
import joblib
import numpy as np

st.title("🔎 Real-Time Fraud Detection")

st.markdown("""
### 📥 Instructions
To check whether a transaction is **fraudulent or legitimate**, please input:
- **Transaction Amount** (e.g., 150.75)
- **Feature V1 – V5** (numerical values representing anonymized credit card features)

👉 These features come from the credit card dataset where original sensitive details are anonymized into `V1, V2, … V28`.
""")

# Sidebar - Model Selection
models = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}
st.sidebar.header("⚙️ Choose Models")
selected_models = st.sidebar.multiselect("Select models", list(models.keys()), default=list(models.keys()))

# Inputs
amount = st.number_input("💰 Transaction Amount", min_value=0.0, step=0.01)

input_features = []
for i in range(1, 6):  # Example: 5 anonymized features
    val = st.number_input(f"Feature V{i}", value=0.0, step=0.1)
    input_features.append(val)

transaction = np.array([input_features + [amount]])

if st.button("🔎 Check Fraud"):
    cols = st.columns(len(selected_models))
    for idx, model_name in enumerate(selected_models):
        model = joblib.load(f"models/{models[model_name]}")
        prediction = model.predict(transaction)[0]
        proba = model.predict_proba(transaction)[0][1]

        with cols[idx]:
            st.markdown(f"### {model_name}")
            if prediction == 1:
                st.error(f"🚨 Fraudulent (Probability={proba:.2f})")
            else:
                st.success(f"✅ Legitimate (Probability={proba:.2f})")
