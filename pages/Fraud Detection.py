import streamlit as st
import joblib
import numpy as np
import tensorflow as tf

st.title("🔎 Real-Time Fraud Detection")

st.markdown("""
Enter all **30 features** of the transaction:

- Time  
- V1–V28  
- Amount  

Select if you want models trained **with SMOTE** or **without SMOTE**.
""")

# ----------------------------
# LOAD SCALER
# ----------------------------
scaler = joblib.load("models/scaler.pkl")

# ----------------------------
# MODEL CHOICE
# ----------------------------
model_type = st.sidebar.radio(
    "Choose Model Training Type:",
    ["ML (With SMOTE)", "ML (Without SMOTE)", "Deep Learning Ensemble"],
)

ml_paths = {
    "With SMOTE": "models/smote",
    "Without SMOTE": "models/nosmote"
}

dl_path = "models/dl/dl_ensemble.pkl"

# ----------------------------
# INPUT FEATURES (30)
# ----------------------------
feature_names = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]

st.subheader("📊 Enter Transaction Features")
col1, col2 = st.columns(2)

values = []
for idx, f in enumerate(feature_names):
    with (col1 if idx % 2 == 0 else col2):
        v = st.number_input(f, value=0.0)
        values.append(v)

transaction_raw = np.array([values])
transaction_scaled = scaler.transform(transaction_raw)

# ----------------------------
# PREDICT
# ----------------------------
if st.button("🔎 Check Fraud"):

    if model_type != "Deep Learning Ensemble":

        folder = "With SMOTE" if model_type == "ML (With SMOTE)" else "Without SMOTE"
        base = ml_paths[folder]

        st.write(f"📌 Using ML model from: `{base}`")

        models = ["logistic_regression", "random_forest", "xgboost"]

        for m in models:
            model = joblib.load(f"{base}/{m}.pkl")
            pred = model.predict(transaction_scaled)[0]
            proba = model.predict_proba(transaction_scaled)[0][1]

            st.subheader(m.upper())
            st.write("Fraud!" if pred==1 else "Legitimate")
            st.write("Probability:", proba)

    else:
        ensemble = joblib.load(dl_path)
        pred = ensemble.predict(transaction_scaled)[0][0]
        proba = ensemble.predict_proba(transaction_scaled)[0][0]

        st.subheader("Deep Learning Ensemble")
        st.write("Fraud!" if pred==1 else "Legitimate")
        st.write("Probability:", proba)
