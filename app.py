import streamlit as st

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("💳 Fraud Detection System")

st.markdown("""
Welcome to the **Fraud Transaction Detection Dashboard**.

📌 Use the left sidebar to navigate:
- **Fraud Detection** → Enter transaction details and check fraud in real-time.
- **Model Evaluation** → Compare machine learning models on test dataset.
""")
