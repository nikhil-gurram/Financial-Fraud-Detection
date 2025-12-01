import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Fraud Detection - EDA", layout="wide")

st.title("📊 Exploratory Data Analysis (EDA)")

uploaded_file = st.file_uploader("Upload Dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Show dataset head
    st.subheader("🔍 Sample Data")
    st.dataframe(df.head(10))

    # Fraud distribution
    st.subheader("⚖️ Class Balance (Fraud vs Non-Fraud)")
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.countplot(x="Class", hue="Class", data=df, ax=ax, palette="Set2", legend=False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Not Fraud (0)", "Fraud (1)"])
    st.pyplot(fig, width="stretch")

    fraud_ratio = df["Class"].mean() * 100
    st.metric("Fraudulent Transactions %", f"{fraud_ratio:.2f}%")

    # Summary statistics
    st.subheader("📌 Summary Statistics (Amount)")
    st.write(df.groupby("Class")["Amount"].describe())

    # ---------------- Class Distribution ----------------
    st.subheader("⚖️ Class Distribution: Before vs After Balancing")

    col1, col2, col3 = st.columns(3)

    # Before balancing
    with col1:
        fig1, ax1 = plt.subplots(figsize=(4,3))
        sns.countplot(x="Class", data=df, ax=ax1, palette=["green", "red"])
        ax1.set_title("Before Balancing")
        ax1.set_xticklabels(["Not Fraud (0)", "Fraud (1)"])
        st.pyplot(fig1, width="stretch")

        counts_before = df["Class"].value_counts()
        perc_before = df["Class"].value_counts(normalize=True) * 100
        stats_before = pd.DataFrame({"Count": counts_before, "Percentage": perc_before.round(2)})
        st.markdown("**Class Distribution (Before)**")
        st.dataframe(stats_before)

    # Apply SMOTE for balancing
    X = df.drop(columns=["Class"])
    y = df["Class"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # After balancing
    with col2:
        fig2, ax2 = plt.subplots(figsize=(4,3))
        sns.countplot(x=y_res, ax=ax2, palette=["green", "red"])
        ax2.set_title("After SMOTE Balancing")
        ax2.set_xticklabels(["Not Fraud (0)", "Fraud (1)"])
        st.pyplot(fig2, width="stretch")

        counts_after = pd.Series(y_res).value_counts()
        perc_after = pd.Series(y_res).value_counts(normalize=True) * 100
        stats_after = pd.DataFrame({"Count": counts_after, "Percentage": perc_after.round(2)})
        st.markdown("**Class Distribution (After)**")
        st.dataframe(stats_after)

    with col3:
        cols = st.columns(2)  # create two columns in one container

        stats = df["Class"].value_counts(normalize=True)
        stats_after = counts_after / counts_after.sum()  # normalize after balancing


        fig3, ax3 = plt.subplots(figsize=(4,3))
        ax3.pie(
            stats,
            labels=["Not Fraud", "Fraud"],
            autopct="%.2f%%",
            colors=["green", "red"],
            startangle=90
        )
        ax3.set_title("Imbalance (Pie Chart)")
        st.pyplot(fig3, width="stretch")

        
        fig_compare2, ax_c2 = plt.subplots(figsize=(4,3))
        ax_c2.pie(
            stats_after,
            labels=["Not Fraud", "Fraud"],
            autopct="%.2f%%",
            colors=["green", "red"],
            startangle=90
        )
        ax_c2.set_title("After Balancing (SMOTE)")
        st.pyplot(fig_compare2, width="stretch")



    # Boxplot of amount
    st.subheader("💰 Transaction Amount by Class")
    fig2, ax2 = plt.subplots(figsize=(8,4))
    sns.boxplot(x="Class", y="Amount", hue="Class", data=df, ax=ax2, palette="Set2", legend=False)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Not Fraud", "Fraud"])
    st.pyplot(fig2, width="stretch")

    # Correlation barplot (Top 12)
    st.subheader("📈 Top Correlations with Fraud")
    corr = df.corr(numeric_only=True).abs()["Class"].sort_values(ascending=False).head(12)
    fig3, ax3 = plt.subplots(figsize=(6,3))
    sns.barplot(x=corr.values, y=corr.index, ax=ax3, hue=corr.index, legend=False, palette="viridis")
    ax3.set_title("Features Most Related to Fraud")
    st.pyplot(fig3, width="stretch")

    # Heatmap of selected features
    st.subheader("🔗 Correlation Heatmap (Top Features)")
    top_corr = corr.index
    fig4, ax4 = plt.subplots(figsize=(10,4))
    sns.heatmap(df[top_corr].corr(), annot=True, cmap="coolwarm", ax=ax4)
    st.pyplot(fig4, width="stretch")

    # Time of day analysis (if Time column exists)
    if "Time" in df.columns:
        st.subheader("⏱ Fraud Analysis by Time of Day")

        # Convert time (seconds) → hour of day
        df["Hour"] = (df["Time"] // 3600) % 24  

        # Create two plots side by side
        col1, col2 = st.columns(2)

        # 🔹 Plot 1: Fraud transaction counts by hour
        with col1:
            st.markdown("**Fraud Transaction Counts by Hour**")
            fraud_only = df[df["Class"] == 1]
            fig1, ax1 = plt.subplots(figsize=(6,3))
            sns.countplot(x="Hour", data=fraud_only, color="red", ax=ax1)
            ax1.set_title("Fraud Counts by Hour")
            ax1.set_ylabel("Count")
            plt.xticks(rotation=90)
            st.pyplot(fig1, width="stretch")

        # 🔹 Plot 2: Fraud percentage (fraud rate) by hour
        with col2:
            st.markdown("**Fraud Percentage by Hour**")
            fraud_rate = df.groupby("Hour")["Class"].mean() * 100
            fig2, ax2 = plt.subplots(figsize=(6,3))
            fraud_rate.plot(kind="bar", color="tomato", ax=ax2)
            ax2.set_title("Fraud Rate (%) by Hour")
            ax2.set_ylabel("Fraud %")
            st.pyplot(fig2, width="stretch")
