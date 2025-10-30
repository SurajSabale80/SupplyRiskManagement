import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# ---------------------------------------------
# Streamlit Page Config
# ---------------------------------------------
st.set_page_config(
    page_title="Supply Risk Management",
    layout="wide",
    page_icon="📊"
)

# ---------------------------------------------
# Title & Description
# ---------------------------------------------
st.title("📊 Supply Risk Management System")
st.markdown("""
This app helps you **analyze supplier performance** and assess potential **supply risks**.
Upload a dataset, explore visualizations, and (optionally) predict risk levels using a trained ML model.
""")

# ---------------------------------------------
# Sidebar - File Upload
# ---------------------------------------------
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your supplier data (CSV format)", type=["csv"])

# Optional model load (if you have one)
# Example: model = joblib.load("model.pkl")
model = None

# ---------------------------------------------
# Main App Logic
# ---------------------------------------------
if uploaded_file is not None:
    # Read CSV file
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # Data Summary
    st.subheader("📊 Dataset Summary")
    st.write(df.describe())

    # Correlation heatmap
    st.subheader("📈 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    # Missing values
    st.subheader("🚨 Missing Value Check")
    missing = df.isnull().sum()
    st.write(missing[missing > 0] if missing.sum() > 0 else "✅ No missing values detected!")

    # Optional: Model Prediction
    st.subheader("🔮 Risk Prediction (Optional)")
    if model is not None:
        if st.button("Predict Risk Levels"):
            try:
                predictions = model.predict(df)
                df["Predicted_Risk"] = predictions
                st.success("✅ Risk Prediction Completed!")
                st.dataframe(df)
            except Exception as e:
                st.error(f"Prediction Error: {e}")
    else:
        st.info("💡 To enable predictions, train and save a model (e.g., model.pkl) in your project folder.")

else:
    st.info("👈 Upload a CSV file from the sidebar to begin your analysis.")

# ---------------------------------------------
# Footer
# ---------------------------------------------
st.markdown("---")
st.caption("Developed by **Your Name** | Data Science Project | © 2025 Supply Risk Management")
