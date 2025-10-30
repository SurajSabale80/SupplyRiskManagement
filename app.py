import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(page_title="ML Prediction Algorithms", page_icon="🤖", layout="wide")

st.title("🤖 Machine Learning Prediction App")
st.markdown("""
Upload your dataset, choose a prediction algorithm, and train your model right here!
""")

# ------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------
uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Display basic info
    st.write("**Shape of data:**", df.shape)
    st.write("**Missing values:**")
    st.write(df.isnull().sum())

    # ------------------------------------------------
    # SELECT FEATURES AND TARGET
    # ------------------------------------------------
    all_columns = df.columns.tolist()
    target_col = st.selectbox("🎯 Select target column (label)", all_columns)
    feature_cols = st.multiselect("🧩 Select feature columns", [c for c in all_columns if c != target_col])

    if feature_cols and target_col:
        X = df[feature_cols]
        y = df[target_col]

        # ------------------------------------------------
        # SPLIT DATA
        # ------------------------------------------------
        test_size = st.slider("Select test size", 0.1, 0.5, 0.2)
        random_state = st.number_input("Random state", 0, 100, 42)

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # ------------------------------------------------
        # CHOOSE ALGORITHM
        # ------------------------------------------------
        st.sidebar.header("⚙️ Model Settings")
        model_choice = st.sidebar.selectbox(
            "Choose Algorithm",
            ("Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine")
        )

        # Model initialization
        if model_choice == "Logistic Regression":
            model = LogisticRegression()
        elif model_choice == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_choice == "Random Forest":
            model = RandomForestClassifier()
        else:
            model = SVC(probability=True)

        # ------------------------------------------------
        # TRAIN MODEL
        # ------------------------------------------------
        if st.button("🚀 Train Model"):
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            acc = accuracy_score(y_test, y_pred)
            st.success(f"✅ Model trained successfully! Accuracy: **{acc*100:.2f}%**")

            # Confusion Matrix
            st.subheader("📉 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", ax=ax)
            st.pyplot(fig)

            # Classification Report
            st.subheader("📄 Classification Report")
            st.text(classification_report(y_test, y_pred))

            # Save model (optional)
            st.download_button(
                label="💾 Download Trained Model",
                data=pd.Series(model.__class__.__name__).to_csv(index=False),
                file_name="trained_model_name.csv",
                mime="text/csv"
            )

            # ------------------------------------------------
            # PREDICT ON NEW DATA
            # ------------------------------------------------
            st.subheader("🔮 Try Prediction on Custom Input")

            input_data = {}
            for feature in feature_cols:
                val = st.number_input(f"Enter value for {feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
                input_data[feature] = val

            if st.button("Predict"):
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)[0]
                st.success(f"🎯 Predicted value: **{prediction}**")

else:
    st.info("👈 Please upload a dataset to start.")

# ------------------------------------------------
# FOOTER
# ------------------------------------------------
st.markdown("---")
st.caption("Developed by **Your Name** | © 2025 | Machine Learning App using Streamlit")

