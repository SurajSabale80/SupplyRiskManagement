import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(page_title="ML Prediction App", page_icon="🤖", layout="wide")

st.title("🤖 Machine Learning App — KNN | Naive Bayes | Logistic Regression | SVM")
st.markdown("""
Upload your dataset, select features, choose an algorithm, and let the app train and predict automatically!
""")

# ------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------
uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.write("**Shape:**", df.shape)
    st.write("**Missing Values:**")
    st.write(df.isnull().sum())

    # ------------------------------------------------
    # SELECT FEATURES AND TARGET
    # ------------------------------------------------
    all_columns = df.columns.tolist()
    target_col = st.selectbox("🎯 Select Target Column", all_columns)
    feature_cols = st.multiselect("🧩 Select Feature Columns", [c for c in all_columns if c != target_col])

    if feature_cols and target_col:
        X = df[feature_cols]
        y = df[target_col]

        # ------------------------------------------------
        # TRAIN TEST SPLIT
        # ------------------------------------------------
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
        random_state = st.number_input("Random State", 0, 100, 42)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # ------------------------------------------------
        # FEATURE SCALING
        # ------------------------------------------------
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        # ------------------------------------------------
        # CHOOSE MODEL
        # ------------------------------------------------
        st.sidebar.header("⚙️ Choose Algorithm")
        model_choice = st.sidebar.selectbox(
            "Select Algorithm",
            ("K-Nearest Neighbors (KNN)", "Naive Bayes", "Logistic Regression", "Support Vector Machine (SVM)")
        )

        # Initialize model
        if model_choice == "K-Nearest Neighbors (KNN)":
            n_neighbors = st.sidebar.slider("Number of Neighbors (K)", 1, 20, 5)
            model = KNeighborsClassifier(n_neighbors=n_neighbors)

        elif model_choice == "Naive Bayes":
            model = GaussianNB()

        elif model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)

        elif model_choice == "Support Vector Machine (SVM)":
            kernel = st.sidebar.selectbox("Kernel Type", ("linear", "rbf", "poly", "sigmoid"))
            model = SVC(kernel=kernel, probability=True)

        # ------------------------------------------------
        # TRAIN MODEL
        # ------------------------------------------------
        if st.button("🚀 Train Model"):
            model.fit(x_train_scaled, y_train)
            y_pred = model.predict(x_test_scaled)
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

            # ------------------------------------------------
            # CUSTOM PREDICTION
            # ------------------------------------------------
            st.subheader("🔮 Make a Custom Prediction")
            input_data = {}
            for feature in feature_cols:
                val = st.number_input(f"Enter value for {feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
                input_data[feature] = val

            if st.button("Predict"):
                input_df = pd.DataFrame([input_data])
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)[0]
                st.success(f"🎯 Predicted Output: **{prediction}**")

else:
    st.info("👈 Please upload a dataset to begin.")

# ------------------------------------------------
# FOOTER
# ------------------------------------------------
st.markdown("---")
st.caption("Developed by **Your Name** | © 2025 | Machine Learning App using Streamlit")


