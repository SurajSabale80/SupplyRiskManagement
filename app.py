# ==========================================
# 📊 Supply Risk Management - ML Prediction App (with Graphs)
# ==========================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# ------------------------------------------
# Streamlit App Title and Description
# ------------------------------------------
st.title("📦 Supply Risk Management - Machine Learning App")

st.markdown("""
### 🧠 Predict Supply Chain Risks using Multiple ML Algorithms
Upload your dataset, choose your target and features, and this app will:
- Train **KNN**, **Naive Bayes**, **Logistic Regression**, and **SVM**
- Compare their accuracies with **graphs**
- Display **confusion matrix heatmaps** for each model
- Show the **best performing model**
""")

# ------------------------------------------
# File Upload
# ------------------------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Preview of Uploaded Data")
    st.write(df.head())

    # Column selection
    target_col = st.selectbox("🎯 Select Target Column (what you want to predict):", df.columns)
    feature_cols = st.multiselect("🧩 Select Feature Columns:", [col for col in df.columns if col != target_col])

    # Continue only if features are selected
    if len(feature_cols) > 0 and st.button("🚀 Train Models"):
        X = df[feature_cols]
        y = df[target_col]

        # Encode categorical target
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Encode categorical features
        X = pd.get_dummies(X, drop_first=True)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Define models
        models = {
            "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Support Vector Machine (SVM)": SVC()
        }

        results = {}
        confusion_matrices = {}

        # Train and evaluate each model
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            results[name] = acc
            confusion_matrices[name] = cm

        # Show results table
        st.subheader("📈 Model Accuracy Comparison")
        results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
        st.dataframe(results_df)

        # Bar chart for accuracy comparison
        st.subheader("📊 Accuracy Comparison Chart")
        fig, ax = plt.subplots()
        ax.bar(results_df["Model"], results_df["Accuracy"], color=['skyblue', 'orange', 'green', 'red'])
        ax.set_xlabel("Algorithms")
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Performance Comparison")
        plt.xticks(rotation=15)
        st.pyplot(fig)

        # Confusion Matrix charts
        st.subheader("🧩 Confusion Matrix for Each Algorithm")
        for name, cm in confusion_matrices.items():
            st.write(f"**{name}**")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f"Confusion Matrix - {name}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        # Best model
        best_model = max(results, key=results.get)
        best_acc = results[best_model]
        st.success(f"🏆 Best Model: {best_model} with Accuracy: {best_acc*100:.2f}%")

else:
    st.info("👆 Please upload a CSV file to start.")






