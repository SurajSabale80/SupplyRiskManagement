import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(page_title="Best ML Algorithm Finder", page_icon="🤖", layout="wide")

st.title("🤖 Machine Learning Algorithm Comparator")
st.markdown("""
Upload your dataset, and this app will **train and evaluate 4 algorithms**:
- 🧠 Logistic Regression  
- 🤝 K-Nearest Neighbors (KNN)  
- 📈 Support Vector Machine (SVM)  
- 🧮 Naive Bayes  

Then, it will automatically show which algorithm performs **best**!
""")

# ------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------
uploaded_file = st.file_uploader("📁 Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
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
        # SPLIT & SCALE
        # ------------------------------------------------
        test_size = st.slider("Test Size (portion for testing)", 0.1, 0.5, 0.2)
        random_state = st.number_input("Random State", 0, 100, 42)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        # ------------------------------------------------
        # MODELS
        # ------------------------------------------------
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "SVM": SVC(kernel="rbf"),
            "Naive Bayes": GaussianNB()
        }

        results = {}
        st.subheader("🚀 Training Models...")

        for name, model in models.items():
            model.fit(x_train_scaled, y_train)
            y_pred = model.predict(x_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            results[name] = acc

            st.write(f"✅ **{name} Accuracy:** {acc*100:.2f}%")

        # ------------------------------------------------
        # SHOW BEST MODEL
        # ------------------------------------------------
        best_model = max(results, key=results.get)
        st.success(f"🏆 Best Performing Algorithm: **{best_model}** with Accuracy: **{results[best_model]*100:.2f}%**")

        # ------------------------------------------------
        # PLOT COMPARISON
        # ------------------------------------------------
        st.subheader("📊 Algorithm Accuracy Comparison")
        fig, ax = plt.subplots()
        sns.barplot(x=list(results.keys()), y=list(results.values()), palette="coolwarm", ax=ax)
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        for i, v in enumerate(list(results.values())):
            plt.text(i, v + 0.01, f"{v*100:.2f}%", ha='center', fontweight='bold')
        st.pyplot(fig)

        # ------------------------------------------------
        # CLASSIFICATION REPORT & CONFUSION MATRIX FOR BEST MODEL
        # ------------------------------------------------
        st.subheader(f"📄 Detailed Report for {best_model}")

        best_model_instance = models[best_model]
        y_pred_best = best_model_instance.predict(x_test_scaled)

        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred_best))

        cm = confusion_matrix(y_test, y_pred_best)
        st.subheader("📉 Confusion Matrix")
        fig2, ax2 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
        st.pyplot(fig2)

        # ------------------------------------------------
        # CUSTOM PREDICTION USING BEST MODEL
        # ------------------------------------------------
        st.subheader(f"🔮 Predict Using the Best Model ({best_model})")
        input_data = {}
        for feature in feature_cols:
            val = st.number_input(
                f"Enter value for {feature}",
                float(df[feature].min()),
                float(df[feature].max()),
                float(df[feature].mean())
            )
            input_data[feature] = val

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            prediction = best_model_instance.predict(input_scaled)[0]
            st.success(f"🎯 Predicted Output: **{prediction}**")

else:
    st.info("👈 Please upload a dataset to start.")

# ------------------------------------------------
# FOOTER
# ------------------------------------------------
st.markdown("---")
st.caption("Developed by **Your Name** | © 2025 | ML Auto Algorithm Comparator using Streamlit")



