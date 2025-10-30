import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Streamlit App Title
st.title("📊 Supply Risk Management - ML Prediction App")

st.write("""
Upload your dataset, select features and target,
and the app will train multiple ML algorithms (KNN, NB, LR, SVM)
to find the **best-performing model** automatically.
""")

# Upload Dataset
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.write(df.head())

    # Select target and




