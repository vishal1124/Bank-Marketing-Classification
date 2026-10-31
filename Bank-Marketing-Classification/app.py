import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay
)

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="Bank Marketing Classification", layout="wide")

st.title("🏦 Bank Marketing Classification")
st.write("Upload the test_data.csv file and evaluate different Machine Learning models.")

# ----------------------------
# Load Models
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

models = {
    "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl")),
    "Decision Tree": joblib.load(os.path.join(MODEL_DIR, "decision_tree.pkl")),
    "KNN": joblib.load(os.path.join(MODEL_DIR, "knn.pkl")),
    "Naive Bayes": joblib.load(os.path.join(MODEL_DIR, "naive_bayes.pkl")),
    "Support Vector Machine": joblib.load(os.path.join(MODEL_DIR, "support_vector_machine.pkl"))
}

scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

# Models that require scaled data
scaled_models = [
    "Logistic Regression",
    "KNN",
    "Support Vector Machine"
]

# ----------------------------
# File Upload
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type="csv"
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.dataframe(df.head())

    if "Target" not in df.columns:
        st.error("Target column not found.")
        st.stop()

    X = df.drop("Target", axis=1)
    y = df["Target"]

    selected_model = st.selectbox(
        "Select Model",
        list(models.keys())
    )

    model = models[selected_model]

    # Scale only when required
    if selected_model in scaled_models:
        X_input = scaler.transform(X)
    else:
        X_input = X

    # Prediction
    y_pred = model.predict(X_input)

    # Probability / Decision score
    try:
        y_score = model.predict_proba(X_input)[:, 1]
        auc = roc_auc_score(y, y_score)
    except:
        auc = "Not Available"

    st.header("Evaluation Metrics")

    st.write("Accuracy :", accuracy_score(y, y_pred))
    st.write("Precision :", precision_score(y, y_pred))
    st.write("Recall :", recall_score(y, y_pred))
    st.write("F1 Score :", f1_score(y, y_pred))
    st.write("MCC :", matthews_corrcoef(y, y_pred))
    st.write("AUC :", auc)

    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots()

    ConfusionMatrixDisplay.from_predictions(
        y,
        y_pred,
        ax=ax
    )

    st.pyplot(fig)