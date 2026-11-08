from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_FILES = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Decision Tree": "Decision_Tree.pkl",
    "KNN": "KNN.pkl",
    "Naive Bayes": "Naive_Bayes.pkl",
    "Support Vector Machine": "SVM.pkl",
}
TARGET = "Revenue"

st.set_page_config(page_title="Online Shoppers Purchase Predictor",
                   page_icon="🛍️", layout="wide")
st.title("🛍️ Online Shoppers Purchasing Intention")
st.caption("Evaluate five saved classification models on an uploaded labelled test CSV.")

@st.cache_resource
def load_artifacts():
    """Load already-trained models and fitted preprocessing pipeline. Never fit here."""
    missing = [str(MODEL_DIR / f) for f in list(MODEL_FILES.values()) + ["preprocessor.pkl"]
               if not (MODEL_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Required saved artifacts are missing: " + ", ".join(missing) +
            ". Run `python train_models.py` before deployment and commit the generated "
            "model/*.pkl files to GitHub."
        )
    models = {name: joblib.load(MODEL_DIR / filename)
              for name, filename in MODEL_FILES.items()}
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
    return models, preprocessor

try:
    models, preprocessor = load_artifacts()
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.sidebar.header("Evaluation settings")
selected_model = st.sidebar.selectbox("Model for detailed report", list(MODEL_FILES))
st.sidebar.write("All five saved models are evaluated on the same uploaded test rows.")

st.subheader("1. Upload labelled test data")
st.write(
    "Upload your test dataset. The CSV must contain every "
    "original input feature plus the target column `Revenue`."
)
uploaded = st.file_uploader("Choose test_data.csv", type=["csv"])
if uploaded is None:
    default_path = ROOT / "test_data.csv"
    if default_path.exists():
        st.info("No file uploaded. The app will use the repository's test_data.csv.")
        test_df = pd.read_csv(default_path)
    else:
        st.warning("Upload a labelled test CSV to begin.")
        st.stop()
else:
    try:
        test_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.stop()

if test_df.empty:
    st.error("The uploaded CSV has no rows.")
    st.stop()
if TARGET not in test_df.columns:
    st.error(f"Target column `{TARGET}` is required in the uploaded CSV.")
    st.stop()

y_test = test_df[TARGET]
if y_test.isna().any():
    st.error(f"`{TARGET}` contains missing values. Please clean the test CSV.")
    st.stop()
# Accept boolean or 0/1 values only; map booleans to integer labels.
if y_test.dtype == bool:
    y_test = y_test.astype(int)
else:
    try:
        y_test = pd.to_numeric(y_test, errors="raise").astype(int)
    except Exception:
        st.error("Revenue must contain binary values (0/1 or True/False).")
        st.stop()
if not set(y_test.unique()).issubset({0, 1}) or len(y_test.unique()) < 2:
    st.error("Revenue must contain both binary classes: 0 and 1.")
    st.stop()

X_test = test_df.drop(columns=[TARGET])
# Derive the expected raw input columns from the fitted transformer.
expected_cols = list(preprocessor.feature_names_in_)
missing_cols = [c for c in expected_cols if c not in X_test.columns]
extra_cols = [c for c in X_test.columns if c not in expected_cols]
if missing_cols:
    st.error(f"Missing required input feature columns: {missing_cols}")
    st.stop()
if extra_cols:
    st.warning(f"Extra columns will be ignored: {extra_cols}")
X_test = X_test[expected_cols]

st.success(f"Loaded {len(test_df):,} labelled test rows and {len(expected_cols)} input features.")
with st.expander("Preview uploaded test data"):
    st.dataframe(test_df.head(20), use_container_width=True)

# Only transform with the saved fitted preprocessor. No fitting or retraining occurs here.
try:
    X_transformed = preprocessor.transform(X_test)
except Exception as exc:
    st.error(f"Preprocessing failed. Check feature types and values. Details: {exc}")
    st.stop()

st.subheader("2. Evaluation results for all five models")
results = {}
predictions = {}
probabilities = {}
for name, classifier in models.items():
    try:
        y_pred = classifier.predict(X_transformed)
        y_prob = classifier.predict_proba(X_transformed)[:, 1]
        predictions[name] = y_pred
        probabilities[name] = y_prob
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_prob),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y_test, y_pred),
        }
    except Exception as exc:
        st.error(f"Evaluation failed for {name}: {exc}")
        st.stop()

results_df = pd.DataFrame(results).T
results_df.index.name = "ML Model Name"
st.dataframe(results_df.style.format("{:.4f}"), use_container_width=True)

metric_long = results_df.reset_index().melt(
    id_vars="ML Model Name", var_name="Metric", value_name="Score"
)
fig = px.bar(metric_long, x="Metric", y="Score", color="ML Model Name",
             barmode="group", range_y=[-1, 1],
             title="Comparison of all five classifiers")
st.plotly_chart(fig, use_container_width=True)

st.subheader("3. Confusion matrix and classification report")
st.write(f"Detailed results for **{selected_model}**")
chosen_pred = predictions[selected_model]
cm = confusion_matrix(y_test, chosen_pred, labels=[0, 1])
cm_fig = px.imshow(
    cm, text_auto=True, aspect="auto",
    x=["Predicted 0 (No purchase)", "Predicted 1 (Purchase)"],
    y=["Actual 0 (No purchase)", "Actual 1 (Purchase)"],
    labels=dict(x="Prediction", y="Actual", color="Count"),
    title=f"{selected_model} – Confusion Matrix"
)
st.plotly_chart(cm_fig, use_container_width=True)
report = classification_report(
    y_test, chosen_pred, labels=[0, 1],
    target_names=["No purchase (0)", "Purchase (1)"],
    output_dict=True, zero_division=0
)
st.dataframe(pd.DataFrame(report).T.style.format("{:.4f}"),
             use_container_width=True)

st.subheader("4. Download evaluation results")
st.download_button(
    "Download metrics for all five models (CSV)",
    results_df.reset_index().to_csv(index=False).encode("utf-8"),
    "uploaded_test_metrics.csv", "text/csv"
)
