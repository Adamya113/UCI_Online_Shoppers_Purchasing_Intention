"""Train the five classifiers and save all deployment artifacts."""
from pathlib import Path
import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

def train():
    dataset = fetch_ucirepo(id=468)
    X = dataset.data.features.copy()
    y = dataset.data.targets["Revenue"].astype(int)

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore",
                                                   sparse_output=False))]), cat_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision_Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive_Bayes": GaussianNB(),
        "SVM": SVC(probability=True, random_state=42),
    }
    rows = []
    for name, model in models.items():
        model.fit(X_train_t, y_train)
        pred = model.predict(X_test_t)
        prob = model.predict_proba(X_test_t)[:, 1]
        rows.append({
            "Model": name.replace("_", " "),
            "Accuracy": accuracy_score(y_test, pred),
            "AUC": roc_auc_score(y_test, prob),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "MCC": matthews_corrcoef(y_test, pred),
        })
        joblib.dump(model, MODEL_DIR / f"{name}.pkl")

    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.pkl")
    test_df = X_test.copy()
    test_df["Revenue"] = y_test.values
    test_df.to_csv(ROOT / "test_data.csv", index=False)
    pd.DataFrame(rows).to_csv(ROOT / "model_metrics.csv", index=False)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    print(train().to_string(index=False))
