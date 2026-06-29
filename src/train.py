from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.graph_features import add_graph_features
from src.simulate_transactions import simulate_transactions

def train_model():
    df = add_graph_features(simulate_transactions())
    y = df["is_fraud"]
    X = df.drop(columns=["is_fraud", "transaction_id"])

    cat = ["user_id", "merchant_id", "device", "channel"]
    num = [c for c in X.columns if c not in cat]

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat),
    ])

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([("pre", pre), ("clf", clf)])

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipe.fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(yte, p),
        "pr_auc": average_precision_score(yte, p),
        "report": classification_report(yte, (p >= 0.5).astype(int), output_dict=True),
    }
    return pipe, metrics

if __name__ == "__main__":
    model, metrics = train_model()

    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    joblib.dump(model, "models/fraud_model.joblib")
    pd.DataFrame([{
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "fraud_precision": metrics["report"]["1"]["precision"],
        "fraud_recall": metrics["report"]["1"]["recall"],
        "fraud_f1": metrics["report"]["1"]["f1-score"],
    }]).to_csv("reports/metrics.csv", index=False)

    print("Model saved to models/fraud_model.joblib")
    print("Metrics saved to reports/metrics.csv")
