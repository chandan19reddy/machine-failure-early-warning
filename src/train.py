"""Train and evaluate two failure classifiers on the AI4I dataset."""
from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ai4i2020.csv"
FEATURES = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
            "Torque [Nm]", "Tool wear [min]"]
TARGET = "Machine failure"

def load_data():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}. See README for download instructions.")
    df = pd.read_csv(DATA)
    return df

def main():
    df = load_data()
    missing = df[FEATURES + [TARGET]].isna().sum().to_dict()
    counts = df[TARGET].value_counts().sort_index().to_dict()
    print(f"Rows/columns: {df.shape}; missing: {missing}; target counts: {counts}")
    X, y = df[FEATURES], df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    prep_lr = ColumnTransformer([("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), FEATURES)])
    prep_rf = ColumnTransformer([("numeric", SimpleImputer(strategy="median"), FEATURES)])
    models = {
        "Logistic Regression": Pipeline([("preprocess", prep_lr),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))]),
        "Random Forest": Pipeline([("preprocess", prep_rf),
            ("model", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                               random_state=42, n_jobs=-1, min_samples_leaf=2))]),
    }
    results = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        pred, prob = pipe.predict(X_test), pipe.predict_proba(X_test)[:, 1]
        results[name] = {"accuracy": accuracy_score(y_test, pred),
                         "precision": precision_score(y_test, pred, zero_division=0),
                         "recall": recall_score(y_test, pred, zero_division=0),
                         "f1": f1_score(y_test, pred, zero_division=0),
                         "roc_auc": roc_auc_score(y_test, prob),
                         "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
                         "classification_report": classification_report(y_test, pred, zero_division=0, output_dict=True)}
        print(f"\n{name}: {json.dumps({k: round(v, 4) for k,v in results[name].items() if isinstance(v, float)})}")
        print("Confusion matrix [[TN, FP], [FN, TP]]:", results[name]["confusion_matrix"])
    # Recall is the primary selection metric: missed failures can be costly.
    selected = max(models, key=lambda name: (results[name]["recall"], results[name]["f1"]))
    chosen = models[selected]
    artifact = {"pipeline": chosen, "features": FEATURES, "target": TARGET,
                "model_name": selected, "metrics": results[selected],
                "all_metrics": results, "dataset_note": "AI4I 2020 is itself a synthetic benchmark dataset."}
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(artifact, ROOT / "models" / "failure_pipeline.joblib")
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / "metrics.json").write_text(json.dumps({
        "model_name": selected, "metrics": results,
        "dataset_note": artifact["dataset_note"],
        "features": FEATURES,
    }, indent=2, default=float))
    print(f"\nSelected by test recall then F1: {selected}; saved models/failure_pipeline.joblib")

if __name__ == "__main__":
    main()
