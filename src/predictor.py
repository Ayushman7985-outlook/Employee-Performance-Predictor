"""Train, evaluate, save, and use classification pipelines."""
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_fscore_support
)
from src.analysis import FEATURE_COLUMNS, TARGET_COLUMN

CATEGORICAL_FEATURES = ["department", "job_role"]
NUMERIC_FEATURES = [
    "experience_years", "training_hours", "attendance_pct",
    "job_satisfaction", "previous_performance_score"
]


def build_preprocessor() -> ColumnTransformer:
    """Build transformations that are fitted only on the training partition."""
    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    return ColumnTransformer(transformers=[
        ("categorical", categorical, CATEGORICAL_FEATURES),
        ("numeric", numeric, NUMERIC_FEATURES)
    ])


def train_and_evaluate(df: pd.DataFrame, model_dir: str | Path = "models",
                       output_dir: str | Path = "outputs",
                       random_state: int = 42) -> dict:
    """Compare two baseline classifiers on the same held-out test data."""
    model_dir, output_dir = Path(model_dir), Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, min_samples_leaf=2, class_weight="balanced",
            random_state=random_state, n_jobs=-1
        ),
    }
    results = {}
    fitted = {}

    for name, estimator in candidates.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", estimator)
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average="weighted", zero_division=0
        )
        results[name] = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "weighted_precision": round(float(precision), 4),
            "weighted_recall": round(float(recall), 4),
            "weighted_f1": round(float(f1), 4),
            "classification_report": classification_report(
                y_test, predictions, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=["Low", "Average", "High"]
            ).tolist(),
        }
        fitted[name] = (pipeline, predictions)

    # Choose by weighted F1 rather than accuracy alone.
    best_name = max(results, key=lambda name: results[name]["weighted_f1"])
    best_pipeline, best_predictions = fitted[best_name]
    joblib.dump(best_pipeline, model_dir / "performance_model.joblib")

    summary = {
        "best_model": best_name,
        "test_rows": int(len(y_test)),
        "class_labels": ["Low", "Average", "High"],
        "models": results,
        "synthetic_data_notice": (
            "Metrics are from synthetic data and do not establish real-world validity."
        ),
    }
    (output_dir / "model_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "classification_report.json").write_text(
        json.dumps(results[best_name]["classification_report"], indent=2),
        encoding="utf-8"
    )

    cm = confusion_matrix(y_test, best_predictions, labels=["Low", "Average", "High"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Low", "Average", "High"]
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {best_name}")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    return summary


def load_model(path: str | Path = "models/performance_model.joblib"):
    """Load a previously trained pipeline."""
    return joblib.load(path)


def predict_employee(model, employee: dict) -> dict:
    """Predict a category for one hypothetical employee profile."""
    row = pd.DataFrame([employee])[FEATURE_COLUMNS]
    prediction = str(model.predict(row)[0])
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(max(model.predict_proba(row)[0]))
    return {"predicted_category": prediction, "model_confidence": confidence}
