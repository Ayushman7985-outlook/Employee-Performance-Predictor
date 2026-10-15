"""Data validation, summaries, and exploratory analysis charts."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FEATURE_COLUMNS = [
    "department", "job_role", "experience_years", "training_hours",
    "attendance_pct", "job_satisfaction", "previous_performance_score"
]
TARGET_COLUMN = "performance_category"
REQUIRED_COLUMNS = FEATURE_COLUMNS + ["performance_score", TARGET_COLUMN]


def load_and_validate(path: str | Path = "data/employees.csv") -> pd.DataFrame:
    """Load the CSV and fail clearly when required columns or ranges are invalid."""
    df = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Dataset contains no rows.")
    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Required dataset fields contain missing values.")
    if not df["attendance_pct"].between(0, 100).all():
        raise ValueError("attendance_pct must be between 0 and 100.")
    if not df["performance_score"].between(0, 100).all():
        raise ValueError("performance_score must be between 0 and 100.")
    if not df["job_satisfaction"].between(1, 5).all():
        raise ValueError("job_satisfaction must be between 1 and 5.")
    return df


def summarize_data(df: pd.DataFrame) -> dict:
    """Return compact aggregate summaries for terminal output and the dashboard."""
    return {
        "rows": int(len(df)),
        "departments": int(df["department"].nunique()),
        "average_score": round(float(df["performance_score"].mean()), 2),
        "median_score": round(float(df["performance_score"].median()), 2),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "category_counts": df["performance_category"].value_counts().to_dict(),
        "department_scores": (
            df.groupby("department", observed=False)["performance_score"]
            .mean().round(2).sort_values(ascending=False).to_dict()
        ),
    }


def save_eda_charts(df: pd.DataFrame, output_dir: str | Path = "outputs") -> list[Path]:
    """Save charts to disk for project evidence and the README."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="notebook", style="whitegrid")
    paths = []

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="performance_score", bins=20, kde=True, color="steelblue", ax=ax)
    ax.set_title("Synthetic Employee Performance Score Distribution")
    ax.set_xlabel("Performance score (0–100)")
    fig.tight_layout()
    p = output_dir / "performance_score_distribution.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(9, 5))
    order = df.groupby("department")["performance_score"].mean().sort_values().index
    sns.barplot(data=df, x="department", y="performance_score", order=order, color="steelblue", ax=ax)
    ax.set_title("Average Synthetic Performance Score by Department")
    ax.set_xlabel("Department")
    ax.set_ylabel("Average score")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    p = output_dir / "department_average_score.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(data=df, x="performance_category",
                  order=["Low", "Average", "High"], color="steelblue", ax=ax)
    ax.set_title("Performance Category Counts")
    ax.set_xlabel("Synthetic category")
    ax.set_ylabel("Number of records")
    fig.tight_layout()
    p = output_dir / "performance_category_counts.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    return paths
