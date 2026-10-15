"""Generate a reproducible synthetic employee dataset for educational use."""
from pathlib import Path
import numpy as np
import pandas as pd

DEPARTMENTS = ["Technology", "Sales", "Finance", "Operations", "Human Resources"]
ROLES = ["Associate", "Specialist", "Senior Specialist", "Team Lead", "Manager"]


def generate_employees(n: int = 500, random_state: int = 42) -> pd.DataFrame:
    """Return n fictional employee records with a simulated performance outcome."""
    if n < 30:
        raise ValueError("n must be at least 30 to support a useful train/test split.")

    rng = np.random.default_rng(random_state)
    department = rng.choice(DEPARTMENTS, n)
    role = rng.choice(ROLES, n, p=[0.25, 0.28, 0.22, 0.15, 0.10])
    experience = np.clip(rng.gamma(shape=2.2, scale=3.0, size=n), 0, 30).round(1)
    training_hours = np.clip(rng.normal(24, 12, n), 0, 80).round(1)
    attendance = np.clip(rng.normal(94, 5, n), 70, 100).round(1)
    satisfaction = rng.integers(1, 6, n)
    prior_score = np.clip(rng.normal(70, 12, n), 25, 98).round(1)

    dept_effect = pd.Series(department).map({
        "Technology": 2.0, "Sales": 1.0, "Finance": 1.5,
        "Operations": 0.0, "Human Resources": 0.5
    }).to_numpy()
    role_effect = pd.Series(role).map({
        "Associate": -1.5, "Specialist": 0.0, "Senior Specialist": 1.0,
        "Team Lead": 1.5, "Manager": 1.0
    }).to_numpy()

    noise = rng.normal(0, 8, n)
    score = (
        0.48 * prior_score
        + 0.18 * attendance
        + 0.20 * training_hours
        + 2.2 * satisfaction
        + 0.35 * experience
        + dept_effect + role_effect + noise
    )
    score = np.clip(score, 0, 100).round(1)

    category = pd.cut(
        score,
        bins=[-0.1, 59.9, 79.9, 100],
        labels=["Low", "Average", "High"]
    ).astype(str)

    df = pd.DataFrame({
        "employee_id": [f"EMP{i:04d}" for i in range(1, n + 1)],
        "department": department,
        "job_role": role,
        "experience_years": experience,
        "training_hours": training_hours,
        "attendance_pct": attendance,
        "job_satisfaction": satisfaction,
        "previous_performance_score": prior_score,
        "performance_score": score,
        "performance_category": category,
    })
    return df


def save_dataset(path: str | Path = "data/employees.csv", n: int = 500,
                 random_state: int = 42) -> pd.DataFrame:
    """Generate and save the dataset, creating the destination folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_employees(n=n, random_state=random_state)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    df = save_dataset()
    print(f"Created {len(df)} synthetic employee records at data/employees.csv")
    print(df.head().to_string(index=False))
    print("\nPerformance category counts:")
    print(df["performance_category"].value_counts())
