"""Basic tests for the synthetic employee performance project."""
from src.data_generator import generate_employees
from src.analysis import summarize_data, REQUIRED_COLUMNS
from src.predictor import train_and_evaluate, load_model, predict_employee


def test_generated_dataset_has_expected_columns_and_ranges():
    df = generate_employees(n=100, random_state=7)
    assert set(REQUIRED_COLUMNS).issubset(df.columns)
    assert len(df) == 100
    assert df["employee_id"].is_unique
    assert df["performance_score"].between(0, 100).all()
    assert set(df["performance_category"].unique()).issubset({"Low", "Average", "High"})


def test_summary_reports_rows():
    df = generate_employees(n=100, random_state=7)
    summary = summarize_data(df)
    assert summary["rows"] == 100
    assert summary["missing_values"] == 0


def test_training_and_prediction(tmp_path):
    df = generate_employees(n=150, random_state=13)
    model_dir = tmp_path / "models"
    output_dir = tmp_path / "outputs"
    results = train_and_evaluate(df, model_dir=model_dir, output_dir=output_dir)
    assert results["best_model"] in {"Logistic Regression", "Random Forest"}
    assert (model_dir / "performance_model.joblib").exists()
    model = load_model(model_dir / "performance_model.joblib")
    profile = {
        "department": "Technology",
        "job_role": "Specialist",
        "experience_years": 4.0,
        "training_hours": 30.0,
        "attendance_pct": 96.0,
        "job_satisfaction": 4,
        "previous_performance_score": 75.0,
    }
    prediction = predict_employee(model, profile)
    assert prediction["predicted_category"] in {"Low", "Average", "High"}
