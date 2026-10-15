"""Run the complete synthetic data and model-training pipeline."""
import json
from src.data_generator import save_dataset
from src.analysis import load_and_validate, summarize_data, save_eda_charts
from src.predictor import train_and_evaluate, load_model, predict_employee


def main():
    df_generated = save_dataset("data/employees.csv", n=500, random_state=42)
    print(f"\n1. Generated {len(df_generated)} synthetic employee records.")
    df = load_and_validate("data/employees.csv")

    summary = summarize_data(df)
    print("\n2. Data summary:")
    print(json.dumps(summary, indent=2))

    chart_paths = save_eda_charts(df, "outputs")
    print("\n3. EDA charts saved:")
    for path in chart_paths:
        print(f"   - {path}")

    metrics = train_and_evaluate(df)
    print("\n4. Held-out test evaluation (synthetic data only):")
    for name, values in metrics["models"].items():
        print(
            f"   {name}: accuracy={values['accuracy']:.3f}, "
            f"weighted F1={values['weighted_f1']:.3f}"
        )
    print(f"   Selected model: {metrics['best_model']}")

    model = load_model()
    example = {
        "department": "Technology",
        "job_role": "Specialist",
        "experience_years": 4.0,
        "training_hours": 30.0,
        "attendance_pct": 96.0,
        "job_satisfaction": 4,
        "previous_performance_score": 75.0,
    }
    print("\n5. Demonstration prediction for a hypothetical profile:")
    print(json.dumps(predict_employee(model, example), indent=2))
    print("\nPipeline complete. Run the dashboard with: streamlit run app.py")


if __name__ == "__main__":
    main()
