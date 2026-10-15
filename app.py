"""Interactive Streamlit dashboard for synthetic employee analytics."""
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from src.analysis import load_and_validate, summarize_data
from src.predictor import load_model, predict_employee
from src.data_generator import save_dataset

st.set_page_config(
    page_title="Employee Performance Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "employees.csv"
MODEL_PATH = ROOT / "models" / "performance_model.joblib"
METRICS_PATH = ROOT / "outputs" / "model_metrics.json"


@st.cache_data
def get_data():
    if not DATA_PATH.exists():
        save_dataset(DATA_PATH, n=500, random_state=42)
    return load_and_validate(DATA_PATH)


@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        return None
    return load_model(MODEL_PATH)


df = get_data()
summary = summarize_data(df)

st.title("📊 Employee Performance Predictor")
st.caption(
    "Synthetic HR analytics demonstration • Educational use only • "
    "Not validated for real employment decisions"
)

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose a section",
        ["Overview", "Explore Data", "Model Evaluation", "Demo Prediction"]
    )
    st.divider()
    st.subheader("Dataset filters")
    departments = st.multiselect(
        "Department",
        sorted(df["department"].unique()),
        default=sorted(df["department"].unique())
    )
    categories = st.multiselect(
        "Performance category",
        ["Low", "Average", "High"],
        default=["Low", "Average", "High"]
    )

filtered = df[
    df["department"].isin(departments)
    & df["performance_category"].isin(categories)
].copy()

if page == "Overview":
    st.subheader("Workforce snapshot")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employee records", f"{len(filtered):,}")
    c2.metric(
        "Average score",
        f"{filtered['performance_score'].mean():.1f}" if len(filtered) else "—"
    )
    c3.metric("Departments", filtered["department"].nunique())
    c4.metric("Missing values", int(filtered.isna().sum().sum()))

    left, right = st.columns(2)
    with left:
        if len(filtered):
            fig = px.histogram(
                filtered, x="performance_score", nbins=20,
                color="performance_category",
                title="Performance Score Distribution",
                labels={"performance_score": "Synthetic score"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least one department and category.")
    with right:
        if len(filtered):
            dept = filtered.groupby("department", as_index=False)["performance_score"].mean()
            fig = px.bar(
                dept, x="department", y="performance_score",
                title="Average Score by Department",
                labels={"performance_score": "Average synthetic score"}
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset preview")
    st.dataframe(filtered.head(20), use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_synthetic_employees.csv",
        mime="text/csv"
    )

elif page == "Explore Data":
    st.subheader("Explore synthetic workforce patterns")
    left, right = st.columns(2)
    with left:
        fig = px.box(
            filtered, x="department", y="performance_score",
            color="department", title="Score Spread by Department"
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.scatter(
            filtered, x="training_hours", y="performance_score",
            color="performance_category", hover_data=["department", "job_role"],
            title="Training Hours vs. Synthetic Performance Score"
        )
        st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(
        filtered, x="experience_years", color="performance_category",
        title="Experience Distribution by Synthetic Category", nbins=18
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "These are simulated relationships. They do not show that training or "
        "experience causes performance changes."
    )

elif page == "Model Evaluation":
    st.subheader("Held-out model evaluation")
    if not METRICS_PATH.exists():
        st.warning("Model metrics are not available yet. Run `python main.py` first.")
    else:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.info(
            f"Selected model: {metrics['best_model']} | "
            f"Test records: {metrics['test_rows']}"
        )
        rows = []
        for name, result in metrics["models"].items():
            rows.append({
                "Model": name,
                "Accuracy": result["accuracy"],
                "Weighted precision": result["weighted_precision"],
                "Weighted recall": result["weighted_recall"],
                "Weighted F1": result["weighted_f1"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        cm_path = ROOT / "outputs" / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion matrix for selected model")
        st.warning(
            "These scores measure performance on a held-out synthetic test set. "
            "They are not evidence of accuracy on real employees."
        )

else:
    st.subheader("Try a hypothetical employee profile")
    st.write("Adjust the values below to demonstrate how the trained model responds.")
    left, right = st.columns(2)
    with left:
        department = st.selectbox(
            "Department", ["Technology", "Sales", "Finance", "Operations", "Human Resources"]
        )
        role = st.selectbox(
            "Job role", ["Associate", "Specialist", "Senior Specialist", "Team Lead", "Manager"]
        )
        experience = st.slider("Experience (years)", 0.0, 30.0, 4.0, 0.5)
        training = st.slider("Training hours", 0.0, 80.0, 30.0, 1.0)
    with right:
        attendance = st.slider("Attendance (%)", 70.0, 100.0, 96.0, 0.5)
        satisfaction = st.slider("Job satisfaction (1–5)", 1, 5, 4)
        previous = st.slider("Previous score", 25.0, 98.0, 75.0, 1.0)

    if st.button("Run demonstration prediction", type="primary"):
        model = get_model()
        if model is None:
            st.error("Model is missing. Run `python main.py` in the project folder first.")
        else:
            profile = {
                "department": department,
                "job_role": role,
                "experience_years": experience,
                "training_hours": training,
                "attendance_pct": attendance,
                "job_satisfaction": satisfaction,
                "previous_performance_score": previous,
            }
            result = predict_employee(model, profile)
            st.metric("Predicted synthetic category", result["predicted_category"])
            if result["model_confidence"] is not None:
                st.metric("Model confidence (maximum class probability)",
                          f"{result['model_confidence']:.1%}")
            st.json(profile)
            st.caption(
                "This is a model demonstration on synthetic data, not an assessment "
                "of a real person. Confidence is not a guarantee of correctness."
            )
