# 📊 Employee Performance Predictor

An end-to-end Data Science and Machine Learning project that analyzes employee performance data, identifies performance patterns, evaluates classification models, and predicts employee performance categories through an interactive Streamlit dashboard.

## 📌 Project Overview

The Employee Performance Predictor demonstrates a complete machine learning workflow, from synthetic data generation and exploratory data analysis to model training, evaluation, prediction, and visualization.

The project uses a dataset of **500 synthetic employee records** containing employee-related attributes and performance scores. It analyzes the data, visualizes important patterns, and trains machine learning models to classify employee performance into categories.

This project is developed for educational and analytical demonstration purposes. It does not use real employee records or establish a validated method for making employment decisions.

## ✨ Key Features

- 📁 **Synthetic Dataset Generation:** Generates 500 employee records for analysis and experimentation.
- 🧹 **Data Cleaning:** Checks data quality, missing values, and duplicate records.
- 📊 **Exploratory Data Analysis:** Examines performance scores, department-level patterns, and performance category distributions.
- 📈 **Data Visualization:** Produces charts for performance score distribution, average department scores, and category counts.
- 🤖 **Machine Learning:** Trains and evaluates Logistic Regression and Random Forest classification models.
- 🎯 **Performance Prediction:** Predicts a performance category from the supplied employee attributes.
- 🖥️ **Interactive Dashboard:** Uses Streamlit to present analytics and model predictions in an accessible interface.
- 🧪 **Automated Testing:** Includes project tests to check important functionality.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning, preprocessing, and evaluation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Streamlit | Interactive web dashboard |
| Joblib | Saving and loading trained models |
| Pytest | Automated testing |
| Git and GitHub | Version control and project hosting |

## 📂 Project Structure

```text
Employee-Performance-Predictor/
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── expenses.csv / employee dataset files
│
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── analysis.py
│   ├── preprocessing.py
│   ├── model_training.py
│   └── prediction.py
│
├── models/
│   └── Saved model artifacts
│
├── outputs/
│   ├── performance_score_distribution.png
│   ├── department_average_score.png
│   └── performance_category_counts.png
│
└── tests/
    └── test_project.py
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```powershell
git clone https://github.com/Ayushman7985-outlook/mployee-Performance-Predictor.git
cd mployee-Performance-Predictor
```

### 2. Create a Virtual Environment

```powershell
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## ▶️ Running the Project

### Run the Complete Data Science Pipeline

```powershell
python main.py
```

This executes the project pipeline, including data generation, data analysis, chart creation, model training, model evaluation, and a demonstration prediction.

Generated visualizations are saved in the `outputs/` directory.

### Launch the Interactive Dashboard

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

The dashboard provides an interactive interface for exploring employee performance analytics and running model-based predictions.

## 📊 Data Analysis and Visualizations

The project generates the following visualizations:

1. **Performance Score Distribution:** Shows how employee performance scores are distributed.
2. **Department Average Performance:** Compares average performance scores across departments.
3. **Performance Category Distribution:** Displays the number of records in each performance category.

These charts help illustrate the structure of the synthetic dataset and the relationships represented in the analysis.

## 🤖 Machine Learning Models

Two classification algorithms are evaluated:

### Logistic Regression
A baseline classification model used to learn relationships between input features and performance categories.

### Random Forest Classifier
An ensemble learning algorithm that combines multiple decision trees to classify employee performance.

### Evaluation Metrics

The models are evaluated using classification accuracy and weighted F1-score.

| Model | Accuracy | Weighted F1-score |
|---|---:|---:|
| Logistic Regression | 52.8% | 0.512 |
| Random Forest | 58.4% | 0.587 |

**Selected model:** Random Forest Classifier, based on the evaluation results from the current synthetic dataset.

These scores describe performance on the project's held-out test data. They are not evidence of real-world employee assessment accuracy, and results may vary if the generated dataset or training configuration changes.

## 🧪 Running Tests

Execute the automated tests from the project root:

```powershell
python -m pytest -q
```

The tests cover core project functionality, including data processing and machine learning workflow components.

## 📦 Dependencies

Project dependencies are listed in `requirements.txt`. The main libraries include:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `streamlit`
- `joblib`
- `pytest`

Install the dependencies with:

```powershell
pip install -r requirements.txt
```

## 🔄 Workflow

```text
Synthetic Data Generation
          ↓
Data Cleaning and Validation
          ↓
Exploratory Data Analysis
          ↓
Feature Preparation
          ↓
Model Training
          ↓
Model Evaluation
          ↓
Performance Category Prediction
          ↓
Streamlit Dashboard and Visualizations
```

## 🎯 Learning Outcomes

This project demonstrates practical experience with:

- Structuring a Python data science project into reusable modules.
- Generating and validating datasets with Pandas and NumPy.
- Performing exploratory data analysis and creating visualizations.
- Preparing data for machine learning.
- Training and comparing classification algorithms.
- Evaluating models using standard classification metrics.
- Building an interactive application with Streamlit.
- Running automated tests with Pytest.
- Managing project files and version control with Git and GitHub.

## ⚠️ Data and Responsible Use

All employee records are synthetic and intended for learning and demonstration. The predictions are experimental outputs from models trained on generated data. They should not be used to rank, hire, promote, discipline, or terminate real employees.

## 👨‍💻 Author

**Ayushman**

GitHub: [Ayushman7985-outlook](https://github.com/Ayushman7985-outlook)

## 📄 License

No license has been specified for this repository.
