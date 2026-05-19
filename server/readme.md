<div align="center">

# 🤖 Auto-DataScientist

**An Automated Machine Learning, Data Profiling, and Executive Reporting Assistant**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0%2B-FF4B4B.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0%2B-F7931E.svg)](https://scikit-learn.org/)
[![SQLite](https://img.shields.io/badge/SQLite-3.x-07405E.svg)](https://sqlite.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Transform raw data into actionable business insights, predictive models, and professional PDF reports in minutes—without writing a single line of code.*

---

</div>

## 📖 Project Description

**Auto-DataScientist** is a lightweight, end-to-end Automated Machine Learning (AutoML) platform designed for speed, privacy, and simplicity. It enables users to upload raw datasets, automatically profile data health, optimize memory usage, train machine learning models, and generate professional business reports—all through an intuitive, web-based UI.

### ❓ Problem Statement

Traditional ML workflows require extensive boilerplate code for data cleaning, preprocessing, model selection, and reporting. Additionally:

- **Hosting Constraints:** Processing large datasets often causes Out-Of-Memory (OOM) failures on free-tier or budget servers.
- **Privacy & Cost:** Generating human-like insights typically requires sending sensitive data to paid external LLM APIs (e.g., OpenAI), raising privacy concerns and increasing operational costs.

### 💡 The Solution

This project automates the entire pipeline locally using **rule-based heuristic engines** to generate deep business insights without external API dependencies. It also actively **downcasts memory types** and caps row parsing to ensure stable performance on any hardware footprint.

---

## ✨ Key Features

- 🔍 **Intelligent Data Profiling:** Automatically calculates Health Scores, detects missing values, identifies class imbalances, and flags high-correlation features.
- 💾 **Defensive Memory Optimization:** Automatically downcasts 64-bit data structures to 32-bit types and caps processing at 50,000 rows to prevent server crashes.
- 🎯 **Smart Task Detection:** Analyzes target variables to automatically choose between Regression and Classification workflows.
- 🔧 **Dynamic Preprocessing Pipelines:** Builds leak-proof `scikit-learn` ColumnTransformers with median imputation, standard scaling, and one-hot encoding.
- 🧠 **Heuristic AI Commentator:** Generates human-like analytical insights and business context without paid external API calls.
- 📊 **Embedded Experiment Tracking:** Automatically saves pipeline runs, metrics, and models to a local SQLite database for cross-comparison.
- 📄 **Automated PDF Reporting:** Compiles multi-page business summaries featuring metrics, insights, and actionable next steps via `ReportLab`.
- 📈 **Tiered User Experience:** Toggle between Beginner, Intermediate, and Advanced modes to control metric and hyperparameter visibility.

---

## 🛠️ Tech Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| **Frontend / UI** | Streamlit | Reactive web interface and state management |
| **Data Processing** | Pandas, NumPy | Core dataframe operations and memory optimization |
| **Machine Learning** | Scikit-Learn | Preprocessing pipelines, model training, evaluation |
| **Visualizations** | Plotly Express | Interactive charts, correlation heatmaps |
| **Database** | SQLite3 | Embedded experiment tracking and history storage |
| **Reporting** | ReportLab | Programmatic PDF executive summary generation |

---

## 🏗️ System Architecture

The application uses decoupled modules to ensure clean separation between the UI layer, processing engines, and data storage.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                            │
│  [Upload] → [Explore] → [Target] → [Train] → [Insights/Report]     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                    Core Logic & Services                            │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │  Data Layer  │  │   ML Layer   │  │  Insights & Reporting    │   │
│  ├──────────────┤  ├──────────────┤  ├──────────────────────────┤   │
│  │ DataProfiler │  │ TaskDetector │  │ AICommentator            │   │
│  │ Visualizer   │  │SmartRecmdr   │  │ BusinessInsights         │   │
│  │              │  │ Preprocessor │  │ Explainer                │   │
│  │              │  │ Trainer      │  │ PDFReporter              │   │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                      Storage Layer                                  │
│                 SQLite (experiment_history.db)                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 🔄 Data & Pipeline Flow

```
[Raw CSV]
   │
   ▼
┌─────────────────────────────────────────────────────────┐
│ DataProfiler                                            │
│ • RAM Optimization (downcast float64→float32, etc.)    │
│ • Health Scoring & Statistical Extraction              │
└─────────────────────────────────┬───────────────────────┘
                                  │
   ▼
[Target Variable Selection]
   │
   ▼
┌─────────────────────────────────────────────────────────┐
│ TaskDetector → Classification vs. Regression            │
└─────────────────────────────────┬───────────────────────┘
                                  │
   ▼
┌─────────────────────────────────────────────────────────┐
│ SmartRecommender → Optimal Model Suggestions            │
└─────────────────────────────────┬───────────────────────┘
                                  │
   ▼
┌─────────────────────────────────────────────────────────┐
│ Preprocessing Pipeline                                  │
│ • Numeric: Median Imputation + Standard Scaling        │
│ • Categorical: Constant Imputation + One-Hot Encoding  │
└─────────────────────────────────┬───────────────────────┘
                                  │
   ▼
┌─────────────────────────────────────────────────────────┐
│ Model Training (80/20 Train/Test Split)                 │
└─────────────────────────────────┬───────────────────────┘
                                  │
   ▼
┌─────────────────────────────────────────────────────────┐
│ Evaluation (Accuracy/F1 or R²/MAE)                      │
└─────────────────────────────────┬───────────────────────┘
                                  │
   ▼
[Output] → PDF Report, SQLite Tracking, Business Insights
```

### 📁 Folder Structure

```
auto-data-scientist/
│
├── .gitignore                    # Git exclusions (venv, pycache, .db files)
├── requirements.txt              # Python dependency definitions
├── app.py                        # Main Streamlit application entry point
├── experiment_history.db         # Embedded SQLite database (auto-generated)
│
├── .streamlit/
│   └── config.toml               # Streamlit UI theming configuration
│
├── src/                          # Core Application Source Code
│   ├── __init__.py
│   ├── ai_commentator.py         # Rule-based text insight generation
│   ├── business_insights.py      # Translates metrics to business logic
│   ├── config.py                 # Application settings & user capability tiers
│   ├── dashboard_layout.py       # Reusable Streamlit UI components
│   ├── data_profiler.py          # RAM optimization & dataset health scoring
│   ├── difficulty_estimator.py   # ML problem complexity calculation
│   ├── experiment_tracker.py     # SQLite CRUD operations
│   ├── explainer.py              # Natural language explanations
│   ├── pdf_reporter.py           # ReportLab PDF generation logic
│   ├── preprocess.py             # Scikit-learn pipelines
│   ├── smart_recommender.py      # Zero-shot algorithm recommendations
│   ├── task_detector.py          # Classification vs Regression detection
│   ├── train_eval.py             # Model fitting & scoring orchestration
│   ├── visualizer.py             # Plotly interactive chart generation
│   └── workflow_manager.py       # UI state & step-by-step navigation
│
└── utils/
    ├── __init__.py
    └── prompt_templates.py       # Formatting templates for text explanations
```

---

## 💻 Installation Guide

Follow these steps to set up the project locally.

### 1. Prerequisites

- Python 3.10 or higher
- Git installed on your system

### 2. Clone the Repository

```bash
git clone https://github.com/iamfaiz1/auto-datascientist.git
cd auto-datascientist
```

### 3. Set Up Virtual Environment

It is highly recommended to use an isolated virtual environment.

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

---

## ⚙️ Configuration & Environment

### Zero External API Keys Required!

Unlike many modern AI applications, this project uses a custom-built heuristic engine (`ai_commentator.py`). **No OpenAI, Anthropic, or other external API keys are needed.**

### User Tiers (`src/config.py`)

The application includes built-in UI configurations tailored to different experience levels:

- **Beginner Mode:** Auto-manages preprocessing, auto-selects optimal models, displays comprehensive text explanations.
- **Intermediate Mode:** Exposes detailed metrics, preprocessing steps, and model comparisons.
- **Advanced Mode:** Provides full control over hyperparameters, pipeline inspection, and algorithm selection.

---

## 🚀 Usage Instructions

### Step 1: Upload Dataset
Drag and drop a raw CSV file into the sidebar uploader.

### Step 2: Review Data Health
The DataProfiler displays a health score (0-100), flags correlations, and identifies missing values at a glance.

### Step 3: Select Target Variable
Choose the column you want the AI to predict. The system automatically detects Classification or Regression problems.

### Step 4: View Difficulty & Recommendations
Review the heuristic evaluation of dataset difficulty and view zero-shot model recommendations.

### Step 5: Train Models
Click "Run AI Analysis". The system will:
- Build scikit-learn pipelines
- Evaluate multiple models (Random Forest, Logistic Regression, etc.)
- Plot feature importance and model performance

### Step 6: Generate Report
Review business insights, save the run to SQLite history, and click "Generate PDF Report" to download a formatted executive summary.

---

## 🧠 ML / AI Workflow

### 1. Smart Preprocessing (`src/preprocess.py`)

To prevent data leakage, all operations are chained using `sklearn.compose.ColumnTransformer`:

- **Numerical Features:** `SimpleImputer(strategy='median')` → `StandardScaler()`
- **Categorical Features:** `SimpleImputer(strategy='constant', fill_value='missing')` → `OneHotEncoder(handle_unknown='ignore')`

### 2. Algorithm Roster (`src/train_eval.py`)

**Classification Tasks:**
- Logistic Regression
- Random Forest Classifier

**Regression Tasks:**
- Linear Regression
- Random Forest Regressor

*Future expansion planned for XGBoost, LightGBM, and SVMs.*

### 3. Explainability

Tree-based models trigger feature importance extraction via `visualizer.py`, plotting top drivers influencing the target variable.

---

## 🗄️ Database Design

The application uses an embedded SQLite3 database (`experiment_history.db`) to log pipeline runs with the following schema:

- **Run Metadata:** Timestamp, dataset name, task type
- **Model Details:** Algorithm, hyperparameters, training duration
- **Evaluation Metrics:** Accuracy/F1 (classification) or R²/MAE (regression)
- **Pipeline Configuration:** Preprocessing steps, feature transformations

---

## 🛡️ Security & Performance Considerations

### Data Privacy
All logic, including AI commentary, runs locally via Python heuristics. Your data never leaves your machine.

### OOM Prevention
`DataProfiler.optimize_memory()` actively downcasts `float64` and `int64` datatypes to `float32` and `int32`, cutting RAM consumption by up to 50%.

### Row Capping
Datasets exceeding 50,000 rows are safely sampled down to prevent Streamlit memory buffer crashes.

### Input Validation
Scikit-learn's `OneHotEncoder` is set to `handle_unknown='ignore'` to prevent pipeline crashes from unseen categorical data in test splits.

---

## 🐳 Docker Setup (Optional)

To containerize the application for consistent deployment:

### 1. Create a Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

### 2. Build and Run

```bash
docker build -t auto-datascientist .
docker run -p 8501:8501 auto-datascientist
```

---

## 🔮 Future Scope

- **Hyperparameter Optimization:** Integrate RandomizedSearchCV or Optuna for automated model tuning.
- **Advanced Imbalance Handling:** Add SMOTE integration directly into preprocessing pipelines.
- **Time-Series Forecasting:** Implement detection for datetime indices and rolling-window evaluation splits.
- **Cloud Database Connectors:** Enable ExperimentTracker to connect to remote PostgreSQL/MySQL instances.
- **Model Explainability:** SHAP values integration for advanced feature importance analysis.
- **AutoML Ensembling:** Combine multiple models for improved prediction accuracy.

---

## 🙏 Acknowledgements

- **Streamlit** – For an incredible front-end Python framework
- **Scikit-Learn** – For robust and reliable ML primitives
- **Plotly** – For beautiful, interactive charting
- **ReportLab** – For programmatic PDF generation

---

## 📬 Contact & Links

- **GitHub:** [github.com/iamfaiz1](https://github.com/iamfaiz1)
- **Project Repository:** [auto-datascientist](https://github.com/iamfaiz1/auto-datascientist)

---

## 📄 License

This project is licensed under the **MIT License** – see the LICENSE file for details.

---

<div align="center">

**Made with ❤️ for the data science community**

</div>