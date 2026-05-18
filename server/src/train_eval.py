import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.preprocess import DataPreprocessor

class Trainer:
    def __init__(self, df: pd.DataFrame, target_col: str, task_type: str):
        self.df = df
        self.target_col = target_col
        self.task_type = task_type
        self.results = []

    def run_pipeline(self):
        # 1. Initialize Preprocessing
        preprocessor_obj = DataPreprocessor(self.df, self.target_col)
        preprocessor, X, y = preprocessor_obj.build_pipeline()

        # 2. Split Data (80% Train, 20% Test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Define Lightweight Models [cite: 3, 23]
        if self.task_type == "Classification":
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=5)
            }
            metric_name = "Accuracy"
            score_fn = accuracy_score
        else:
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=5)
            }
            metric_name = "R2 Score"
            score_fn = r2_score

        # 4. Train and Evaluate
        for name, model in models.items():
            pipe = Pipeline(steps=[("preprocessor", preprocessor), ("regressor_or_classifier", model)])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            score = score_fn(y_test, preds)
            
            self.results.append({
                "model_name": name,
                "score": round(score, 4),
                "metric_name": metric_name,
                "pipeline": pipe
            })

        # Sort results to find the best model
        self.results = sorted(self.results, key=lambda x: x["score"], reverse=True)
        return self.results