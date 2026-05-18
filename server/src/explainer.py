from utils.prompt_templates import TASK_EXPLANATION, MODEL_SELECTION_EXPLANATION

class AIExplainer:
    @staticmethod
    def explain_task(target_col, n_unique, dtype, task_type):
        """Generates the 'Why this task?' explanation[cite: 37]."""
        return TASK_EXPLANATION.format(
            target_col=target_col, 
            n_unique=n_unique, 
            dtype=dtype, 
            task_type=task_type
        )

    @staticmethod
    def explain_model_selection(best_model_name, metric_name, score):
        """Generates the 'Why this model?' explanation[cite: 37]."""
        return MODEL_SELECTION_EXPLANATION.format(
            best_model=best_model_name,
            metric_name=metric_name,
            score=score
        )

    @staticmethod
    def get_model_reasoning(model_name, task_type):
        """Heuristic-based reasoning for why a model was chosen."""
        reasoning = {
            "Logistic Regression": "It provides a clear baseline and works well when relationships are linear.",
            "Random Forest": "It captures complex, non-linear patterns and is robust to outliers in your data.",
            "Linear Regression": "It is the gold standard for understanding direct linear relationships.",
            "Random Forest Regressor": "It handles non-linear trends better than simple linear models."
        }
        return reasoning.get(model_name, "This model showed the most stable performance.")

    @staticmethod
    def generate_business_insights(best_model_name, score, task_type):
        """Translates metrics into Business Impact."""
        if task_type == "Classification":
            return f"With {score*100:.1f}% accuracy, this model can automate categorization, reducing manual review time."
        return f"The model explains {score*100:.1f}% of the variance, allowing for predictable budgeting."