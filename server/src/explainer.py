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