"""
Configuration for user levels and feature availability.
"""

USER_LEVELS = {
    "Beginner": {
        "show_explanations": True,
        "auto_model_selection": True,
        "auto_preprocessing": True,
        "model_choices": ["Random Forest"],  # Auto-select best
        "show_hyperparameters": False,
        "show_preprocessing_details": False,
        "show_advanced_metrics": False,
    },
    "Intermediate": {
        "show_explanations": True,
        "auto_model_selection": False,
        "auto_preprocessing": True,
        "model_choices": ["Logistic Regression", "Random Forest"],
        "show_hyperparameters": False,
        "show_preprocessing_details": True,
        "show_advanced_metrics": True,
    },
    "Advanced": {
        "show_explanations": False,
        "auto_model_selection": False,
        "auto_preprocessing": False,
        "model_choices": ["Logistic Regression", "Random Forest", "SVM", "Gradient Boosting"],
        "show_hyperparameters": True,
        "show_preprocessing_details": True,
        "show_advanced_metrics": True,
    },
}