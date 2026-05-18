"""
Smart model recommendation based on dataset characteristics.
Avoids unnecessary training by predicting suitable models upfront.
"""

import numpy as np
import pandas as pd

class SmartRecommender:
    """Recommends models based on data properties—no training required."""
    
    @staticmethod
    def recommend_classification_models(df, target_col, profile):
        """Recommends classification models based on dataset characteristics."""
        recommendations = []
        
        n_rows = len(df)
        n_features = len(df.columns) - 1
        health = profile["Health Score"]
        
        target = df[target_col]
        n_classes = target.nunique()
        imbalance_ratio = target.value_counts().max() / target.value_counts().min()
        
        # Logistic Regression
        numeric_features = len(df.select_dtypes(include=[np.number]).columns)
        if numeric_features >= (n_features * 0.5):  # Mostly numeric
            recommendations.append({
                "model": "Logistic Regression",
                "reason": "Dataset is mostly numeric with linear decision boundaries",
                "score": 85,
            })
        
        # Random Forest (general purpose)
        if n_rows > 100 and health > 60:
            recommendations.append({
                "model": "Random Forest",
                "reason": "Handles mixed features and non-linear patterns robustly",
                "score": 90,
            })
        
        # Weighted recommendation if imbalanced
        if imbalance_ratio > 3:
            recommendations.append({
                "model": "Random Forest (with class weights)",
                "reason": "Class imbalance detected; ensemble with weighting recommended",
                "score": 88,
            })
        
        # Sort by score
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:3]  # Top 3

    @staticmethod
    def recommend_regression_models(df, target_col, profile):
        """Recommends regression models based on dataset characteristics."""
        recommendations = []
        
        n_rows = len(df)
        n_features = len(df.columns) - 1
        health = profile["Health Score"]
        
        target = df[target_col]
        skewness = target.skew()
        
        # Linear Regression
        if abs(skewness) < 0.5 and health > 75:
            recommendations.append({
                "model": "Linear Regression",
                "reason": "Target is normally distributed; linear relationships likely",
                "score": 85,
            })
        
        # Random Forest Regressor (general purpose)
        if n_rows > 100 and health > 60:
            recommendations.append({
                "model": "Random Forest Regressor",
                "reason": "Captures non-linear relationships without assuming linearity",
                "score": 88,
            })
        
        # For high-skew data
        if abs(skewness) > 1:
            recommendations.append({
                "model": "Random Forest Regressor",
                "reason": "Highly skewed target; tree-based models handle outliers better",
                "score": 92,
            })
        
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:3]

    @staticmethod
    def recommend_models(df, target_col, task_type, profile):
        """Main recommendation function."""
        if task_type == "Classification":
            return SmartRecommender.recommend_classification_models(df, target_col, profile)
        else:
            return SmartRecommender.recommend_regression_models(df, target_col, profile)

    @staticmethod
    def generate_recommendation_text(recommendations, task_type):
        """Formats recommendations as readable text."""
        if not recommendations:
            return "No specific recommendations available."
        
        top = recommendations[0]
        text = f"**🎯 Recommended:** {top['model']}\n\n"
        text += f"**Why:** {top['reason']}\n\n"
        text += "**Alternative options:**\n"
        for rec in recommendations[1:]:
            text += f"• {rec['model']}: {rec['reason']}\n"
        
        return text