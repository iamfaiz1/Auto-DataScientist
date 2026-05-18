import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

class Visualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def plot_target_distribution(self, target_col, task_type):
        """Visualizes how the target variable is distributed."""
        if task_type == "Classification":
            fig = px.pie(self.df, names=target_col, title=f"Distribution of {target_col}")
        else:
            fig = px.histogram(self.df, x=target_col, title=f"Distribution of {target_col}", marginal="box")
        return fig

    def plot_correlation_heatmap(self):
        """Generates a heatmap for numerical features."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlation Heatmap", color_continuous_scale='RdBu_r')
        return fig

    def plot_feature_importance(self, model_pipeline, target_col):
        """
        Extracts and plots feature importance from the best model.
        Note: This is a simplified version for CPU-friendly execution.
        """
        try:
            # Get the model from the end of the pipeline
            model = model_pipeline.named_steps['regressor_or_classifier']
            
            # Try to get feature importances
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                # Get approximate feature names from the dataset
                features = [col for col in self.df.columns if col != target_col][:len(importances)]
                
                importance_df = pd.DataFrame({'Feature': features, 'Importance': importances[:len(features)]})
                importance_df = importance_df.sort_values(by='Importance', ascending=True)
                
                fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title="Top Predictors (Feature Importance)")
                return fig
        except Exception:
            return None
        return None