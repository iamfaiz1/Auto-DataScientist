"""
Heuristic-based AI commentary generator for datasets and models.
Produces human-like insights without LLM calls.
"""

import numpy as np
import pandas as pd

class AICommentator:
    @staticmethod
    def analyze_dataset_quality(df, profile):
        """Generates AI commentary on dataset quality."""
        health = profile["Health Score"]
        missing_pct = profile["Missing Pct"]
        issues = profile["Issues"]
        
        commentaries = []
        
        # Quality assessment
        if health >= 85:
            commentaries.append("✨ **Dataset Quality:** This dataset is exceptionally clean. You're in a great position to train robust models.")
        elif health >= 70:
            commentaries.append("📊 **Dataset Quality:** The dataset is reasonably well-maintained with minor data quality issues.")
        elif health >= 50:
            commentaries.append("⚠️ **Dataset Quality:** This dataset shows moderate noise and missing values. Preprocessing is crucial.")
        else:
            commentaries.append("🔧 **Dataset Quality:** Significant data quality issues detected. Heavy preprocessing required before modeling.")
        
        # Missing data insight
        if missing_pct == 0:
            commentaries.append("✅ **Missingness:** Complete data—no missing values to handle.")
        elif missing_pct < 5:
            commentaries.append(f"📌 **Missingness:** Minimal missing data ({missing_pct:.1f}%). Simple imputation should suffice.")
        elif missing_pct < 20:
            commentaries.append(f"⚡ **Missingness:** Moderate missingness ({missing_pct:.1f}%). Consider advanced imputation strategies.")
        else:
            commentaries.append(f"🚨 **Missingness:** High missingness ({missing_pct:.1f}%). This may limit model performance significantly.")
        
        return commentaries

    @staticmethod
    def analyze_feature_landscape(df, target_col):
        """Generates commentary on features and relationships."""
        commentaries = []
        X = df.drop(columns=[target_col])
        
        # Feature count
        n_features = len(X.columns)
        if n_features < 5:
            commentaries.append(f"🎯 **Feature Count:** Only {n_features} features. The problem is well-scoped—avoid overfitting.")
        elif n_features < 20:
            commentaries.append(f"📈 **Feature Count:** {n_features} features provide balanced coverage without curse of dimensionality.")
        else:
            commentaries.append(f"🌌 **Feature Count:** {n_features} features detected. Feature selection may improve model efficiency.")
        
        # Numeric correlation
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr().abs()
            high_corr_pairs = (corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)] > 0.8).sum()
            
            if high_corr_pairs == 0:
                commentaries.append("🎪 **Feature Independence:** Features show low correlation—good for linear models.")
            elif high_corr_pairs < 3:
                commentaries.append("🔗 **Feature Relationships:** Moderate feature correlation detected. Tree-based models may excel.")
            else:
                commentaries.append(f"🔀 **Feature Relationships:** High correlation ({high_corr_pairs} pairs). Consider dimensionality reduction.")
        
        return commentaries

    @staticmethod
    def analyze_target_distribution(df, target_col, task_type):
        """Generates commentary on target variable."""
        commentaries = []
        target = df[target_col]
        
        if task_type == "Classification":
            class_dist = target.value_counts()
            imbalance_ratio = class_dist.max() / class_dist.min()
            
            if imbalance_ratio < 1.5:
                commentaries.append("⚖️ **Class Balance:** Classes are well-balanced. Standard metrics (accuracy) are reliable.")
            elif imbalance_ratio < 3:
                commentaries.append(f"📊 **Class Balance:** Moderate imbalance ({imbalance_ratio:.1f}:1). Use F1-score or AUC for evaluation.")
            else:
                commentaries.append(f"⚠️ **Class Balance:** Severe imbalance ({imbalance_ratio:.1f}:1). Use stratified sampling and weighted metrics.")
        else:
            # Regression: check skewness
            skewness = target.skew()
            if abs(skewness) < 0.5:
                commentaries.append("📈 **Target Distribution:** Nearly normal distribution. Linear models should perform well.")
            elif abs(skewness) < 1:
                commentaries.append("📊 **Target Distribution:** Moderate skewness. Log transformation may help.")
            else:
                commentaries.append("🔄 **Target Distribution:** Highly skewed. Consider log/box-cox transformation.")
        
        return commentaries

    @staticmethod
    def analyze_problem_complexity(df, profile, task_type, target_col):
        """Estimates problem complexity based on heuristics."""
        commentaries = []
        
        n_rows = len(df)
        n_features = len(df.columns) - 1
        health = profile["Health Score"]
        
        # Data size vs features
        ratio = n_rows / max(n_features, 1)
        
        if task_type == "Classification":
            if ratio < 10:
                commentaries.append(f"⚡ **Data Sufficiency:** Few samples ({n_rows}) for {n_features} features. Risk of overfitting.")
            elif ratio < 100:
                commentaries.append(f"✅ **Data Sufficiency:** Reasonable sample-to-feature ratio ({ratio:.0f}:1). Good for generalization.")
            else:
                commentaries.append(f"🎯 **Data Sufficiency:** Abundant data ({n_rows} samples). Complex patterns can be captured.")
        else:
            if ratio < 5:
                commentaries.append(f"⚠️ **Data Sufficiency:** Very few samples ({n_rows}) relative to features. High risk of overfitting.")
            elif ratio < 50:
                commentaries.append(f"📊 **Data Sufficiency:** Moderate sample size. Regularization recommended.")
            else:
                commentaries.append(f"💪 **Data Sufficiency:** Sufficient data for robust regression modeling.")
        
        # Overall difficulty
        if health < 60 and n_features > 15:
            commentaries.append("🔥 **Problem Difficulty:** HIGH — Noisy data + high dimensionality. Expect iterative refinement.")
        elif health < 70 or n_features > 20:
            commentaries.append("⚡ **Problem Difficulty:** MODERATE — Some noise or dimensionality. Standard techniques should work.")
        else:
            commentaries.append("✨ **Problem Difficulty:** LOW — Clean, well-structured problem. Quick wins expected.")
        
        return commentaries

    @staticmethod
    def generate_full_commentary(df, profile, task_type, target_col):
        """Combines all commentary into a cohesive analysis."""
        all_commentaries = []
        
        all_commentaries.extend(AICommentator.analyze_dataset_quality(df, profile))
        all_commentaries.extend(AICommentator.analyze_feature_landscape(df, target_col))
        all_commentaries.extend(AICommentator.analyze_target_distribution(df, target_col, task_type))
        all_commentaries.extend(AICommentator.analyze_problem_complexity(df, profile, task_type, target_col))
        
        return all_commentaries