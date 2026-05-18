"""
Estimates dataset difficulty for ML prediction.
Uses heuristic scoring based on multiple factors.
"""

import numpy as np
import pandas as pd

class DifficultyEstimator:
    """Estimates problem difficulty: Easy, Moderate, Hard."""
    
    @staticmethod
    def estimate_difficulty(df, profile, task_type, target_col):
        """
        Calculate difficulty score (0-100) and return label.
        Higher score = harder problem.
        """
        score = 50  # Start at neutral
        factors = {}
        
        # --- FACTOR 1: Data Quality & Missingness ---
        health = profile["Health Score"]
        if health >= 85:
            factors["Quality"] = ("Excellent", 0)
        elif health >= 70:
            factors["Quality"] = ("Good", 5)
        elif health >= 50:
            factors["Quality"] = ("Fair", 15)
        else:
            factors["Quality"] = ("Poor", 25)
        
        score += factors["Quality"][1]
        
        # --- FACTOR 2: Class/Target Imbalance ---
        target = df[target_col]
        if task_type == "Classification":
            class_dist = target.value_counts()
            imbalance_ratio = class_dist.max() / class_dist.min()
            
            if imbalance_ratio < 1.5:
                factors["Balance"] = ("Balanced", 0)
            elif imbalance_ratio < 3:
                factors["Balance"] = ("Moderate", 10)
            elif imbalance_ratio < 10:
                factors["Balance"] = ("Severe", 20)
            else:
                factors["Balance"] = ("Extreme", 30)
        else:
            # For regression, check target skewness
            skewness = abs(target.skew())
            if skewness < 0.5:
                factors["Balance"] = ("Normal", 0)
            elif skewness < 1:
                factors["Balance"] = ("Moderate Skew", 10)
            else:
                factors["Balance"] = ("Severe Skew", 15)
        
        score += factors["Balance"][1]
        
        # --- FACTOR 3: Feature Complexity ---
        n_features = len(df.columns) - 1
        n_rows = len(df)
        
        if n_features <= 5:
            factors["Dimensionality"] = ("Low", 0)
        elif n_features <= 15:
            factors["Dimensionality"] = ("Moderate", 5)
        elif n_features <= 30:
            factors["Dimensionality"] = ("High", 15)
        else:
            factors["Dimensionality"] = ("Very High", 25)
        
        score += factors["Dimensionality"][1]
        
        # --- FACTOR 4: Sample Size Sufficiency ---
        sample_feature_ratio = n_rows / max(n_features, 1)
        
        if sample_feature_ratio > 100:
            factors["Sample Size"] = ("Abundant", -10)
        elif sample_feature_ratio > 50:
            factors["Sample Size"] = ("Sufficient", 0)
        elif sample_feature_ratio > 10:
            factors["Sample Size"] = ("Moderate", 10)
        elif sample_feature_ratio > 5:
            factors["Sample Size"] = ("Limited", 20)
        else:
            factors["Sample Size"] = ("Scarce", 30)
        
        score += factors["Sample Size"][1]
        
        # --- FACTOR 5: Feature Multicollinearity ---
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr().abs()
            high_corr_pairs = (corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)] > 0.8).sum()
            
            if high_corr_pairs == 0:
                factors["Multicollinearity"] = ("None", 0)
            elif high_corr_pairs < 3:
                factors["Multicollinearity"] = ("Low", 5)
            elif high_corr_pairs < 8:
                factors["Multicollinearity"] = ("Moderate", 10)
            else:
                factors["Multicollinearity"] = ("High", 15)
        else:
            factors["Multicollinearity"] = ("N/A", 0)
        
        score += factors["Multicollinearity"][1]
        
        # --- FACTOR 6: Target Distribution Complexity ---
        n_unique = target.nunique()
        
        if task_type == "Classification":
            if n_unique == 2:
                factors["Target Complexity"] = ("Binary", 0)
            elif n_unique <= 5:
                factors["Target Complexity"] = ("Multi-class", 5)
            else:
                factors["Target Complexity"] = ("Many Classes", 15)
        else:
            # Regression
            if profile["Missing Pct"] > 20:
                factors["Target Complexity"] = ("High Variance", 10)
            else:
                factors["Target Complexity"] = ("Normal", 0)
        
        score += factors["Target Complexity"][1]
        
        # Ensure score is in valid range
        score = max(0, min(100, score))
        
        # --- CLASSIFY DIFFICULTY ---
        if score < 35:
            difficulty = "🟢 Easy"
            description = "Well-structured problem. Standard algorithms should perform well."
        elif score < 65:
            difficulty = "🟡 Moderate"
            description = "Some complexity. May require careful preprocessing and tuning."
        else:
            difficulty = "🔴 Hard"
            description = "Complex problem. Advanced techniques and extensive iteration needed."
        
        return {
            "score": int(score),
            "difficulty": difficulty,
            "description": description,
            "factors": factors,
        }
    
    @staticmethod
    def generate_difficulty_report(difficulty_data):
        """Generate formatted difficulty assessment."""
        report = f"**Difficulty Level:** {difficulty_data['difficulty']}\n\n"
        report += f"**Score:** {difficulty_data['score']}/100\n\n"
        report += f"**Assessment:** {difficulty_data['description']}\n\n"
        report += "**Contributing Factors:**\n"
        
        for factor_name, (factor_status, factor_score) in difficulty_data['factors'].items():
            score_text = f"+{factor_score}" if factor_score >= 0 else f"{factor_score}"
            report += f"• **{factor_name}:** {factor_status} ({score_text})\n"
        
        return report

    @staticmethod
    def get_mitigation_strategies(difficulty_data, task_type):
        """Suggest strategies based on difficulty."""
        strategies = []
        score = difficulty_data['score']
        
        # Quality issues
        quality_score = difficulty_data['factors'].get('Quality', ('', 0))[1]
        if quality_score > 10:
            strategies.append("**Data Cleaning:** Invest time in handling missing values and outliers.")
        
        # Imbalance
        balance_score = difficulty_data['factors'].get('Balance', ('', 0))[1]
        if balance_score > 10:
            if task_type == "Classification":
                strategies.append("**Class Imbalance:** Use stratified sampling, class weights, or SMOTE.")
            else:
                strategies.append("**Skewed Target:** Consider log/Box-Cox transformation.")
        
        # High dimensionality
        dim_score = difficulty_data['factors'].get('Dimensionality', ('', 0))[1]
        if dim_score > 10:
            strategies.append("**Feature Selection:** Reduce dimensionality via PCA, correlation analysis, or domain knowledge.")
        
        # Sample size
        sample_score = difficulty_data['factors'].get('Sample Size', ('', 0))[1]
        if sample_score > 15:
            strategies.append("**Limited Data:** Use cross-validation, regularization, and simpler models.")
        
        # Multicollinearity
        multi_score = difficulty_data['factors'].get('Multicollinearity', ('', 0))[1]
        if multi_score > 10:
            strategies.append("**Multicollinearity:** Drop correlated features or use ridge/lasso regression.")
        
        if score >= 65:
            strategies.append("**Iteration:** This is a hard problem—be prepared for multiple cycles of refinement.")
        
        return strategies