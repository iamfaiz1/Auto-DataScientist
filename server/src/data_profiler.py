import pandas as pd
import numpy as np

class DataProfiler:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def optimize_memory(self):
        """
        Crucial for Free-Tier deployment. 
        Downcasts 64-bit types to 32-bit to save RAM, and limits rows.
        """
        # Limit rows to prevent Out-Of-Memory (OOM) crashes
        if len(self.df) > 50000:
            self.df = self.df.sample(50000, random_state=42)
            
        # Downcast floats
        float_cols = self.df.select_dtypes(include=['float64']).columns
        self.df[float_cols] = self.df[float_cols].astype('float32')
        
        # Downcast integers
        int_cols = self.df.select_dtypes(include=['int64']).columns
        self.df[int_cols] = self.df[int_cols].astype('int32')
        
        return self.df

    def calculate_health_score(self):
        """
        Calculates comprehensive health score (0-100).
        Returns: score, issues_list, recommendations_list
        """
        score = 100
        issues = []
        recommendations = []
        
        # 1. Missing Values (up to -30)
        missing_pct = (self.df.isnull().sum().sum() / self.df.size) * 100
        if missing_pct > 50:
            score -= 30
            issues.append(f"Severe missingness ({missing_pct:.1f}%)")
            recommendations.append("Consider dropping rows/columns with extreme missing values")
        elif missing_pct > 20:
            score -= 20
            issues.append(f"High missingness ({missing_pct:.1f}%)")
            recommendations.append("Use mean/median imputation for numerical features")
        elif missing_pct > 5:
            score -= 10
            issues.append(f"Moderate missingness ({missing_pct:.1f}%)")
            recommendations.append("Apply imputation strategy before modeling")
        
        # 2. Duplicates (up to -15)
        dup_pct = (self.df.duplicated().sum() / len(self.df)) * 100
        if dup_pct > 10:
            score -= 15
            issues.append(f"High duplicates ({dup_pct:.1f}%)")
            recommendations.append("Remove duplicate rows to reduce noise")
        elif dup_pct > 0:
            score -= 5
            issues.append(f"Minor duplicates ({dup_pct:.1f}%)")
        
        # 3. High Correlation (up to -10)
        numeric_df = self.df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr().abs()
            high_corr = (corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)] > 0.95).sum()
            if high_corr > 0:
                score -= min(10, high_corr * 2)
                issues.append(f"High multicollinearity ({high_corr} pairs > 0.95)")
                recommendations.append("Consider dropping highly correlated features")
        
        # 4. Class Imbalance (up to -15) - checked only for classification targets
        # This will be checked per-target in explainer
        
        # 5. Outliers (simple IQR check, up to -10)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_count = 0
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.df[col] < Q1 - 1.5 * IQR) | (self.df[col] > Q3 + 1.5 * IQR)).sum()
            outlier_count += outliers
        
        if outlier_count > len(self.df) * 0.05:
            score -= 10
            issues.append(f"Potential outliers detected ({outlier_count} records)")
            recommendations.append("Review outliers—they may be valid or require treatment")
        
        # Ensure score stays in bounds
        score = max(0, min(100, score))
        
        return int(score), issues, recommendations

    def generate_profile(self):
        """Generates a statistical summary and health metrics."""
        missing_count = int(self.df.isnull().sum().sum())
        total_cells = self.df.size
        missing_pct = (missing_count / total_cells) * 100 if total_cells > 0 else 0
        
        health_score, issues, recommendations = self.calculate_health_score()
        
        # Detect Personality
        rows, cols = self.df.shape
        if health_score > 90: personality = "Clean Professional"
        elif missing_pct > 10: personality = "Noisy Real-World"
        elif cols > 20: personality = "High-Dimensional"
        else: personality = "Standard Academic"

        return {
            "Total Rows": rows,
            "Total Columns": cols,
            "Missing Values": missing_count,
            "Numerical Columns": len(self.df.select_dtypes(include=[np.number]).columns),
            "Health Score": health_score,
            "Personality": personality,
            "Missing Pct": round(missing_pct, 2),
            "Issues": issues,
            "Recommendations": recommendations,
        }