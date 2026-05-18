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

    def generate_profile(self):
        """Generates a statistical summary of the dataset."""
        profile = {
            "Total Rows": self.df.shape[0],
            "Total Columns": self.df.shape[1],
            "Missing Values": int(self.df.isnull().sum().sum()),
            "Numerical Columns": len(self.df.select_dtypes(include=[np.number]).columns),
            "Categorical Columns": len(self.df.select_dtypes(exclude=[np.number]).columns)
        }
        return profile