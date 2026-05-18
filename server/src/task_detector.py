import pandas as pd
import numpy as np

class TaskDetector:
    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df
        self.target_col = target_col
        self.task_type = None

    def detect(self) -> str:
        """
        Rule-based heuristic to determine ML task.
        """
        if self.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataset.")

        target_series = self.df[self.target_col]
        unique_vals = target_series.nunique()
        dtype = target_series.dtype

        # If it's text, boolean, or categorical -> Classification [cite: 30]
        if dtype == 'object' or dtype == 'category' or dtype == 'bool':
            self.task_type = 'Classification'
            
        # If it's a number...
        elif np.issubdtype(dtype, np.number):
            # If it has very few unique numeric values, it's likely a class label (e.g., 0, 1, 2) [cite: 32]
            if unique_vals <= 20: 
                self.task_type = 'Classification'
            # If it has many unique numeric continuous values -> Regression [cite: 31]
            else:
                self.task_type = 'Regression'
        else:
            self.task_type = 'Unknown'

        return self.task_type