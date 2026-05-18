import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class DataPreprocessor:
    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df
        self.target_col = target_col

    def build_pipeline(self):
        """
        Dynamically builds a scikit-learn preprocessing pipeline based on column types.
        """
        # Separate features (X) from the target (y)
        X = self.df.drop(columns=[self.target_col])
        
        # Identify numerical and categorical columns
        numeric_features = X.select_dtypes(include=['int32', 'int64', 'float32', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        # Pipeline for numerical data: Fill missing with Median, then Scale
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Pipeline for categorical data: Fill missing with 'missing', then One-Hot Encode
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine both pipelines into one unified preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
            
        return preprocessor, X, self.df[self.target_col]