import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_and_clean_data(filepath):
    """Loads dataset and performs initial cleaning."""
    df = pd.read_csv(filepath)
    
    # Replace '?' with standard NaN
    df.replace('?', np.nan, inplace=True)
    
    # Drop irrelevant columns and identifiers to prevent leakage
    cols_to_drop = ['_c39', 'policy_number', 'incident_location', 'policy_bind_date', 'incident_date']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    return df

def get_preprocessor(X):
    """Creates a ColumnTransformer pipeline for numerical and categorical features."""
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
    
    # Numerical Pipeline: Impute missing with median, then scale
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical Pipeline: Impute missing with most frequent, then one-hot encode
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine into a ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])
    
    return preprocessor, numerical_cols, categorical_cols

def preprocess_data(filepath):
    """Full preprocessing workflow."""
    df = load_and_clean_data(filepath)
    
    # Separate features and target
    X = df.drop('fraud_reported', axis=1)
    y = df['fraud_reported'].map({'Y': 1, 'N': 0})
    
    # Stratified Train-Test Split (80/20) to handle class imbalance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Build preprocessor
    preprocessor, num_cols, cat_cols = get_preprocessor(X_train)
    
    # Fit ONLY on training data (prevents data leakage)
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Transform test data
    X_test_processed = preprocessor.transform(X_test)
    
    # Feature names after one-hot encoding
    try:
        cat_feature_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(cat_cols)
        feature_names = num_cols + list(cat_feature_names)
    except Exception:
        feature_names = None
        
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_names

if __name__ == '__main__':
    # Test the preprocessing script
    import sys
    import os
    
    # Ensure correct path assuming this is run from root directory
    data_path = os.path.join('data', 'raw', 'insurance_claims.csv')
    if not os.path.exists(data_path):
        # Fallback if run from src/
        data_path = os.path.join('..', 'data', 'raw', 'insurance_claims.csv')
        
    print("Running Preprocessing...")
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(data_path)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape} (Fraud: {y_train.sum()} instances)")
    print(f"Number of extracted features: {len(feature_names) if feature_names else 'Unknown'}")
    print("Preprocessing completed successfully with no data leakage!")
