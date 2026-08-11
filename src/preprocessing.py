import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib, os

# ─── Feature groups ────────────────────────────────────────────────────────────
# UI_FEATURES: Fields shown to the user in the 3-step stepper
UI_FEATURES = [
    'months_as_customer', 'age', 'policy_deductable', 'policy_annual_premium',
    'policy_csl', 'umbrella_limit',
    'incident_type', 'collision_type', 'incident_severity', 'authorities_contacted',
    'incident_hour_of_the_day', 'number_of_vehicles_involved',
    'property_damage', 'bodily_injuries', 'witnesses', 'police_report_available',
    'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim',
    'auto_make', 'auto_year',
]

# HIDDEN_DEFAULTS: Features required by Model A but not shown to the user.
# Values are actual statistical mode/median from the training data.
HIDDEN_DEFAULTS = {
    'insured_sex': 'FEMALE',
    'insured_education_level': 'JD',
    'insured_occupation': 'machine-op-inspct',
    'insured_hobbies': 'reading',
    'insured_relationship': 'own-child',
    'capital-gains': 0,
    'capital-loss': -23250,
    'incident_city': 'Springfield',
    'incident_state': 'NY',
    'auto_model': 'RAM',
    'policy_state': 'OH',
}

# ALL features the model was trained on (identifiers + empty dropped, rest kept)
DROP_ALWAYS = ['_c39', 'policy_number', 'insured_zip', 'incident_location',
               'policy_bind_date', 'incident_date']

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df.replace('?', np.nan, inplace=True)
    df = df.drop(columns=DROP_ALWAYS, errors='ignore')
    return df

def get_model_features(df):
    """Returns the full ordered list of features the model expects."""
    return [c for c in df.columns if c != 'fraud_reported']

def build_preprocessor(X_train):
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'string']).columns.tolist()

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ]), num_cols, cat_cols

def preprocess_data(filepath):
    df = load_and_clean_data(filepath)
    X = df.drop(columns=['fraud_reported'], errors='ignore')
    y = df['fraud_reported'].map({'Y': 1, 'N': 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    try:
        cat_enc_names = preprocessor.named_transformers_['cat']['encoder']\
                        .get_feature_names_out(cat_cols).tolist()
        feature_names = num_cols + cat_enc_names
    except Exception:
        feature_names = None

    return X_train_p, X_test_p, y_train, y_test, preprocessor, feature_names, list(X_train.columns)

if __name__ == '__main__':
    import sys
    data_path = os.path.join('data', 'raw', 'insurance_claims.csv')
    print("Running new preprocessing pipeline...")
    X_train, X_test, y_train, y_test, prep, feat_names, model_cols = preprocess_data(data_path)
    print(f"X_train: {X_train.shape}")
    print(f"Model expects {len(model_cols)} input columns")
    print("Preprocessing OK — no leakage.")
