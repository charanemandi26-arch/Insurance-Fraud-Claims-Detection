import os, sys
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess_data, UI_FEATURES, HIDDEN_DEFAULTS

def save_final_model():
    data_path = os.path.join('data', 'raw', 'insurance_claims.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join('..', 'data', 'raw', 'insurance_claims.csv')

    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor, feature_names, model_cols = preprocess_data(data_path)

    print("Training Random Forest (Model A — 33 features)...")
    model = RandomForestClassifier(
        class_weight='balanced', n_estimators=150, max_depth=10, random_state=42
    )
    model.fit(X_train, y_train)

    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/model.pkl')
    joblib.dump(preprocessor, 'model/preprocessing.pkl')
    if feature_names:
        joblib.dump(feature_names, 'model/feature_names.pkl')

    # Save the ordered list of columns the model expects
    joblib.dump(model_cols, 'model/model_columns.pkl')
    # Save UI features and hidden defaults for the Flask app
    joblib.dump(UI_FEATURES, 'model/ui_features.pkl')
    joblib.dump(HIDDEN_DEFAULTS, 'model/hidden_defaults.pkl')

    print(f"Saved: model.pkl, preprocessing.pkl, feature_names.pkl")
    print(f"Saved: model_columns.pkl ({len(model_cols)} cols), ui_features.pkl, hidden_defaults.pkl")

if __name__ == '__main__':
    save_final_model()
