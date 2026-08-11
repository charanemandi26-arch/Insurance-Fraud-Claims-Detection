import os
import sys
import joblib
from sklearn.ensemble import RandomForestClassifier

# Add src to path to import preprocessing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess_data

def save_final_model():
    data_path = os.path.join('data', 'raw', 'insurance_claims.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join('..', 'data', 'raw', 'insurance_claims.csv')
        
    print("Loading and Preprocessing Data...")
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(data_path)
    
    print("Training Final Random Forest Model...")
    model = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=150, max_depth=10)
    model.fit(X_train, y_train)
    
    # Save artifacts
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/model.pkl')
    joblib.dump(preprocessor, 'model/preprocessing.pkl')
    
    # Save feature names for explainability (Phase 6)
    if feature_names:
        joblib.dump(feature_names, 'model/feature_names.pkl')
    
    print("Model and preprocessor saved successfully to model/ folder!")

if __name__ == '__main__':
    save_final_model()
