import os
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

# Add src to path to import preprocessing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess_data

def evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42, max_depth=7),
        'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=150, max_depth=10),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100)
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'ROC-AUC': roc,
            'CM': cm
        })
        
    return results, trained_models

if __name__ == '__main__':
    data_path = os.path.join('data', 'raw', 'insurance_claims.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join('..', 'data', 'raw', 'insurance_claims.csv')
        
    print("Loading and Preprocessing Data...")
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(data_path)
    
    print("Training and Evaluating Models...")
    results, trained_models = evaluate_models(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    for r in results:
        print(f"\n[{r['Model']}]")
        print(f"Accuracy:  {r['Accuracy']:.4f}")
        print(f"Precision: {r['Precision']:.4f}")
        print(f"Recall:    {r['Recall']:.4f}")
        print(f"F1-score:  {r['F1']:.4f}")
        print(f"ROC-AUC:   {r['ROC-AUC']:.4f}")
        print(f"Confusion Matrix:\n{r['CM']}")
    print("\n" + "="*50)
