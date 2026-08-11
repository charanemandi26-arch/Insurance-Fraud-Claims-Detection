from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load models at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'model.pkl')
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'preprocessing.pkl')

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
except Exception as e:
    print(f"Error loading model artifacts: {e}")
    model, preprocessor = None, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not preprocessor:
        return render_template('result.html', error="Model not loaded properly. Please train the model first.")
        
    try:
        # Collect data from the form
        data = request.form.to_dict()
        
        # Convert numerical fields to appropriate types
        num_fields = ['months_as_customer', 'age', 'policy_deductable', 'policy_annual_premium', 
                      'umbrella_limit', 'insured_zip', 'capital-gains', 'capital-loss', 
                      'incident_hour_of_the_day', 'number_of_vehicles_involved', 'bodily_injuries', 
                      'witnesses', 'total_claim_amount', 'injury_claim', 'property_claim', 
                      'vehicle_claim', 'auto_year']
                      
        for field in num_fields:
            if field in data:
                # Handle floats and ints
                if field == 'policy_annual_premium':
                    data[field] = float(data[field])
                else:
                    data[field] = int(data[field])
                    
        # Create DataFrame (1 row)
        df = pd.DataFrame([data])
        
        # Preprocess
        X_processed = preprocessor.transform(df)
        
        # Predict
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0][1]
        
        # Interpret Risk
        if probability >= 0.7:
            risk_level = "HIGH RISK"
            recommendation = "This claim has an elevated predicted fraud risk and requires immediate manual investigation."
            risk_class = "danger"
        elif probability >= 0.4:
            risk_level = "MEDIUM RISK"
            recommendation = "This claim shows some suspicious patterns. A standard review is recommended."
            risk_class = "warning"
        else:
            risk_level = "LOW RISK"
            recommendation = "This claim aligns with normal patterns. Standard automated processing can proceed."
            risk_class = "success"
            
        return render_template('result.html', 
                               risk_level=risk_level, 
                               probability=f"{probability * 100:.1f}%", 
                               recommendation=recommendation,
                               risk_class=risk_class)
                               
    except Exception as e:
        return render_template('result.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
