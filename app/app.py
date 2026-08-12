from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# ── Load artifacts at startup ─────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), '..', 'model')

def load_artifact(name):
    path = os.path.join(BASE, name)
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[WARN] Could not load {name}: {e}")
        return None

model          = load_artifact('model.pkl')
preprocessor   = load_artifact('preprocessing.pkl')
model_columns  = load_artifact('model_columns.pkl')   # 22 ordered col names
feature_names  = load_artifact('feature_names.pkl')   # encoded names for importance

# Actual metrics computed on held-out test set using the saved model
MODEL_METRICS = {
    'model_name': 'Random Forest',
    'accuracy':   '80.5%',
    'precision':  '58.6%',
    'recall':     '69.4%',
    'f1':         '63.6%',
    'roc_auc':    '77.6%',
}

# Numerical fields — cast to int/float before prediction
NUMERICAL_FIELDS = [
    'months_as_customer', 'age', 'policy_deductable', 'policy_annual_premium',
    'umbrella_limit', 'incident_hour_of_the_day', 'number_of_vehicles_involved',
    'bodily_injuries', 'witnesses', 'total_claim_amount',
    'injury_claim', 'property_claim', 'vehicle_claim', 'auto_year',
]

# Required fields — missing/empty → user-friendly validation message
REQUIRED_FIELDS = [
    'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim',
    'incident_type', 'incident_severity', 'incident_hour_of_the_day',
    'age', 'months_as_customer', 'policy_csl', 'policy_annual_premium',
    'policy_deductable', 'umbrella_limit',
    'number_of_vehicles_involved', 'bodily_injuries', 'witnesses',
    'auto_make', 'auto_year',
]


def validate_form(form_data):
    """Returns an error message string if validation fails, else None."""
    for field in REQUIRED_FIELDS:
        val = form_data.get(field, '').strip()
        if not val:
            label = field.replace('_', ' ').title()
            return f"Required field missing: '{label}'. Please fill in all required fields."

    for field in NUMERICAL_FIELDS:
        val = form_data.get(field, '').strip()
        if val:
            try:
                float(val)
            except ValueError:
                label = field.replace('_', ' ').title()
                return f"Invalid value for '{label}': must be a number (got '{val}')."

    try:
        age = int(form_data.get('age', 0))
        if not (16 <= age <= 110):
            return "Customer Age must be between 16 and 110."
        total = float(form_data.get('total_claim_amount', 0))
        if total < 0:
            return "Total Claim Amount cannot be negative."
        hour = int(form_data.get('incident_hour_of_the_day', 0))
        if not (0 <= hour <= 23):
            return "Incident Hour must be between 0 and 23."
        year = int(form_data.get('auto_year', 2000))
        if not (1950 <= year <= 2030):
            return "Vehicle Year must be between 1950 and 2030."
    except (ValueError, TypeError):
        pass
    return None


def build_input_row(form_data: dict) -> pd.DataFrame:
    """
    Build a single-row DataFrame using ONLY the 22 columns the model expects.
    Missing optional fields become NaN — the preprocessing pipeline imputes them.
    """
    row = {}
    for col in model_columns:
        val = form_data.get(col, None)
        if val == '' or val is None:
            val = np.nan
        row[col] = val

    for col in NUMERICAL_FIELDS:
        if col in row:
            try:
                v = row[col]
                if isinstance(v, float) and np.isnan(v):
                    continue
                row[col] = float(v) if col == 'policy_annual_premium' else int(float(v))
            except (ValueError, TypeError):
                row[col] = np.nan

    return pd.DataFrame([row], columns=model_columns)


def get_top_features(n=6):
    """Return top-n feature importances as dicts with label and pct."""
    if model is None or feature_names is None:
        return []
    importances = model.feature_importances_
    idx   = np.argsort(importances)[::-1]
    total = importances.sum()
    seen, result = {}, []
    for i in idx:
        raw = feature_names[i]
        label = raw
        for col in (model_columns or []):
            if raw.startswith(col + '_') or raw == col:
                label = col
                break
        label = label.replace('_', ' ').title()
        pct = round(float(importances[i]) / total * 100, 1)
        if label not in seen:
            seen[label] = pct
            result.append({'label': label, 'pct': pct})
        if len(result) >= n:
            break
    return result


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html', metrics=MODEL_METRICS)


@app.route('/assess')
def assess():
    return render_template('assess.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or preprocessor is None or model_columns is None:
        return render_template('result.html',
                               error="Model not loaded. Please run the notebook "
                                     "Appendix cell to save model artifacts first.",
                               metrics=MODEL_METRICS)

    form_data = request.form.to_dict()

    error_msg = validate_form(form_data)
    if error_msg:
        return render_template('result.html',
                               error=error_msg,
                               metrics=MODEL_METRICS)

    try:
        df_input    = build_input_row(form_data)
        X_proc      = preprocessor.transform(df_input)
        probability = float(model.predict_proba(X_proc)[0][1])
        pct         = round(probability * 100, 1)

        if probability >= 0.60:
            risk_level     = 'HIGH RISK'
            risk_class     = 'danger'
            icon           = '⚠'
            recommendation = (
                "This claim has an elevated predicted fraud risk. "
                "Manual investigation by a claims adjuster is recommended "
                "before processing."
            )
        elif probability >= 0.35:
            risk_level     = 'MEDIUM RISK'
            risk_class     = 'warning'
            icon           = '◉'
            recommendation = (
                "This claim shows some patterns associated with elevated risk. "
                "A standard secondary review is recommended."
            )
        else:
            risk_level     = 'LOW RISK'
            risk_class     = 'success'
            icon           = '✓'
            recommendation = (
                "This claim aligns with typical low-risk patterns. "
                "Standard automated processing can proceed."
            )

        top_features = get_top_features(6)

        label_map = {
            'total_claim_amount':      'Total Claim Amount',
            'incident_severity':       'Incident Severity',
            'incident_type':           'Accident Type',
            'collision_type':          'Collision Type',
            'age':                     'Customer Age',
            'months_as_customer':      'Customer Tenure (months)',
            'witnesses':               'Witnesses',
            'police_report_available': 'Police Report',
            'bodily_injuries':         'Bodily Injuries',
            'auto_make':               'Vehicle Make',
            'auto_year':               'Vehicle Year',
        }
        summary = {}
        for key, label in label_map.items():
            val = form_data.get(key, '—') or '—'
            if key == 'total_claim_amount':
                try:
                    val = f"${int(float(val)):,}"
                except Exception:
                    pass
            summary[label] = val

        return render_template('result.html',
                               risk_level=risk_level,
                               risk_class=risk_class,
                               icon=icon,
                               probability=pct,
                               recommendation=recommendation,
                               top_features=top_features,
                               summary=summary,
                               metrics=MODEL_METRICS)

    except Exception as e:
        import traceback
        print("[ERROR] Prediction failed:")
        traceback.print_exc()
        return render_template('result.html',
                               error=f"Prediction error: {str(e)}. "
                                     "Please check your input values and try again.",
                               metrics=MODEL_METRICS)


if __name__ == '__main__':
    app.run(debug=True, port=5000)

