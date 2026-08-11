from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# ── Load artifacts at startup ─────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), '..', 'model')

def load_artifact(name):
    try:
        return joblib.load(os.path.join(BASE, name))
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return None

model          = load_artifact('model.pkl')
preprocessor   = load_artifact('preprocessing.pkl')
model_columns  = load_artifact('model_columns.pkl')   # 33 ordered col names
feature_names  = load_artifact('feature_names.pkl')   # encoded names for importance
hidden_defaults = load_artifact('hidden_defaults.pkl') # statistical defaults

# Actual model metrics (computed during training, Phase 4-5)
MODEL_METRICS = {
    'model_name': 'Random Forest',
    'accuracy':   '83.5%',
    'precision':  '62.9%',
    'recall':     '79.6%',
    'f1':         '70.3%',
    'roc_auc':    '83.2%',
}

NUMERICAL_FIELDS = [
    'months_as_customer', 'age', 'policy_deductable', 'policy_annual_premium',
    'umbrella_limit', 'incident_hour_of_the_day', 'number_of_vehicles_involved',
    'bodily_injuries', 'witnesses', 'total_claim_amount',
    'injury_claim', 'property_claim', 'vehicle_claim', 'auto_year',
]


def build_input_row(form_data: dict) -> pd.DataFrame:
    """
    Merge user-supplied form fields with statistical hidden defaults,
    producing a single-row DataFrame in the exact column order the
    preprocessor/model expects.
    """
    row = dict(hidden_defaults)   # start with hidden statistical defaults
    row.update(form_data)          # override with whatever the user provided

    # Type-cast numerics
    for col in NUMERICAL_FIELDS:
        if col in row and row[col] not in (None, ''):
            try:
                row[col] = float(row[col]) if col == 'policy_annual_premium' else int(row[col])
            except (ValueError, TypeError):
                row[col] = np.nan

    # Build a 1-row DataFrame with exactly the columns the model expects
    df = pd.DataFrame([row], columns=model_columns)
    return df


def get_top_features(n=6):
    """Return top-n feature importances from the model as (label, pct) pairs."""
    if model is None or feature_names is None:
        return []
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:n]
    total = importances[idx].sum()
    result = []
    for i in idx:
        raw  = feature_names[i]
        # Clean up encoded name: remove cat__/num__ prefix and make readable
        label = raw.replace('cat__', '').replace('num__', '')
        # Drop one-hot suffix like _YES, _NO, _Police, etc.
        if '_' in label:
            parts = label.split('_')
            # Keep first meaningful part
            label = parts[0].replace('-', ' ').title()
        label = label.replace('_', ' ').title()
        pct = round(float(importances[i]) / total * 100)
        result.append({'label': label, 'pct': pct})
    # Deduplicate by label, keep highest
    seen = {}
    for item in result:
        if item['label'] not in seen or item['pct'] > seen[item['label']]['pct']:
            seen[item['label']] = item
    return list(seen.values())[:n]


@app.route('/')
def home():
    return render_template('index.html', metrics=MODEL_METRICS)


@app.route('/assess')
def assess():
    return render_template('assess.html')


@app.route('/predict', methods=['POST'])
def predict():
    if not model or not preprocessor or not model_columns:
        return render_template('result.html',
                               error="Model not loaded. Please run save_final_model.py first.",
                               metrics=MODEL_METRICS)
    try:
        form_data = request.form.to_dict()
        df_input  = build_input_row(form_data)
        X_proc    = preprocessor.transform(df_input)

        probability = float(model.predict_proba(X_proc)[0][1])
        pct         = round(probability * 100, 1)

        # Documented risk thresholds (based on optimal F1 trade-off at 0.35/0.60)
        if probability >= 0.60:
            risk_level   = 'HIGH RISK'
            risk_class   = 'danger'
            icon         = '⚠'
            recommendation = (
                "This claim has an elevated predicted fraud risk. "
                "Manual investigation by a claims adjuster is recommended "
                "before processing."
            )
        elif probability >= 0.35:
            risk_level   = 'MEDIUM RISK'
            risk_class   = 'warning'
            icon         = '◉'
            recommendation = (
                "This claim shows some patterns associated with elevated risk. "
                "A standard secondary review is recommended."
            )
        else:
            risk_level   = 'LOW RISK'
            risk_class   = 'success'
            icon         = '✓'
            recommendation = (
                "This claim aligns with typical low-risk patterns. "
                "Standard automated processing can proceed."
            )

        top_features = get_top_features(6)

        # Build a clean claim summary dict for the result page
        label_map = {
            'total_claim_amount':  'Total Claim Amount',
            'incident_severity':   'Incident Severity',
            'incident_type':       'Accident Type',
            'collision_type':      'Collision Type',
            'age':                 'Customer Age',
            'months_as_customer':  'Customer Tenure (months)',
            'witnesses':           'Witnesses',
            'police_report_available': 'Police Report',
            'bodily_injuries':     'Bodily Injuries',
            'auto_make':           'Vehicle Make',
            'auto_year':           'Vehicle Year',
        }
        summary = {}
        for key, label in label_map.items():
            val = form_data.get(key, hidden_defaults.get(key, '—'))
            if key == 'total_claim_amount':
                try:
                    val = f"${int(val):,}"
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
        return render_template('result.html',
                               error=str(e),
                               metrics=MODEL_METRICS)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
