# Insurance Fraud Claims Detection Engine

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> An end-to-end Machine Learning prototype that assesses the **fraud risk** of auto insurance claims using a trained Random Forest classifier, served via a modern Flask web application.

---

## 1. Project Overview

The Insurance Fraud Claims Detection Engine is a complete ML project that:

- Loads and analyses **1,000 historical auto insurance claims**.
- Trains and compares **4 classification models** (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting).
- Selects the best model based on **Recall and F1-score** — metrics appropriate for fraud detection.
- Exposes a professional **4-step web form** where a claim can be assessed and a risk score returned in real time.

---

## 2. Problem Statement

Insurance fraud costs the global industry **billions of dollars annually**, raising premiums for honest customers. Manual review of every claim is slow and error-prone. This project demonstrates how a machine learning system can automatically flag suspicious claims for priority investigation.

---

## 3. Objectives

- Automate the initial screening of auto insurance claims.
- Maximise **Recall** — catching as many real fraud cases as possible.
- Provide a clean, user-friendly web interface for non-technical investigators.
- Save and serve the model as a production-ready Flask API.

---

## 4. Technology Stack

| Layer | Technologies |
|---|---|
| Language | Python 3.11 |
| Data Processing | pandas, NumPy |
| Machine Learning | scikit-learn, joblib |
| Visualisation | matplotlib, seaborn |
| Web Framework | Flask |
| Frontend | HTML5, Vanilla CSS (Glassmorphism / Dark Mode) |

---

## 5. Dataset

| Attribute | Value |
|---|---|
| Source | Auto Insurance Claims (Synthetic/Historical) |
| Total Records | 1,000 |
| Raw Features | 40 |
| Features Used by Model | 22 (user input) + 11 (statistical defaults) = 33 |
| Target Variable | `fraud_reported` (Y = Fraud, N = Not Fraud) |
| Class Distribution | 75.3% Non-Fraud / 24.7% Fraud |

---

## 6. System Architecture

```
User / Claim Data
       ↓
4-Step Web Form (Flask)
       ↓
Input Validation + Statistical Defaults
       ↓
Preprocessing Pipeline (Imputation → Scaling → OneHotEncoding)
       ↓
Random Forest Classifier (150 trees)
       ↓
Fraud Probability Score
       ↓
Risk Level: LOW / MEDIUM / HIGH
       ↓
Result Page + Feature Influence Bars
```

---

## 7. Project Structure

```
insurance-fraud-detection/
│
├── app/
│   ├── app.py                  # Flask application (routes, prediction logic)
│   ├── static/
│   │   └── style.css           # Dark glassmorphism UI design
│   └── templates/
│       ├── index.html          # Dashboard / landing page
│       ├── assess.html         # 4-step claim input form
│       └── result.html         # Fraud risk result page
│
├── data/
│   ├── raw/
│   │   └── insurance_claims.csv  # Original dataset (1,000 records, 40 columns)
│   └── processed/              # Reserved for processed data exports
│
├── model/
│   ├── model.pkl               # Trained Random Forest model
│   ├── preprocessing.pkl       # Fitted ColumnTransformer pipeline
│   ├── model_columns.pkl       # Ordered column list (33 features)
│   ├── feature_names.pkl       # Encoded feature names (157 after OHE)
│   └── hidden_defaults.pkl     # Statistical defaults for 11 background features
│
├── notebooks/
│   └── insurance_fraud_analysis.ipynb   # Complete ML analysis (55 cells, 8 figures)
│
├── src/
│   ├── preprocessing.py        # Reusable preprocessing functions (used by save_final_model.py)
│   └── save_final_model.py     # Standalone script to retrain and save model artifacts
│
├── tests/
│   └── test_app.py             # Flask application tests (TC01-TC05)
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 8. Model Results

Four models were trained and compared. **Random Forest** was selected.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest ✓** | **83.5%** | **62.9%** | **79.6%** | **70.3%** | **83.2%** |
| Logistic Regression | 82.5% | 63.0% | 69.4% | 66.0% | 82.9% |
| Decision Tree | 81.5% | 60.3% | 71.4% | 65.4% | 74.5% |
| Gradient Boosting | 80.0% | 58.8% | 61.2% | 60.0% | 84.5% |

> **Why Recall over Accuracy?** With 75% non-fraud claims, a naive model achieves 75% accuracy by predicting everything as non-fraud — but catches 0% of real fraud. Recall (79.6%) ensures the majority of actual fraud cases are flagged for investigation.

---

## 9. Notebook

The complete ML analysis lives in:

```
notebooks/insurance_fraud_analysis.ipynb
```

**55 cells | 8 embedded figures | all outputs pre-executed**

Sections:
1. Problem Statement
2. Objectives
3. Dataset Description
4. Import Libraries
5. Load Dataset
6. Initial Data Inspection (shape, dtypes, missing values, duplicates)
7. Exploratory Data Analysis (7 visualisations)
8. Feature Audit
9. Feature Selection
10. Data Preprocessing
11. Train / Test Split
12. Model Training (4 models)
13. Model Comparison (comparison table)
14. Confusion Matrices (all 4 models)
15. ROC-AUC Curves (all 4 models)
16. Feature Importance (top 15 features)
17. Final Model Selection (justified rationale)
18. Final Model Evaluation
19. Save Final Model
20. Sample End-to-End Prediction
21. Conclusion

---

## 10. Installation

```bash
# 1. Clone the repository
git clone <YOUR_GITHUB_URL>
cd insurance-fraud-detection

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 11. Running Locally

```bash
python app/app.py
```

Navigate to: **http://127.0.0.1:5000**

---

## 12. How to Retrain the Model

The notebook is the authoritative training source. To retrain without opening Jupyter:

```bash
python src/save_final_model.py
```

This re-runs the full preprocessing + training pipeline and overwrites all artifacts in `model/`.

---

## 13. Limitations

- Trained on a synthetic/historical dataset; may not generalise to all real-world markets.
- Self-reported incident details (e.g., severity) can be falsified by fraudsters.
- Feature importance shows statistical correlation, not causation.
- The current prototype does not support real-time data feeds or external API integration.

---

## 14. Future Enhancements

- Computer Vision to analyse submitted car damage photos automatically.
- NLP to extract structured signals from free-text police/incident reports.
- Explainability layer using SHAP values for per-prediction feature attribution.
- User authentication and audit logging for production deployment.

---

## 15. Disclaimer

> This is an academic project prototype. All predictions represent statistical risk modelling and are intended as a **decision-support tool** for human investigators. A high-risk prediction does **not** prove insurance fraud and must never be used as the sole reason to deny a claim without proper human review.

---

## 16. License

MIT License — see [LICENSE](LICENSE) for details.
