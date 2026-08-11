# Insurance Fraud Claims Detection Engine

## 1. Project Overview
The Insurance Fraud Claims Detection Engine is an end-to-end Machine Learning prototype designed to assess the risk of fraud in auto insurance claims. By analyzing historical claim data, the system flags suspicious claims as High, Medium, or Low Risk, acting as an automated decision-support tool for human investigators.

## 2. Problem Statement
Insurance fraud costs the industry billions of dollars annually, leading to increased premiums for honest customers. Manual investigation of every single claim is slow, expensive, and inefficient. A data-driven approach is required to prioritize which claims need immediate investigation.

## 3. Objectives
* Automate the initial screening of auto insurance claims.
* Reduce false positives to ensure investigators only spend time on highly suspicious claims.
* Maintain a high recall rate to avoid missing actual fraudulent claims.
* Provide a modern, user-friendly web interface for claim intake.

## 4. Proposed Solution
We propose a Machine Learning-based classification pipeline. The system ingests claim details (policy info, incident severity, claim amounts), preprocesses the data using a Scikit-Learn pipeline, and evaluates it using a trained Random Forest classifier. A Flask web application provides the frontend interface.

## 5. Dataset
* **Source:** Auto Insurance Claims Data (Synthetic/Historical combination)
* **Total Records:** 1,000
* **Features Used:** 34 (after dropping identifiers and empty columns)
* **Target Variable:** `fraud_reported` ('Y' or 'N')
* **Class Distribution:** 75.3% Non-Fraud, 24.7% Fraud (Imbalanced)

## 6. Technology Stack
* **Language:** Python 3
* **Data Processing:** pandas, NumPy
* **Machine Learning:** scikit-learn, joblib
* **Data Visualization:** matplotlib, seaborn
* **Web Framework:** Flask
* **Frontend:** HTML5, CSS3 (Modern Glassmorphism Design)

## 7. System Architecture
```text
User / Claim Data 
       ↓ 
Input Validation 
       ↓ 
Preprocessing Pipeline (Imputation, Scaling, Encoding)
       ↓ 
Trained ML Model (Random Forest)
       ↓ 
Fraud Probability / Class Prediction
       ↓ 
Risk Interpretation (Low / Medium / High Risk)
       ↓ 
Flask Web Application 
       ↓ 
User Result + Basic Explanation
```

## 8. Development Workflow
```text
Dataset
   ↓
EDA (Exploratory Data Analysis)
   ↓
Preprocessing (Imputation & Encoding)
   ↓
Feature Engineering / Dropping Identifiers
   ↓
Model Training (LR, DT, RF, GB)
   ↓
Evaluation (Accuracy, Precision, Recall, F1)
   ↓
Best Model Selection
   ↓
Flask Prototype Development
   ↓
Testing (TC01-TC05)
   ↓
GitHub Version Control
   ↓
Free Deployment
```

## 9. Folder Structure
```text
insurance-fraud-detection/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── insurance_fraud_analysis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── save_final_model.py
│   ├── explain_model.py
│   └── eda.py
├── model/
│   ├── model.pkl
│   ├── preprocessing.pkl
│   └── feature_names.pkl
├── app/
│   ├── app.py
│   ├── templates/
│   │   ├── index.html
│   │   └── result.html
│   └── static/
│       └── style.css
├── docs/
│   ├── screenshots/
│   ├── figures/
│   └── case-study/
├── tests/
│   └── test_app.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## 10. Installation
1. Clone the repository:
   ```bash
   git clone <YOUR_GITHUB_URL>
   cd insurance-fraud-detection
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 11. Running Locally
1. Start the Flask server:
   ```bash
   python app/app.py
   ```
2. Open your browser and navigate to: `http://127.0.0.1:5000`

## 12. Model Results
We evaluated 4 models. The **Random Forest** classifier was selected for its superior balance of Precision and Recall.

* **Accuracy:** 84.50%
* **Precision:** 65.00%
* **Recall:** 79.59%
* **F1-Score:** 71.56%
* **ROC-AUC:** 84.01%

*Note: Recall (79.5%) is prioritized to ensure the majority of actual fraud cases are flagged.*

## 13. Screenshots

### Home Page
![Home](docs/screenshots/home_page.png)

### Fraud Prediction
![Prediction](docs/screenshots/fraud_prediction.png)

### Model Insights
![Insights](docs/screenshots/model_insights.png)

*(Note: Add the screenshots to `docs/screenshots/` to display them here.)*

## 14. GitHub Repository
**URL:** [Pending Phase 11]

## 15. Live Demo
**URL:** [Pending Phase 12]

## 16. Limitations
* **Imbalanced Data:** While handled via class weights, extreme edge cases may still lean toward the majority class.
* **Feature Dependency:** The model relies heavily on self-reported inputs (like incident severity) which can be lied about by fraudsters.
* **Geographic Limits:** The training data is limited to a few states (OH, IN, IL). It may not generalize well to other regions.

## 17. Future Enhancements
* Incorporate image processing (Computer Vision) to analyze car damage photos automatically.
* Add NLP (Natural Language Processing) to analyze police report text.
* Integrate an API to pull real-time weather data at the time of the incident to verify road conditions.

## 18. Disclaimer
This is an academic project prototype. The prediction represents statistical risk modeling and is designed as a decision-support tool, not as absolute proof of fraud. It should never be used as the sole reason to deny a claim without human investigation.
