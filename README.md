# Auto Insurance Fraud Detection Engine

An end-to-end Machine Learning solution to assess the risk of auto insurance claims, featuring a trained Random Forest model served through a clean Flask web application.

🌐 **Live Web Application:** [https://insurance-fraud-detection-b8k5.onrender.com](https://insurance-fraud-detection-b8k5.onrender.com)

![App Screenshot](screenshot.png)


---

## 🚀 Key Features

* **Data-Driven Risk Profiling:** Leverages 1,000 auto insurance claims to model risk patterns.
* **Balanced Classification:** Random Forest Classifier trained with class weight balancing to maximize recall.
* **4-Step User Interface:** Intuitive, secure form to inputs claim, customer, and vehicle details.
* **Real-time Explanation:** Provides risk probability (Low, Medium, High) along with the top contributing features.
* **Robust Input Validation:** Complete client and server-side verification to handle missing or invalid data gracefully.

---

## 📊 Model Performance

Three classifiers were evaluated on a stratified 20% test split. The **Random Forest** model was chosen as the production model to optimize the balance between precision and recall.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 80.5% | 57.8% | 75.5% | 65.5% | 80.3% |
| **Random Forest** | **80.5%** | **58.6%** | **69.4%** | **63.6%** | **77.6%** |
| Decision Tree | 72.0% | 44.9% | 63.3% | 52.5% | 65.0% |

---

## 🛠️ Project Setup & Installation

### Prerequisites
* Python 3.11+
* Git

### Local Installation
```bash
# 1. Clone the repository
git clone https://github.com/charanemandi26-arch/Insurance-Fraud-Claims-Detection.git
cd Insurance-Fraud-Claims-Detection

# 2. Set up virtual environment
python -m venv .venv
.venv\Scripts\activate      # On Windows
# source .venv/bin/activate # On macOS/Linux

# 3. Install required packages
pip install -r requirements.txt
```

### Running the Web Application
```bash
python app/app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000`**

### Re-running Model Training
To retrain the model and export updated artifacts (`model.pkl`, `preprocessing.pkl`), run the complete pipeline:
```bash
python src/save_final_model.py
```
You can also run the full exploratory analysis directly in the Jupyter Notebook:
`notebooks/insurance_fraud_analysis.ipynb`

---

## 📁 Repository Structure

```
├── app/                  # Flask web application (backend logic, templates, CSS)
├── data/raw/             # Cleaned historical insurance claims dataset
├── model/                # Saved pipeline, model weights, and metadata artifacts
├── notebooks/            # Jupyter Notebook with full EDA and model training
├── src/                  # Helper scripts for preprocessing and model exporting
└── tests/                # Unit tests for Flask app integration
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
