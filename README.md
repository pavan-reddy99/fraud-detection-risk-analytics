# fraud-detection-risk-analytics
ML pipeline detecting fraudulent transactions using PySpark &amp; Scikit-learn with Power BI risk dashboard


![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-E25A1C?style=flat-square&logo=apachespark&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Power BI](https://img.shields.io/badge/Power%20BI-F2C811?style=flat-square&logo=powerbi&logoColor=black)
![XGBoost](https://img.shields.io/badge/XGBoost-189B34?style=flat-square&logo=xgboost&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

> An end-to-end ML classification pipeline that detects fraudulent transactions from engineered behavioral and transactional features — surfacing ranked risk scores in a Power BI dashboard with daily account-level review queues.

---

## 📌 Problem Statement

Manual fraud review is slow, inconsistent, and unscalable. Analysts spend hours triaging thousands of accounts with no systematic prioritization — high-risk accounts get buried, and low-risk ones consume valuable time.

This system **automates the triage process**: every account is scored daily, ranked by fraud risk, and surfaced in a Power BI dashboard so risk teams focus exclusively on the highest-risk cases first.

---

## 🏗️ Architecture

```
Raw Transaction Data (CSV / DB)
        │
        ▼
PySpark Feature Engineering
  ├── Behavioral features (velocity, frequency, time patterns)
  ├── Transactional features (amounts, merchant categories, deviations)
  └── Account-level aggregations
        │
        ▼
Scikit-learn Model Training & Evaluation
  ├── Random Forest
  ├── XGBoost  ◄── Selected (best precision-recall trade-off)
  └── Logistic Regression (baseline)
        │
        ▼
Risk Scoring Pipeline
  └── Ranked risk scores per account
        │
        ▼
Power BI Dashboard
  └── Daily review queue with account-level risk tiers
```

---

## ✨ Features

- **Large-scale feature engineering** using PySpark for distributed processing of transaction datasets
- **Multi-model evaluation** with Random Forest, XGBoost, and Logistic Regression — final model selected based on precision-recall trade-off analysis
- **Threshold tuning** to balance fraud catch rate vs. false positive rate for the risk team's tolerance
- **Ranked risk scoring** output that feeds a Power BI dashboard with tiered review queues (High / Medium / Low)
- **Evaluation suite** including confusion matrices, ROC-AUC curves, classification reports, and feature importance plots

---

## 📁 Project Structure

```
fraud-detection-risk-analytics/
├── data/
│   ├── sample_transactions.csv       # Sample input data (anonymized)
│   └── feature_store/                # Engineered features output
├── src/
│   ├── feature_engineering.py        # PySpark feature generation
│   ├── model_train.py                # Model training & evaluation
│   ├── scoring_pipeline.py           # Daily risk scoring pipeline
│   └── utils.py                      # Shared helpers
├── notebooks/
│   └── eda_and_modeling.ipynb        # Exploratory analysis & model selection
├── dashboard/
│   └── fraud_risk_dashboard.pbix     # Power BI dashboard file
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | PySpark, Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Evaluation | Matplotlib, Seaborn |
| Visualization | Power BI |
| Language | Python 3.9+ |

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/pavan-reddy99/fraud-detection-risk-analytics.git
cd fraud-detection-risk-analytics
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run feature engineering**
```bash
python src/feature_engineering.py --input data/sample_transactions.csv --output data/feature_store/
```

**4. Train and evaluate the model**
```bash
python src/model_train.py --features data/feature_store/ --output models/
```

**5. Run the scoring pipeline**
```bash
python src/scoring_pipeline.py --model models/xgb_fraud_model.pkl --output data/risk_scores.csv
```

**6. Load `data/risk_scores.csv` into Power BI** using the included `.pbix` template

---

## 📊 Results

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.71 | 0.64 | 0.67 | 0.82 |
| Random Forest | 0.83 | 0.76 | 0.79 | 0.91 |
| **XGBoost** | **0.88** | **0.81** | **0.84** | **0.94** |

> XGBoost selected as final model. Threshold tuned to 0.42 to optimize recall for the risk team's tolerance of ~12% false positive rate.

---

## 📬 Contact

**Pavan Reddy** · pavanhere99@gmail.com · Saint Louis, MO
