# 💳 Loan Default Risk Prediction with Business Cost Optimization

## 📌 Objective
Build a binary classification model to predict the likelihood of loan default and adjust the decision threshold to minimize overall business loss using a cost-sensitive evaluation approach.

---

## 📂 Dataset
**Home Credit Default Risk Dataset**  
- Source: Kaggle
- Main Features:
  - Demographics (age, gender, education)
  - Financial info (income, credit amount)
  - Loan status (target: 0 = no default, 1 = default)

---

## 🚀 Project Workflow

### ✅ 1. Data Cleaning & Preprocessing
- Imputed missing values
- Encoded categorical variables
- Normalized numeric features
- Handled class imbalance (optional: SMOTE or class weighting)

### ✅ 2. Model Training
- 🔹 Logistic Regression (baseline)
- 🔹 CatBoostClassifier (high-performance gradient boosting)

### ✅ 3. Cost-Benefit Analysis
- Defined:
  - False Positive cost: $500 (good client rejected)
  - False Negative cost: $5000 (bad client accepted)
- Calculated business cost for various threshold values

### ✅ 4. Threshold Optimization
- Found threshold that **minimizes total business cost**
- Compared it to default (0.5) threshold performance

### ✅ 5. Feature Importance
- Visualized most predictive features using CatBoost

---

## 📊 Evaluation Metrics
- Accuracy, Precision, Recall, F1
- Confusion Matrix
- Cost-based evaluation using business loss matrix
- ROC AUC score

---

## 📈 Visualizations
- Cost vs Threshold Curve
- ROC Curve
- Feature Importance Bar Chart

---

## 🧰 Tools & Libraries
- `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`
- `Scikit-learn`
- `CatBoost`
- `Imbalanced-learn` (optional)
- `Joblib` for model saving

---

## 💡 Skills Gained
- Binary classification with cost-sensitive decisions
- Business cost modeling
- Threshold tuning for profit optimization
- Feature importance analysis

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/loan-default-risk-optimization.git
cd loan-default-risk-optimization
