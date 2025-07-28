import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
from utils.cost_utils import evaluate_cost
import joblib

# Load data
df = pd.read_csv("data/application_train.csv")

# Target
TARGET = "TARGET"

# Drop irrelevant or high-missing columns
drop_cols = [col for col in df.columns if df[col].isnull().mean() > 0.4]
df.drop(columns=drop_cols, inplace=True)

# Impute missing values
for col in df.select_dtypes(include='number'):
    df[col].fillna(df[col].median(), inplace=True)
for col in df.select_dtypes(include='object'):
    df[col].fillna(df[col].mode()[0], inplace=True)

# Label encode categorical
df = pd.get_dummies(df)

# Feature-target split
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------
# Logistic Regression
lr = LogisticRegression(class_weight='balanced', max_iter=1000)
lr.fit(X_train_scaled, y_train)
probs_lr = lr.predict_proba(X_test_scaled)[:, 1]

# --------------------------
# CatBoost
cat_model = CatBoostClassifier(verbose=0, scale_pos_weight=10)
cat_model.fit(X_train, y_train)
probs_cat = cat_model.predict_proba(X_test)[:, 1]

# --------------------------
# Threshold Optimization
thresholds = [i/100 for i in range(1, 100)]
results = []

for t in thresholds:
    cost_lr = evaluate_cost(probs_lr, y_test, t, fp_cost=5000, fn_cost=100000)
    cost_cat = evaluate_cost(probs_cat, y_test, t, fp_cost=5000, fn_cost=100000)
    results.append((t, cost_lr, cost_cat))

# Find optimal thresholds
best_lr = min(results, key=lambda x: x[1])
best_cat = min(results, key=lambda x: x[2])

print(f"Best LR Threshold: {best_lr[0]} | Total Cost: {best_lr[1]:,.0f}")
print(f"Best CatBoost Threshold: {best_cat[0]} | Total Cost: {best_cat[2]:,.0f}")

# Save best model
joblib.dump(cat_model, "models/final_model_catboost.pkl")
