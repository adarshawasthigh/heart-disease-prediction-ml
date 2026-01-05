import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, recall_score, f1_score, ConfusionMatrixDisplay

# -------------------------------------------------
# 1. LOAD & CLEAN DATA
# -------------------------------------------------
url = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
df = pd.read_csv(url)
df = df.apply(pd.to_numeric, errors="coerce").dropna().drop_duplicates()

# -------------------------------------------------
# 2. FEATURE CORRELATION
# -------------------------------------------------
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="RdYlGn", fmt=".2f")
plt.title("Clinical Feature Correlation Matrix")
plt.show()

# -------------------------------------------------
# 3. OUTLIER REMOVAL (IQR)
# -------------------------------------------------
initial_rows = len(df)
for col in df.columns[:-1]:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

print(f"Removed {initial_rows - len(df)} outliers\n")

# -------------------------------------------------
# 4. SPLIT DATA (NO SCALING YET)
# -------------------------------------------------
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# -------------------------------------------------
# 5. PIPELINES FOR ALL MODELS
# -------------------------------------------------
pipelines = {
    "Logistic Regression": Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "KNN": Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", KNeighborsClassifier())
    ]),

    "SVM": Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", SVC(probability=True))
    ]),

    "Random Forest": Pipeline([
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=5,
            random_state=42
        ))
    ]),

    "XGBoost": Pipeline([
        ("model", XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        ))
    ])
}

# -------------------------------------------------
# 6. TRAIN, CROSS-VALIDATE & EVALUATE
# -------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"{'Model':<20} | {'CV Acc':<8} | {'Train Acc':<10} | {'Test Acc':<9} | {'Recall':<8} | {'F1':<6}")
print("-" * 80)

results = []

for name, pipe in pipelines.items():
    # Cross-validation (NO leakage)
    cv_acc = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="accuracy").mean()

    # Train
    pipe.fit(X_train, y_train)

    # Predictions
    y_train_pred = pipe.predict(X_train)
    y_test_pred = pipe.predict(X_test)

    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    print(f"{name:<20} | {cv_acc:.2%} | {train_acc:.2%} | {test_acc:.2%} | {recall:.2%} | {f1:.2%}")

    results.append({
        "Model": name,
        "CV Accuracy": cv_acc,
        "Train Accuracy": train_acc,
        "Test Accuracy": test_acc,
        "Recall": recall,
        "F1": f1
    })

# -------------------------------------------------
# 7. AUTO-SELECT BEST MODEL (BY TEST ACC)
# -------------------------------------------------
best_model_name = max(results, key=lambda x: x["Test Accuracy"])["Model"]
best_pipeline = pipelines[best_model_name]

print(f"\nBest Model Selected: {best_model_name}")

# -------------------------------------------------
# 8. CONFUSION MATRIX (BEST MODEL)
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    best_pipeline.predict(X_test),
    display_labels=["Low Risk", "High Risk"],
    cmap="Reds",
    ax=ax
)
plt.title(f"Confusion Matrix – {best_model_name}")
plt.show()

