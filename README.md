# Heart Disease Prediction using Machine Learning

## Overview
This project builds a **robust, leakage-free machine learning pipeline** to predict heart disease risk using clinical data.  
Multiple classification models are evaluated using **proper preprocessing, cross-validation, and pipeline-based design**.

---

## Key ML Practices Used
- ✅ Train–test split before preprocessing
- ✅ Scikit-learn Pipelines to prevent data leakage
- ✅ Stratified cross-validation
- ✅ Multi-model comparison
- ✅ Recall & F1-score prioritization (healthcare-safe)
- ✅ Automatic best-model selection

---

## Models Evaluated
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest (regularized)
- XGBoost (regularized)

---

## Evaluation Metrics
- Cross-validation Accuracy
- Train Accuracy
- Test Accuracy
- Recall
- F1-score
- Confusion Matrix

> In healthcare problems, recall and generalization are prioritized over raw accuracy.

---

## Dataset
Heart Disease UCI Dataset  
Source: https://github.com/sharmaroshan/Heart-UCI-Dataset

---

## How to Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/heart-disease-prediction-ml.git
cd heart-disease-prediction-ml

