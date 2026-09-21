# Experiment 1: Generalized Machine Learning Pipeline with Automatic Task Detection and Feature Selection

Student: S.S.Janane
Register Number: 3122247001024
Degree: M.Tech Integrated CSE(5 yrs)
Course: ICS1512 - Machine Learning Algorithms Laboratory

## Objective
To build a generalized machine learning pipeline that automatically loads and cleans a dataset, detects whether the task is classification or regression from the target column, performs feature selection, and trains and evaluates multiple candidate models, demonstrated on a health outcome prediction dataset.

## Dataset
- Name: Diabetes Prediction Dataset
- File: diabetes_prediction_dataset.csv
- Target: diabetes (binary classification, 0 = No, 1 = Yes)
- Features: patient health attributes such as age, BMI, blood glucose level, HbA1c level, hypertension, heart disease, gender, and smoking history

## Methodology
- Preprocessing: automatic loading, duplicate removal, empty row/column removal, missing value symbol normalization, whitespace stripping on text columns.
- Exploratory Data Analysis: shape, dtypes, missing value summary, duplicate check, unique value counts, summary statistics, missing value heatmap, correlation analysis.
- Task Detection: automatically classifies the target column as classification or regression based on data type and number of unique values.
- Feature Selection: SelectKBest with chi2, f_classif (ANOVA), or f_regression scoring functions, depending on the detected task.
- Train/Validation/Test Split: stratified split for classification tasks.
- Model Training: Logistic Regression, Decision Tree, Random Forest, KNN, SVM, and Gaussian Naive Bayes for classification tasks (Linear Regression and related models for regression tasks).
- Metrics: Accuracy, Precision, Recall, F1-Score for classification; MAE, MSE, R2 for regression.

## Installation and Execution
Install dependencies:
```bash
pip install -r requirements.txt
```

Run the notebook:
```bash
jupyter notebook exp1jan.ipynb
```
