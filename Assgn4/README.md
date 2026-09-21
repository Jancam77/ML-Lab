# Experiment 4: Binary Classification of Spam Email using Logistic Regression and SVM with Hyperparameter Tuning

Student: S.S.Janane
Register Number: 3122247001024
Degree: M.Tech Integrated CSE(5 yrs)
Course: ICS1512 - Machine Learning Algorithms Laboratory

## Objective
To implement and compare Logistic Regression and Support Vector Machine classifiers for spam email detection, tune hyperparameters using GridSearchCV and RandomizedSearchCV, and evaluate the effect of SVM kernel choice and cross-validation.

## Dataset
- Name: Spambase Dataset
- File: spambase.csv
- Instances: 4601 emails
- Features: 57 word/character frequency features plus 3 capital run length features
- Target: class (0 = Ham, 1 = Spam)

## Methodology
- Preprocessing: missing value and duplicate checks, StandardScaler feature scaling, stratified 80/20 train-test split.
- Exploratory Data Analysis: class balance, top features correlated with spam, boxplots of key features by class.
- Logistic Regression: baseline model, then tuned using GridSearchCV and RandomizedSearchCV over penalty (l1/l2), C, and solver (liblinear/saga), with the better of the two searches selected.
- SVM: kernel comparison (linear, poly, rbf, sigmoid) as a baseline, then tuned using GridSearchCV over C, gamma, kernel, and degree.
- Validation: 5-fold Stratified Cross-Validation comparing the tuned Logistic Regression and tuned SVM.
- Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrices, Classification Reports.

## Installation and Execution
Install dependencies:
```bash
pip install -r requirements.txt
```

Run the notebook:
```bash
jupyter notebook exp4jan.ipynb
```
