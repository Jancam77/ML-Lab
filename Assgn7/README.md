# Experiment 7: Binary Classification on Breast Cancer using Bagging, Boosting and Stacking

Student: S.S.Janane
Register Number: 3122247001024
Degree: M.Tech Integrated CSE(5 yrs)
Course: ICS1512 - Machine Learning Algorithms Laboratory

## Objective
To implement and compare Bagging, Boosting (AdaBoost and Gradient Boosting), and Stacking ensemble methods on the Wisconsin Diagnostic Breast Cancer dataset, evaluating hyperparameter tuning with 5-fold cross-validation, feature importance, ROC analysis, and bias-variance trade-offs.

## Dataset
- Name: Wisconsin Diagnostic Breast Cancer (WDBC), loaded via sklearn.datasets.load_breast_cancer
- Instances: 569 samples (357 Benign, 212 Malignant)
- Features: 30 numerical measurements describing cell nuclei
- Target: Binary class (0 = Malignant, 1 = Benign)

## Methodology
- Preprocessing: stratified 80/20 train-test split, StandardScaler feature scaling.
- Exploratory Data Analysis: class distribution, feature correlation heatmap, boxplots of key features by diagnosis.
- Bagging: Decision Tree base estimator, GridSearchCV (5-fold) tuning n_estimators, max_samples, and max_features.
- Boosting: AdaBoost and Gradient Boosting, each tuned with GridSearchCV (5-fold) over n_estimators, learning_rate, and (for Gradient Boosting) max_depth.
- Stacking: heterogeneous ensemble combining Support Vector Classifier (SVM), Gaussian Naive Bayes, and Decision Tree base learners with a Logistic Regression meta-learner, evaluated via 5-fold cross-validation.
- Validation: 5-fold cross-validation accuracy and F1-score for all three ensemble approaches, held-out test set evaluation.
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrices, ROC Curves.

## Installation and Execution
Install dependencies:
```bash
pip install -r requirements.txt
```

Run the notebook:
```bash
jupyter notebook exp7jan.ipynb
```
