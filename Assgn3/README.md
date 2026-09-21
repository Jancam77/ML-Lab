# Experiment 3: Loan Amount Prediction using Linear and Regularized Regression Models

Student: S.S.Janane
Register Number: 3122247001024
Degree: M.Tech Integrated CSE(5 yrs)
Course: ICS1512 - Machine Learning Algorithms Laboratory

## Objective
To implement and compare Linear Regression, Ridge, Lasso, and Elastic Net regression models for predicting loan amount, tune regularization strength using GridSearchCV, and analyze overfitting, learning curves, and feature coefficients.

## Dataset
- Name: Loan Prediction Dataset
- File: train_loan.csv
- Target: LoanAmount (continuous)
- Features: numerical fields (such as applicant income, coapplicant income, loan term, credit history) and categorical fields (such as gender, marital status, education, property area), after dropping the Loan_ID identifier and the Loan_Status classification column

## Methodology
- Preprocessing: missing value imputation (median for numeric, most frequent for categorical), one-hot encoding for categorical features, standard scaling for numeric features, all combined in a ColumnTransformer and Pipeline.
- Train/Test Split: 80/20 split.
- Exploratory Data Analysis: target distribution, feature vs target scatter plots, correlation heatmap.
- Baseline Model: Linear Regression.
- Regularized Models: Ridge, Lasso, and Elastic Net, each tuned with GridSearchCV (5-fold) over alpha, and l1_ratio for Elastic Net.
- Validation: 5-fold cross-validation comparing MAE, MSE, RMSE, and R2 across all four models.
- Diagnostics: predicted vs actual plot, residual plot, learning curves (training vs validation error), coefficient comparison across models, train-test R2 gap analysis for overfitting.
- Metrics: MAE, MSE, RMSE, R2 Score.

## Installation and Execution
Install dependencies:
```bash
pip install -r requirements.txt
```

Run the notebook:
```bash
jupyter notebook exp3jan.ipynb
```
