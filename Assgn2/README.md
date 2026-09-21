# Experiment 2: Email Spam Classification using Naive Bayes and KNN with Hyperparameter Tuning

Student: S.S.Janane
Register Number: 3122247001024
Degree: M.Tech Integrated CSE(5 yrs)
Course: ICS1512 - Machine Learning Algorithms Laboratory

## Objective
To implement and compare Gaussian, Multinomial, and Bernoulli Naive Bayes with K-Nearest Neighbors for spam email classification, tune KNN using GridSearchCV and RandomizedSearchCV, and analyze the effect of cross-validation, distance metrics, and train-test split ratios.

## Dataset
- Name: Spambase Dataset
- File: spambase.csv
- Instances: 4601 emails
- Features: 57 word/character frequency features plus 3 capital run length features
- Target: class (0 = Ham, 1 = Spam)

## Methodology
- Preprocessing: automatic loading, duplicate removal, missing value handling, feature scaling (StandardScaler for Gaussian NB and KNN, MinMaxScaler for Multinomial NB, Binarizer for Bernoulli NB).
- Train/Validation/Test Split: stratified 70/10/20 split.
- Baseline Models: Gaussian NB, Multinomial NB, Bernoulli NB, and KNN (k=5), each evaluated on validation and test sets.
- KNN Analysis: effect of varying k (1 to 11) on train vs validation accuracy.
- Hyperparameter Tuning: GridSearchCV and RandomizedSearchCV over n_neighbors, weights, algorithm, and distance metric for KNN, compared on best score and search time.
- KDTree vs BallTree: training and prediction time comparison for the tuned KNN.
- Validation: 5-fold Stratified Cross-Validation for the best Naive Bayes variant and tuned KNN.
- Additional Analysis: timing comparison across all models, train-test split ratio sensitivity (60/40 to 90/10), distance metric comparison, weighting scheme comparison.
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrices, ROC Curves, Precision-Recall Curves.

## Installation and Execution
Install dependencies:
```bash
pip install -r requirements.txt
```

Run the notebook:
```bash
jupyter notebook exp2jan.ipynb
```
