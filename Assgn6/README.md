# Experiment 6: Dimensionality Reduction and Model Evaluation With and Without PCA on Breast Cancer Data

Student: S.S.Janane
Register Number: 3122247001024
Degree: M.Tech Integrated CSE(5 yrs)
Course: ICS1512 - Machine Learning Algorithms Laboratory

## Objective
To apply Principal Component Analysis for dimensionality reduction on the Wisconsin Diagnostic Breast Cancer dataset, and compare the performance of 10 classifiers with and without PCA, each tuned with GridSearchCV and evaluated using 5-fold cross-validation.

## Dataset
- Name: Wisconsin Diagnostic Breast Cancer (WDBC), loaded via sklearn.datasets.load_breast_cancer
- Instances: 569 samples
- Features: 30 numerical measurements describing cell nuclei (reduced via PCA for the With-PCA setting)
- Target: Diagnosis (0 = Malignant, 1 = Benign)

## Methodology
- Preprocessing: stratified 80/20 train-test split, StandardScaler feature scaling.
- Dimensionality Reduction: full PCA fit to determine explained variance, scree plot and cumulative variance plot, number of components selected to retain 95% variance (10 components out of 30).
- Classifiers Compared (No-PCA vs With-PCA): SVM, Gaussian Naive Bayes, KNN, Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost, and a Stacking ensemble (SVM + Random Forest + KNN base learners, Logistic Regression meta-learner).
- Hyperparameter Tuning: GridSearchCV (5-fold) applied separately to each classifier in both the original feature space and the PCA-reduced space.
- Validation: 5-fold cross-validation accuracy and F1-score, fold-wise spread, and standard deviation comparison between No-PCA and With-PCA settings for every classifier.
- Test Evaluation: confusion matrices and ROC curves for selected models (SVM, Random Forest, XGBoost, Stacking) in both settings.
- Metrics: Accuracy, F1-Score, ROC-AUC, Confusion Matrices, ROC Curves.

## Installation and Execution
Install dependencies:
```bash
pip install -r requirements.txt
```

Run the notebook:
```bash
jupyter notebook exp6jan.ipynb
```
