# Experiment 5: Breast Cancer Classification using Decision Tree and Random Forest

Student: S.S.Janane
Register Number: 3122247001024
Degree: M.Tech Integrated CSE(5 yrs)
Course: ICS1512 - Machine Learning Algorithms Laboratory

## Objective
To implement and compare Decision Tree and Random Forest classifiers on the Wisconsin Diagnostic Breast Cancer dataset, tune hyperparameters using 5-fold cross-validation with GridSearchCV, and analyze feature importance and overfitting through tree depth.

## Dataset
- Name: Wisconsin Diagnostic Breast Cancer (WDBC), loaded via sklearn.datasets.load_breast_cancer
- Instances: 569 samples
- Features: 30 numerical measurements describing cell nuclei
- Target: Diagnosis (Benign / Malignant), label-encoded

## Methodology
- Preprocessing: label encoding of diagnosis, stratified 80/20 train-test split.
- Exploratory Data Analysis: class distribution, feature correlation heatmap, top features correlated with the target.
- Decision Tree: baseline model, then tuned with GridSearchCV (5-fold) over criterion, max_depth, min_samples_split, and min_samples_leaf; visualization of the top levels of the best tree.
- Random Forest: tuned with GridSearchCV (5-fold) over n_estimators, max_depth, max_features, and bootstrap.
- Validation: fold-by-fold 5-fold cross-validation accuracy comparison between the two tuned models.
- Diagnostics: confusion matrices, ROC curves, classification reports, Random Forest feature importance ranking, and a training vs cross-validation accuracy plot across tree depths (1 to 15) to study overfitting.
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.

## Installation and Execution
Install dependencies:
```bash
pip install -r requirements.txt
```

Run the notebook:
```bash
jupyter notebook exp5jan.ipynb
```
