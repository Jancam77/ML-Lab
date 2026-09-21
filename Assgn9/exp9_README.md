# Experiment 9: Perceptron vs Multilayer Perceptron (A/B Experiment) with Hyperparameter Tuning

Student: S.S.Janane
Register Number: 3122247001024
Degree: M.Tech Integrated CSE(5 yrs)
Course: ICS1512 - Machine Learning Algorithms Laboratory

## Objective
To implement a Single Layer Perceptron (PLA) from scratch and compare it against a tuned Multilayer Perceptron (MLP) on a handwritten character recognition task, selecting MLP hyperparameters such as activation function, optimizer, learning rate, hidden layer sizes, and batch size through grid search.

## Dataset
- Name: English Handwritten Characters Dataset
- Instances: 3410 images (55 images per class)
- Classes: 62 (digits 0-9, uppercase A-Z, lowercase a-z)
- Preprocessing: images resized to 32x32, converted to grayscale, flattened to 1024 features, and normalized to the 0-1 range

## Methodology
- Preprocessing: grayscale conversion, resizing, flattening, normalization, label encoding, stratified 80/20 train-test split.
- Model A (PLA): implemented from scratch using a step activation function and the perceptron weight update rule, extended to multiclass via a one-vs-rest scheme; training error tracked across epochs.
- Model B (MLP): tuned using GridSearchCV (3-fold) over hidden layer sizes, activation function (ReLU, Tanh), optimizer (SGD, Adam), learning rate, and batch size; trained with early stopping and the best configuration selected by cross-validation accuracy.
- Evaluation: Accuracy, Precision, Recall, F1-Score, Confusion Matrices, and micro/macro-averaged ROC curves for both models.
- Convergence Analysis: PLA training error vs epochs, MLP training loss curve.
- A/B Comparison: side-by-side metric and ROC AUC comparison highlighting the impact of hyperparameter tuning and the effect of hidden layers, optimizer choice, and possible overfitting.

## Installation and Execution
Install dependencies:
```bash
pip install -r requirements.txt
```

Run the notebook:
```bash
jupyter notebook exp9.ipynb
```
