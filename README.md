# Classification Model Picker

A Python utility class that runs six classification models on any dataset and compares their accuracy in one call.

## What It Does

Pass in your train/test data and it runs all six classifiers, returns the trained model, confusion matrix, and accuracy score for each. Or call `basic_classifier_model_picker()` to run all six at once and see which performs best.

## Models Included

- Decision Tree
- Random Forest
- Naive Bayes
- K-Nearest Neighbors
- Logistic Regression
- Support Vector Machine (SVM)

## Usage

```python
from classification_model_picker import ClassificationModelPicker

picker = ClassificationModelPicker()

# run all models at once
picker.basic_classifier_model_picker(X_train, X_test, y_train, y_test)

# or run individually
model, cm, accuracy = picker.random_forest(X_train, X_test, y_train, y_test, n_estimators=100)
```

## Parameters

Each model has sensible defaults but you can customise:
- `decision_tree(criterion="entropy")`
- `random_forest(criterion="entropy", n_estimators=10)`
- `k_nearest(n_neighbors=10, p=2)`
- `svm(kernel="rbf")`

## Requirements
```
pip install numpy scikit-learn
```
