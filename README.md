# Housing Price Prediction with Support Vector Regression

## Overview

This project builds a Support Vector Machine, used as a regressor (SVR), to predict housing prices from 13 features such as crime rate, number of rooms, and tax rate. After training, prediction quality is evaluated with mean squared error and explained variance. The regression output is then converted into a binary classification problem by splitting prices at a threshold of 25.00, enabling a confusion matrix and classification report to measure how well the model separates cheaper homes from more expensive ones.

## Tech Stack

Python, NumPy, scikit-learn, Matplotlib

## How It Works

The dataset (506 samples) is shuffled with a fixed random seed for reproducibility and split 80/20 into training and test sets (404 / 102 samples). An SVR model with a linear kernel (`C=1.0`, `epsilon=0.1`) is trained on the training set and evaluated on the test set using mean squared error and explained variance.

To frame the problem as classification, both the predicted and actual test prices are binarized at a threshold of 25.00 (1 = at or above, 0 = below). This produces two aligned label lists — predicted and actual — used to generate a confusion matrix (plotted in grayscale) and a classification report.

**Compatibility fix:** the original assignment used `sklearn.datasets.load_boston()`, which was removed in newer versions of scikit-learn. The script tries `load_boston()` first and automatically falls back to a bundled `BostonHousing.csv` if it's unavailable, so it runs regardless of scikit-learn version.

## Results

```
Mean squared error       = 15.38
Explained variance score = 0.82

Confusion matrix:
[[69 10]
 [ 4 19]]

                    precision    recall  f1-score   support
 Class-0 (< 25.00)       0.95      0.87      0.91        79
Class-1 (>= 25.00)       0.66      0.83      0.73        23
          accuracy                           0.86       102
```

The model explains about 82% of the variance in housing prices with a linear kernel. On the classification side, it reaches 86% overall accuracy — very strong at identifying affordable homes (0.95 precision) and reasonably good at catching expensive ones, though it occasionally over-predicts "expensive" (0.66 precision on that class).

## Running It

```
pip install numpy scikit-learn matplotlib
python M5_Problem1_SVM_Housing.py
```
