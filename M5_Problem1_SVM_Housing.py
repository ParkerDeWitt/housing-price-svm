# Step 1 - Import the required packages

import numpy as np
from sklearn import datasets, preprocessing
from sklearn.svm import SVR
from sklearn.utils import shuffle
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    mean_squared_error,
    explained_variance_score,
)
import matplotlib.pyplot as plt



# Step 2 - Load the housing dataset and shuffle the data
#          (shuffling removes any ordering bias before we split)
#          NOTE!!! my version of python doesnt have the boston dataset. 
#          I am not going to regress my python version, so i found the dataset and load it from csv

def load_boston_dataset():
 
    try:
        # Works on scikit-learn < 1.2 (the version the assignment assumes)
        boston = datasets.load_boston()
        return boston.data, boston.target
    except (ImportError, AttributeError):
        # scikit-learn >= 1.2: load_boston was removed.
        # Fallback A: identical data from a local CSV shipped with this script
        # (columns in the same order as sklearn's load_boston).
        import os
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "BostonHousing.csv")
        if os.path.exists(csv_path):
            raw = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
            return raw[:, :13], raw[:, 13]
        # Fallback B: rebuild it from the original public source.
        raw = np.loadtxt("http://lib.stat.cmu.edu/datasets/boston", skiprows=22)
        data = np.hstack([raw[::2, :], raw[1::2, :2]])
        target = raw[1::2, 2]
        return data, target


data_X, data_y = load_boston_dataset()

# Shuffle so we don't bias the analysis (fixed random_state for reproducibility)
X, y = shuffle(data_X, data_y, random_state=7)



# Step 3 - Split the dataset into training and testing sets (80/20)

num_training = int(0.8 * len(X))          # 80% of the data for training
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]



# Step 4 - Create and train the Support Vector Regressor (linear kernel)
#          C       : penalty on the error term
#          epsilon : width of the no-penalty tube around the prediction

sv_regressor = SVR(kernel='linear', C=1.0, epsilon=0.1)
sv_regressor.fit(X_train, y_train)



# Step 5 - Run the regressor on the testing data (predicted values)

y_test_pred = sv_regressor.predict(X_test)



# Step 6 - Evaluate the regressor: Mean Squared Error & Explained Variance

mse = mean_squared_error(y_test, y_test_pred)
evs = explained_variance_score(y_test, y_test_pred)

print("#### Support Vector Regressor performance ####")
print("Mean squared error       =", round(mse, 2))
print("Explained variance score =", round(evs, 2))
print()



# Step 7 - Binarize predicted and actual values (threshold 25.00)
#          Prices >= 25.00 -> label 1 ("expensive")
#          Prices <  25.00 -> label 0 ("affordable")
#          We binarize the actual TEST prices (y_test) so the two label arrays
#          line up 1-to-1 with the predictions for the confusion matrix.

threshold = 25.00
binarizer = preprocessing.Binarizer(threshold=threshold)

y_pred_label = binarizer.transform(y_test_pred.reshape(-1, 1)).ravel().astype(int)
y_test_label = binarizer.transform(y_test.reshape(-1, 1)).ravel().astype(int)



# Step 8 - Build the confusion matrix (true labels vs. predicted labels)

confusion_mat = confusion_matrix(y_test_label, y_pred_label)
print("#### Confusion matrix ####")
print(confusion_mat)
print()



# Step 9 - Visualize the confusion matrix

plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Confusion matrix')
plt.colorbar()
ticks = np.arange(2)
plt.xticks(ticks, ticks)
plt.yticks(ticks, ticks)
plt.ylabel('True labels')
plt.xlabel('Predicted labels')
plt.savefig('confusion_matrix.png', dpi=120, bbox_inches='tight')
plt.show()



# Step 10 - Print the classification report based on the confusion matrix

target_names = ['Class-0 (< 25.00)', 'Class-1 (>= 25.00)']
print("#### Classification report ####")
print(classification_report(y_test_label, y_pred_label, target_names=target_names))
