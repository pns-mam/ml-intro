import os.path
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.datasets import load_digits

# TODO: Add necessary imports here



# The lines below shall not be modified!

# The following will be replaced by our own 
if os.path.isfile("test_data.npy"):
    X_test = np.load("test_data.npy")
    y_test = np.load("test_labels.npy")
    X_train, y_train = load_digits(return_X_y=True)
else:
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# TODO: Prepare your learning pipeline and set all the parameters for your final algorithm.

clf = Pipeline([('classifier', DummyClassifier())])


# The next lines shall not be modified
clf.fit(X_train, y_train)
print(f"Score on the test set {clf.score(X_test, y_test)}")