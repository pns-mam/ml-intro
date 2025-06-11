import os.path
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from skimage.filters import sobel

from sklearn.datasets import load_digits

from sklearn.base import BaseEstimator, TransformerMixin

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
## Functions for feature engineering
def extract_zone_features(images):
    zone_features = []
    for img_vector in images:
        img = img_vector.reshape((8,8))
        top = np.mean(img[:3, :])  # Top region
        middle = np.mean(img[3:5, :])  # Middle region
        bottom = np.mean(img[5:, :])  # Bottom region
        zone_features.append([top, middle, bottom])
    return np.array(zone_features)

class EdgeInfoPreprocessing(BaseEstimator, TransformerMixin):
    '''A class used to compute an average Sobel estimator on the image
       This class can be used in conjunction of other feature engineering
       using Pipelines or FeatureUnion
    '''
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self # No fitting needed for this processing
    
    def transform(self, X):
        sobel_feature = np.array([np.mean(sobel(img.reshape((8,8)))) for img in X]).reshape(-1, 1)
        return sobel_feature

class ZonalInfoPreprocessing(BaseEstimator, TransformerMixin):
    '''A class used to compute zone information on the image
       This class can be used in conjunction of other feature engineering
       using Pipelines or FeatureUnion

       TODO: Continue this work
    '''
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self # No fitting needed for this processing
    
    def transform(self, X):
        return extract_zone_features(X)
        # return X[:,1]

all_features = FeatureUnion([('pca', PCA(n_components=20)), ('zones', ZonalInfoPreprocessing()), ('sobel', EdgeInfoPreprocessing())])

clf = Pipeline([('classifier', DummyClassifier())])
clf = Pipeline([('prescale', MinMaxScaler()), ('features', all_features), ('postscale', StandardScaler()), ('classifier', SVC(kernel='linear'))])


# The next lines shall not be modified
clf.fit(X_train, y_train)
print(f"Score on the test set {clf.score(X_test, y_test)}")