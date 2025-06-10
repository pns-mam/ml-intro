# Split the data 
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.multiclass import  OneVsOneClassifier,OneVsRestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
# Load processed feature matrix and labels
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from skimage.filters import sobel
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


# TODO: Add any util functions you may have from the previous script


# TODO: Load the raw data
X,y = None, None

#####
#In machine learning, we must train the model on one subset of data and test it on another.
#This prevents the model from memorizing the data and instead helps it generalize to unseen examples.
#The dataset is typically divided into:
#Training set → Used for model learning.
#Testing set → Used for evaluating model accuracy.
# The training set is also split as a training set and validation set for hyper-parameter tunning. This is done later
#
# Split dataset into training & testing sets


##########################################
## Train/test split and distributions
##########################################


# 1- Split dataset into training & testing sets
# TODO: FILL OUT THE CORRECT SPLITTING HERE
X_train, X_test, y_train, y_test = X[0:-100,:], X[-100:,:], y[0:-100], y[-100:]
### If you want, you could save the data, this would be a good way to test your final script in the same evaluation mode as what we will be doing
# np.save("X_train.npy", X_train)
# np.save("test_data.npy", X_test)
# np.save("y_train.npy", y_train)
# np.save("test_label.npy", y_test)
####

# TODO: Print dataset split summary...


# TODO: ... and plot graphs of the three distributions in a readable and useful manner (bar graph, either side by side, or with some transparancy)
plt.figure()
plt.show()


# TODO: (once the learning has started, and to be documented in your report) - Impact: Changing test_size affects model training & evaluation.


##########################################
## Prepare preprocessing pipeline
##########################################

# We are trying to combine some global features fitted from the training set
# together with some hand-computed features.
# 
# The PCA shall not be fitted using the test set. 
# The handmade features are computed independently from the PCA
# We therefore need to concatenate the PCA computed features with the zonal and 
# edge features. 
# This is done with the FeatureUnion class of sklearn and then combining everything in
# a Pipeline.
# 
# All elements included in the FeatureUnion and Pipeline shall have at the very least a
# .fit and .transform method. 
#
# Check this documentation to understand how to work with these things 
# https://scikit-learn.org/stable/auto_examples/compose/plot_feature_union.html#sphx-glr-auto-examples-compose-plot-feature-union-py

# Example of wrapper for adding a new feature to the feature matrix
from sklearn.base import BaseEstimator, TransformerMixin

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

# TODO: Fill out the useful code for this class
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
        return X[:,1]

# TODO: Create a single sklearn object handling the computation of all features in parallel
all_features = None

F = all_features.fit(X_train,y).transform(X_train)
# Let's make sure we have the number of dimensions that we expect!
print("Nb features computed: ", F.shape[1])

# Now combine everything in a Pipeline
# The clf variable is the one which plays the role of the learning algorithms
# The Pipeline simply allows to include the data preparation step into it, to 
# avoid forgetting a scaling, or a feature, or ...
# 
# TODO: Write your own pipeline, with a linear SVC classifier as the prediction
clf = Pipeline([('classifier', DummyClassifier())])

##########################################
## Premier entrainement d'un SVC
##########################################

# TODO: Train your model via the pipeline

# TODO: Predict the outcome of the learned algorithm on the train set and then on the test set 
predict_test = [1]*len(X_test)
predict_train = [1]*len(X_train)

print("Accuracy of the SVC on the test set: ", sum(y_test==predict_test)/len(y_test))
print("Accuracy of the SVC on the train set: ", sum(y_train==predict_train)/len(y_train))

# TODO: Look at confusion matrices from sklearn.metrics and 
# 1. Display a print of it
# 2. Display a nice figure of it
# 3. Report on how you understand the results


# TODO: Work out the following questions (you may also use the score function from the classifier)
print("\n Question: How does changing test_size influence accuracy?")
print("Try different values like 0.1, 0.3, etc., and compare results.\n")


##########################################
## Hyper parameter tuning and CV
##########################################
# TODO: Change from the linear classifier to an rbf kernel
# TODO: List all interesting parameters you may want to adapt from your preprocessing and algorithm pipeline
# TODO: Create a dictionary with all the parameters to be adapted and the ranges to be tested

# TODO: Use a GridSearchCV on 5 folds to optimize the hyper parameters
grid_search = GridSearchCV(DummyClassifier()) #, verbose=10)
# TODO: fit the grid search CV and 
# 1. Check the results
# 2. Update the original pipeline (or create a new one) with all the optimized hyper parameters
# 3. Retrain on the whol train set, and evaluate on the test set
# 4. Answer the questions below and report on your findings

print(" K-Fold Cross-Validation Results:")
print(f"- Best Cross-validation score: {grid_search.best_score_}")
print(f"- Best parameters found: {grid_search.best_estimator_}")
#####
print("\n Question: What happens if we change K from 5 to 10?")
print("Test different K values and compare the accuracy variation.\n")


##########################################
## OvO and OvR
##########################################
# TODO: Using the best found classifier, analyse the impact of one vs one versus one vs all strategies
# Analyse in terms of time performance and accuracy


# Print OvO results
print(" One-vs-One (OvO) Classification:")
print(f"- Test score: {clf.score(X_test, y_test)}")
print(f"- Number of classifiers trained: {len(clf.get_params('classifier__estimators_'))}")
print("- Impact: Suitable for small datasets but increases complexity.")

print("\n Question: How does OvO compare to OvR in execution time?")
print("Try timing both methods and analyzing efficiency.\n")
###################
# TODO:  One-vs-Rest (OvR) Classification


# Print OvR results
print(" One-vs-Rest (OvR) Classification:")
print(f"- Test score: {clf.score(X_test, y_test)}")
print(f"- Number of classifiers trained: {len(clf.get_params('classifier__estimators_'))}")
print("- Impact: Better for large datasets but less optimal for highly imbalanced data.")

print("\n Question: When would OvR be better than OvO?")
print("Analyze different datasets and choose the best approach!\n")
########



