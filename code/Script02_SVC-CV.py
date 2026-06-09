import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# Ensure you have your utils.py file or these functions available in your environment
try:
    from Script01_PreprocessingExploration import compute_hog, my_PCA, TARGET_SIZE
except ImportError:
    # Safe defaults if utils are missing during initial setup
    TARGET_SIZE = (64, 64)
    def my_PCA(data, n_components=5): pass
    def compute_hog(image, nb_h_cells, nb_w_cells, nb_bins): return np.zeros(nb_h_cells*nb_w_cells*nb_bins)
    def load_dict(f): return {0: 'Chihuahua', 1: 'Pug', 2: 'Malamute', 3: 'Beagle'}

OUTPUT_DIR = os.path.join('..', 'folder_code2')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Define or reload precalculated vector states from Lab 1
if os.path.exists("X_train_standard.npy"):
    X_train = np.load("X_train_standard.npy")
    X_test = np.load("X_test_standard.npy")
    y_train = np.load("y_train_standard.npy")
    y_test = np.load("y_test_standard.npy")
    label_names = load_dict("lbl_names.npy")
    labels = np.load("labels.npy")
else:
    raise FileNotFoundError(" Cache arrays not found. Run the Lab 1 setup script to generate the design matrices.")


# =========================================================================
# TASK 1: QUANTIFYING TRAIN/TEST DISTRIBUTION RESEMBLANCE 
# =========================================================================
# Prevent the "Time Machine Effect". We must ensure our random splits 
# have similar class distributions before applying any transformations.

### STUDENT IMPLEMENTATION START ###
# TODO: 1. Calculate the empirical distribution arrays (class probabilities) for y_train and y_test.
#          Hint: Use np.unique(..., return_counts=True) and divide by the total length.
# TODO: 2. Implement the Cosine Similarity metric to quantify the resemblance:
#          Formula: (A dot B) / (norm(A) * norm(B))
# What other similarity metrics could you have used here? (e.g., KL Divergence, Wasserstein Distance, etc.) How would you implement them mathematically?

similarity_score = 0.0  # Replace this placeholder with your math

### STUDENT IMPLEMENTATION END ###
print(f"   - Train/Test Resemblance (Cosine Similarity): {similarity_score:.6f}")


# =========================================================================
# TASK 2: CUSTOM PIPELINE TRANSFORMER WRAPPERS 
# =========================================================================
# The "Lego Block" Philosophy: We wrap our custom functions into classes
# that inherit from BaseEstimator and TransformerMixin.

class EdgeInfoPreprocessing(BaseEstimator, TransformerMixin):
    def __init__(self, nb_h_cells=4, nb_w_cells=4, nb_bins=8):
        self.nb_h_cells = nb_h_cells
        self.nb_w_cells = nb_w_cells
        self.nb_bins = nb_bins
        
    def fit(self, X, y=None):
        return self # HOG requires no training, so fit does nothing
        
    def transform(self, X):
        return np.array([compute_hog(img.reshape(TARGET_SIZE), self.nb_h_cells, self.nb_w_cells, self.nb_bins) for img in X])


class PCAInfoPreprocessing(BaseEstimator, TransformerMixin):
    """
    In Module 1, class-specific PCA was computed manually via loops.
    To prevent Data Leakage, we automate this inside a scikit-learn transformer.
    """
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.pca_per_class = []
        
    def fit(self, X, y=None):
        if y is None:
            raise ValueError("Supervised target array 'y' is required to fit class-specific subspaces.")
        
        ### STUDENT IMPLEMENTATION START ###
        # TODO: 1. Clear historical instances tracked in self.pca_per_class.
        # TODO: 2. Iterate through unique class identifiers present within vector 'y'.
        # TODO: 3. Isolate matching instance data slices from 'X', train separate PCA models,
        #          and append each fitted model to self.pca_per_class.
        
        pass 
        
        ### STUDENT IMPLEMENTATION END ###
        return self
        
    def transform(self, X):
        out = np.zeros((len(X), 0))
        
        ### STUDENT IMPLEMENTATION START ###
        # TODO: 1. Iterate through the fitted PCA models saved in self.pca_per_class.
        # TODO: 2. Transform the input matrix 'X' using each PCA model.
        # TODO: 3. Horizontally stack the outputs together using np.hstack.
        
        ### STUDENT IMPLEMENTATION END ###
        return out


# =========================================================================
# TASK 3: END-TO-END AUTOMATED PIPELINE ARCHITECTURES (See Slide 5)
# =========================================================================
print("\n Step 2: Assembling automated Pipeline and FeatureUnion components...")

### STUDENT IMPLEMENTATION START ###
# TODO: 1. Construct a 'FeatureUnion' merging 'PCAInfoPreprocessing' and 'EdgeInfoPreprocessing'.
#          Name it 'all_features'.
# TODO: 2. Build the end-to-end processing pipeline using scikit-learn's 'Pipeline' class.
#          The sequence MUST be: MinMaxScaler -> FeatureUnion -> StandardScaler -> SVC(kernel='linear').
#          Name it 'pipeline_svc'.

all_features = None  # Build using FeatureUnion([...])
pipeline_svc = None  # Build using Pipeline([...])



# =========================================================================
# TASK 4: HYPERPARAMETER OPTIMIZATION SELECTION SPACE (See Slide 6)
# =========================================================================
print("\n Step 3: Tuning via GridSearchCV...")

if pipeline_svc is not None:
    # Switch to RBF kernel for complex boundary mapping
    pipeline_svc.set_params(classifier__kernel='rbf')
    
    ### STUDENT IMPLEMENTATION START ###
    # TODO: Set up a dictionary search parameter grid matching your pipeline's exact component names.
    # Grid Requirements: 
    # - PCA components: [5, 10]
    # - SVC Cost C: [0.1, 1, 10]
    # - SVC Gamma: [0.01, 0.1]
    
    param_grid = {}  # Fill with proper keys (e.g., 'features__pca__n_components') and value lists
    
    ### STUDENT IMPLEMENTATION END ###
    
    # We use 3-Fold Cross Validation to test the parameters safely
    grid_search = GridSearchCV(pipeline_svc, param_grid=param_grid, cv=3, n_jobs=-1)
    
    print("   Training the Grid Search ")
    # Uncomment the line below once your pipeline and param_grid are built!
    # grid_search.fit(X_train, y_train)
    # print(f"    Optimal Parameters Identified: {grid_search.best_params_}")
    # print(f"    Generalization Score on Testing Partition: {grid_search.score(X_test, y_test)*100:.2f}%")


# =========================================================================
# TASK 5: MULTICLASS STRATEGY COMPARISON (See Slides 7 & 8)
# =========================================================================
# Support Vector Machines are binary. Let's compare the "Round Robin" (OvO)
# strategy against the "Me Against the World" (OvR) strategy.
print("\n Step 4: Comparing  OvO vs. OvR ")

# TODO: 1. Build a new pipeline named 'pipeline_ovo'. Use the same scaling and feature
#          steps as before, but wrap the final SVC inside a OneVsOneClassifier().
# TODO: 2. Fit 'pipeline_ovo' on the training data and score it on the test data.
# TODO: 3. Repeat the exact same process for a new pipeline named 'pipeline_ovr', 
#          but use a OneVsRestClassifier() instead.

# --- OvO Implementation ---
pipeline_ovo = None 
ovo_score = 0.0

# --- OvR Implementation ---
pipeline_ovr = None
ovr_score = 0.0


print(f"   One-vs-One (OvO) Strategy Score: {ovo_score*100:.2f}%")
print(f"   One-vs-Rest (OvR) Strategy Score: {ovr_score*100:.2f}%")

