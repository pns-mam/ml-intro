# TODO: Import necessary libraries



##########################################
## Data loading and first visualisation
##########################################

# Load the handwritten digits dataset
digits = # TODO: Write your own code!

# Visualize some images
# TODO: Graph the first4 images from the data base 

# Display at least one random sample par class (some repetitions of class... oh well)
def plot_multi(data, y):
    '''Plots 16 digits'''
    nplots = 16
    nb_classes = len(np.unique(y))
    cur_class = 0
    fig = plt.figure(figsize=(15,15))
    for j in range(nplots):
        plt.subplot(4,4,j+1)
        to_display_idx = np.random.choice(np.where(y == cur_class)[0])
        plt.imshow(data[to_display_idx].reshape((8,8)), cmap='binary')
        plt.title(cur_class)
        plt.axis('off')
        cur_class = (cur_class + 1) % nb_classes
    plt.show()


plot_multi(digits.data, digits.target)

##########################################
## Data exploration and first analysis
##########################################

def get_statistics_text(targets):
    # TODO: Write your code here, returning at least the following useful infos:
    # * Label names
    # * Number of elements per class
    return None

# TODO: Call the previous function and generate graphs and prints for exploring and visualising the database



##########################################
## Start data preprocessing
##########################################

# Access the whole dataset as a matrix where each row is an individual (an image in our case) 
# and each column is a feature (a pixel intensity in our case)
## X = [
#  [Pixel1, Pixel2, ..., Pixel64],  # Image 1 as a row
#  [Pixel1, Pixel2, ..., Pixel64],  # Image 2 as a row
#  [Pixel1, Pixel2, ..., Pixel64],  # Image 3 as a row
#  [Pixel1, Pixel2, ..., Pixel64]   # Image 4 as a row
#]

# TODO: Create a feature matrix and a vector of labels
X = None
y = None

# Print dataset shape
print(f"Feature matrix shape: {X.shape}. Max value = {np.max(X)}, Min value = {np.min(X)}, Mean value = {np.mean(X)}")
print(f"Labels shape: {y.shape}")


# TODO: Normalize pixel values to range [0,1]
F = None  # Feature matrix after scaling

# Print matrix shape
print(f"Feature matrix F shape: {F.shape}. Max value = {np.max(F)}, Min value = {np.min(F)}, Mean value = {np.mean(F)}")

##########################################
## Dimensionality reduction
##########################################


### just an example to test, for various number of PCs
sample_index = 0
original_image = F[sample_index].reshape(8, 8)  # Reshape back to 8×8 for visualization

# TODO: Using the specific sample above, iterate the following:
# * Generate a PCA model with a certain value of principal components
# * Compute the approximation of the sample with this PCA model
# * Reconstruct a 64 dimensional vector from the reduced dimensional PCA space
# * Reshape the resulting approximation as an 8x8 matrix
# * Quantify the error in the approximation
# Finally: plot the original image and the 15 approximation on a 4x4 subfigure

#### TODO: Expolore the explanined variance of PCA and plot 

# Create the visualization plot


### TODO: Display the whole database in 2D: 


### TODO: Create a 20 dimensional PCA-based feature matrix

F_pca = None

# Print reduced feature matrix shape
print(f"Feature matrix F_pca shape: {F_pca.shape}")


##########################################
## Feature engineering
##########################################
### # Function to extract zone-based features
###  Zone-Based Partitioning is a feature extraction method
### that helps break down an image into smaller meaningful regions to analyze specific patterns.
def extract_zone_features(images):
    '''Break down an 8x8 image in 3 zones: row 1-3, 4-5, and 6-8'''
    # TODO: Fill in code
    return np.array([])

# Apply zone-based feature extraction
F_zones = extract_zone_features('''some data''')

# Print extracted feature shape
print(f"Feature matrix F_zones shape: {F_zones.shape}")


### Edge detection features

## TODO: Get used to the Sobel filter by applying it to an image and displaying both the original image 
# and the result of applying the Sobel filter side by side


# TODO: Compute the average edge intensity for each image and return it as an n by 1 array
F_edges = None

# Print feature shape after edge extraction
print(f"Feature matrix F_edges shape: {F_edges.shape}")

### connect all the features together

# TODO: Concatenate PCA, zone-based, and edge features
F_final = None 

# TODO: Normalize final features
F_final = F_final

# Print final feature matrix shape
print(f"Final feature matrix F_final shape: {F_final.shape}")

