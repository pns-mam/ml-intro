import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# TODO: Load data and splits if necessary


### TensorFlow is a powerful open-source deep learning framework developed by Google. 
## It enables building and training neural networks efficiently, 
##using optimized tensor computations. 
## It supports CPU, GPU, and TPU acceleration, making it ideal for complex AI models.

#Keras is a high-level neural network API that runs on top of TensorFlow. 
#It provides an easy-to-use interface for defining deep learning models without needing complex implementations.
#TensorFlow integrates Keras natively, making it ideal for rapid AI development.


# TODO: Create your first fully connected neural network with the following caracteristics:
# * 1 input layer
# * 1 hidden layer using 32 neurons
# * Input and hidden layers shall have relu activation functions
# * Output layer is activated with softmax
# * As input, we consider the pixels directly -- we'll look later into using engineered features
model = None # Use keras.Sequential



# TODO: Compile the model (setting parameters on how the model will learn from the data)
# Use adam optimizer, sparse categorical cross entropy as loss, and at least the accuracy as the metric



# TODO: Train the model and store training and validation history
history = None




# Extract loss values
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot loss vs epochs
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss', marker='o')
plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss', marker='s')
plt.xlabel("Epochs")
plt.yscale("log")  # Apply log scale to the y-axis
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.legend()
plt.grid()
plt.show()

# Save the trained model
model.save("digit_classifier_model.keras")  # Native Keras format

# TODO: 
# For the remaining time of the class, find a way to combine preprocessing + learning algorithm to achieve the best possible scores.
# Your final tuned model should be written in the test script provided for us to test during the oral exam. 
# 
# Enjoy, and good luck!
