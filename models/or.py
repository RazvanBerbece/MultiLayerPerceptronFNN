"""
    (RUNNABLE) 
    Modelling the OR logic gate using a perceptron FNN
"""

# Imports #
import numpy as np
from classes.NeuralNet import FeedforwardNeuralNet
from functions.activation.sigmoid import sigmoid

# Hyperparameters #
INPUT_NODES = 2
HIDDEN_NODES = 3
OUTPUT_NODES = 1
BIAS = 0.25 # global bias
OUTPUT_ACTIVATION_FUNCTION = sigmoid # use sigmoid as we have a binary classification problem and the output can be squished between 1 and 0

# Allocate and config Neural Net with hyperparams #
FNN = FeedforwardNeuralNet(
    input_size=INPUT_NODES, 
    hidden_size=HIDDEN_NODES, 
    output_size=OUTPUT_NODES,
    output_activation=OUTPUT_ACTIVATION_FUNCTION, 
    bias=BIAS)

# Init Neural Net #
# Random data to be used in training
input_features = np.array([[0,0],[1,0],[1,1]])
output_target = np.array([[0,1,1]])
# Method calls for init
# Data
FNN.init_input(input_array=input_features)
FNN.init_output(output_array=output_target)
# Weights
FNN.init_weights()

# Training #
LR = 0.05
EPOCHS = 50000
FNN.train(epochs=EPOCHS, learning_rate=LR)

# Predicting #
FNN.predict(np.array([[6,4]]))

# Debugging #
# FNN.display_network()
