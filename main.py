"""
    (RUNNABLE) 
    Modelling 2-integer summation using a perceptron FNN
"""

# Imports #
import numpy as np
from classes.NeuralNet import FeedforwardNeuralNet
from functions.activation.linear import linear

# Hyperparameters #
INPUT_NODES = 2
HIDDEN_NODES = 4
OUTPUT_NODES = 1
BIAS = 0.25 # global bias
OUTPUT_ACTIVATION_FUNCTION = linear # use linear as we don't have to squish the output between 0 and 1 in the case of regression

# Allocate and config Neural Net with hyperparams #
FNN = FeedforwardNeuralNet(
    input_size=INPUT_NODES, 
    hidden_size=HIDDEN_NODES, 
    output_size=OUTPUT_NODES,
    output_activation=OUTPUT_ACTIVATION_FUNCTION, 
    bias=BIAS)

# Init Neural Net #
# Random data to be used in training
input_features = np.array([[0,0],[1,0],[1,1],[2,1],[3,1],[3,2],[3,3],[4,3],[4,4],[8,1],[7,3],[2,2]])
output_target = np.array([[0,1,2,3,4,5,6,7,8,9,10,4]])
# Method calls for init
# Data
FNN.init_input(input_array=input_features)
FNN.init_output(output_array=output_target)
# Weights
FNN.init_weights()

# Training #
LR = 0.05
EPOCHS = 500000
FNN.train(epochs=EPOCHS, learning_rate=LR)

# Predicting #
FNN.predict(np.array([[4,2]]))

# Debugging #
# FNN.display_network()
