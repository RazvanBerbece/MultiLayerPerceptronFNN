# Imports
import numpy as np
from NeuralNet import FeedforwardNeuralNet

# Hyperparameters
input_nodes = 2
hidden_nodes = 3
output_nodes = 1
bias = 0.25
lr = 0.05

# Allocate and config Neural Net with hyperparams
NN = FeedforwardNeuralNet(input_size=input_nodes, hidden_size=hidden_nodes, output_size=output_nodes, bias=bias, learning_rate=lr)

# Init Neural Net
input_features = np.array([[1,0],[1,1],[0,1],[0,0]])
NN.init_input(input_array=input_features)
output_target = np.array([[1,1,1,0]])
NN.init_output(output_array=output_target)
NN.init_weights()

# Debugging
NN.display_network()
