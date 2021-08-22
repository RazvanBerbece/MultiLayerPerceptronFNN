"""
    FEEDFORWARD NEURAL NET IMPLEMENTATION

    Specs :
        INPUT NEURONS [NUMBER OF FEATURES OF ONE ENTRY]
        HIDDEN LAYERS [NUMBER OF HIDDEN LAYERS]
            LAYER 1   [SHOULD BE LESS THAN 2 x INPUT NEURONS]
            ...
            ..
            .
            TODO LAYER N 
        OUTPUT LAYER  [NUMBER OF CLASSES; 1 IF REGRESSION OR SIGMOIDAL OUTPUT]

    Notes :
        x is a matrix; each row is 1 entry with |col| number of features
        y is a matrix; each row is the target (class, targetValue, etc.) of 1 entry (row) from x (in order)
            To check: y (number of rows) = x (number of rows) as each entry has a target

    References :
        1. https://pub.towardsai.net/building-neural-networks-with-python-code-and-math-in-detail-ii-bbe8accbf3d1#8c76
"""

# Imports
import numpy as np
from functions.sigmoid import sigmoid
from functions.mse import MSE

class FeedforwardNeuralNet:

    # Network Hyperparameters Initialiser 
    def __init__(self, input_size, hidden_size, output_size, bias, learning_rate):
        self.input_size  = input_size  # nodes in the input layer
        self.hidden_size = hidden_size  # nodes in the hidden layer
        self.output_size = output_size  # nodes in the output layer

        self.bias = bias
        self.learning_rate = learning_rate
    
    def init_input(self, input_array):
        self.x = input_array
    
    def init_output(self, output_array):
        output_array = output_array.reshape(self.x.shape[0], 1) # make sure that the results are in the right shape
        self.y = output_array

    def init_weights(self):
        """
            Uses random sigmoidal values (i.e. between 0 and 1)

            Needed for hidden layer : |Input| * |Hidden|
            Needed for output layer : |Hidden| * |Output|
        """
        self.hidden_weights = np.random.uniform(low=0, high=1, size=(self.input_size, self.hidden_size))  
        self.y_weights = np.random.uniform(low=0, high=1, size=(self.hidden_size, self.output_size))

    def init_hidden_pipe(self):
        """
        Does matrix multiplication as per the layer output formula of Neural Nets (output = i1 * w1 + i2 * w2 + ... + in * wn)
        """
        self.hidden_input = np.dot(self.x, self.hidden_weights)
        self.hidden_output = sigmoid(self.hidden_input, derivative=False)
    
    def init_output_pipe(self):
        """
        self.output_input  = the input of the output layer
        self.output_output = the output of the output layer
        """
        self.output_input = np.dot(self.hidden_output, self.y_weights)
        self.output_output = sigmoid(self.output_input, derivative=False)
    
    def update_weights(self, derivative_error_hidden, derivative_error_output):
        """
        Increase/Decrease weigths for layers according to the resulting errors from the gradient descent algo
        """
        self.hidden_weights -= self.learning_rate * derivative_error_hidden
        self.y_weights -= self.learning_rate * derivative_error_output
    
    def descent(self, epochs):
        """
        Updates the weight values of the neural net using gradient descent
        """
        for epoch in range(epochs):

            # These will be updated after calculating the phase 1 and 2 derivatives
            derivative_error_hidden = 0
            derivative_error_output = 0

            self.init_hidden_pipe()
            self.init_output_pipe()

            # Phase 1 -> update output layer weights
            MSE(self.output_output, self.y) # display MSE

            # Phase 1 Derivatives TODO
            # Phase 2 -> update hidden layer weights TODO

            # Update weights
            self.update_weights(derivative_error_hidden, derivative_error_output)

    # Debugging
    def display_network(self):
        print(self.x)
        print(self.y)
        print(self.hidden_weights)
        print(self.y_weights)
        print(self.hidden_input)
        print(self.hidden_output)
        print(self.output_output)