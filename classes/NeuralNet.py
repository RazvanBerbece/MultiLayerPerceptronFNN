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
        2. https://www.kdnuggets.com/2019/11/build-artificial-neural-network-scratch-part-1.html
        3. https://brilliant.org/wiki/backpropagation/
"""

# Imports
import numpy as np
from functions.sigmoid import sigmoid
from functions.mse import MSE
from functions.linear import linear

class FeedforwardNeuralNet:

    # Network Hyperparameters Initialiser 
    def __init__(self, input_size, hidden_size, output_size, bias):
        self.input_size  = input_size  # nodes in the input layer
        self.hidden_size = hidden_size  # nodes in the hidden layer
        self.output_size = output_size  # nodes in the output layer

        self.bias = bias
    
    def init_input(self, input_array):
        self.x = input_array
    
    def init_output(self, output_array):
        output_array = output_array.T # make sure that the results are in the right shape using the transpose
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
        self.hidden_input = np.dot(self.x, self.hidden_weights) + self.bias
        self.hidden_output = sigmoid(self.hidden_input, derivative=False)
    
    def init_output_pipe(self):
        """
        self.output_input  = the input of the output layer
        self.output_output = the output of the output layer
        """
        self.output_input = np.dot(self.hidden_output, self.y_weights)
        self.output_output = linear(self.output_input, derivative=False)
    
    def update_weights(self, derivative_error_hidden, derivative_error_output, lr):
        """
        Increase/Decrease weigths for layers according to the resulting errors from the gradient descent algo
        """
        self.hidden_weights -= lr * derivative_error_hidden
        self.y_weights -= lr * derivative_error_output
    
    def train(self, epochs, learning_rate):
        """
        In Deep Learning (and most of ML), training means finding the minimum of the error (cost) function
        Updates the weight values of the neural net using gradient descent
        """
        for epoch in range(epochs):

            print("EPOCH ", epoch + 1) # epoch + 1 in order print epochs indexed from 1

            # Propagate from input layer to hidden layer and then to output layer & update output layer weights
            self.init_hidden_pipe()
            self.init_output_pipe()
            mean_squared_err = MSE(self.output_output, self.y, vector_size=self.output_size) # display MSE
            print("MSE : ", mean_squared_err.sum())

            # GRADIENT DESCENT (d = curl operator)
            # Phase 1 Derivatives (process output layer gradients) -- assets/img/ChainRulePhase1.png for visualisation
            derror_douto = self.output_output - self.y # on output layer
            douto_dino = linear(self.output_input, derivative=True)
            dino_dwo = self.hidden_output
            # derror_dwo is the left side of the chain rule 
            derror_dwo = np.dot(dino_dwo.T, derror_douto * douto_dino)

            # Phase 2 Derivatives (process hidden layer gradients) -- assets/img/ChainRulePhase2.png for visualisation
            derror_dino = derror_douto * douto_dino # on output layer
            dino_douth = self.y_weights
            derror_douth = np.dot(derror_dino, dino_douth.T)
            douth_dinh = sigmoid(self.hidden_input, derivative=True)
            dinh_dwh = self.x
            # derror_wh is the left side of the chain rule 
            derror_dwh = np.dot(dinh_dwh.T, douth_dinh * derror_douth)

            # Update weights
            self.update_weights(derror_dwh, derror_dwo, learning_rate)
    
    def predict(self, input):
        # Forward to Hidden Layer
        hidden_input = np.dot(input, self.hidden_weights)
        hidden_output = sigmoid(hidden_input, derivative=False)
        # Forward to Output Layer
        output_input = np.dot(hidden_output, self.y_weights)
        output_output = linear(output_input, derivative=False)
        print(output_output) # final prediction on input

    # Debugging
    def display_network(self):
        print(self.x)
        print(self.y)
        print(self.hidden_weights)
        print(self.y_weights)
        print(self.hidden_input)
        print(self.hidden_output)
        print(self.output_output)