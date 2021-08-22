"""

    FEEDFORWARD NEURAL NET IMPLEMENTATION

    Specs :
        INPUT NEURONS = 2 [NUMBER OF FEATURES OF ONE ENTRY]
        HIDDEN LAYERS = 1
            LAYER 1 = 3   [SHOULD BE LESS THAN 2 x INPUT NEURONS]
        OUTPUT LAYER = 1  [NUMBER OF CLASSES; 1 IF REGRESSION OR SIGMOIDAL OUTPUT]

    Notes :
        x is a matrix; each row is 1 entry with |col| number of features
        y is a matrix; each row is the target (class, targetValue, etc.) of 1 entry (row) from x (in order)
            To check: y (number of rows) = x (number of rows) as each entry has a target
        y weights have to be 
"""

# Imports
import numpy as np

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
            Uses sigmoidal values (i.e. between 0 and 1)

            Needed for hidden layer : |Input| * |Hidden|
            Needed for output layer : |Hidden| * |Output|
        """
        self.weights_y = np.random.uniform(low=0, high=1, size=(self.input_size, self.hidden_size))
        self.weights_hidden = np.random.uniform(low=0, high=1, size=(self.hidden_size, self.output_size))  

    def display_network(self):
        print(self.x)
        print(self.y)
        print(self.weights_hidden)