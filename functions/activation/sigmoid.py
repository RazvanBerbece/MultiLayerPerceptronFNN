import numpy as np

# Sigmoid function with added derivative functionality through param if true
def sigmoid(x, derivative):

    if derivative == True:
        """
        Return derivative of function sigmoid(x)
        """
        return sigmoid(x, derivative=False) * (1 - sigmoid(x, derivative=False))

    return 1 / (1 + np.exp(-x))