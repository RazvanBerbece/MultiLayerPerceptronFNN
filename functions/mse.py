import numpy as np

def MSE(predicted, real, vector_size):
    """
    Mean Squared Error function that returns the vector of errors for each pair (predicted_i, real_i)
    """
    mse = ((1 / vector_size) * (np.power((predicted - real), 2)))
    # print(mse)
    return mse 