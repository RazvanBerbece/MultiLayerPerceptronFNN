import numpy as np

def MSE(predicted, real):
    """
    Mean Squared Error function
    """
    output_error = ((1 / 2) * (np.power((predicted - real), 2)))
    mse = output_error.sum()
    print(mse)
    # return mse