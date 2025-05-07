import numpy as np

# MSE Calculation
def rmse(actual, predicted):
    return np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2))

def qlike(actual, predicted):
    return(np.mean(np.log(predicted) + np.array(actual) / np.array(predicted)))