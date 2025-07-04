import numpy as np

# Loss functions Calculation
def rmse(actual, predicted):
    return np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2))

def mae(actual, predicted):
    return np.mean(np.abs((np.array(actual) - np.array(predicted))))

def qlike(actual, predicted):
    return(np.mean(np.log(predicted) + np.array(actual) / np.array(predicted)))