import numpy as np
from model_confidence_set import ModelConfidenceSet

# Loss functions Calculation
def rmse(actual, predicted):
    return np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2))

def mae(actual, predicted):
    return np.mean(np.abs((np.array(actual) - np.array(predicted))))

def qlike(actual, predicted):
    return(np.mean(np.log(predicted) + np.array(actual) / np.array(predicted)))

def ratio_p(actual, predicted):
    return np.sum((np.array(actual) - np.array(predicted)) ** 2) / np.sum((np.array(actual) - np.mean(np.array(actual))) ** 2)

def hmse(actual, predicted):
    return np.mean((1 - np.array(actual) / np.array(predicted)) ** 2)

def wlfreq(predicted, predicted_bm, actual):
    predicted = np.array(predicted)
    predicted_bm = np.array(predicted_bm)
    actual = np.array(actual)
    
    diff_pred = np.abs(predicted - actual)
    diff_bm = np.abs(predicted_bm - actual)

    winners = (diff_pred < diff_bm).astype(int)
    min_diffs = np.minimum(diff_pred, diff_bm)
    max_diffs = np.maximum(diff_pred, diff_bm)
    win_pred = np.sum(winners == 1)
    win_bm = np.sum(winners == 0)

    return diff_pred, diff_bm, win_pred, win_bm, min_diffs, max_diffs, winners

def mcs(predicted, predicted_bm, actual):
    predicted = np.array(predicted)
    predicted_bm = np.array(predicted_bm)
    actual = np.array(actual)
    
    diff_pred = np.abs(predicted - actual)
    diff_bm = np.abs(predicted_bm - actual)

    loss_mat = np.column_stack((diff_bm, diff_pred))
    mcs = ModelConfidenceSet(loss_mat, n_boot=5000, alpha=0.7)

    mcs.compute()
    return mcs.results()  