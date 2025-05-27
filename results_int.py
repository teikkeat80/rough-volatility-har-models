import numpy as np
from scipy.optimize import minimize
import math
import pandas as pd
import scipy.integrate as integrate
from matplotlib import pyplot as plt
import pickle
from time import time
from hurst import Hurst
from misc import rmse

indices = ["SPX", "GDAXI", "FCHI", "FTSE", "OMXSPI", "N225", "KS11", "HSI"]
# indices = ["SPX"]
# idx = indices[0]
# log_rv = np.log(load_rv_one('data/rv_dataset.csv', f'.{idx}')) 
# for idx in indices:
# # log_rv = np.log(load_rv('data/SNP500_RV_5min.csv', 'RV'))
#     log_rv = np.log(load_rv_one('data/rv_dataset.csv', f'.{idx}')) 
# # rq = 2 * np.array(load_rv('data/SP500_RQ_5min.csv', 'RQ')) / np.exp(log_rv) ** 2

timeframe = 500

for ix in indices:
    with open(f'estm1_result/HARK2_{ix}_EST.pickle', 'rb') as file:
        result = pickle.load(file)
        print(ix)
        print("-------------------------")
        print(result)

    dfhark = pd.read_csv(f'fcst1_result/HARK_{ix}_FCST.csv')
    dfhark2 = pd.read_csv(f'fcst1_result/HARK2_{ix}_FCST.csv')

    predictedhark2 = dfhark2['predicted'].values[-timeframe:]
    actualhark2 = dfhark2['actual'].values[-timeframe:]
    
    predictedhark = dfhark['predicted'].values[-timeframe:]
    actualhark = dfhark['actual'].values[-timeframe:]

    rmsehark2 = rmse(actualhark2, predictedhark2)
    print(f"{ix} HARK2 Out of Sample rmse: {round(rmsehark2, 6)}")

    rmsehark = rmse(actualhark, predictedhark)
    print(f"{ix} HARK Out of Sample rmse: {round(rmsehark, 6)}")  

    if rmsehark2 < rmsehark:
        print('success')

    param = 1 / (1 + np.exp(- dfhark2['h'].values))
    param = param[-timeframe:]
    print(np.mean(param))

    plt.figure(figsize=(10, 6))
    plt.plot(param)
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.grid(True)
    plt.show()