import numpy as np
from scipy.optimize import minimize
import math
import pandas as pd
import scipy.integrate as integrate
from matplotlib import pyplot as plt
import pickle
from time import time
from hurst import Hurst
from misc import rmse, qlike, mae
import visualisation as viz
from data_processing import load_rv_one, load_rv

indices = ["SPX", "GDAXI", "FCHI", "FTSE", "OMXSPI", "N225", "KS11", "HSI"]
# indices = ["SPX"]
# idx = indices[0]
# for idx in indices:
log_rv = np.log(load_rv('data/SNP500_RV_5min.csv', 'RV'))
    # log_rv = np.log(load_rv_one('data/rv_dataset.csv', f'.{idx}')) 
rq = (2 / 78) * np.array(load_rv('data/SP500_RQ_5min.csv', 'RQ')) / np.exp(log_rv) ** 2

# timeframe = 500

# for ix in indices:
# with open(f'estm_result/HARK_SPX_EST.pickle', 'rb') as file:
#     result = pickle.load(file)
# print("-------------------------")
# print(result)
# np.set_printoptions(suppress=True)
# print('Estimated Params: ', np.round(result.x, 4))
# print('LL: ', np.round(- result.fun, 4))
# print('AIC: ', np.round(2 * len(result.x) - 2 * (- result.fun), 4))

dfhark = pd.read_csv(f'isa_result/HARK2QC_RV_FCST.csv')

variancehark = np.array(dfhark['var'].values)
rhark = np.array(dfhark['r'].values) ** 2
# rhark = 0
predictedhark = dfhark['predicted'].values
actualhark = dfhark['actual'].values

rvpredictedhark = np.exp(predictedhark + ((variancehark + rhark) / 2))
rvactualhark = np.exp(actualhark)

# RMSE
rmsehark = rmse(rvactualhark, rvpredictedhark)
print(f"rmse: {round(rmsehark, 4)}")
# QLIKE
qlikehark = qlike(rvactualhark, rvpredictedhark)
print(f"qlike: {round(qlikehark, 4)}")
# MAE
maehark = mae(rvactualhark, rvpredictedhark)
print(f"mae: {round(maehark, 4)}")  

viz.plot_comparison(rvactualhark, rvpredictedhark)