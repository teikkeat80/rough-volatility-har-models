import numpy as np
from scipy.optimize import minimize
import math
import pandas as pd
import scipy.integrate as integrate
from time import time
import pickle

class HARK:
    def __init__(self, b0, b1, b2, b3, q):
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.q = q ** 2

    def construct_kf(self):
        self.k = np.vstack((np.array([self.b0]), np.zeros((21, 1))))
        self.t = np.vstack((np.array([self.b1] + [self.b2 / 4] * 4 + [self.b3 / 17] * 17), np.hstack((np.eye(21), np.zeros((21, 1))))))
        self.Q = np.diag([self.q] + [0] * 21)
        self.g = np.eye(22)
        self.m = np.concatenate([np.ones(1), np.zeros(21)]).reshape(1, 22)

    def initialise_a(self, mean):
        self.a = (np.ones(22) * mean).reshape(22, 1)
    
    def initialise_p(self, var_iv):
        self.p = np.diag(np.ones(22) * var_iv)
    
    def predict(self):
        a_pred = self.k + self.t @ self.a
        p_pred = self.t @ self.p @ self.t.T + self.g @ self.Q @ self.g.T

        self.a = a_pred
        self.p = p_pred
        return a_pred, p_pred
    
    def update(self, obs, r):
        v = obs - self.m @ self.a
        f = self.m @ self.p @ self.m.T + r
        kg = self.p @ self.m.T @ np.linalg.inv(f)
        a_upd = self.a + kg @ v
        p_upd = self.p - kg @ self.m @ self.p

        self.a = a_upd
        self.p = p_upd
        return v, f, a_upd, p_upd

def log_likelihood(params, rv, rq): 
    b0, b1, b2, b3, q = params
    x = HARK(b0, b1, b2, b3, q)
    x.construct_kf()
    x.initialise_a(np.mean(rv))
    x.initialise_p(np.var(rv))
    sum_ll = 0

    for t in range(len(rv)):
        x.predict()
        v, f, _, _ = x.update(rv[t], rq[t])
        sum_ll += math.log(abs(f)) + v.T @ np.linalg.inv(f) @ v

    ll = - (22 / 2) * len(rv) * math.log(2 * math.pi) - (1 / 2) * sum_ll
    return -ll

# Load Data
rv_path = 'C:\\Users\\teikkeattee\\ProjProg\\SNP500_RV_5min.csv'
rv_df = pd.read_csv(rv_path).sort_values(by='Date', ignore_index=True)
rv = rv_df['RV'].tolist()
log_rv = np.log(rv)
rq_path = 'C:\\Users\\teikkeattee\\ProjProg\\SP500_RQ_5min.csv'
rq_df = pd.read_csv(rq_path).sort_values(by='Date', ignore_index=True)
rq = rq_df['RQ'].tolist()

# Initialise Parameters
initial_params = [0.001, 0.5, 0.5, 0.5, 0.1]

start_time = time()

# Estimation
result = minimize(
    log_likelihood,
    initial_params,
    args=(log_rv, rq),
    method='Nelder-Mead',
    options={'xatol': 1e-6, 'fatol': 1e-2, 'maxfev': 8000}
)

# Record Estimation Result
est_params = result.x
loglik = - result.fun
b0, b1, b2, b3, q = est_params
aic = (2 * len(initial_params)) - (2 * loglik)
print(result)
np.set_printoptions(suppress=True)
print('Estimated Params: ', np.round(est_params, 4))
print('AIC: ', aic)
with open(f'C:\\Users\\teikkeattee\\ProjProg\\HARK_RVRQ_EST.pickle', 'wb') as file:
    pickle.dump(result, file)

        
# Record Time
end_time = time()
print(f"Elapsed time: {end_time - start_time} seconds")

