import numpy as np
from scipy.optimize import minimize
import math
import pandas as pd
from time import time
import os

class HARK:
    def __init__(self, b0, b1, b2, b3, q, r):
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.q = q ** 2
        self.r = r ** 2

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
    
    def update(self, obs, r=None):
        if r is None:
            r = self.r
        v = obs - self.m @ self.a
        f = self.m @ self.p @ self.m.T + r
        kg = self.p @ self.m.T @ np.linalg.inv(f)
        a_upd = self.a + kg @ v
        p_upd = self.p - kg @ self.m @ self.p

        self.a = a_upd
        self.p = p_upd
        return v, f, a_upd, p_upd

def log_likelihood_harkred(params, rv): 
    b0, b1, b2, b3, q, r = params
    x = HARK(b0, b1, b2, b3, q, r)
    x.construct_kf()
    x.initialise_a(np.mean(rv))
    x.initialise_p(np.var(rv))
    sum_ll = 0

    for t in range(len(rv)):
        x.predict()
        v, f, _, _ = x.update(rv[t])
        sum_ll += math.log(abs(f)) + v.T @ np.linalg.inv(f) @ v

    ll = - (22 / 2) * len(rv) * math.log(2 * math.pi) - (1 / 2) * sum_ll
    return -ll

def log_likelihood_hark(params, rv, rq): 
    b0, b1, b2, b3, q = params
    x = HARK(b0, b1, b2, b3, q, 1) # dummy r
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
rv_path = '/Users/teikkeattee/Workplace/UM_MSC_STATS/UM_STATS_Research_Project/Project_Placeholder/data/SNP500_RV_5min.csv'
rv_df = pd.read_csv(rv_path).sort_values(by='Date', ignore_index=True)
rv = rv_df['RV'].tolist()[-1500:]
log_rv = np.log(rv)
# rq_path = '/Users/teikkeattee/Workplace/UM_MSC_STATS/UM_STATS_Research_Project/Project_Placeholder/data/SP500_RQ_5min.csv'
# rq_df = pd.read_csv(rq_path).sort_values(by='Date', ignore_index=True)
# rq = rq_df['RQ'].tolist()[-1500:]
# rq = (2 / 78) * (np.array(rq) / np.exp(log_rv) ** 2)


# Output file path
output_file = '/Users/teikkeattee/Workplace/UM_MSC_STATS/UM_STATS_Research_Project/Project_Placeholder/osa_result/HARK_RV_FCST.csv'
columns = ['iteration', 'b0', 'b1', 'b2', 'b3', 'q', 'r', 'loglik', 'predicted', 'var', 'actual']

# Determine where to resume from (if file already exists)
if os.path.exists(output_file):
    existing_df = pd.read_csv(output_file)
    start_iter = existing_df['iteration'].max() + 1
else:
    start_iter = 0
    # Write header if file doesn't exist
    pd.DataFrame(columns=columns).to_csv(output_file, index=False)

# Initialise Parameters
window = 500
initial_params = [0.001, 0.5, 0.5, 0.5, 0.1, 0.1]

for i in range(start_iter, len(log_rv) - window):
    start_time = time()

    try:
        # Select window
        series = log_rv[i: window + i]
        # rq_series = rq[i: window + i]

        # Estimation
        result = minimize(
            log_likelihood_harkred,
            initial_params,
            args=(series),
            method='Nelder-Mead',
            options={'xatol': 1e-6, 'fatol': 1e-2, 'maxfev': 2000}
        )

        # Record Estimation Result
        est_params = result.x
        loglik = - result.fun
        b0, b1, b2, b3, q, r = est_params

        # Initialise Filter
        y = HARK(b0, b1, b2, b3, q, r)
        y.construct_kf()
        y.initialise_a(np.mean(rv))
        y.initialise_p(np.var(rv))

        # Run filter
        for l in range(len(series)):
            y.predict()
            y.update(series[l])

        # Generate prediction and record actual
        a_pred, p_pred = y.predict()
        predicted = (y.m @ a_pred).item()
        var = (y.m @ p_pred @ y.m.T).item()
        actual = log_rv[window + i]

        # Combine into rows
        row = pd.DataFrame([{
            'iteration': i,
            'b0': b0,
            'b1': b1,
            'b2': b2,
            'b3': b3,
            'q': q,
            'r': r,
            'loglik': loglik,
            'predicted': predicted,
            'var': var,
            'actual': actual
        }])

        # Append to csv
        print(row)
        row.to_csv(output_file, mode='a', index=False, header=False)
    
    except Exception as e:
        print(f'Error at iteration{i}: {e}')
        continue
        
    # Record Time
    end_time = time()
    print(f"Elapsed time: {end_time - start_time} seconds")

