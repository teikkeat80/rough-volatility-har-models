import numpy as np
from scipy.optimize import minimize
import math
import pandas as pd
import scipy.integrate as integrate
from time import time
import pickle

class HARK2:
    def __init__(self, b0, b1, b2, b3, q, r, h):
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.q = np.exp(q)
        self.r = np.exp(r)
        self.h = 1 / (1 + np.exp(- h))
    
    def construct_z(self, n):
        self.j = math.floor(2 * n ** math.log(1 + 0.25) * math.log(n))     # change h to 0.25?
        self.FBM_CONSTANT = math.sqrt((math.pi * self.h * ((2 * self.h) - 1)) / (math.gamma(2 - (2 * self.h)) * math.gamma(self.h + .5) ** 2 * math.sin(math.pi * (self.h - .5))))
        self.zeta_ratio = ((self.j ** (4 - 2 * (self.h + .5))) / (self.j ** ((- 2) * (self.h + .5)))) ** (1 / self.j)
        self.zetas = [(self.j ** ((- 2) * (self.h + .5))) * (self.zeta_ratio ** i) for i in range(self.j + 1)]
        self.c = np.array([integrate.quad(lambda x: self.FBM_CONSTANT * x ** (- self.h - .5) / math.gamma(.5 - self.h), self.zetas[i], self.zetas[i + 1])[0] for i in range(self.j)])
        self.kappa = np.array([(1 / self.c[i]) * integrate.quad(lambda x: self.FBM_CONSTANT * x * x ** (- self.h - .5) / math.gamma(.5 - self.h), self.zetas[i], self.zetas[i + 1])[0] for i in range(self.j)])

    def construct_kf(self):
        self.k = np.vstack((np.zeros((self.j, 1)), np.array([self.b0]), np.zeros((21, 1))))
        self.t = np.vstack((np.hstack((np.diag(np.exp(- self.kappa)), np.zeros((self.j, 22)))), np.hstack((np.zeros((22, self.j)), np.vstack((np.array([self.b1] + [self.b2 / 4] * 4 + [self.b3 / 17] * 17), np.hstack((np.eye(21), np.zeros((21, 1))))))))))
        self.Q = np.diag([1] * self.j + [self.q] + [0] * 21)
        self.g = np.diag(np.concatenate([np.sqrt((1 - np.exp(- 2 * self.kappa)) / (2 * self.kappa)), np.ones(22)]))
        self.m = np.concatenate([self.c, np.ones(1), np.zeros(21)]).reshape(1, self.j + 22)

    def initialise_a(self, mean):
        self.a = np.concatenate([np.zeros(self.j), np.ones(22) * mean]).reshape(self.j + 22, 1)
    
    def initialise_p(self, var_iv, var_z=1):
        self.p = np.diag(np.concatenate([np.ones(self.j) * var_z, np.ones(22) * var_iv]))
    
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

def log_likelihood(params, rv):
    b0, b1, b2, b3, q, r, h = params
    x = HARK2(b0, b1, b2, b3, q, r, h)
    x.construct_z(len(rv))
    x.construct_kf()
    x.initialise_a(np.mean(rv))
    x.initialise_p(var_iv=np.var(rv), var_z=0.001)
    sum_ll = 0

    for t in range(len(rv)):
        x.predict()
        v, f, _, _ = x.update(rv[t])
        sum_ll += math.log(abs(f)) + v.T @ np.linalg.inv(f) @ v

    ll = - (22 / 2) * len(rv) * math.log(2 * math.pi) - (1 / 2) * sum_ll
    return -ll

# Load Data
indices = ["SPX", "GDAXI", "FCHI", "FTSE", "OMXSPI", "N225", "KS11", "HSI"]
# indices = ['SPX']
path = 'C:\\Users\\teikkeattee\\ProjProg\\rv_dataset.csv'
df = pd.read_csv(path)
dict = df.iloc[:, 1:].to_dict(orient='list')

for idx in indices:
    # Load Data
    rv = dict[f'.{idx}']
    log_rv = np.log(rv)

    # Initialise Parameters
    initial_params = [0.001, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1]

    start_time = time()

    # Estimation
    result = minimize(
        log_likelihood,
        initial_params,
        args=(log_rv),
        method='Nelder-Mead',
        options={'xatol': 1e-6, 'fatol': 1e-2, 'maxfev': 4000}
    )

    # Record Estimation Result
    est_params = result.x
    loglik = - result.fun
    b0, b1, b2, b3, q, r, h = est_params
    aic = (2 * len(initial_params)) - (2 * loglik)
    print(result)
    np.set_printoptions(suppress=True)
    print('Estimated Params: ', np.round(est_params, 4))
    print('AIC: ', aic)
    with open(f'C:\\Users\\teikkeattee\\ProjProg\\HARK2_{idx}_EST.pickle', 'wb') as file:
        pickle.dump(result, file)

            
    # Record Time
    end_time = time()
    print(f"Elapsed time: {end_time - start_time} seconds")