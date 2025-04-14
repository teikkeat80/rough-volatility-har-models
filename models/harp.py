import numpy as np
from scipy.stats import gaussian_kde
import math
import pandas as pd
from scipy.optimize import minimize

class HARP:
    def __init__(self, beta_0, beta_1, beta_2, beta_3, q, rv, h, m):
        self.rv = rv
        self.h = h
        self.m = m
        self.a = self.initialise()
        self.t = np.vstack((np.array([beta_1] + [beta_2 / 4] * 4 + [beta_3 / 17] * 17), np.hstack((np.eye(21), np.zeros((21, 1))))))
        self.c = np.vstack((np.array([beta_0]), np.zeros((21, 1))))
        self.q = np.vstack((np.array([q] + [0] * 21), np.zeros((21, 22))))
        self.z = np.array([1] + [0] * 21).reshape(1, 22)
        self.w = np.ones(m) * (1 / m)

    def initialise(self):
        a_0 = []
        for _ in range(self.m):
            a_0.append(gaussian_kde(self.rv).resample(22).flatten().tolist())
        return np.array(a_0)
    
    def update(self):
        a_update = np.zeros_like(self.a)
        for i in range(self.m):
            a_update[i, :] = np.random.multivariate_normal((self.c + self.t @ self.a[i, :].reshape(22, 1)).flatten(), self.q)
        self.a = a_update
        return a_update
    
    def weights(self, rv, h):
        weights = np.zeros_like(self.w)
        for i in range(self.m):
            weights[i] = (np.exp(-(1 / 2) * ((rv - self.z @ self.a[i, :]) ** 2 / h)) / np.sqrt(2 * math.pi * h))
        # maxl_weights = np.max(np.log(weights))
        # sl_weights = np.log(weights) - maxl_weights
        # norm_weights = np.exp(sl_weights) / np.sum(np.exp(sl_weights))
        norm_weights = weights / np.sum(weights)
        self.w = norm_weights
        return weights, norm_weights
    
    def resample(self):
        n_eff = 1 / np.sum(self.w ** 2)
        if n_eff < self.m / 3:
            indices = np.random.choice(self.m, self.m, p=self.w)
            self.a = self.a[indices, :]
            self.w = np.ones(self.m) * (1 / self.m)

# Define log likelihood function

def log_likelihood(params, rv, h, m):
    beta_0, beta_1, beta_2, beta_3, q = params
    x = HARP(beta_0, beta_1, beta_2, beta_3, q, rv, h, m)
    ll = 0

    for t in range(len(x.rv)):
        a_update = x.update()
        weights, _ = x.weights(x.rv[t], x.h[t])
        x.resample()
        mu = (1 / x.m) * np.sum(weights)
        sigma = (1 / (x.m - 1)) * np.sum((weights - mu) ** 2)
        # ll += np.log(np.exp(maxl_weights + np.log(np.sum(np.exp(sl_weights))) - np.log(x.m)))
        ll += np.log(mu) + (sigma / (2 * x.m * mu ** 2))
        print(np.sum(weights))
        if np.isnan(ll):
            print(f"mu: {mu}")
            print(f"sigma: {sigma}")
            print(f"ll: {ll}")
            print(f"rv:{x.rv[t]}")
            print(f"h: {x.h[t]}")
            print(f"time:{t}")
            print(a_update)
            print("---wei---")
            print(weights)
            raise SystemExit
    
    return -ll

def callback(params):
    print(f"Current Params: {params}, Current LL: {log_likelihood(params, rv, h, m)}")

# Define EM algorithm

# def em(init_params, rv, h, m):
#     beta_0, beta_1, beta_2, beta_3, q = init_params
#     x = HARP(beta_0, beta_1, beta_2, beta_3, q, rv, h, m)
#     ll = 0



# Test

def load_rv(path, x):
    df_raw = pd.read_csv(path)
    df_sorted = df_raw.sort_values(by='Date', ignore_index=True)
    rv = df_sorted[x].tolist()
    return rv

psn = 500
rv = load_rv('data/SNP500_RV_5min.csv', 'RV')[:psn]
h = load_rv('data/SP500_RQ_5min.csv', 'RQ')[:psn]
m = 600

np.random.seed(123)
# x = HARP(0.1, 0.5, 0.3, 0.1, 0.1, rv, h, m)
# # print(x.w[0])
# # print(np.zeros_like(x.w))
# # print(np.exp(-(1 / 2) * ((x.rv[0] - x.z @ x.a[0, :]) ** 2 / x.h[0])) / np.sqrt(2 * math.pi * x.h[0]))
# a_update = x.update()
# x.weights(x.rv[0], x.h[0])
# x.resample()
# print(x.a)
# print("---")
# print(a_update)
# # print(x.w)


initial_params = [0.1, 0.5, 0.3, 0.1, 0.1]
ll = log_likelihood(initial_params, rv, h, m)
print(f"initial likelihood: {ll}")
result = minimize(
    log_likelihood,
    initial_params,
    args=(rv, h, m),
    method='L-BFGS-B',
    callback=callback
)

est_params = result.x
print(result)
print('Estimated Params: ', np.round(est_params, 4))