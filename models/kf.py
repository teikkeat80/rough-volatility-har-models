import numpy as np
from scipy.optimize import minimize
import math
import pandas as pd
from particle_test import Particle

# HARK Class
class HARK:
    def __init__(self, beta_0, beta_1, beta_2, beta_3, q, rv, h):
        self.rv = rv
        self.h = h
        self.a = self.initialise()[0]
        self.p = self.initialise()[1]
        self.t = np.vstack((np.array([beta_1] + [beta_2 / 4] * 4 + [beta_3 / 17] * 17), np.hstack((np.eye(21), np.zeros((21, 1))))))
        self.c = np.vstack((np.array([beta_0]), np.zeros((21, 1))))
        self.q = np.vstack((np.array([q ** 2] + [0] * 21), np.zeros((21, 22))))
        self.z = np.array([1] + [0] * 21).reshape(1, 22)

    def initialise(self):
        mean_rv = np.mean(self.rv)
        var_rv = np.var(self.rv)
        a_0 = np.ones((22, 1)) * mean_rv
        p_0 = np.eye(22) * var_rv
        return a_0, p_0

    def predict(self):
        # a_t+1 = c + T a_t
        a_pred = self.c + self.t @ self.a
        # P_t+1 = T P_t T' + Q
        p_pred = self.t @ self.p @ self.t.T + self.q

        self.a = a_pred
        self.p = p_pred
        return a_pred, p_pred

    def update(self, rv, h):
        # v_t = RV_t - Z a_t
        v = rv - self.z @ self.a
        # F_t = Z P_t Z^T + h_t
        f = self.z @ self.p @ self.z.T + np.eye(1) * h
        # K_t = T P_t Z^T F_t^-1
        # k = self.t @ self.p @ self.z.T @ np.linalg.inv(f)
        k = self.p @ self.z.T @ np.linalg.inv(f)
        # a_t+1 = c + T a_t + K_t v_t
        # a_upd = self.c + self.t @ self.a + k @ v
        a_upd = self.a + k @ v
        # P_t+1 = T P_t (T - K_t Z)^T + Q
        # p_upd = self.t @ self.p @ (self.t - k @ self.z).T + self.q
        p_upd = self.p - k @ self.z @ self.p

        self.a = a_upd
        self.p = p_upd
        return v, f, a_upd, p_upd

def log_likelihood(params, rv, h):
    beta_0, beta_1, beta_2, beta_3, q = params
    x = HARK(beta_0, beta_1, beta_2, beta_3, q, rv, h)
    sum_ll = 0

    for t in range(len(x.rv)):
        a_pred, p_pred = x.predict()
        v, f, a_upd, p_upd = x.update(x.rv[t], x.h[t])
        sum_ll += math.log(abs(f)) + v.T @ np.linalg.inv(f) @ v
        # print("a_pred: ", a_pred, "| p_pred: ", p_pred)
        # print("a_upd: ", a_upd, "| p_upd: ", p_upd)

    ll = - (22 / 2) * len(x.rv) * math.log(2 * math.pi) - (1 / 2) * sum_ll
    return -ll

def callback(params):
    print(f"Current Params: {params}, Current LL: {log_likelihood(params, rv, h)}")

# Test
def load_rv(path, x):
    df_raw = pd.read_csv(path)
    df_sorted = df_raw.sort_values(by='Date', ignore_index=True)
    rv = df_sorted[x].tolist()
    return rv

def load_rv_all(path):
    df_raw = pd.read_csv(path)
    rv_all = df_raw.iloc[:, 1:].to_dict(orient='list')
    return rv_all

def load_rv_one(path, select):
    rv_all = load_rv_all(path)
    rv_select = rv_all[select]
    return rv_select
psn = 1000
rv = load_rv_one('data/rv_dataset.csv', '.SPX')[:psn]
# rv = load_rv('data/SNP500_RV_5min.csv', 'RV')[:psn]
np.random.seed(123)
h = Particle(rv, h=0.14, delta=1, m=600, t=psn).recursive()
# h = load_rv('data/SP500_RQ_5min.csv', 'RQ')[:psn]
print('h done.')

rv = rv[200:]
h = h[200:]

initial_params = [0.7, 0.5, 0.3, 0.1, 1]
ll = log_likelihood(initial_params, rv, h)
print(ll)
result = minimize(
    log_likelihood,
    initial_params,
    args=(rv, h),
    method='L-BFGS-B',
    callback=callback
)

est_params = result.x
print(result)
print('Estimated Params: ', np.round(est_params, 4))



# # In sample
# beta_0, beta_1, beta_2, beta_3, q = est_params
# hark = HARK(beta_0, beta_1, beta_2, beta_3, q, rv, h)

# x = HARK(2, 2, 2, 2, 2, rv, h)
# a_pred, p_pred = x.predict()
# v, f, _, _ = x.update(1, 2)
# print(x.z.shape)
# print(a_pred.shape)
# print(p_pred.shape)
# print(v.shape)
# print(f.shape)

# Test with RQ data
# Test with lesser data points
# Test with sacling parameter on fbm
# Test with exp q