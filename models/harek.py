import numpy as np
from scipy.optimize import minimize
import math
import pandas as pd
import scipy.integrate as integrate
from matplotlib import pyplot as plt

class HARK2:
    def __init__(self, b0, b1, b2, b3, q, r, h):
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.q = q ** 2
        self.r = r ** 2
        self.h = h
    
    def construct_z(self, n):
        self.j = math.floor(2 * n ** math.log(1 + self.h) * math.log(n))
        self.FBM_CONSTANT = math.sqrt((math.pi * self.h * ((2 * self.h) - 1)) / math.gamma(2 - (2 * self.h)) * math.gamma(self.h + .5) ** 2 * math.sin(math.pi * (self.h - .5)))
        self.zeta_ratio = ((self.j ** (4 - 2 * (h + .5))) / (self.j ** ((- 2) * (h + .5)))) ** (1 / self.j)
        self.zetas = [(self.j ** ((- 2) * (self.h + .5))) * (self.zeta_ratio ** i) for i in range(self.j + 1)]
        self.c = np.array([integrate.quad(lambda x: self.FBM_CONSTANT * x ** (- self.h - .5) / math.gamma(.5 - self.h), self.zetas[i], self.zetas[i + 1])[0] for i in range(self.j)])
        self.kappa = np.array([(1 / self.c[i]) * integrate.quad(lambda x: self.FBM_CONSTANT * x * x ** (- self.h - .5) / math.gamma(.5 - self.h), self.zetas[i], self.zetas[i + 1])[0] for i in range(self.j)])

    def construct_kf(self, extended=True):
        if extended:
            self.k = np.vstack((np.zeros((self.j, 1)), np.array([self.b0]), np.zeros((21, 1))))
            self.t = np.vstack((np.hstack((np.diag(np.exp(- self.kappa)), np.zeros((self.j, 22)))), np.hstack((np.zeros((22, self.j)), np.vstack((np.array([self.b1] + [self.b2 / 4] * 4 + [self.b3 / 17] * 17), np.hstack((np.eye(21), np.zeros((21, 1))))))))))
            self.Q = np.diag([1] * self.j + [self.q] + [0] * 21)
            self.g = np.diag(np.concatenate([np.sqrt((1 - np.exp(- 2 * self.kappa)) / (2 * self.kappa)), np.ones(22)]))
            self.m = np.concatenate([self.c, np.ones(1), np.zeros(21)]).reshape(1, self.j + 22)
        else:
            self.k = np.vstack((np.array([self.b0]), np.zeros((21, 1))))
            self.t = np.vstack((np.array([self.b1] + [self.b2 / 4] * 4 + [self.b3 / 17] * 17), np.hstack((np.eye(21), np.zeros((21, 1))))))
            self.Q = np.diag([self.q] + [0] * 21)
            self.g = np.eye(22)
            self.m = np.concatenate([np.ones(1), np.zeros(21)]).reshape(1, 22)

    def initialise_a(self, mean, extended=True):
        if extended:
            self.a = np.concatenate([np.zeros(self.j), np.ones(22) * mean]).reshape(self.j + 22, 1)
        else:
            self.a = (np.ones(22) * mean).reshape(22, 1)
    
    def initialise_p(self, var_iv, var_z=1, extended=True):
        if extended:
            self.p = np.diag(np.concatenate([np.random.normal(0, var_z, size=self.j), np.ones(22) * var_iv]))
        else:
            self.p = np.diag(np.ones(22) * var_iv)

    def simulate(self, n, mean):
        self.construct_z(n)
        self.construct_kf()
        self.initialise_a(mean)
        state = []
        state.append(self.a.flatten().tolist())
        obs = []
        obs.append((self.m @ self.a + np.random.normal(0, self.r)).item())
        for _ in range(n - 1):
            self.a = self.k + self.t @ self.a + self.g @ np.random.multivariate_normal(np.zeros(self.j + 22), self.Q).reshape(self.j + 22, 1)
            state.append(self.a.flatten().tolist())
            rv = self.m @ self.a + np.random.normal(0, self.r)
            obs.append(rv.item())
        return state, obs
    
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

    def simulate_iv(self, n, mean):
        self.construct_z(n)
        self.construct_kf()
        self.initialise_a(mean)
        a = self.a[-22:, :]
        t = self.t[-22:, -22:]
        m = self.m[:, -22:]
        g = np.diag(self.g[-22:, -22:]).reshape(22, 1)
        k = self.k[-22:, :]
        Q = self.Q[-22:, -22:]
        collector = []
        collector.append((m @ a + np.random.normal(0, self.r)).item())
        for _ in range(n):
            a = k + t @ a + (g @ np.random.multivariate_normal(np.zeros(22), Q)).reshape(22, 1)
            x = m @ a + np.random.normal(0, self.r)
            collector.append(x.item())
        return collector
    
    def simulate_z(self, n, mean):
        self.construct_z(n)
        self.construct_kf()
        self.initialise_a(mean)
        a = self.a[:self.j, :]
        t = self.t[:self.j, :self.j]
        m = self.m[:, :self.j]
        g = np.diag(self.g[:self.j, :self.j]).reshape(self.j, 1)
        k = self.k[:self.j, :]
        collector = []
        collector.append((m @ a + np.random.normal(0, self.r)).item())
        for _ in range(n):
            a = k + t @ a + (g @ np.random.normal(0, self.q, 1)).reshape(self.j, 1)
            x = m @ a + np.random.normal(0, self.r)
            collector.append(x.item())
        return collector

def load_rv(path, x):
    df_raw = pd.read_csv(path)
    df_sorted = df_raw.sort_values(by='Date', ignore_index=True)
    rv = df_sorted[x].tolist()
    return rv

np.random.seed(123)

log_rv = np.log(load_rv('data/SNP500_RV_5min.csv', 'RV')[:1000])
rq = 2 * np.array(load_rv('data/SP500_RQ_5min.csv', 'RQ')[:1000]) / np.exp(log_rv) ** 2
b0 = 0.1
b1 = 0.5
b2 = 0.3
b3 = 0.1
q = 0.1
r = 0.01
h = 0.14

# y = HARK2(b0, b1, b2, b3, q, r, h)
# state, obs = y.simulate(10000, np.mean(log_rv))
# obs = obs[-1000:]
# plt.plot(obs, label="predicted")
# plt.xlabel("Time")
# plt.ylabel("Volatility")
# plt.title("Simulation")
# plt.legend()
# plt.show()

# y = HARK2(-0.0203, 0.4366, 0.4015, 0.4283, 0.3843, r, h)         # L-BFGS-B                rmse: 0.5240754153765077
# # # y = HARK2(0.002, 0.463, 0.274, 0.1366, 0.5285, r, h)             # HARK with constant r    rmse: 0.49509427602675643
# # # y = HARK2(0.0019, 0.4662, 0.2722, 0.1359, 0.5259, r, h)          # HARK with rq r          rmse: 0.4949339274697459
# # # y = HARK2(-0.0024, 0.4703, 0.1326, 0.2656, 0.3821, r, h)         # Powell                  rmse: 0.49649219795913263   
# # # y = HARK2(0.002, 0.4885, 0.2259, 0.1670, 0.3837, r, h)           # Nelder-Mead             rmse: 0.49410641637176334
# # y = HARK2(0.002, 0.5096, 0.2153, 0.1607, 0.3706, 0.0909, h)      # Nelder-Mead with est r  rmse: 0.49405790302076347
# y.construct_z(len(log_rv))
# y.construct_kf()
# y.initialise_a(mean=np.mean(log_rv))
# y.initialise_p(var_iv=np.var(log_rv), var_z=0.01)

# # # y = HARK2(0.002, 0.463, 0.274, 0.1366, 0.5285, r, h)             # rmse: 0.49797885654056717
# # # y = HARK2(0.0019, 0.4662, 0.2722, 0.1359, 0.5259, r, h)          # rmse: 0.49774300990806014
# # # y.construct_kf(extended=False)
# # # y.initialise_a(mean=np.mean(log_rv), extended=False)
# # # y.initialise_p(var_iv=np.var(log_rv), extended=False)

# predicted = []
# filtered = []
# iv_filt = []
# z_filt = []

# for i in range(len(log_rv)):
#     pred, _ = y.predict()
#     _, _, a, _ = y.update(log_rv[i])
#     iv_filt.append(a[-22].item())
#     z_filt.append((y.m[:, :y.j] @ y.a[:y.j, :]).item())
#     filtered.append((y.m @ a).item())
#     predicted.append((y.m @ pred).item())

# print(f"rmse: {np.sqrt(np.mean((np.array(log_rv) - np.array(predicted)) ** 2))}")

# plt.plot(log_rv, label="True")
# plt.plot(iv_filt, label="IV")
# plt.plot(z_filt, label="Error")
# # plt.plot(filtered, label="predicted")
# plt.xlabel("Time")
# plt.ylabel("Volatility")
# plt.title("Simulation")
# plt.legend()
# plt.show()

def log_likelihood(params, h, rv):
    b0, b1, b2, b3, q, r = params
    x = HARK2(b0, b1, b2, b3, q, r, h)
    x.construct_z(len(rv))
    x.construct_kf()
    x.initialise_a(np.mean(rv))
    x.initialise_p(var_iv=np.var(rv), var_z=0.01)
    sum_ll = 0

    for t in range(len(rv)):
        x.predict()
        v, f, _, _ = x.update(rv[t])
        sum_ll += math.log(abs(f)) + v.T @ np.linalg.inv(f) @ v

    ll = - (22 / 2) * len(rv) * math.log(2 * math.pi) - (1 / 2) * sum_ll
    return -ll

# def log_likelihood(params, r, h, rv, rq):
#     b0, b1, b2, b3, q = params
#     x = HARK2(b0, b1, b2, b3, q, r, h)
#     x.construct_kf(extended=False)
#     x.initialise_a(mean=np.mean(rv), extended=False)
#     x.initialise_p(var_iv=np.var(rv), extended=False)
#     sum_ll = 0

#     for t in range(len(rv)):
#         x.predict()
#         v, f, _, _ = x.update(rv[t], rq[t])
#         sum_ll += math.log(abs(f)) + v.T @ np.linalg.inv(f) @ v

#     ll = - (22 / 2) * len(rv) * math.log(2 * math.pi) - (1 / 2) * sum_ll
#     return -ll

def callback(params):
    print(f"Current Params: {params}, Current LL: {log_likelihood(params, h, log_rv)}")

# actual_params = [b0, b1, b2, b3]
initial_params = [0.002, 0.5, 0.5, 0.5, 0.1, 0.1]
init_ll = log_likelihood(initial_params, h, log_rv)
# act_ll = log_likelihood(actual_params, q=0.2, r=0.2, h=0.15, rv=obs)
print(f"initial likelihood: {init_ll}")
# print(f"actual likelihood: {act_ll}")
result = minimize(
    log_likelihood,
    initial_params,
    args=(h, log_rv),
    method='BFGS',
    options={'eps': 1e-3, 'xrtol': 1e-3},  ## 'xatol': 1e-6, 'fatol': 1e-3
    callback=callback
)

est_params = result.x
print(result)
print('Estimated Params: ', np.round(est_params, 4))

