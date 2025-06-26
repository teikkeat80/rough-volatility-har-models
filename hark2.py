import numpy as np
import math
import scipy.integrate as integrate

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
        self.j = math.floor(2 * n ** math.log(1 + 0.25) * math.log(n))     # change h to 0.25?
        self.FBM_CONSTANT = math.sqrt((math.pi * self.h * ((2 * self.h) - 1)) / (math.gamma(2 - (2 * self.h)) * math.gamma(self.h + .5) ** 2 * math.sin(math.pi * (self.h - .5))))
        self.zeta_ratio = ((self.j ** (4 - 2 * (self.h + .5))) / (self.j ** ((- 2) * (self.h + .5)))) ** (1 / self.j)
        self.zetas = [(self.j ** ((- 2) * (self.h + .5))) * (self.zeta_ratio ** i) for i in range(self.j + 1)]
        self.c = np.array([integrate.quad(lambda x: self.FBM_CONSTANT * x ** (- self.h - .5) / math.gamma(.5 - self.h), self.zetas[i], self.zetas[i + 1])[0] for i in range(self.j)])
        self.kappa = np.array([(1 / self.c[i]) * integrate.quad(lambda x: self.FBM_CONSTANT * x * x ** (- self.h - .5) / math.gamma(.5 - self.h), self.zetas[i], self.zetas[i + 1])[0] for i in range(self.j)])

    def construct_kf(self):
        self.k = np.vstack((np.zeros((self.j, 1)), np.array([self.b0]), np.zeros((21, 1))))
        self.t = np.vstack((np.hstack((np.diag(np.exp(- self.kappa)), np.zeros((self.j, 22)))), np.hstack((np.zeros((22, self.j)), np.vstack((np.array([self.b1] + [self.b2 / 4] * 4 + [self.b3 / 17] * 17), np.hstack((np.eye(21), np.zeros((21, 1))))))))))
        self.Q = np.vstack((np.hstack((np.ones((self.j, self.j)), np.zeros((self.j, 22)))), np.hstack((np.zeros((22, self.j)), np.diag([self.q] + [0] * 21)))))
        self.g = np.diag(np.concatenate([np.sqrt((1 - np.exp(- 2 * self.kappa)) / (2 * self.kappa)), np.ones(22)]))
        self.m = np.concatenate([self.c, np.ones(1), np.zeros(21)]).reshape(1, self.j + 22)

    def initialise_a(self, mean):
        self.a = np.concatenate([np.zeros(self.j), np.ones(22) * mean]).reshape(self.j + 22, 1)
    
    def initialise_p(self, var_iv, var_z=0.001):
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
    
    def simulate(self, n, mean):
        self.construct_z(n)
        self.construct_kf()
        self.initialise_a(mean)
        state = []
        state.append(self.a)
        obs = []
        obs.append((self.m @ self.a + np.random.normal(0, self.r)).item())
        zfilt = []
        zfilt.append((self.m[:, :self.j] @ self.a[:self.j, :]).item())
        ivfilt = []
        ivfilt.append((self.m[:, -22:] @ self.a[-22:, :]).item())
        for _ in range(n - 1):
            self.a = self.k + self.t @ self.a + self.g @ np.random.multivariate_normal(np.zeros(self.j + 22), self.Q).reshape(self.j + 22, 1)
            state.append(self.a)
            rv = self.m @ self.a + np.random.normal(0, self.r)
            z = self.m[:, :self.j] @ self.a[:self.j, :]
            iv = self.m[:, -22:] @ self.a[-22:, :]
            obs.append(rv.item())
            zfilt.append(z.item())
            ivfilt.append(iv.item())       
        return state, zfilt, ivfilt, obs

def log_likelihood_hark2(params, rv):
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