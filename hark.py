import numpy as np
import math

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

def log_likelihood(params, rv): 
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