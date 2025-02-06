import numpy as np
import math
import pandas as pd
import scipy.integrate as integrate
from matplotlib import pyplot as plt
from time import time
from fbm import FBM
from scipy.stats import poisson

class Particle:
    def __init__(self, rv, h: float, delta: int, m: int, t: int):
        self.rv = rv[: t]
        self.h = h
        self.m = m
        self.delta = delta
        self.n = math.floor(t / delta)
        self.j = math.floor(2 * self.n ** math.log(1 + h) * math.log(self.n))
        self.fbm_c = math.sqrt((math.pi * self.h * ((2 * h) - 1)) / math.gamma(2 - (2 * h)) * math.gamma(h + .5) ** 2 * math.sin(math.pi * (h - .5)))
        self.zeta_0 = self.j ** ((- 2) * (h + .5))
        self.zeta_j = self.j ** (4 - 2 * (h + .5))
        self.zeta_ratio = (self.zeta_j / self.zeta_0) ** (1 / self.j)
        self.zetas = [self.zeta_0 * (self.zeta_ratio ** i) for i in range(self.j + 1)]
        self.coef = np.array([integrate.quad(lambda x: self.fbm_c * x ** (- h - .5) / math.gamma(.5 - h), self.zetas[i], self.zetas[i + 1])[0] for i in range(self.j)])
        self.kappa = np.array([(1 / self.coef[i]) * integrate.quad(lambda x: self.fbm_c * x * x ** (- h - .5) / math.gamma(.5 - h), self.zetas[i], self.zetas[i + 1])[0] for i in range(self.j)])
        self.z_0 = self.initialise()
        self.w_0 = np.ones(m) * (1 / m)

    def _simple_resampling(self, weight):
        indices = np.random.choice(self.m, self.m, p=weight)
        return indices
    
    def _systematic_resampling(self, weight):
        c = []
        c.append(weight[0])

        for i in range(self.m - 1):
            c.append(c[i] + weight[i + 1])

        start = np.random.uniform(low=0.0, high = 1 / (self.m))
        indices = []

        for j in range(self.m):
            current = start + (1 / self.m) * (j)
            s = 0
            while (current > c[s]):
                s += 1
            indices.append(s)
        
        return indices
    
    def initialise(self):
        mean_0 = np.zeros(self.j)
        variance_0 = 0.01 * np.eye(self.j)
        z_0 = np.random.multivariate_normal(mean_0, variance_0, size=self.m)
        return z_0
    
    def recursive(self, resampling_method : str = 'simple'):
        z = self.z_0
        w = self.w_0
        x_array = []
        w_array = []
        output = []

        for y in range(self.n):
            # Step (a): Update
            z_update = np.zeros_like(z)
            for i in range(self.m):
                v = np.random.randn()
                z_update[i, :] = z[i, :] * np.exp(- self.kappa * self.delta) + np.sqrt((1 - np.exp(- 2 * self.kappa * self.delta)) / (2 * self.kappa)) * v

            # Step (b): Compute weights
            x = np.sum(self.coef * z, axis=1)
            x = np.exp(x)
            w_update = np.exp(-(1 / 2) * (self.rv[y] ** 2 / x)) / np.sqrt(2 * math.pi * x) * w
            w_update /= np.sum(w_update)

            output.append(np.sum(x * w_update))
            w_array.append(w_update)
            x_array.append(x)

            # Step (c): Resample
            if resampling_method == 'simple':
                resample = self._simple_resampling
            elif resampling_method == 'systematic':
                resample = self._systematic_resampling
            else:
                raise ValueError("Invalid resampling method.")

            n_eff = 1 / np.sum(w_update ** 2)

            if n_eff < self.m / 3:
                indices = resample(w_update)
                z = z_update[indices, :]
                w = self.w_0
            else:
                z = z_update
                w = w_update

        return output

# Test
def load_rv_all(path):
    df_raw = pd.read_csv(path)
    rv_all = df_raw.iloc[:, 1:].to_dict(orient='list')
    return rv_all

def load_rv_one(path, select):
    rv_all = load_rv_all(path)
    rv_select = rv_all[select]
    return rv_select

start_time = time()

rv = load_rv_one('data/rv_dataset.csv', '.SPX')


p = Particle(rv, h=0.14, delta=1, m=600, t=1000)
output = p.recursive()
print(p.z_0.shape)

# # # for i in range(600):
# # #     print(z_initialised[i])

plt.figure(figsize=(10, 6))
plt.plot(range(p.n), p.rv, label='RV', color='red')
plt.plot(range(p.n), output, label='IV', alpha=0.5, color='blue')
plt.xlabel('Time')
plt.ylabel('Values')
plt.grid(True)
plt.show()

# plt.figure(figsize=(10, 6))
# plt.hist(output, density=True, bins=80)
# plt.show()

# end_time = time()
# print(f"Elapsed time: {end_time - start_time} seconds")