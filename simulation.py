import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
import math
import statistics

# Real Data
def load_rv_all(path):
    df_raw = pd.read_csv(path)
    rv_all = df_raw.iloc[:, 1:].to_dict(orient='list')
    return rv_all

def load_rv_one(path, select):
    rv_all = load_rv_all(path)
    rv_select = rv_all[select]
    return rv_select

rv_f = load_rv_one('data/rv_dataset.csv', '.SPX')
kde = gaussian_kde(rv_f)
cov = statistics.variance(rv_f)
mean = statistics.mean(rv_f)
print(f'cov: {cov}, mean: {mean}')

# HAR Simulation
# Parameters definition
beta_0 = mean
beta_1 = 0.36
beta_2 = 0.28
beta_3 = 0.28
nsim = 1000
ntime = 3000

# Initialisation
rv_0 = kde.resample(22).flatten().tolist()
rv = rv_0.copy()

for i in range(ntime):
    rv_d = rv[-1]
    rv_w = np.mean(rv[-5:])
    rv_m = np.mean(rv[-22:])
    while True:
        error = np.random.normal(0, 0.05)
        new_rv = beta_0 + beta_1 * rv_d + beta_2 * rv_w + beta_3 * rv_m + error
        if new_rv >= 0:
            break
    rv.append(new_rv)

plt.plot(rv, label="Simulated Realized Volatility")
plt.xlabel("Time")
plt.ylabel("Realized Volatility")
plt.title("HAR Model Simulation")
plt.legend()
plt.show()

# iv_0 = np.random.lognormal()

# # IV and RV Simulation
# # Parameters
# T = 1.0             # Time horizon
# N = 1000            # Number of time steps
# dt = T / N          # Time step size
# S0 = 100            # Initial asset price
# np.random.seed(42)  # For reproducibility

# # Initialize arrays
# t = np.linspace(0, T, N+1)
# W = np.zeros(N+1)  # Brownian motion W_t
# B = np.zeros(N+1)  # Brownian motion B_t
# S = np.zeros(N+1)  # Asset price
# sigma = np.zeros(N+1)  # Instantaneous volatility

# # Initial values
# S[0] = S0

# # Simulate Brownian motions and price process
# for i in range(N):
#     Z1 = np.random.normal(0, 1)
#     Z2 = np.random.normal(0, 1)
#     W[i+1] = W[i] + np.sqrt(dt) * Z1
#     B[i+1] = B[i] + np.sqrt(dt) * Z2
#     sigma[i] = abs(W[i])  # Instantaneous volatility
#     Z3 = np.random.normal(0, 1)
#     S[i+1] = S[i] + sigma[i] * S[i] * np.sqrt(dt) * Z3

# # Compute realized volatility
# returns = np.diff(S)
# realized_volatility = np.sqrt(np.sum(returns**2))

# # Plot results
# plt.figure(figsize=(12, 6))
# plt.plot(t[:-1], sigma[:-1], label="Instantaneous Volatility (σ_t)")
# plt.plot(t[:-1], realized_volatility[:-1], label="Realized Volatility (RV_t)")
# plt.xlabel("Time")
# plt.legend()
# plt.show()


# def load_rv(path, x):
#     df_raw = pd.read_csv(path)
#     df_sorted = df_raw.sort_values(by='Date', ignore_index=True)
#     rv = df_sorted[x].tolist()
#     return rv

# def load_rv_all(path):
#     df_raw = pd.read_csv(path)
#     rv_all = df_raw.iloc[:, 1:].to_dict(orient='list')
#     return rv_all

# def load_rv_one(path, select):
#     rv_all = load_rv_all(path)
#     rv_select = rv_all[select]
#     return rv_select

# rv_f = load_rv_one('data/rv_dataset.csv', '.SPX')
# rv_0 = [x for x in rv_f[:22]]

# nsteps = 1000
# nsim = 1000
# params = [np.mean(rv_0), 0.3, 0.3, 0.2, 0.1]
# beta_0, beta_1, beta_2, beta_3, qm = params
# # h = Particle(rv_f, h=0.14, delta=1, m=600, t=1022).recursive()[22:]
# # # h = load_rv('data/SP500_RQ_5min.csv', 'RQ')[22:1022]

# rv_mean = []
# for j in range(nsim):
#     rv = rv_0.copy()
#     a_t = rv_0
#     rv_s = []
#     a_s = []


#     for i in range(nsteps):
#     #     t = np.vstack((np.array([beta_1] + [beta_2 / 4] * 4 + [beta_3 / 17] * 17), np.hstack((np.eye(21), np.zeros((21, 1))))))
#     #     c = np.vstack((np.array([beta_0]), np.zeros((21, 1))))
#     #     q = np.vstack((np.array([qm ** 2] + [0] * 21), np.zeros((21, 22))))
#     #     z = np.array([1] + [0] * 21).reshape(1, 22)
#     #     n = np.random.normal(0, qm, (22, 1))
#     #     e = np.random.normal(0, h[i], (1, 1))

#     #     rv_t = z @ a_t + e
#     #     a_t = (t @ a_t).reshape(22, 1) + n
#     #     rv_s.append(rv_t[0])
#     #     a_s.append(a_t[0])

#         rv_d = rv[-1]
#         rv_w = np.mean(rv[-5:])
#         rv_m = np.mean(rv[-22:])
#         err = np.random.normal(0, 0.1)
#         rv.append(beta_0 + beta_1 * rv_d + beta_2 * rv_w + beta_3 * rv_m + err)
#     rv_mean.append(rv)
#     print(j)

# rv_mean = np.mean(rv_mean, axis=0)

# plt.figure(figsize=(10, 6))
# plt.plot(range(nsteps + 22), rv_mean, label='RV', color='red')
# plt.xlabel('Time')
# plt.ylabel('Values')
# plt.grid(True)
# plt.show()