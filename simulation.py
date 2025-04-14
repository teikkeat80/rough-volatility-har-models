import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
import math
import statistics
from scipy.optimize import minimize
import particles  # core module
from particles import distributions as dists  # where probability distributions are defined
from particles import state_space_models as ssm  # where state-space models are defined
from particles.collectors import Moments
import seaborn as sb
from time import time


# Real Data
def load_rv_all(path):
    df_raw = pd.read_csv(path)
    rv_all = df_raw.iloc[:, 1:].to_dict(orient='list')
    return rv_all

def load_rv_one(path, select):
    rv_all = load_rv_all(path)
    rv_select = rv_all[select]
    return rv_select

def load_rv(path, x):
    df_raw = pd.read_csv(path)
    df_sorted = df_raw.sort_values(by='Date', ignore_index=True)
    rv = df_sorted[x].tolist()
    return rv

np.random.seed(123)
start_time = time()

# HARP Simulation
index = np.random.choice(500)
rv = np.log(load_rv('data/SNP500_RV_5min.csv', 'RV'))
mean = statistics.mean(rv)
cov = statistics.variance(rv)
print(mean)
print(cov)

# Parameters
c = 0.1
t = 0.5
q = 5
h = cov
ntime = 10000
a = mean
a_samples = []
rv_samples = []

for i in range(ntime):
    eta = np.random.normal(0, math.sqrt(q))
    a = c + t * a + eta
    epsilon = np.random.normal(0, math.sqrt(h))
    rv = a + epsilon
    rv_samples.append(rv)
    a_samples.append(a)

    # while True:
    #     eta = np.random.normal(0, q)
    #     a = c + t * a + eta
    #     if a >= 0:
    #         break
    # a_samples.append(a)
    # while True:
    #     epsilon = np.random.normal(0, h)
    #     rv = a + epsilon
    #     if rv >= 0:
    #         break
    # rv_samples.append(rv)

a_final = a_samples[-1000:]
rv_final = rv_samples[-1000:]
a_0_mean = statistics.mean(a_final)
p_0_cov = statistics.variance(a_final)

# Simulation Plot
# plt.plot(rv_final, label="Simulated Realized Volatility")
# plt.plot(a_final, label="Simulated Integrated Volatility")
# plt.xlabel("Time")
# plt.ylabel("Volatility")
# plt.title("Simulation")
# plt.legend()
# plt.show()

class Kalman:
    def __init__(self, c, t, q, rv, h):
        self.rv = rv
        self.h = h
        self.a = a_0
        self.p = p_0
        self.t = t
        self.c = c
        self.q = q
    
    def predict(self):
        a_pred = self.c + self.t * self.a
        p_pred = (self.t ** 2) * self.p + self.q

        self.a = a_pred
        self.p = p_pred
        return a_pred, p_pred

    def update(self, rv):
        v = rv - self.a
        f = self.p + self.h
        k = self.p / f
        a_upd = self.a + k * v
        p_upd = (1 - k) * self.p

        self.a = a_upd
        self.p = p_upd
        return v, f, a_upd, p_upd
    
def log_likelihood_kf(params, rv, h):
    c, t, q = params
    x = Kalman(c, t, q, rv, h)
    sum_ll = 0

    for t in range(len(x.rv)):
        x.predict()
        v, f, _, _ = x.update(x.rv[t])
        sum_ll += math.log(abs(f)) + (v ** 2) / f

    ll = - (1 / 2) * len(x.rv) * math.log(2 * math.pi) - (1 / 2) * sum_ll
    return -ll

class Particle:
    def __init__(self, t, c, q, rv, h, m):
        self.rv = rv
        self.c = c
        self.h = h
        self.m = m
        self.a = self.initialise()
        self.t = t
        self.q = q
        self.w = np.ones(m) * (1 / m)

    def initialise(self):
        return np.random.normal(loc=a_0_mean, scale=np.sqrt(p_0_cov), size=self.m)
    
    def update(self):
        a_update = np.zeros_like(self.a)
        for i in range(self.m):
            a_update[i] = np.random.normal(self.c + self.t * self.a[i], math.sqrt(self.q))
        self.a = a_update
        return a_update
    
    def weights(self, rv):
        weights = np.zeros_like(self.w)
        for i in range(self.m):
            weights[i] = (np.exp(-(1 / 2) * ((rv - self.a[i]) ** 2 / self.h)) / np.sqrt(2 * math.pi * self.h)) * self.w[i]
        norm_weights = weights / np.sum(weights)
        self.w = norm_weights
        return weights, norm_weights
    
    def resample(self):
        n_eff = 1 / np.sum(self.w ** 2)
        if n_eff < self.m / 3:
            indices = np.random.choice(self.m, self.m, p=self.w)
            self.a = self.a[indices]
            self.w = np.ones(self.m) * (1 / self.m)
    
    def filter(self):
        particles_history = []
        weights_history = []
        exp_filtered_values = []
        for t in range(len(self.rv)):
            ph = self.update()
            _, norm_weights = self.weights(self.rv[t])
            self.resample()
            particles_history.append(ph)
            weights_history.append(norm_weights)
            exp_filtered_values.append(np.sum(ph * norm_weights))
        return particles_history, weights_history, exp_filtered_values
    

# Define log likelihood function

def log_likelihood_particle(params, c, q, rv, h, m):
    t = params
    x = Particle(t, c, q, rv, h, m)
    ll = 0

    for t in range(len(x.rv)):
        a_update = x.update()
        weights, _ = x.weights(x.rv[t])
        x.resample()
        mu = (1 / x.m) * np.sum(weights)
        # sigma = (1 / (x.m - 1)) * np.sum((weights - mu) ** 2)
        ll += np.log(mu) # + (sigma / (2 * x.m * mu ** 2))
        if np.isnan(ll):
            print(f"mu: {mu}")
            # print(f"sigma: {sigma}")
            print(f"ll: {ll}")
            print(f"rv:{x.rv[t]}")
            print(f"time:{t}")
            print(a_update)
            print("---wei---")
            print(weights)
            raise SystemExit
    
    return -ll

def callback(params):
    print(f"Current Params: {params}, Current LL: {log_likelihood_particle(params, c, q, rv, h, m)}")

#################################### (UNIVARIATE)

class SSModel(ssm.StateSpaceModel):
    def PX0(self):
        return dists.Normal(loc=-0.42, scale=np.sqrt(0.95))
    def PX(self, t, xp):
        return dists.Normal(loc=self.cons_m + self.t_coefm * xp, scale=np.sqrt(5))
    def PY(self, t, xp, x):
        return dists.Normal(loc=x, scale=np.sqrt(0.95))
    
ss_model = SSModel(cons_m=0.1, t_coefm=0.5)
state, data = ss_model.simulate(1000)
# ss_model_alt = SSModel(cons_m=2.1, t_coefm=0.45)
# fk_model = ssm.Bootstrap(ssm=ss_model_alt, data=data)
# pf = particles.SMC(fk=fk_model, N=1000, resampling='multinomial', collect=[Moments()], store_history=True)
# pf.run()

# plt.figure()
# plt.plot(data, label="data")
# plt.plot([m['mean'] for m in pf.summaries.moments], label="filtered")
# plt.legend()
# plt.show()

# rv = data.copy()
# c = 0.1
# h = 0.28
# q = 7
# m = 300
# a_filtered = []
# print(f"x: {5 / np.sum(np.array(data) ** 2)}")

prior_dict = {'cons_m': dists.Normal(1.0, 1), 't_coefm': dists.Normal(0.1, 1)}
prior = dists.StructDist(prior_dict)
from particles import mcmc
pmmh = mcmc.PMMH(ssm_cls=SSModel, prior=prior, data=data, Nx=50, niter=1000)
pmmh.run()
burnin = 100  # discard the 100 first iterations
for p in prior_dict.keys():
    spl = pmmh.chain.theta[p][burnin:]
    print(f"{p}: {np.mean(spl)}")
    plt.figure()
    plt.plot(spl)
    plt.title(p)
    plt.show()


#################################### (MULTIVARIATE)

# class SSModel(ssm.StateSpaceModel):

#     default_params = {
#         'b0': 0.1,
#         'b1': 0.5,
#         'b2': 0.3,
#         'b3': 0.1,
#         'q': 5
#     }

#     def PX0(self):
#         return dists.IndepProd(dists.Normal(loc=-0.42, scale=np.sqrt(0.95)), 
#                                dists.Normal(loc=-0.42, scale=np.sqrt(0.95)),
#                                dists.Normal(loc=-0.42, scale=np.sqrt(0.95)),
#                                dists.Normal(loc=-0.42, scale=np.sqrt(0.95)),
#                                dists.Normal(loc=-0.42, scale=np.sqrt(0.95)),
#                                dists.Normal(loc=-0.42, scale=np.sqrt(0.95)),
#                                dists.Normal(loc=-0.42, scale=np.sqrt(0.95)),
#                                dists.Normal(loc=-0.42, scale=np.sqrt(0.95)),
#                                dists.Normal(loc=-0.42, scale=np.sqrt(0.95)),
#                                dists.Normal(loc=-0.42, scale=np.sqrt(0.95)),
#                                dists.Normal(loc=-0.42, scale=np.sqrt(0.95)),
#                                dists.Normal(loc=-0.42, scale=np.sqrt(0.95)),
#                                dists.Normal(loc=-0.42, scale=np.sqrt(0.95)),
#                                dists.Normal(loc=-0.42, scale=np.sqrt(0.95)),
#                                dists.Normal(loc=-0.42, scale=np.sqrt(0.95)),
#                                dists.Normal(loc=-0.42, scale=np.sqrt(0.95)),
#                                dists.Normal(loc=-0.42, scale=np.sqrt(0.95)),
#                                dists.Normal(loc=-0.42, scale=np.sqrt(0.95)),
#                                dists.Normal(loc=-0.42, scale=np.sqrt(0.95)),
#                                dists.Normal(loc=-0.42, scale=np.sqrt(0.95)),
#                                dists.Normal(loc=-0.42, scale=np.sqrt(0.95)),
#                                dists.Normal(loc=-0.42, scale=np.sqrt(0.95))
#                               )
#     def PX(self, t, xp):
#         return dists.IndepProd(dists.Normal(loc=self.b0 + self.b1 * xp[:, 0] + (self.b2 / 4) * xp[:, 1] + (self.b2 / 4) * xp[:, 2] + (self.b2 / 4) * xp[:, 3] + (self.b2 / 4) * xp[:, 4] + (self.b3 / 17) * xp[:, 5] + (self.b3 / 17) * xp[:, 6] + (self.b3 / 17) * xp[:, 7] + (self.b3 / 17) * xp[:, 8] + (self.b3 / 17) * xp[:, 9] + (self.b3 / 17) * xp[:, 10] + (self.b3 / 17) * xp[:, 11] + (self.b3 / 17) * xp[:, 12] + (self.b3 / 17) * xp[:, 13] + (self.b3 / 17) * xp[:, 14] + (self.b3 / 17) * xp[:, 15] + (self.b3 / 17) * xp[:, 16] + (self.b3 / 17) * xp[:, 17] + (self.b3 / 17) * xp[:, 18] + (self.b3 / 17) * xp[:, 19] + (self.b3 / 17) * xp[:, 20] + (self.b3 / 17) * xp[:, 21], scale=np.sqrt(self.q)), 
#                                dists.Normal(xp[:, 0], scale=0),
#                                dists.Normal(xp[:, 1], scale=0),
#                                dists.Normal(xp[:, 2], scale=0),
#                                dists.Normal(xp[:, 3], scale=0),
#                                dists.Normal(xp[:, 4], scale=0),
#                                dists.Normal(xp[:, 5], scale=0),
#                                dists.Normal(xp[:, 6], scale=0),
#                                dists.Normal(xp[:, 7], scale=0),
#                                dists.Normal(xp[:, 8], scale=0),
#                                dists.Normal(xp[:, 9], scale=0),
#                                dists.Normal(xp[:, 10], scale=0),
#                                dists.Normal(xp[:, 11], scale=0),
#                                dists.Normal(xp[:, 12], scale=0),
#                                dists.Normal(xp[:, 13], scale=0),
#                                dists.Normal(xp[:, 14], scale=0),
#                                dists.Normal(xp[:, 15], scale=0),
#                                dists.Normal(xp[:, 16], scale=0),
#                                dists.Normal(xp[:, 17], scale=0),
#                                dists.Normal(xp[:, 18], scale=0),
#                                dists.Normal(xp[:, 19], scale=0),
#                                dists.Normal(xp[:, 20], scale=0)
#                               )
#     def PY(self, t, xp, x):
#         return dists.Normal(loc=x[:, 0], scale=np.sqrt(0.95))
    
# ss_model = SSModel(b0=0.2, b1=0.5, b2=0.3, b3=0.1, q=5)
# state, data = ss_model.simulate(1000)
# ss_model_alt = SSModel(b0=0.3596, b1=0.579, b2=0.4012, b3=-0.0655, q=1.2853)
# fk_model = ssm.Bootstrap(ssm=ss_model_alt, data=data)
# pf = particles.SMC(fk=fk_model, N=300, resampling='multinomial', collect=[Moments()], store_history=True)
# pf.run()

# plt.figure()
# plt.plot(data, label="data")
# plt.plot([m['mean'][0] for m in pf.summaries.moments], label="filtered")
# plt.legend()
# plt.show()

# prior_dict = {
#     'b0': dists.Normal(0.1, 0.1), 
#     'b1': dists.Normal(0.5, 0.1),
#     'b2': dists.Normal(0.5, 0.1),
#     'b3': dists.Normal(0.5, 0.1),
#     'q': dists.Normal(1.0, 0.1)
# }
# prior = dists.StructDist(prior_dict)
# from particles import mcmc
# pmmh = mcmc.PMMH(ssm_cls=SSModel, prior=prior, data=data, Nx=300, niter=3000)
# pmmh.run()
# burnin = 1000  # discard the 100 first iterations
# for p in prior_dict.keys():
#     spl = pmmh.chain.theta[p][burnin:]
#     print(f"{p}: {np.mean(spl)}")
#     # plt.figure()
#     # plt.hist(spl, 50)
#     # plt.title(p)
#     # plt.show()

####################################

# rv = rv_final.copy()
# c = 0.1
# h = 0.28
# q = 7
# m = 300
# # a_filtered = []

# # x = Particle(0.5, c, q, rv, h, m)
# # for t in range(len(x.rv)):
# #     a_update = x.update()
# #     _, norm_weights = x.weights(x.rv[t])
# #     x.resample()
# #     a_filtered.append(np.sum(a_update * norm_weights))

# # plt.plot(a_filtered, label="Filtered")
# # plt.plot(a_final, label="True")
# # plt.xlabel("Time")
# # plt.ylabel("Volatility")
# # plt.title("Simulation")
# # plt.legend()
# # plt.show()


# initial_params = 0.1
# ll = log_likelihood_particle(initial_params, c, q, rv, h, m)
# print(f"initial likelihood: {ll}")
# result = minimize(
#     log_likelihood_particle,
#     initial_params,
#     args=(c, q, rv, h, m),
#     method='Powell',
#     callback=callback
# )

# est_params = result.x
# print(result)
# print('Estimated Params: ', np.round(est_params, 4))


#########################

# rv_f = load_rv_one('data/rv_dataset.csv', '.SPX')
# kde = gaussian_kde(rv_f)
# cov = statistics.variance(rv_f)
# mean = statistics.mean(rv_f)
# print(f'cov: {cov}, mean: {mean}')

# # HAR Simulation
# # Parameters definition
# beta_0 = mean
# beta_1 = 0.36
# beta_2 = 0.28
# beta_3 = 0.28
# nsim = 1000
# ntime = 3000

# # Initialisation
# rv_0 = kde.resample(22).flatten().tolist()
# rv = rv_0.copy()

# for i in range(ntime):
#     rv_d = rv[-1]
#     rv_w = np.mean(rv[-5:])
#     rv_m = np.mean(rv[-22:])
#     while True:
#         error = np.random.normal(0, 0.001)
#         new_rv = beta_0 + beta_1 * rv_d + beta_2 * rv_w + beta_3 * rv_m + error
#         if new_rv >= 0:
#             break
#     rv.append(new_rv)

# plt.plot(rv, label="Simulated Realized Volatility")
# plt.xlabel("Time")
# plt.ylabel("Realized Volatility")
# plt.title("HAR Model Simulation")
# plt.legend()
# plt.show()

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

end_time = time()
print(f"Elapsed time: {end_time - start_time} seconds")