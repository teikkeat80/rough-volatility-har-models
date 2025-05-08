import numpy as np
from scipy.optimize import minimize
import math
import pandas as pd
import scipy.integrate as integrate
from matplotlib import pyplot as plt
import pickle
from time import time
from hurst import Hurst

class HARK2:
    def __init__(self, b0, b1, b2, b3, q, r, h):
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.q = q ** 2
        self.r = r ** 2
        self.h = np.clip(h, 0.001, 0.999)
    
    def construct_z(self, n):
        self.j = math.floor(2 * n ** math.log(1 + 0.25) * math.log(n))     # change h to 0.25?
        self.FBM_CONSTANT = math.sqrt((math.pi * self.h * ((2 * self.h) - 1)) / (math.gamma(2 - (2 * self.h)) * math.gamma(self.h + .5) ** 2 * math.sin(math.pi * (self.h - .5))))
        self.zeta_ratio = ((self.j ** (4 - 2 * (self.h + .5))) / (self.j ** ((- 2) * (self.h + .5)))) ** (1 / self.j)
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
            self.p = np.diag(np.concatenate([np.ones(self.j) * var_z, np.ones(22) * var_iv]))
        else:
            self.p = np.diag(np.ones(22) * var_iv)

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

def log_likelihood(params, rv):                # Have to change r and h when with rq
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

############################
#        LOAD DATA         #
############################

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


indices = ["SPX", "GDAXI", "FCHI", "FTSE", "OMXSPI", "N225", "KS11", "HSI"]
idx = indices[0]
log_rv = np.log(load_rv_one('data/rv_dataset.csv', f'.{idx}')) 
# for idx in indices:
# # log_rv = np.log(load_rv('data/SNP500_RV_5min.csv', 'RV'))
#     log_rv = np.log(load_rv_one('data/rv_dataset.csv', f'.{idx}')) 
# # rq = 2 * np.array(load_rv('data/SP500_RQ_5min.csv', 'RQ')) / np.exp(log_rv) ** 2

############################
#        SIMULATION        #
############################

# np.random.seed(456)
# b0 = 0.001
# b1 = 0.3
# b2 = 0.2
# b3 = 0.2
# q = 0.3
# r = 0.1
# h = 0.2
# y = HARK2(b0, b1, b2, b3, q, r, h)
# state, zfilt, ivfilt,obs = y.simulate(10000, np.mean(log_rv))
# obs = obs[-1000:]
# zfilt = zfilt[-1000:]
# ivfilt = ivfilt[-1000:]
# plt.plot(obs, label="RV")
# plt.plot(ivfilt, label="IV")
# plt.plot(zfilt, label="z")
# plt.xlabel("Time")
# plt.ylabel("Volatility")
# plt.title("Simulation")
# plt.legend()
# plt.show()
# q_list = [0.5, 1, 1.5, 2, 3]
# max_delta = 30
# zh = Hurst(zfilt, q_list, max_delta, 'overlap')
# ivh = Hurst(ivfilt, q_list, max_delta, 'overlap')
# zh_est = zh.est_h()
# ivh_est = ivh.est_h()
# print(zh_est)
# print(ivh_est)
# h.plot_scale_m_delta()
# h.plot_scale_zeta_q()
# plt.acorr(obs, maxlags=50)
# plt.xlabel("lag")
# plt.ylabel("ACF")
# plt.show()

############################
#        ESTIMATION        #
############################

# def log_likelihood(params, rv):                # Have to change r and h when with rq
#     b0, b1, b2, b3, q, r, h = params
#     x = HARK2(b0, b1, b2, b3, q, r, h)
#     x.construct_z(len(rv))
#     x.construct_kf()
#     x.initialise_a(np.mean(rv))
#     x.initialise_p(var_iv=np.var(rv), var_z=0.001)
#     sum_ll = 0

#     for t in range(len(rv)):
#         x.predict()
#         v, f, _, _ = x.update(rv[t])
#         if t > 22:
#             sum_ll += math.log(abs(f)) + v.T @ np.linalg.inv(f) @ v

#     ll = - (22 / 2) * len(rv) * math.log(2 * math.pi) - (1 / 2) * sum_ll
#     return -ll

# def log_likelihood(params, h, rv):              # h (and r) is not influencing
#     b0, b1, b2, b3, q, r = params
#     x = HARK2(b0, b1, b2, b3, q, r, h)
#     x.construct_kf(extended=False)
#     x.initialise_a(mean=np.mean(rv), extended=False)
#     x.initialise_p(var_iv=np.var(rv), extended=False)
#     sum_ll = 0

#     for t in range(len(rv)):
#         x.predict()
#         v, f, _, _ = x.update(rv[t])
#         if t > 22:
#           sum_ll += math.log(abs(f)) + v.T @ np.linalg.inv(f) @ v

#     ll = - (22 / 2) * len(rv) * math.log(2 * math.pi) - (1 / 2) * sum_ll
#     return -ll

# def callback(params):
#     print(f"Current Params: {params}, Current LL: {log_likelihood(params, h, log_rv)}")

    # initial_params = [0.001, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1]
    # # initial_params = [ 0.0006,  0.2599,  0.3766,  0.3636,  0.1100,  0.0327,  0.1612]
    # # init_ll = log_likelihood(initial_params, log_rv) 
    # # print(f"initial likelihood: {init_ll}")

    # start_time = time()
    # result = minimize(
    #     log_likelihood,
    #     initial_params,
    #     args=(log_rv),
    #     method='Nelder-Mead',
    #     options={'xatol': 1e-6, 'fatol': 1e-2, 'maxfev': 4000}  ## NM['xatol': 1e-6, 'fatol': 1e-3] | BGFS['eps': 1e-3, 'xrtol': 1e-3]
    # )
    # end_time = time()

    # est_params = result.x
    # final_ll = - result.fun
    # aic = (2 * len(initial_params)) - (2 * final_ll)
    # print(idx)
    # print("-------------------------")
    # print(result)
    # np.set_printoptions(suppress=True)
    # print('Estimated Params: ', np.round(est_params, 4))
    # print('AIC: ', aic)
    # with open(f'result/HARK2_{idx}_EST.pickle', 'wb') as file:
    #     pickle.dump(result, file)

    # print(f"Elapsed time: {end_time - start_time} seconds")

############################
#    ESTIMATION RESULT     #
############################

# for ix in indices:
# with open(f'result/HARK2_{idx}_FULL_EST_WOLA.pickle', 'rb') as file:
#     result = pickle.load(file)
#     print(idx)
#     print("-------------------------")
#     print(result.x[0])

# [RVDATA_1000] HARK WITH RQ                  [0.0019, 0.4662, 0.2722, 0.1360, 0.5259]                   LL: 20077.999010541876
# [RVDATA_1000] HARK WITH ESTIMATED R         [0.0017, 0.6165, 0.1837, 0.1016, 0.4158, 0.2865]           LL: 20078.117941831013
# [RVDATA_1000] HARK2                         [0.0023, 0.5674, 0.2024, 0.1342, 0.3264, -0.0556, 0.1188]  LL: 20078.258425686472
# [RVDATA_1000] HARK2 WITH RQ                 [0.0020, 0.3575, 0.6279, -0.0604, 0.2449, 0.0959]          LL: 20083.291744964972                         

# [RVDATA_FULL] HARK WITH RQ                  [-0.0199, 0.4830, 0.3527, 0.1179, 0.4880]                   LL: 81931.44872459522
# [RVDATA_FULL] HARK WITH ESTIMATED R         [-0.0150, 0.6522, 0.2321, 0.0807, 0.3751, 0.2731]           LL: 81927.6698051827
# [RVDATA_FULL] HARK2                         [0.0030, 0.7154, 0.2126, 0.0586, 0.2396, -0.0988, 0.1008]   LL: 81931.20725129756
# [RVDATA_FULL] HARK2 WITH RQ                 [-0.0135, 0.6528, 0.2448, 0.0708, 0.3454, 0.0115]           LL: 81923.07833454263


# [SPX_FULL] HARK WITH ESTIMATED R       [ 0.0005,  0.8240,  0.0640,  0.1117,  0.1985,  0.2101]           LL: 51175.66006358707   AIC: 102363.32012717414
# [SPX_FULL] HARK2                       [ 0.0006,  0.2599,  0.3766,  0.3636,  0.1100,  0.0327,  0.1612]  LL: 51170.5074265299    AIC: 102355.0148530598  *
# [SPX_FULL] HARK WITH ESTIMATED R       [-0.1401,  0.8192,  0.0612,  0.0912,  0.1983,  0.2107]           LL: 51189.23216113838   AIC: 102390.46432227676 
# [SPX_FULL] HARK2                       [ 0.0010,  0.7435, -0.0657,  0.3222,  0.1087,  0.1094,  0.1724]  LL: 51187.75817799031   AIC: 102389.51635598062 *

# [GDAXI_FULL] HARK WITH ESTIMATED R     [ 0.0009,  0.8514,  0.0544,  0.0941,  0.1432,  0.1840]           LL: 50637.460505579686  AIC: 101286.92101115937                                      
# [GDAXI_FULL] HARK2                     [ 0.0012,  0.0921,  0.2192,  0.6889,  0.1318,  0.0374,  0.2052]  LL: 50632.273330495394  AIC: 101278.54666099079 *
# [GDAXI_FULL] HARK WITH ESTIMATED R     [ 0.0012,  0.8428,  0.0619,  0.0953,  0.1448,  0.1831]           LL: 50654.27852826023   AIC: 101320.55705652045 #
# [GDAXI_FULL] HARK2                     [-0.0012,  0.525 ,  0.2779,  0.1967,  0.1816,  0.1584,  0.3318]  LL: 50666.96021412977   AIC: 101347.92042825955

# [FCHI_FULL] HARK WITH ESTIMATED R      [ 0.0007,  0.7260,  0.1783,  0.0956,  0.1693,  0.1525]           LL: 50498.018350165206  AIC: 101008.03670033041                               
# [FCHI_FULL] HARK2                      [ 0.001 ,  0.3952,  0.343 ,  0.262 ,  0.0833,  0.1073,  0.2153]  LL: 50490.35612641261   AIC: 100994.71225282522 *
# [FCHI_FULL] HARK WITH ESTIMATED R      [ 0.0005,  0.711 ,  0.1904,  0.0984,  0.1726,  0.1501]           LL: 50517.70508697081   AIC: 101047.41017394162 #
# [FCHI_FULL] HARK2                      [ 0.0018,  0.685 ,  0.274 ,  1.4806, -0.1237,  0.1506,  0.2773]  LL: 51297.312538395534  AIC: 102608.62507679107

# [FTSE_FULL] HARK WITH ESTIMATED R      [ 0.0031,  0.8709,  0.0618,  0.0677,  0.1262, -0.229 ]           LL: 50937.78231822865   AIC: 101887.5646364573                                             
# [FTSE_FULL] HARK2                      [ 0.0031,  0.2452,  0.1716,  0.5838,  0.0095,  0.1971,  0.2195]  LL: 50930.15520599725   AIC: 101874.3104119945  *
# [FTSE_FULL] HARK WITH ESTIMATED R      [-0.1171,  0.8497,  0.0754,  0.0504,  0.1294,  0.2281]           LL: 50950.09046540441   AIC: 101912.18093080883
# [FTSE_FULL] HARK2                      [ 0.0011,  0.62  , -0.4274,  0.8076,  0.0201,  0.2008,  0.2236]  LL: 50947.76089784027   AIC: 101909.52179568054 *

# [OMXSPI_FULL] HARK WITH ESTIMATED R    [-0.0002,  0.936 , -0.0122,  0.0759,  0.1218,  0.2143]           LL: 50820.5730916689    AIC: 101653.1461833378
# [OMXSPI_FULL] HARK2                    [ 0.0042,  0.3075, -0.0328,  0.7263,  0.0035,  0.1652,  0.2088]  LL: 50813.50232651696   AIC: 101641.00465303392 *
# [OMXSPI_FULL] HARK WITH ESTIMATED R    [ 0.0016,  0.9367, -0.013 ,  0.0765,  0.1219,  0.2144]           LL: 50840.206897375385  AIC: 101692.41379475077
# [OMXSPI_FULL] HARK2                    [ 0.0011, -0.1636,  0.2294,  0.9346,  0.1041,  0.0977,  0.1975]  LL: 50832.30442002403   AIC: 101678.60884004807 *

# [N225_FULL] HARK WITH ESTIMATED R      [ 0.0002,  0.7364,  0.1548,  0.1085,  0.19  ,  0.1826]           LL: 50886.41570522758   AIC: 101784.83141045515
# [N225_FULL] HARK2                      [-0.0006,  0.3014, -0.2823,  0.9809,  0.088 ,  0.1107,  0.1907]  LL: 50869.20841364947   AIC: 101752.41682729893 *
# [N225_FULL] HARK WITH ESTIMATED R      [-0.    ,  0.7388,  0.1532,  0.1077,  0.1899,  0.1835]           LL: 50910.02047105667   AIC: 101832.04094211334
# [N225_FULL] HARK2                      [ 0.0011,  0.3049, -0.1703,  0.8658,  0.1103,  0.1182,  0.1998]  LL: 50893.57887703671   AIC: 101801.15775407341 *

# [KS11_FULL] HARK WITH ESTIMATED R      [ 0.0009,  0.7893,  0.1036,  0.1071,  0.1402,  0.1627]           LL: 50387.52793491119   AIC: 100787.05586982238  
# [KS11_FULL] HARK2                      [-0.0009,  0.6251, -0.1347,  0.5094,  0.0728,  0.1435,  0.2473]  LL: 50377.18199524665   AIC: 100768.3639904933  *
# [KS11_FULL] HARK WITH ESTIMATED R      [ 0.001 ,  0.7929,  0.1003,  0.1068,  0.1398,  0.1633]           LL: 50412.37747081936   AIC: 100836.75494163872 #
# [KS11_FULL] HARK2                      [-0.0235,  0.3854,  0.3454,  2.7973, -0.1496,  0.1193,  0.2834]  LL: 51199.88076661632   AIC: 102413.76153323264

# [HSI_FULL] HARK WITH ESTIMATED R       [-0.0009,  0.905 , -0.0009,  0.0956,  0.101 ,  0.2016]           LL: 50553.19558548151   AIC: 101118.39117096302
# [HSI_FULL] HARK2                       [ 0.0017,  0.7899, -0.1547,  0.3651, -0.0544,  0.1855,  0.2665]  LL: 50549.79274379712   AIC: 101113.58548759425 *
# [HSI_FULL] HARK WITH ESTIMATED R       [ 0.0016,  0.8982,  0.008 ,  0.094 ,  0.1022,  0.2012]           LL: 50571.24377914674   AIC: 101154.48755829348 #
# [HSI_FULL] HARK2                       [ 0.0015,  1.3234, -0.3139,  0.3452, -0.0222,  0.1876,  0.2758]  LL: 51380.034872630946  AIC: 102774.06974526189

############################
#        EVALUATION        #
############################

# y = HARK2(0.0030, 0.7154, 0.2126, 0.0586, 0.2396, -0.0988, 0.1008)
# y.construct_z(len(log_rv))
# y.construct_kf()
# y.initialise_a(mean=np.mean(log_rv))
# y.initialise_p(var_iv=np.var(log_rv), var_z=0.001)

# # y = HARK2(-0.0150, 0.6522, 0.2321, 0.0807, 0.3751, 0.2731, h)
# # y.construct_kf(extended=False)
# # y.initialise_a(mean=np.mean(log_rv), extended=False)
# # y.initialise_p(var_iv=np.var(log_rv), extended=False)

# predicted = []
# # filtered = []
# # iv_filt = []
# # z_filt = []

# for i in range(len(log_rv)):
#     pred, _ = y.predict()
#     _, _, a, _ = y.update(log_rv[i])
#     # iv_filt.append(a[-22].item())
#     # z_filt.append((y.m[:, :y.j] @ y.a[:y.j, :]).item())
#     # filtered.append((y.m @ a).item())
#     predicted.append((y.m @ pred).item())

# print(f"rmse: {np.sqrt(np.mean((np.array(log_rv) - np.array(predicted)) ** 2))}")

# plt.plot(log_rv, label="True")
# # plt.plot(iv_filt, label="IV")
# # plt.plot(z_filt, label="Error")
# plt.plot(predicted, label="predicted")
# plt.xlabel("Time")
# plt.ylabel("Volatility")
# plt.title("Simulation")
# plt.legend()
# plt.show()

############################
#        ROLLING FC        #
############################

log_rv = log_rv[-505:]
window = 500
initial_params = [0.001, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1]
# h = 0
i = 0
predicted = []
actual = []
hurst = []

while window + i < len(log_rv):
    print(i)
    series = log_rv[i: window + i]
    start_time = time()
    result = minimize(
        log_likelihood,
        initial_params,
        args=(series),
        method='Nelder-Mead',
        options={'xatol': 1e-6, 'fatol': 1e-2, 'maxfev': 4000}
    )
    end_time = time()
    print(f"Elapsed time: {end_time - start_time} seconds")
    est_params = result.x
    b0, b1, b2, b3, q, r, h = est_params
    # hurst.append(h)
    y = HARK2(b0, b1, b2, b3, q, r, h)
    y.construct_z(len(series))
    y.construct_kf()
    y.initialise_a(mean=np.mean(series))
    y.initialise_p(var_iv=np.var(series), var_z=0.001)
    # y.construct_kf(extended=False)
    # y.initialise_a(mean=np.mean(series), extended=False)
    # y.initialise_p(var_iv=np.var(series), extended=False)

    for l in range(len(series)):
        y.predict()
        y.update(series[l])
    pred, var = y.predict()
    predicted.append((y.m @ pred).item())
    actual.append(log_rv[window + i])
    print(b0)
    print(b1)
    print(b2)
    print(b3)
    print(q)
    print(r)
    print(h)
    print(- result.fun)
    print((y.m @ pred).item())
    print(log_rv[window + i])
    print((y.m @ var @ y.m.T).item())
    i += 1

# print(hurst)
# print(predicted)
# print(actual)
print(f"rmse: {np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2))}")

# rmse: 0.245530960106272
# rmse: 0.24221746782240985

# SPX
# HARK2 rmse: 0.3163530437838898
# HARK  rmse: 0.31733840261613433

# GDAXI
# HARK2 rmse: 0.24402523492147946
# HARK  rmse: 0.24366988057396213

# FCHI
# HARK2 rmse: 0.25860024653772506
# HARK  rmse: 0.2585008462218376

# FTSE
# HARK2
# HARK  rmse: 0.3357106106923433

############################
#    EVALUATION RESULT     #
############################

# [RVDATA_1000] HARK WITH RQ                 RMSE: 0.528745743767236    ***
# [RVDATA_1000] HARK WITH ESTIMATED R        RMSE: 0.5282859083752761   *
# [RVDATA_1000] HARK2                        RMSE: 0.5283384990745935   **
# [RVDATA_1000] HARK2 WITH RQ                RMSE: 0.5306447516058641   ****

# [SPX_1000] HARK WITH ESTIMATED R
# [SPX_1000] HARK2

# [GDAXI_1000] HARK WITH ESTIMATED R
# [GDAXI_1000] HARK2

# [FCHI_1000] HARK WITH ESTIMATED R
# [FCHI_1000] HARK2

# [FTSE_1000] HARK WITH ESTIMATED R
# [FTSE_1000] HARK2

# [OMXSPI_1000] HARK WITH ESTIMATED R
# [OMXSPI_1000] HARK2

# [N225_1000] HARK WITH ESTIMATED R
# [N225_1000] HARK2

# [KS11_1000] HARK WITH ESTIMATED R
# [KS11_1000] HARK2

# [HSI_1000] HARK WITH ESTIMATED R
# [HSI_1000] HARK2

# [RVDATA_FULL] HARK WITH RQ                 RMSE: 0.4900791996077867   ****
# [RVDATA_FULL] HARK WITH ESTIMATED R        RMSE: 0.48929639733866465  **
# [RVDATA_FULL] HARK2                        RMSE: 0.4897134942826764   ***
# [RVDATA_FULL] HARK2 WITH RQ                RMSE: 0.48907269510591117  *

# [SPX_FULL] HARK WITH ESTIMATED R
# [SPX_FULL] HARK2

# [GDAXI_FULL] HARK WITH ESTIMATED R
# [GDAXI_FULL] HARK2

# [FCHI_FULL] HARK WITH ESTIMATED R
# [FCHI_FULL] HARK2

# [FTSE_FULL] HARK WITH ESTIMATED R
# [FTSE_FULL] HARK2

# [OMXSPI_FULL] HARK WITH ESTIMATED R
# [OMXSPI_FULL] HARK2

# [N225_FULL] HARK WITH ESTIMATED R
# [N225_FULL] HARK2

# [KS11_FULL] HARK WITH ESTIMATED R
# [KS11_FULL] HARK2

# [HSI_FULL] HARK WITH ESTIMATED R
# [HSI_FULL] HARK2

############################################################################################################################################

# OTHER ESTIMATION RESULTS
# Let h be a free parameter to estimate (EXTENDED)
# RV      [0.0016   0.5597  0.2063  0.1368  0.3308  0.0049  0.118 ] 20078.25391164273
# RV_RQ   [ 0.0009  0.6435  0.172   0.1027  0.2856  0.1031 -0.0031] 20076.849939270985
# .SPX    [0.0004   0.3312  0.3864  0.2821  0.1202  0.1302  0.2114] 19503.846386421465
# .GDAXI  [0.0008   0.5447  0.2368  0.2185  0.133   0.0679  0.2278] 19337.73018237853
# .FCHI   [0.0004   0.4839  0.2692  0.2467  0.141   0.0865  0.2475] 19301.926284970676
# .FTSE   [0.       0.2369  0.491   0.2718  0.1445  0.1216  0.237 ] 19425.780228629636
# .OMXSPI [-0.0003  0.167   0.5362  0.297   0.1715  0.132   0.2246] 19541.043368578667
# .N225   [ 0.0014  0.4569  0.5521 -0.0087  0.0055  0.098   0.2064] 19364.233081627142
# .KS11   [0.0004   0.4838  0.2692 0.2469   0.1363   0.0981 0.2556] 19284.172329431865
# .HSI    [-0.0002  0.3423  0.5447  0.1128  0.1086  0.1627  0.2683] 19355.450422036345

# NORMAL
# RV_RQ  [0.0019 0.4662 0.2722 0.136  0.5259]        20077.999010541876
# RV     [0.0017 0.6165 0.1837 0.1016 0.4158 0.2865] 20078.117941831013







# MINIMIZATION ALGORITHM
# y = HARK2(-0.0203, 0.4366, 0.4015, 0.4283, 0.3843, r, h)         # L-BFGS-B                rmse: 0.5240754153765077
# y = HARK2(0.002, 0.463, 0.274, 0.1366, 0.5285, r, h)             # HARK with constant r    rmse: 0.49509427602675643
# y = HARK2(0.0019, 0.4662, 0.2722, 0.1359, 0.5259, r, h)          # HARK with rq r          rmse: 0.4949339274697459
# y = HARK2(-0.0024, 0.4703, 0.1326, 0.2656, 0.3821, r, h)         # Powell                  rmse: 0.49649219795913263   
# y = HARK2(0.002, 0.4885, 0.2259, 0.1670, 0.3837, r, h)           # Nelder-Mead             rmse: 0.49410641637176334
# y = HARK2(0.002, 0.5096, 0.2153, 0.1607, 0.3706, 0.0909, h)      # Nelder-Mead with est r  rmse: 0.49405790302076347
# y = HARK2(0.0015, 0.5555, 0.1925, 0.1455, 0.3446, 0.1504, h)     # BFGS with est r         rmse: 0.49390784440163293, h 0.493777567831167, 1000 0.5283969806720257
# y = HARK2(0.0019, 0.9831, 0.012, 0.0051, 0.0375, -0.0253, h)     # SPX - Nelder-Mead       rmse: 0.31708323244390396
# y = HARK2(1.7764, -0.8461, 0.2614, 1.9586, -0.0006, -0.0003, h)  # SPX - BFGS           0.3
# y = HARK2(0.0016, 0.5597, 0.2063, 0.1368, 0.3308, 0.118, 0.0049)    # T1E RV   RMSE: 0.5283348230544058, FULL 0.4994310051091735, 
# # # T1E RVRQ RMSE: 0.5281029077618499


# # # # y = HARK2(0.002, 0.463, 0.274, 0.1366, 0.5285, r, h)             # rmse: 0.49797885654056717
# # # # y = HARK2(0.0019, 0.4662, 0.2722, 0.1359, 0.5259, r, h)          # rmse: 0.49774300990806014
# # y = HARK2(-0.1325, 0.8239, 0.11, 0.0376, 0.1657, 0.211, h)             # rmse: 0.31819909370727556
# y = HARK2(0.0019, 0.4662, 0.2722, 0.136, 0.5259, h, r)            # T1 RVRQ   RMSE: 0.528745743767236
