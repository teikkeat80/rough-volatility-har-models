import numpy as np
from scipy.optimize import minimize
import math
import pandas as pd
import scipy.integrate as integrate
from matplotlib import pyplot as plt
import pickle
from time import time
from hurst import Hurst
from misc import rmse
import hark
import hark2

def run_hark2_analysis(ix, rv):

    # Compute log rv
    log_rv = np.log(rv)

    # Simulation
    np.random.seed(456)
    sim_b0 = 0.001
    sim_b1 = 0.3
    sim_b2 = 0.2
    sim_b3 = 0.2
    sim_q = 0.3
    sim_r = 0.1
    sim_h = 0.2
    sim_y = hark2.HARK2(sim_b0, sim_b1, sim_b2, sim_b3, sim_q, sim_r, sim_h)
    sim_state, sim_zfilt, sim_ivfilt, sim_obs = sim_y.simulate(10000, np.mean(log_rv))
    sim_obs = sim_obs[-1000:]
    sim_zfilt = sim_zfilt[-1000:]
    sim_ivfilt = sim_ivfilt[-1000:]
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

    # In Sample Estimation Analysis
    np.set_printoptions(suppress=True)

    # HARK2 EST
    init_params = [0.001, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1]

    est = minimize(
        hark2.log_likelihood,
        init_params,
        args=(log_rv),
        method='Nelder-Mead',
        options={'xatol': 1e-6, 'fatol': 1e-2, 'maxfev': 4000}
    )

    with open(f'estm1_result/HARK2_{ix}_EST.pickle', 'wb') as file:
        pickle.dump(est, file)

    with open(f'estm1_result/HARK2_{ix}_EST.pickle', 'rb') as file:
        est = pickle.load(file)

    est_params = est.x
    ll = - est.fun
    aic = (2 * len(init_params)) - (2 * ll)

    print(ix)
    print("-------------------------")
    print(est)
    print('HARK2 Estimated Params: ', np.round(est_params, 4))
    print('HARK2 AIC: ', aic)

    est_b0, est_b1, est_b2, est_b3, est_q, est_r, est_h = est_params
    est_y = hark2.HARK2(est_b0, est_b1, est_b2, est_b3, est_q, est_r, est_h)
    est_y.construct_z(len(log_rv))
    est_y.construct_kf()
    est_y.initialise_a(mean=np.mean(log_rv))
    est_y.initialise_p(var_iv=np.var(log_rv), var_z=0.001)

    est_predicted = []
    est_var = []

    for i in range(len(log_rv)):
        est_a_pred, est_p_pred = est_y.predict()
        est_y.update(log_rv[i])
        est_predicted.append((est_y.m @ est_a_pred).item())
        est_var.append((est_y.m @ est_p_pred @ est_y.m.T).item())

    est_rmse = rmse(log_rv, est_predicted)
    print(f"HARK2 In Sample RMSE: {est_rmse}")

    # plt.plot(log_rv, label="log_rv")
    # plt.plot(est_predicted, label="predicted")
    # plt.xlabel("Time")
    # plt.ylabel("Volatility")
    # plt.title("HARK2 In Sample Estimation")
    # plt.legend()
    # plt.show()

    # Out of Sample Rolling Forecasts
    columns = ['iteration', 'b0', 'b1', 'b2', 'b3', 'q', 'r', 'h', 'loglik', 'predicted', 'var', 'actual']
    output_file = f'fcst1_result/HARK2_{ix}_FCST.csv'
    pd.DataFrame(columns=columns).to_csv(output_file, index=False)
    start_iter = 0
    window = 500

    for i in range(start_iter, len(log_rv) - window):

        # Select window
        series = log_rv[i: window + i]

        # Estimation
        fc = minimize(
            hark2.log_likelihood,
            init_params,
            args=(series),
            method='Nelder-Mead',
            options={'xatol': 1e-6, 'fatol': 1e-2, 'maxfev': 2000}
        )

        # Record Estimation Result
        fc_params = fc.x
        fc_ll = - fc.fun
        fc_b0, fc_b1, fc_b2, fc_b3, fc_q, fc_r, fc_h = fc_params

        # Initialise Filter
        fc_y = hark2.HARK2(fc_b0, fc_b1, fc_b2, fc_b3, fc_q, fc_r, fc_h)
        fc_y.construct_z(len(series))
        fc_y.construct_kf()
        fc_y.initialise_a(mean=np.mean(series))
        fc_y.initialise_p(var_iv=np.var(series), var_z=0.001)

        # Run filter
        for l in range(len(series)):
            fc_y.predict()
            fc_y.update(series[l])

        # Generate prediction and record actual
        fc_a_pred, fc_p_pred = fc_y.predict()
        fc_predicted = (fc_y.m @ fc_a_pred).item()
        fc_var = (fc_y.m @ fc_p_pred @ fc_y.m.T).item()
        fc_actual = log_rv[window + i]

        # Combine into rows
        row = pd.DataFrame([{
            'iteration': i,
            'b0': fc_b0,
            'b1': fc_b1,
            'b2': fc_b2,
            'b3': fc_b3,
            'q': fc_q,
            'r': fc_r,
            'h': fc_h,
            'loglik': fc_ll,
            'predicted': fc_predicted,
            'var': fc_var,
            'actual': fc_actual
        }])

        # Append to csv
        row.to_csv(output_file, mode='a', index=False, header=False)
    
    eval_n = 500
    df = pd.read_csv(output_file)
    fc_predicted_full = df['predicted'].values[-eval_n:]
    fc_actual_full = df['actual'].values[-eval_n:]

    fc_rmse = rmse(fc_actual_full, fc_predicted_full)
    print(f"{ix} HARK2 Out of Sample RMSE: {round(fc_rmse, 6)}")

def run_hark_analysis(ix, rv):

    # Compute log rv
    log_rv = np.log(rv)

    # In Sample Estimation Analysis
    np.set_printoptions(suppress=True)

    # HARK EST
    init_params = [0.001, 0.5, 0.5, 0.5, 0.1, 0.1]

    est = minimize(
        hark.log_likelihood,
        init_params,
        args=(log_rv),
        method='Nelder-Mead',
        options={'xatol': 1e-6, 'fatol': 1e-2, 'maxfev': 4000}
    )

    with open(f'estm1_result/HARK_{ix}_EST.pickle', 'wb') as file:
        pickle.dump(est, file)

    with open(f'estm1_result/HARK_{ix}_EST.pickle', 'rb') as file:
        est = pickle.load(file)

    est_params = est.x
    ll = - est.fun
    aic = (2 * len(init_params)) - (2 * ll)

    print(ix)
    print("-------------------------")
    print(est)
    print('HARK Estimated Params: ', np.round(est_params, 4))
    print('HARK AIC: ', aic)

    est_b0, est_b1, est_b2, est_b3, est_q, est_r = est_params
    est_y = hark.HARK(est_b0, est_b1, est_b2, est_b3, est_q, est_r)
    est_y.construct_kf()
    est_y.initialise_a(mean=np.mean(log_rv))
    est_y.initialise_p(var_iv=np.var(log_rv))

    est_predicted = []
    est_var = []

    for i in range(len(log_rv)):
        est_a_pred, est_p_pred = est_y.predict()
        est_y.update(log_rv[i])
        est_predicted.append((est_y.m @ est_a_pred).item())
        est_var.append((est_y.m @ est_p_pred @ est_y.m.T).item())

    est_rmse = rmse(log_rv, est_predicted)
    print(f"HARK In Sample RMSE: {est_rmse}")

    # plt.plot(log_rv, label="log_rv")
    # plt.plot(est_predicted, label="predicted")
    # plt.xlabel("Time")
    # plt.ylabel("Volatility")
    # plt.title("HARK In Sample Estimation")
    # plt.legend()
    # plt.show()

    # Out of Sample Rolling Forecasts
    columns = ['iteration', 'b0', 'b1', 'b2', 'b3', 'q', 'r', 'loglik', 'predicted', 'var', 'actual']
    output_file = f'fcst1_result/HARK_{ix}_FCST.csv'
    pd.DataFrame(columns=columns).to_csv(output_file, index=False)
    start_iter = 0
    window = 500

    for i in range(start_iter, len(log_rv) - window):

        # Select window
        series = log_rv[i: window + i]

        # Estimation
        fc = minimize(
            hark.log_likelihood,
            init_params,
            args=(series),
            method='Nelder-Mead',
            options={'xatol': 1e-6, 'fatol': 1e-2, 'maxfev': 2000}
        )

        # Record Estimation Result
        fc_params = fc.x
        fc_ll = - fc.fun
        fc_b0, fc_b1, fc_b2, fc_b3, fc_q, fc_r = fc_params

        # Initialise Filter
        fc_y = hark.HARK(fc_b0, fc_b1, fc_b2, fc_b3, fc_q, fc_r)
        fc_y.construct_kf()
        fc_y.initialise_a(mean=np.mean(series))
        fc_y.initialise_p(var_iv=np.var(series))

        # Run filter
        for l in range(len(series)):
            fc_y.predict()
            fc_y.update(series[l])

        # Generate prediction and record actual
        fc_a_pred, fc_p_pred = fc_y.predict()
        fc_predicted = (fc_y.m @ fc_a_pred).item()
        fc_var = (fc_y.m @ fc_p_pred @ fc_y.m.T).item()
        fc_actual = log_rv[window + i]

        # Combine into rows
        row = pd.DataFrame([{
            'iteration': i,
            'b0': fc_b0,
            'b1': fc_b1,
            'b2': fc_b2,
            'b3': fc_b3,
            'q': fc_q,
            'r': fc_r,
            'loglik': fc_ll,
            'predicted': fc_predicted,
            'var': fc_var,
            'actual': fc_actual
        }])

        # Append to csv
        row.to_csv(output_file, mode='a', index=False, header=False)
    
    eval_n = 500
    df = pd.read_csv(output_file)
    fc_predicted_full = df['predicted'].values[-eval_n:]
    fc_actual_full = df['actual'].values[-eval_n:]

    fc_rmse = rmse(fc_actual_full, fc_predicted_full)
    print(f"{ix} HARK Out of Sample RMSE: {round(fc_rmse, 6)}")