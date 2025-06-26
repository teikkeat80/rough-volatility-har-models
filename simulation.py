import pandas as pd
from matplotlib import pyplot as plt
from hurst import Hurst
import visualisation as vis
from data_processing import load_rv
from hark2 import HARK2
import numpy as np


log_rv = np.log(load_rv('data/SNP500_RV_5min.csv', 'RV'))

# Simulation
np.random.seed(12)
sim_b0 = -0.0008
sim_b1 = 0.6
sim_b2 = 0.23
sim_b3 = 0.07
sim_q = 0.36
sim_r = 0.26
sim_h = 0.4
sim_y = HARK2(sim_b0, sim_b1, sim_b2, sim_b3, sim_q, sim_r, sim_h)
sim_y.construct_z(1000)
_, sim_e, sim_liv, sim_lrv = sim_y.simulate(10000, np.mean(log_rv))
sim_lrv = sim_lrv[-1000:]
sim_e = sim_e[-1000:]
sim_liv = sim_liv[-1000:]
sim_rv = np.exp(sim_lrv)
sim_iv = np.exp(sim_liv)
true_lrv = log_rv[-1000:]
true_rv = np.exp(true_lrv)

# Simulation plots
sim_combined = [sim_lrv, sim_liv, sim_e]
vis.plot_superimpose_series(sim_combined, [r'$\log\,RV$', r'$\log\,IV$', r'$\epsilon$'])
vis.plot_kd(sim_lrv)
vis.plot_acorr(sim_rv, 0)

vis.plot_series(true_lrv, r'$\log\,RV$')
vis.plot_series(sim_lrv, r'$\log\,RV$')
vis.plot_series(sim_liv, r'$\log\,IV$')
vis.plot_series(sim_e, r'$\epsilon$')
vis.plot_series(true_rv, r'$RV$')
vis.plot_series(sim_rv, r'$RV$')
vis.plot_series(sim_iv, r'$IV$')

# Density plots
vis.plot_kd(sim_lrv)
vis.plot_kd(true_lrv)

# Hurst estimations
q_list = [0.5, 1, 1.5, 2, 3]
max_delta = 30
eh = Hurst(sim_e, q_list, max_delta, False)
rvh = Hurst(np.sqrt(sim_rv), q_list, max_delta)
ivh = Hurst(np.sqrt(sim_iv), q_list, max_delta)
true_rvh = Hurst(np.sqrt(true_rv), q_list, max_delta)
eh_est = eh.est_h()
rvh_est = rvh.est_h()
ivh_est = ivh.est_h()
true_rvh_est = true_rvh.est_h()
print('H for log me: ', eh_est)
print('H for rv: ', rvh_est)
print('H for iv: ', ivh_est)
print('H for true rv: ', true_rvh_est)
vis.plot_scaling_diagram(eh)
vis.plot_scaling_diagram(rvh)
vis.plot_scaling_diagram(true_rvh)

# ACF
vis.plot_acorr(sim_iv, 0)
vis.plot_acorr(sim_rv, 0)
vis.plot_acorr(true_rv, 0)