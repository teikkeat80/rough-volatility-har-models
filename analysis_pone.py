import numpy as np
import pandas as pd
from hurst import Hurst
from roughvol import RoughVolatility
import visualisation as vis
import data_processing as dp
from ls import Ols
from misc import rmse, qlike, mae

def run_analysis(rv):

    # Compute partial vol and log
    prv_d = rv
    prv_w = dp.ma_rv(rv, 5)
    prv_m = dp.ma_rv(rv, 22)
    lprv_d = np.log(prv_d)
    lprv_w = np.log(prv_w)
    lprv_m = np.log(prv_m)

    # Job Start
    print('Analysis Job Start.............')

    # Hurst Estimation
    print('Estimating Hurst Values.............')
    q_list = [0.5, 1, 1.5, 2, 3]
    max_delta = 30
    hd = Hurst(np.sqrt(prv_d), q_list, max_delta)
    hd_est = hd.est_h()
    nud_est = hd.est_nu()
    hw = Hurst(np.sqrt(prv_w), q_list, max_delta)
    hw_est = hw.est_h()
    nuw_est = hw.est_nu()
    hm = Hurst(np.sqrt(prv_m), q_list, max_delta)
    hm_est = hm.est_h()
    num_est = hm.est_nu()   
    print(f'Daily Series Hurst: {hd_est}, Nu: {nud_est}')
    print(f'Weekly Series Hurst: {hw_est}, Nu: {nuw_est}')
    print(f'Monthly Series Hurst: {hm_est}, Nu: {num_est}')

    # Plot scaling diagrams
    print('Generating Scaling Diagrams.............')
    # vis.plot_scaling_diagram(hd)
    # vis.plot_scaling_diagram(hw)
    # vis.plot_scaling_diagram(hm)

    # Rough Volatility Predictions
    print('Training Data - Generate Rough Volatility Linear Predictions.............')
    error_rhar = 0.22
    error_r = 0.22
    train_size = 500

    roughvol = RoughVolatility(rv, hd_est, error_r, nud_est)
    roughvol_fc = roughvol.moving_window_lpred(train_size=train_size, bt=True)

    rhar_d = RoughVolatility(prv_d, hd_est, error_rhar, nud_est)
    lrhar_x_d = rhar_d.moving_window_lpred(train_size=train_size, bt=False)
    rhar_x_d = rhar_d.moving_window_lpred(train_size=train_size, bt=True)

    rhar_w = RoughVolatility(prv_w, hw_est, error_rhar, nuw_est)
    lrhar_x_w = rhar_w.moving_window_lpred(train_size=train_size, bt=False)
    rhar_x_w = rhar_w.moving_window_lpred(train_size=train_size, bt=True)

    rhar_m = RoughVolatility(prv_m, hm_est, error_rhar, num_est)
    lrhar_x_m = rhar_m.moving_window_lpred(train_size=train_size, bt=False)
    rhar_x_m = rhar_m.moving_window_lpred(train_size=train_size, bt=True)

    # Align starting index and get variables
    y = rv[521:]

    har_x_d = prv_d[520: len(prv_d) - 1]
    har_x_w = prv_w[516: len(prv_w) - 1]
    har_x_m = prv_m[499: len(prv_m) - 1]

    roughvol_fc = roughvol_fc[21:]

    rhar_x_d = rhar_x_d[21:]
    rhar_x_w = rhar_x_w[17:]
    rhar_x_m = rhar_x_m[:]

    lrhar_x_d = lrhar_x_d[21:]
    lrhar_x_w = lrhar_x_w[17:]
    lrhar_x_m = lrhar_x_m[:]

    lhar_x_d = lprv_d[520: len(lprv_d) - 1]
    lhar_x_w = lprv_w[516: len(lprv_w) - 1]
    lhar_x_m = lprv_m[499: len(lprv_m) - 1]

    # Fit and Predict
    rolling_window = 501
    dep = y
    ldep = np.log(y)

    print('rough_har model predicting.............')
    rhar_indep = np.array([rhar_x_d, rhar_x_w, rhar_x_m]).T
    rhar_mod = Ols(dep, rhar_indep, rolling_window)
    rhar_rpred = rhar_mod.rol_predict()
    rhar_fpred = rhar_mod.fol_predict()

    print('har model predicting.............')
    har_indep = np.array([har_x_d, har_x_w, har_x_m]).T
    har_mod = Ols(dep, har_indep, rolling_window)
    har_rpred = har_mod.rol_predict()
    har_fpred = har_mod.fol_predict()

    print('log_rough_har model predicting.............')
    lrhar_indep = np.array([lrhar_x_d, lrhar_x_w, lrhar_x_m]).T
    lrhar_mod = Ols(ldep, lrhar_indep, rolling_window)
    lrhar_rpred = lrhar_mod.rol_predict()
    lrhar_fpred = lrhar_mod.fol_predict()
    lrhar_predvar = 4 * sum([rhar_d.cond_var() * rhar_d.nu ** 2, rhar_w.cond_var() * rhar_w.nu ** 2, rhar_m.cond_var() * rhar_m.nu ** 2])
    lrhar_rpred = np.exp(lrhar_rpred + lrhar_predvar / 2)
    lrhar_fpred = np.exp(lrhar_fpred + lrhar_predvar / 2)

    print('log_har model predicting.............')
    lhar_indep = np.array([lhar_x_d, lhar_x_w, lhar_x_m]).T
    lhar_mod = Ols(ldep, lhar_indep, rolling_window)
    lhar_rpred = lhar_mod.rol_predict()
    lhar_fpred = lhar_mod.fol_predict()

    # In sample estimation
    print('In sample estimation.............')
    rhar_mod.fols(nw=True, pr=True)
    rhar_ise_rmse = rmse(y, rhar_fpred)

    har_mod.fols(nw=True, pr=True)
    har_ise_rmse = rmse(y, har_fpred)

    lrhar_mod.fols(pr=True)
    lrhar_ise_rmse = rmse(y, lrhar_fpred)
    
    lhar_mod.fols(pr=True)
    lhar_ise_rmse = rmse(y, lhar_fpred)

    print(f"RHAR In Sample rmse: {round(rhar_ise_rmse, 6)}")
    print(f"HAR In Sample rmse: {round(har_ise_rmse, 6)}")
    print(f"lRHAR In Sample rmse: {round(lrhar_ise_rmse, 6)}")
    print(f"lHAR In Sample rmse: {round(lhar_ise_rmse, 6)}")

    # Out of sample analysis
    print('Out of sample analysis.............')
    test_size = 500
    actual = y[-test_size:]
    rhar_predicted = rhar_rpred[-test_size:]
    har_predicted = har_rpred[-test_size:]
    lrhar_predicted = lrhar_rpred[-test_size:]
    lhar_predicted = lhar_rpred[-test_size:]
    roughvol_predicted = roughvol_fc[-test_size:]

    rhar_osa_rmse = rmse(actual, rhar_predicted)
    rhar_osa_qlike = qlike(actual, rhar_predicted)
    rhar_osa_mae = mae(actual, rhar_predicted)

    har_osa_rmse = rmse(actual, har_predicted)
    har_osa_qlike = qlike(actual, har_predicted)
    har_osa_mae = mae(actual, har_predicted)

    lrhar_osa_rmse = rmse(actual, lrhar_predicted)
    lrhar_osa_qlike = qlike(actual, lrhar_predicted)
    lrhar_osa_mae = mae(actual, lrhar_predicted)

    lhar_osa_rmse = rmse(actual, lhar_predicted)
    lhar_osa_qlike = qlike(actual, lhar_predicted)
    lhar_osa_mae = mae(actual, lhar_predicted)

    roughvol_osa_rmse = rmse(actual, roughvol_predicted)
    roughvol_osa_qlike = qlike(actual, roughvol_predicted)
    roughvol_osa_mae = mae(actual, roughvol_predicted)

    print(f"RHAR Out of Sample rmse: {round(rhar_osa_rmse, 6)}")
    print(f"HAR Out of Sample rmse: {round(har_osa_rmse, 6)}")
    print(f"lRHAR Out of Sample rmse: {round(lrhar_osa_rmse, 6)}")
    print(f"lHAR Out of Sample rmse: {round(lhar_osa_rmse, 6)}")
    print(f"Rough Volatility Out of Sample rmse: {round(roughvol_osa_rmse, 6)}")

    print(f"RHAR Out of Sample qlike: {round(rhar_osa_qlike, 6)}")
    print(f"HAR Out of Sample qlike: {round(har_osa_qlike, 6)}")
    print(f"lRHAR Out of Sample qlike: {round(lrhar_osa_qlike, 6)}")
    print(f"lHAR Out of Sample qlike: {round(lhar_osa_qlike, 6)}")
    print(f"Rough Volatility Out of Sample qlike: {round(roughvol_osa_qlike, 6)}")

    print(f"RHAR Out of Sample mae: {round(rhar_osa_mae, 6)}")
    print(f"HAR Out of Sample mae: {round(har_osa_mae, 6)}")
    print(f"lRHAR Out of Sample mae: {round(lrhar_osa_mae, 6)}")
    print(f"lHAR Out of Sample mae: {round(lhar_osa_mae, 6)}")
    print(f"Rough Volatility Out of Sample mae: {round(roughvol_osa_mae, 6)}")
    # vis.plot_comparison(actual, rhar_predicted, "rhar")
    # vis.plot_comparison(actual, har_predicted, "har")
    # vis.plot_comparison(actual, roughvol_predicted, "roughvol")














    # # Parameter(s)
    # E = 0.5 # rough error
    # N = 500 # train size
    # W = 1000 + 1 # rolling window

    # # Hurst Estimation
    # print('Estimating Hurst Value.............')
    # q_list = [0.5, 1, 1.5, 2, 3]
    # max_delta = 30
    # h = Hurst(np.sqrt(rv), q_list, max_delta, 'overlap')
    # h_est = h.est_h()
    # nu_est = h.est_nu()
    # print(f'H: {h_est}')

    # # Plot scaling diagrams
    # print('Generating Scaling Diagrams.............')
    # # vis.plot_scaling_diagram(h)

    # # Rough Volatility Forecasts
    # print('Training Data - Generate Rough Volatility Forecasts.............')
    # roughvol = RoughVolatility(rv, h_est, E, nu_est)
    # roughvol_fc_d = roughvol.moving_window_lpred(train_size=N)
    # roughvol_fc_d = roughvol_fc_d[21:]
    # # roughvol_fc_w = roughvol.moving_window_forecast(train_size=5, back_transform=True)
    # # roughvol_fc_m = roughvol.moving_window_forecast(train_size=22, back_transform=True)

    # # Generate Har Variables
    # print('Generating Har Variables.............')
    # har = Har(rv=rv, beg_index=N + 21)
    # har_y = har.y
    # har_x_d = har.xd
    # har_x_w = har.xw
    # har_x_m = har.xm

    # hw = Hurst(np.sqrt(dp.ma_rv(rv, 5)), q_list, max_delta, 'overlap')
    # roughvol_w = RoughVolatility(dp.ma_rv(rv, 5), hw.est_h(), E, hw.est_nu())
    # roughvol_fc_w = roughvol_w.moving_window_lpred(train_size=N)
    # roughvol_fc_w = roughvol_fc_w[17:]
    # hm = Hurst(np.sqrt(dp.ma_rv(rv, 22)), q_list, max_delta, 'overlap')
    # roughvol_m = RoughVolatility(dp.ma_rv(rv, 22), hm.est_h(), E, hm.est_nu())
    # roughvol_fc_m = roughvol_m.moving_window_lpred(train_size=N)

    # # Examine relationship of HAR variables
    # # vis.plot_3d(har_x_d, har_x_w, har_x_m)

    # # Combine into df
    # print('Combining Variables.............')
    # df = pd.DataFrame({
    #     'y' : har_y,
    #     'har_x_d' : har_x_d,
    #     'har_x_w' : har_x_w,
    #     'har_x_m' : har_x_m,
    #     'roughvol_fc_d' : roughvol_fc_d, 
    #     'roughvol_fc_w' : roughvol_fc_w,
    #     'roughvol_fc_m' : roughvol_fc_m
    # })

    # # Fit and Predict
    # print('rough_har model predicting.............')
    # rough_har_mod = HarOls(df, dep='y', indep=['roughvol_fc_d', 'roughvol_fc_w', 'roughvol_fc_m'], rolling_window=W)
    # rough_har_pred = rough_har_mod.predict()
    # rh_fols_par = rough_har_mod.fols_summary()
    # print(rh_fols_par[0])
    # # rough_har_mod.tols_summary()

    # print('har model predicting.............')
    # har_mod = HarOls(df, dep='y', indep=['har_x_d', 'har_x_w', 'har_x_m'], rolling_window=W)
    # har_pred = har_mod.predict()
    # h_fols_par = har_mod.fols_summary()
    # # har_mod.tols_summary()

    # # print('combined model predicting.............')
    # # comb_mod = HarOls(df, dep='y', indep=['roughvol_fc_d', 'roughvol_fc_w', 'har_x_m'], rolling_window=W)
    # # comb_pred = comb_mod.predict()
    # # comb_mod.fols_summary()
    # # # comb_mod.tols_summary()

    # # DataFrame Operations - Merge with Predictions, Back Transform and Extract Test Portion
    # print('Finalising testing data.............')
    # df_full = pd.concat([df, 
    #                      rough_har_pred.rename('rough_har_pred'),
    #                      har_pred.rename('har_pred')
    #                     #  comb_pred.rename('comb_pred')
    #                     ], 
    #                     axis=1)

    # df_test = df_full[1000:]

    # # Job Done
    # print('Data is ready to test. Job Done.')
    
    # return df_test

# # Tuning parameters
# E = 0.2 #rough error
# N = 500 #train size
# W = 1000 + 1 #rolling window

# # Reading data
# df_raw = pd.read_csv('SNP500_RV_5min.csv')
# df_sorted = df_raw.sort_values(by='Date', ignore_index=True)
# rv = df_sorted['RV'].tolist()

# # Hurst Estimation
# q_list = [0.5, 1, 1.5, 2, 3]
# max_delta = 30
# h = Hurst(rv, q_list, max_delta, 'overlap')
# h_est = h.est_h()

# # # Plot scaling diagrams
# h.plot_scale_m_delta()
# h.plot_scale_zeta_q()

# # Rough Volatility Forecasts
# roughvol = RoughVolatility(rv=rv, h=h_est, err=E)
# roughvol_fc_d = roughvol.moving_window_forecast(train_size=N, k=1)
# roughvol_fc_w = roughvol.moving_window_forecast(train_size=N, k=5)
# roughvol_fc_m = roughvol.moving_window_forecast(train_size=N, k=22)

# # Generate Har Variables
# har = Har(rv=rv, beg_index=N)
# har_y = har.y
# har_x_d = har.xd
# har_x_w = har.xw
# har_x_m = har.xm

# # Combine into df
# df = pd.DataFrame({
#     'y' : har_y,
#     'har_x_d' : har_x_d,
#     'har_x_w' : har_x_w,
#     'har_x_m' : har_x_m,
#     'roughvol_fc_d' : roughvol_fc_d,
#     'roughvol_fc_w' : roughvol_fc_w,
#     'roughvol_fc_m' : roughvol_fc_m
# })

# # Fit and Predict
# rough_har_mod = HarOls(df, dep='y', indep=['roughvol_fc_d', 'roughvol_fc_w', 'roughvol_fc_m'], rolling_window=W)
# rough_har_pred = rough_har_mod.predict()
# # rough_har_mod.fols_summary()
# # rough_har_mod.tols_summary()

# har_mod = HarOls(df, dep='y', indep=['har_x_d', 'har_x_w', 'har_x_m'], rolling_window=W)
# har_pred = har_mod.predict()
# # har_mod.fols_summary()
# # har_mod.tols_summary()

# comb_mod = HarOls(df, dep='y', indep=['har_x_d', 'har_x_w', 'har_x_m', 'roughvol_fc_d'], rolling_window=W)
# comb_pred = comb_mod.predict()
# # comb_mod.fols_summary()
# # comb_mod.tols_summary()

# # DataFrame Operations - Merge with Predictions, Back Transform and Extract Test Portion
# df_full = pd.concat([df, rough_har_pred.rename('rough_har_pred'), har_pred.rename('har_pred'), comb_pred.rename('comb_pred')], axis=1)
# df_bt = np.exp(df_full)
# df_test = df_bt[1000:]

# # MSE Calculation
# Mse(df_test, 'y', 'rough_har_pred').print_rmse()
# Mse(df_test, 'y', 'har_pred').print_rmse()
# Mse(df_test, 'y', 'comb_pred').print_rmse()
# Mse(df_test, 'y', 'roughvol_fc_d').print_rmse()

# # Plot
# plt.figure(figsize=(10, 6))
# plt.plot(df_test.index, df_test['y'], label='RV')
# plt.plot(df_test.index, df_test['rough_har_pred'], label='RoughHAR')
# plt.title('Title')
# plt.xlabel('Date')
# plt.ylabel('Values')
# plt.legend()
# plt.grid(True)
# plt.show()