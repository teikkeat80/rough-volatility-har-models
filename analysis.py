import numpy as np
import pandas as pd
from models.hurst import Hurst
from models.roughvol import RoughVolatility
from models.har import Har, HarOls
import visualisation as vis
import data_processing as dp

def run_analysis(rv):
    # Job Start
    print('Analysis Job Start.............')

    # Parameter(s)
    E = 0.21 # rough error
    N = 30 # train size
    W = 1000 + 1 # rolling window

    # Hurst Estimation
    print('Estimating Hurst Value.............')
    q_list = [0.5, 1, 1.5, 2, 3]
    max_delta = 30
    h = Hurst(rv, q_list, max_delta, 'overlap')
    h_est = h.est_h()
    nu_est = h.est_nu()
    print(f'H: {h_est}')

    # Plot scaling diagrams
    print('Generating Scaling Diagrams.............')
    # vis.plot_scaling_diagram(h)

    # Rough Volatility Forecasts
    print('Training Data - Generate Rough Volatility Forecasts.............')
    roughvol = RoughVolatility(rv=rv, h=h_est, err=E, nu=nu_est)
    roughvol_fc_d = roughvol.moving_window_forecast(train_size=N * 1, back_transform=True)
    roughvol_fc_d = roughvol_fc_d[N * (22 - 1) + 21:]
    # roughvol_fc_w = roughvol.moving_window_forecast(train_size=5, back_transform=True)
    # roughvol_fc_m = roughvol.moving_window_forecast(train_size=22, back_transform=True)

    # Generate Har Variables
    print('Generating Har Variables.............')
    har = Har(rv=rv, beg_index=N * 22 + 21)
    har_y = har.y
    har_x_d = har.xd
    har_x_w = har.xw
    har_x_m = har.xm

    hw = Hurst(dp.ma_rv(rv, 5), q_list, max_delta, 'overlap')
    roughvol_w = RoughVolatility(rv=dp.ma_rv(rv, 5), h=hw.est_h(), err=E, nu=hw.est_nu())
    roughvol_fc_w = roughvol_w.moving_window_forecast(train_size=N * 5, back_transform=True)
    roughvol_fc_w = roughvol_fc_w[N * (22 - 5) + 17:]
    hm = Hurst(dp.ma_rv(rv, 22), q_list, max_delta, 'overlap')
    roughvol_m = RoughVolatility(rv=dp.ma_rv(rv, 22), h=hm.est_h(), err=E, nu=hm.est_nu())
    roughvol_fc_m = roughvol_m.moving_window_forecast(train_size=N * 22, back_transform=True)

    # Examine relationship of HAR variables
    # vis.plot_3d(har_x_d, har_x_w, har_x_m)

    # Combine into df
    print('Combining Variables.............')
    df = pd.DataFrame({
        'y' : har_y,
        'har_x_d' : har_x_d,
        'har_x_w' : har_x_w,
        'har_x_m' : har_x_m,
        'roughvol_fc_d' : roughvol_fc_d, 
        'roughvol_fc_w' : roughvol_fc_w,
        'roughvol_fc_m' : roughvol_fc_m
    })

    # Fit and Predict
    print('rough_har model predicting.............')
    rough_har_mod = HarOls(df, dep='y', indep=['roughvol_fc_d', 'roughvol_fc_w', 'roughvol_fc_m'], rolling_window=W)
    rough_har_pred = rough_har_mod.predict()
    rough_har_mod.fols_summary()
    # rough_har_mod.tols_summary()

    print('har model predicting.............')
    har_mod = HarOls(df, dep='y', indep=['har_x_d', 'har_x_w', 'har_x_m'], rolling_window=W)
    har_pred = har_mod.predict()
    har_mod.fols_summary()
    har_mod.tols_summary()

    print('combined model predicting.............')
    comb_mod = HarOls(df, dep='y', indep=['roughvol_fc_d', 'har_x_w', 'har_x_m'], rolling_window=W)
    comb_pred = comb_mod.predict()
    # comb_mod.fols_summary()
    # comb_mod.tols_summary()

    # DataFrame Operations - Merge with Predictions, Back Transform and Extract Test Portion
    print('Finalising testing data.............')
    df_full = pd.concat([df, 
                         rough_har_pred.rename('rough_har_pred'),
                         har_pred.rename('har_pred'),
                         comb_pred.rename('comb_pred')
                        ], 
                        axis=1)
    # df_bt = np.exp(df_full)
    df_bt = df_full
    df_test_intm = df_bt[1000:]

    l = 0.5
    c_pred = l * df_test_intm['har_pred'] + (1 - l) * df_test_intm['roughvol_fc_d']
    df_test = pd.concat([df_test_intm, c_pred.rename('c_pred')], axis=1)

    # Job Done
    print('Data is ready to test. Job Done.')
    
    return df_test

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