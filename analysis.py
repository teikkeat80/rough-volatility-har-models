import numpy as np
from dp import *
from hurst import Hurst
from loss import *
from rhark import RHARK
import fbmsim as fs
import visualisation as vis
from mle import *
from sspred import *
from rfsv import RFSV
from ls import Ols

def p1_empirical_analysis():

    # Load data
    rv_all = load_marketspotrv()

    for i, rv in rv_all.items():

        print(f'Analysis for {i}')

        # Compute partial vol and log
        prv_d = rv
        prv_w = ma(rv, 5)
        prv_m = ma(rv, 22)
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
        vis.plot_scaling_diagram(hd)
        vis.plot_scaling_diagram(hw)
        vis.plot_scaling_diagram(hm)

        # Rough Volatility Pre Samples
        print('Training Data - Generate RFSV Linear Predictions.............')
        error_rhar = 0.22
        error_r = 0.22
        train_size = 500

        roughvol = RFSV(rv, hd_est, error_r, nud_est)
        roughvol_fc = roughvol.moving_window_lpred(train_size=train_size, bt=True)

        rhar_d = RFSV(prv_d, hd_est, error_rhar, nud_est)
        lrhar_x_d = rhar_d.moving_window_lpred(train_size=train_size, bt=False)
        rhar_x_d = rhar_d.moving_window_lpred(train_size=train_size, bt=True)

        rhar_w = RFSV(prv_w, hw_est, error_rhar, nuw_est)
        lrhar_x_w = rhar_w.moving_window_lpred(train_size=train_size, bt=False)
        rhar_x_w = rhar_w.moving_window_lpred(train_size=train_size, bt=True)

        rhar_m = RFSV(prv_m, hm_est, error_rhar, num_est)
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
        lrhar_rpred = np.exp(lrhar_rpred + lrhar_mod.rol_res_var() / 2)
        lrhar_fpred = np.exp(lrhar_fpred + lrhar_mod.fol_res_var() / 2)

        print('log_har model predicting.............')
        lhar_indep = np.array([lhar_x_d, lhar_x_w, lhar_x_m]).T
        lhar_mod = Ols(ldep, lhar_indep, rolling_window)
        lhar_rpred = lhar_mod.rol_predict()
        lhar_fpred = lhar_mod.fol_predict()
        lhar_rpred = np.exp(lhar_rpred + lhar_mod.rol_res_var() / 2)
        lhar_fpred = np.exp(lhar_fpred + lhar_mod.fol_res_var() / 2)

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

        print('.......................................................')
        print(f"RHAR In Sample rmse: {rhar_ise_rmse}")
        print(f"HAR In Sample rmse: {har_ise_rmse}")
        print(f"lRHAR In Sample rmse: {lrhar_ise_rmse}")
        print(f"lHAR In Sample rmse: {lhar_ise_rmse}")
        print('.......................................................')

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

        print('......................RMSE.............................')
        print(f"RHAR Out of Sample rmse: {rhar_osa_rmse}")
        print(f"HAR Out of Sample rmse: {har_osa_rmse}")
        print(f"lRHAR Out of Sample rmse: {lrhar_osa_rmse}")
        print(f"lHAR Out of Sample rmse: {lhar_osa_rmse}")
        print(f"Rough Volatility Out of Sample rmse: {roughvol_osa_rmse}")
        print('.......................................................')

        print('..................QLIKE................................')
        print(f"RHAR Out of Sample qlike: {rhar_osa_qlike}")
        print(f"HAR Out of Sample qlike: {har_osa_qlike}")
        print(f"lRHAR Out of Sample qlike: {lrhar_osa_qlike}")
        print(f"lHAR Out of Sample qlike: {lhar_osa_qlike}")
        print(f"Rough Volatility Out of Sample qlike: {roughvol_osa_qlike}")
        print('.......................................................')

        print('........................MAE...........................')
        print(f"RHAR Out of Sample mae: {rhar_osa_mae}")
        print(f"HAR Out of Sample mae: {har_osa_mae}")
        print(f"lRHAR Out of Sample mae: {lrhar_osa_mae}")
        print(f"lHAR Out of Sample mae: {lhar_osa_mae}")
        print(f"Rough Volatility Out of Sample mae: {roughvol_osa_mae}")
        print('.......................................................')

        vis.plot_comparison(actual, rhar_predicted)
        vis.plot_comparison(actual, har_predicted)
        vis.plot_comparison(actual, roughvol_predicted)
        vis.plot_comparison(actual, lrhar_predicted)
        vis.plot_comparison(actual, lhar_predicted)

def p2_simulation_analysis():

    # Set N
    n = 1000

    # # FBM approximation
    # hurst_set = [0.1, 0.4]

    # for h in hurst_set:
    #     fs.FBMAPP(h, n * 10).simulate()
    #     fs.FBMCLS(h, n).simulate()
    #     fs.FBMDH(h, n).simulate()

    # Rough HARK simulation
    rv = load_spxfuturesrv()
    log_rv = np.log(rv)
    sim_b0, sim_b1, sim_b2, sim_b3, sim_q, sim_r, sim_h = 0.01, 0.6, 0.25, 0.1, 0.35, 0.0, 0.48
    np.random.seed(123)
    sim_y = RHARK(sim_b0, sim_b1, sim_b2, sim_b3, sim_q, sim_r, sim_h)
    sim_y.construct_z(n)
    _, _, _, sim_lrv = sim_y.simulate(n * 10, np.mean(log_rv))
    sim_lrv = sim_lrv[-n:]
    sim_rv = np.exp(sim_lrv)
    true_lrv = log_rv[-n:]
    true_rv = rv[-n:]

    # Time series plot
    vis.plot_series(true_lrv, r'$\log\,RV$')
    vis.plot_series(sim_lrv, r'$\log\,RV$')
    vis.plot_series(true_rv, r'$RV$')
    vis.plot_series(sim_rv, r'$RV$')

    # Density plots
    vis.plot_kd(sim_lrv)
    vis.plot_kd(true_lrv)

    # Hurst estimations
    q_list = [0.5, 1, 1.5, 2, 3]
    max_delta = 30
    rvh = Hurst(np.sqrt(sim_rv), q_list, max_delta)
    true_rvh = Hurst(np.sqrt(true_rv), q_list, max_delta)
    rvh_est = rvh.est_h()
    true_rvh_est = true_rvh.est_h()
    print('H for rv: ', rvh_est)
    print('H for true rv: ', true_rvh_est)
    vis.plot_scaling_diagram(rvh)
    vis.plot_scaling_diagram(true_rvh)

    # ACF
    vis.plot_acorr(sim_rv)
    vis.plot_acorr(true_rv)

def p2_empirical_analysis():

    rv = load_spxfuturesrv()
    log_rv = np.log(rv)
    rq = load_spxfuturesrq()
    onv = (2 / 78) * (np.array(rq) / np.exp(log_rv) ** 2)

    # # HARK EST
    # hark_params = mle_hark(log_rv, onv, 'HARK_RV_EST')

    # # HARK ISP
    # hark_ispp, hark_ispa = full_pred_hark(hark_params, log_rv, onv, 'HARK_RV_FCST')
    # rmsehark = rmse(hark_ispa, hark_ispp)
    # print(f'HARK In-sample rmse: {rmsehark}')
    # qlikehark = qlike(hark_ispa, hark_ispp)
    # print(f'HARK In-sample qlike: {qlikehark}')

    # # HARK OSF
    # hark_osfp, hark_osfa = rolling_pred_hark(log_rv, onv, 'HARK_RV_FCST', 500, 1000)
    # rmsehark = rmse(hark_osfa, hark_osfp)
    # print(f'HARK Out-sample rmse: {rmsehark}')
    # qlikehark = qlike(hark_osfa, hark_osfp)
    # print(f'HARK Out-sample qlike: {qlikehark}') 
    # maehark = mae(hark_osfa, hark_osfp)
    # print(f'HARK Out-sample mae: {maehark}')

    # # HARKRED EST
    # harkred_params = mle_harkred(log_rv, 'HARKRED_RV_EST')

    # # HARKRED ISP
    # harkred_ispp, harkred_ispa = full_pred_harkred(harkred_params, log_rv, 'HARKRED_RV_FCST')
    # rmseharkred = rmse(harkred_ispa, harkred_ispp)
    # print(f'HARKRED In-sample rmse: {rmseharkred}')
    # qlikeharkred = qlike(harkred_ispa, harkred_ispp)
    # print(f'HARKRED In-sample qlike: {qlikeharkred}')

    # # HARKRED OSF
    # harkred_osfp, harkred_osfa = rolling_pred_harkred(log_rv, 'HARKRED_RV_FCST', 500, 1000)
    # rmseharkred = rmse(harkred_osfa, harkred_osfp)
    # print(f'HARKRED Out-sample rmse: {rmseharkred}')
    # qlikeharkred = qlike(harkred_osfa, harkred_osfp)
    # print(f'HARKRED Out-sample qlike: {qlikeharkred}') 
    # maeharkred = mae(harkred_osfa, harkred_osfp)
    # print(f'HARKRED Out-sample mae: {maeharkred}')

    # RHARK EST
    rhark_params = mle_rhark(log_rv, 'RHARK_RV_EST_2')

    # RHARK ISP
    rhark_ispp, rhark_ispa = full_pred_rhark(rhark_params, log_rv, 'RHARK_RV_FCST_2')
    rmserhark = rmse(rhark_ispa, rhark_ispp)
    print(f'RHARK In-sample rmse: {rmserhark}')
    qlikerhark = qlike(rhark_ispa, rhark_ispp)
    print(f'RHARK In-sample qlike: {qlikerhark}')

    # RHARK OSF
    rhark_h = rhark_params[-1]
    rhark_osfp, rhark_osfa = rolling_pred_rhark(log_rv, rhark_h, 'RHARK_RV_FCST_2', 500, 1000)
    rmserhark = rmse(rhark_osfa, rhark_osfp)
    print(f'RHARK Out-sample rmse: {rmserhark}')
    qlikerhark = qlike(rhark_osfa, rhark_osfp)
    print(f'RHARK Out-sample qlike: {qlikerhark}') 
    maerhark = mae(rhark_osfa, rhark_osfp)
    print(f'RHARK Out-sample mae: {maerhark}')

    # vis.plot_comparison(hark_osfa, hark_osfp)
    # vis.plot_comparison(harkred_osfa, harkred_osfp)
    vis.plot_comparison(rhark_osfa, rhark_osfp)

def p2_empirical_analysis_extended():

    rv_all = load_marketspotrv()

    for i, rv in rv_all.items():

        print(f'Analysis for {i}')
        log_rv = np.log(rv)

        # HARKRED EST
        harkred_params = mle_harkred(log_rv, f'HARKRED_{i}_EST')

        # HARKRED ISP
        harkred_ispp, harkred_ispa = full_pred_harkred(harkred_params, log_rv, f'HARKRED_{i}_FCST')
        rmseharkred = rmse(harkred_ispa, harkred_ispp)
        print(f'HARKRED In-sample rmse: {rmseharkred}')
        qlikeharkred = qlike(harkred_ispa, harkred_ispp)
        print(f'HARKRED In-sample qlike: {qlikeharkred}')

        # HARKRED OSF
        harkred_osfp, harkred_osfa = rolling_pred_harkred(log_rv, f'HARKRED_{i}_FCST', 500, 500)
        rmseharkred = rmse(harkred_osfa, harkred_osfp)
        print(f'HARKRED Out-sample rmse: {rmseharkred}')
        qlikeharkred = qlike(harkred_osfa, harkred_osfp)
        print(f'HARKRED Out-sample qlike: {qlikeharkred}') 
        maeharkred = mae(harkred_osfa, harkred_osfp)
        print(f'HARKRED Out-sample mae: {maeharkred}')

        # RHARK EST
        rhark_params = mle_rhark(log_rv, f'RHARK_{i}_EST')

        # RHARK ISP
        rhark_ispp, rhark_ispa = full_pred_rhark(rhark_params, log_rv, f'RHARK_{i}_FCST')
        rmserhark = rmse(rhark_ispa, rhark_ispp)
        print(f'RHARK In-sample rmse: {rmserhark}')
        qlikerhark = qlike(rhark_ispa, rhark_ispp)
        print(f'RHARK In-sample qlike: {qlikerhark}')

        # RHARK OSF
        rhark_h = rhark_params[-1]
        rhark_osfp, rhark_osfa = rolling_pred_rhark(log_rv, rhark_h, f'RHARK_{i}_FCST', 500, 500)
        rmserhark = rmse(rhark_osfa, rhark_osfp)
        print(f'RHARK Out-sample rmse: {rmserhark}')
        qlikerhark = qlike(rhark_osfa, rhark_osfp)
        print(f'RHARK Out-sample qlike: {qlikerhark}') 
        maerhark = mae(rhark_osfa, rhark_osfp)
        print(f'RHARK Out-sample mae: {maerhark}')  