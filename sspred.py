import pandas as pd
from hark import *
import numpy as np
from rhark import *
from scipy.optimize import minimize
import os
from time import time # remove after


def full_pred_hark(params, log_rv, onv, filename):

    # b0, b1, b2, b3, q = params
    # columns = ['iteration', 'predicted', 'var', 'actual']

    # output_file = f'isa_result/{filename}.csv'
    # pd.DataFrame(columns=columns).to_csv(output_file, index=False)

    # y = HARK(b0, b1, b2, b3, q, 1)
    # y.construct_kf()
    # y.initialise_a(mean=np.mean(log_rv))
    # y.initialise_p(var_iv=np.var(log_rv))

    # for i in range(len(log_rv)):
    #     a_pred, p_pred = y.predict()
    #     y.update(log_rv[i], onv[i])
    #     predicted = (y.m @ a_pred).item()
    #     var = (y.m @ p_pred @ y.m.T).item()
    #     actual = log_rv[i]

    #     row = pd.DataFrame([{
    #         'iteration': i,
    #         'predicted': predicted,
    #         'var': var,
    #         'actual': actual
    #     }])

    #     row.to_csv(output_file, mode='a', index=False, header=False)
    
    df = pd.read_csv(f'isa_result/{filename}.csv')
    variance = np.array(df['var'].values)
    predicted = df['predicted'].values
    actual = df['actual'].values

    rvpredicted = np.exp(predicted + (variance / 2))
    rvactual = np.exp(actual)

    return rvpredicted, rvactual

def full_pred_harkred(params, log_rv, filename):

    b0, b1, b2, b3, q, r = params
    columns = ['iteration', 'predicted', 'var', 'actual']

    output_file = f'isa_result/{filename}.csv'
    pd.DataFrame(columns=columns).to_csv(output_file, index=False)

    y = HARK(b0, b1, b2, b3, q, r)
    y.construct_kf()
    y.initialise_a(mean=np.mean(log_rv))
    y.initialise_p(var_iv=np.var(log_rv))

    for i in range(len(log_rv)):
        a_pred, p_pred = y.predict()
        y.update(log_rv[i])
        predicted = (y.m @ a_pred).item()
        var = (y.m @ p_pred @ y.m.T).item()
        actual = log_rv[i]

        row = pd.DataFrame([{
            'iteration': i,
            'predicted': predicted,
            'var': var,
            'actual': actual
        }])

        row.to_csv(output_file, mode='a', index=False, header=False)
    
    df = pd.read_csv(f'{filename}.csv')
    variance = np.array(df['var'].values)
    predicted = df['predicted'].values
    actual = df['actual'].values

    rvpredicted = np.exp(predicted + ((variance + r ** 2) / 2))
    rvactual = np.exp(actual)

    return rvpredicted, rvactual

def full_pred_rhark(params, log_rv, filename):

    b0, b1, b2, b3, q, r, h = params
    columns = ['iteration', 'predicted', 'var', 'actual']

    output_file = f'isa_result/{filename}.csv'
    pd.DataFrame(columns=columns).to_csv(output_file, index=False)

    y = RHARK(b0, b1, b2, b3, q, r, h)
    y.construct_z(len(log_rv))
    y.construct_kf()
    y.initialise_a(mean=np.mean(log_rv))
    y.initialise_p(var_iv=np.var(log_rv), var_z=0.001)

    for i in range(len(log_rv)):
        a_pred, p_pred = y.predict()
        y.update(log_rv[i])
        predicted = (y.m @ a_pred).item()
        var = (y.m @ p_pred @ y.m.T).item()
        actual = log_rv[i]

        row = pd.DataFrame([{
            'iteration': i,
            'predicted': predicted,
            'var': var,
            'actual': actual
        }])

        row.to_csv(output_file, mode='a', index=False, header=False)
    
    df = pd.read_csv(f'isa_result/{filename}.csv')
    variance = np.array(df['var'].values)
    predicted = df['predicted'].values
    actual = df['actual'].values

    rvpredicted = np.exp(predicted + ((variance + r ** 2) / 2))
    rvactual = np.exp(actual)

    return rvpredicted, rvactual

def full_pred_rharkq(params, log_rv, onv, filename):

    # b0, b1, b2, b3, q, h = params
    # columns = ['iteration', 'predicted', 'var', 'actual']

    # output_file = f'isa_result/{filename}.csv'
    # pd.DataFrame(columns=columns).to_csv(output_file, index=False)

    # y = RHARK(b0, b1, b2, b3, q, 1, h)
    # y.construct_z(len(log_rv))
    # y.construct_kf()
    # y.initialise_a(mean=np.mean(log_rv))
    # y.initialise_p(var_iv=np.var(log_rv), var_z=0.001)

    # for i in range(len(log_rv)):
    #     a_pred, p_pred = y.predict()
    #     y.update(log_rv[i], onv[i])
    #     predicted = (y.m @ a_pred).item()
    #     var = (y.m @ p_pred @ y.m.T).item()
    #     actual = log_rv[i]

    #     row = pd.DataFrame([{
    #         'iteration': i,
    #         'predicted': predicted,
    #         'var': var,
    #         'actual': actual
    #     }])

    #     row.to_csv(output_file, mode='a', index=False, header=False)
    
    df = pd.read_csv(f'isa_result/{filename}.csv')
    variance = np.array(df['var'].values)
    predicted = df['predicted'].values
    actual = df['actual'].values

    rvpredicted = np.exp(predicted + (variance / 2))
    rvactual = np.exp(actual)

    return rvpredicted, rvactual

def rolling_pred_hark(log_rv, onv, filename, window, eval_start, eval_end, log_results=False, med=False):

    # log_rv = log_rv[-1250:]
    # onv = onv[-1250:]

    # # Output file path
    # output_file = f'osa_result/{filename}.csv'
    # columns = ['iteration', 'b0', 'b1', 'b2', 'b3', 'q', 'loglik', 'predicted', 'var', 'actual']

    # # MODIFIED LINES
    # if os.path.exists(output_file):
    #     existing_df = pd.read_csv(output_file)
    #     start_iter = existing_df['iteration'].max() + 1
    # else:
    #     start_iter = 0
    #     # Write header if file doesn't exist
    #     pd.DataFrame(columns=columns).to_csv(output_file, index=False)
    # # MODIFIED LINES

    # # pd.DataFrame(columns=columns).to_csv(output_file, index=False)

    # # Initialise Parameters
    # initial_params = [0.001, 0.5, 0.5, 0.5, 0.1]

    # for i in range(start_iter, len(log_rv) - window):
    #     # REMOVE AFTER
    #     start_timex = time()
    #     print(f'iteration {i}')
    #     # REMOVE AFTER

    #     try:
    #         # Select window
    #         series = log_rv[i: window + i]
    #         onv_series = onv[i: window + i]

    #         # Estimation
    #         result = minimize(
    #             log_likelihood_hark,
    #             initial_params,
    #             args=(series, onv_series),
    #             method='Nelder-Mead',
    #             options={'xatol': 1e-6, 'fatol': 1e-2, 'maxfev': 2000}
    #         )

    #         # Record Estimation Result
    #         est_params = result.x
    #         loglik = - result.fun
    #         b0, b1, b2, b3, q = est_params

    #         # Initialise Filter
    #         y = HARK(b0, b1, b2, b3, q, 1)
    #         y.construct_kf()
    #         y.initialise_a(np.mean(series))
    #         y.initialise_p(np.var(series))

    #         # Run filter
    #         for l in range(len(series)):
    #             y.predict()
    #             y.update(series[l], onv_series[l])

    #         # Generate prediction and record actual
    #         a_pred, p_pred = y.predict()
    #         predicted = (y.m @ a_pred).item()
    #         var = (y.m @ p_pred @ y.m.T).item()
    #         actual = log_rv[window + i]

    #         # Combine into rows
    #         row = pd.DataFrame([{
    #             'iteration': i,
    #             'b0': b0,
    #             'b1': b1,
    #             'b2': b2,
    #             'b3': b3,
    #             'q': q,
    #             'loglik': loglik,
    #             'predicted': predicted,
    #             'var': var,
    #             'actual': actual
    #         }])

    #         # Append to csv
    #         row.to_csv(output_file, mode='a', index=False, header=False)
        
    #     except Exception as e:
    #         print(f'Error at iteration{i}: {e}')
    #         continue

    #     # REMOVE AFTER
    #     end_timex = time()
    #     print(f"Elapsed time: {end_timex - start_timex} seconds")
    #     # REMOVE AFTER
    
    # df = pd.read_csv(f'osa_result/{filename}.csv').iloc[-eval:]
    df = pd.read_csv(f'osa_result/{filename}.csv').iloc[eval_start:eval_end]
    variance = np.array(df['var'].values)
    predicted = df['predicted'].values
    actual = df['actual'].values

    if med:
        rvpredicted = np.exp(predicted)
        rvactual = np.exp(actual)
    else:
        rvpredicted = np.exp(predicted + (variance / 2))
        rvactual = np.exp(actual)

    if log_results:
        return predicted, actual, variance
    else:
        return rvpredicted, rvactual, variance

def rolling_pred_harkred(log_rv, filename, window, eval):

    # Output file path
    output_file = f'osa_result/{filename}.csv'
    columns = ['iteration', 'b0', 'b1', 'b2', 'b3', 'q', 'r', 'loglik', 'predicted', 'var', 'actual']
    pd.DataFrame(columns=columns).to_csv(output_file, index=False)

    # Initialise Parameters
    initial_params = [0.001, 0.5, 0.5, 0.5, 0.1, 0.1]

    for i in range(0, len(log_rv) - window):

        try:
            # Select window
            series = log_rv[i: window + i]

            # Estimation
            result = minimize(
                log_likelihood_harkred,
                initial_params,
                args=(series),
                method='Nelder-Mead',
                options={'xatol': 1e-6, 'fatol': 1e-2, 'maxfev': 2000}
            )

            # Record Estimation Result
            est_params = result.x
            loglik = - result.fun
            b0, b1, b2, b3, q, r = est_params

            # Initialise Filter
            y = HARK(b0, b1, b2, b3, q, r)
            y.construct_kf()
            y.initialise_a(np.mean(series))
            y.initialise_p(np.var(series))

            # Run filter
            for l in range(len(series)):
                y.predict()
                y.update(series[l])

            # Generate prediction and record actual
            a_pred, p_pred = y.predict()
            predicted = (y.m @ a_pred).item()
            var = (y.m @ p_pred @ y.m.T).item()
            actual = log_rv[window + i]

            # Combine into rows
            row = pd.DataFrame([{
                'iteration': i,
                'b0': b0,
                'b1': b1,
                'b2': b2,
                'b3': b3,
                'q': q,
                'r': r,
                'loglik': loglik,
                'predicted': predicted,
                'var': var,
                'actual': actual
            }])

            # Append to csv
            print(row)
            row.to_csv(output_file, mode='a', index=False, header=False)
        
        except Exception as e:
            print(f'Error at iteration{i}: {e}')
            continue
    
    df = pd.read_csv(f'{filename}.csv').iloc[-eval:]
    variance = np.array(df['var'].values)
    rsq = np.array(df['r'].values) ** 2
    predicted = df['predicted'].values
    actual = df['actual'].values

    rvpredicted = np.exp(predicted + ((variance + rsq) / 2))
    rvactual = np.exp(actual)

    return rvpredicted, rvactual

def rolling_pred_rhark(log_rv, h, filename, window, eval):

    # Output file path
    output_file = f'osa_result/{filename}.csv'
    columns = ['iteration', 'b0', 'b1', 'b2', 'b3', 'q', 'r', 'h', 'loglik', 'predicted', 'var', 'actual']
    
    # # MODIFIED LINES
    # if os.path.exists(output_file):
    #     existing_df = pd.read_csv(output_file)
    #     start_iter = existing_df['iteration'].max() + 1
    # else:
    #     start_iter = 0
    #     # Write header if file doesn't exist
    #     pd.DataFrame(columns=columns).to_csv(output_file, index=False)
    # # MODIFIED LINES
    
    pd.DataFrame(columns=columns).to_csv(output_file, index=False)

    initial_params = [0.001, 0.5, 0.5, 0.5, 0.1, 0.1]

    for i in range(0, len(log_rv) - window):

        try:
            # Select window
            series = log_rv[i: window + i]

            # Estimation
            result = minimize(
                log_likelihood_rhark,
                initial_params,
                args=(h, series),
                method='Nelder-Mead',
                options={'xatol': 1e-6, 'fatol': 1e-2, 'maxfev': 4000}
            )

            # Record Estimation Result
            est_params = result.x
            loglik = - result.fun
            b0, b1, b2, b3, q, r = est_params

            # Initialise Filter
            y = RHARK(b0, b1, b2, b3, q, r, h)
            y.construct_z(len(series))
            y.construct_kf()
            y.initialise_a(mean=np.mean(series))
            y.initialise_p(var_iv=np.var(series), var_z=0.001)

            # Run filter
            for l in range(len(series)):
                y.predict()
                y.update(series[l])

            # Generate prediction and record actual
            a_pred, p_pred = y.predict()
            predicted = (y.m @ a_pred).item()
            var = (y.m @ p_pred @ y.m.T).item()
            actual = log_rv[window + i]

            # Combine into rows
            row = pd.DataFrame([{
                'iteration': i,
                'b0': b0,
                'b1': b1,
                'b2': b2,
                'b3': b3,
                'q': q,
                'r': r,
                'h': h,
                'loglik': loglik,
                'predicted': predicted,
                'var': var,
                'actual': actual
            }])

            # Append to csv
            row.to_csv(output_file, mode='a', index=False, header=False)
        
        except Exception as e:
            print(f'Error at iteration{i}: {e}')
            continue

    df = pd.read_csv(f'osa_result/{filename}.csv').iloc[-eval:]
    variance = np.array(df['var'].values)
    rsq = np.array(df['r'].values) ** 2
    predicted = df['predicted'].values
    actual = df['actual'].values

    rvpredicted = np.exp(predicted + ((variance + rsq) / 2))
    rvactual = np.exp(actual)

    return rvpredicted, rvactual

def rolling_pred_rharkq(log_rv, onv, filename, window, eval_start, eval_end, log_results=False, med=False):

    # log_rv = log_rv[-1500:]
    # onv = onv[-1500:]    

    # # Output file path
    # output_file = f'osa_result/{filename}.csv'
    # columns = ['iteration', 'b0', 'b1', 'b2', 'b3', 'q', 'h', 'loglik', 'predicted', 'var', 'actual']
    
    # # MODIFIED LINES
    # if os.path.exists(output_file):
    #     existing_df = pd.read_csv(output_file)
    #     start_iter = existing_df['iteration'].max() + 1
    # else:
    #     start_iter = 0
    #     # Write header if file doesn't exist
    #     pd.DataFrame(columns=columns).to_csv(output_file, index=False)
    # # MODIFIED LINES
    
    # # pd.DataFrame(columns=columns).to_csv(output_file, index=False)

    # initial_params = [0.001, 0.5, 0.5, 0.5, 0.1, 0.1]

    # for i in range(start_iter, len(log_rv) - window):
    #     # REMOVE AFTER
    #     start_timex = time()
    #     print(f'iteration {i}')
    #     # REMOVE AFTER

    #     try:
    #         # Select window
    #         series = log_rv[i: window + i]
    #         onv_series = onv[i: window + i]

    #         # Estimation
    #         result = minimize(
    #             log_likelihood_rharkq,
    #             initial_params,
    #             args=(series, onv_series),
    #             method='Nelder-Mead',
    #             options={'xatol': 1e-6, 'fatol': 1e-2, 'maxfev': 4000}
    #         )

    #         # Record Estimation Result
    #         est_params = result.x
    #         loglik = - result.fun
    #         b0, b1, b2, b3, q, h = est_params

    #         # Initialise Filter
    #         y = RHARK(b0, b1, b2, b3, q, 1, h)
    #         y.construct_z(len(series))
    #         y.construct_kf()
    #         y.initialise_a(mean=np.mean(series))
    #         y.initialise_p(var_iv=np.var(series), var_z=0.001)

    #         # Run filter
    #         for l in range(len(series)):
    #             y.predict()
    #             y.update(series[l], onv_series[l])

    #         # Generate prediction and record actual
    #         a_pred, p_pred = y.predict()
    #         predicted = (y.m @ a_pred).item()
    #         var = (y.m @ p_pred @ y.m.T).item()
    #         actual = log_rv[window + i]

    #         # Combine into rows
    #         row = pd.DataFrame([{
    #             'iteration': i,
    #             'b0': b0,
    #             'b1': b1,
    #             'b2': b2,
    #             'b3': b3,
    #             'q': q,
    #             'h': h,
    #             'loglik': loglik,
    #             'predicted': predicted,
    #             'var': var,
    #             'actual': actual
    #         }])

    #         # Append to csv
    #         row.to_csv(output_file, mode='a', index=False, header=False)
        
    #     except Exception as e:
    #         print(f'Error at iteration{i}: {e}')
    #         continue

    #     # REMOVE AFTER
    #     end_timex = time()
    #     print(f"Elapsed time: {end_timex - start_timex} seconds")
    #     # REMOVE AFTER

    # df = pd.read_csv(f'osa_result/{filename}.csv').iloc[-eval:]
    df = pd.read_csv(f'osa_result/{filename}.csv').iloc[eval_start:eval_end]
    variance = np.array(df['var'].values)
    predicted = df['predicted'].values
    actual = df['actual'].values

    if med:
        rvpredicted = np.exp(predicted)
        rvactual = np.exp(actual)
    else:
        rvpredicted = np.exp(predicted + (variance / 2))
        rvactual = np.exp(actual)

    if log_results:
        return predicted, actual, variance
    else:
        return rvpredicted, rvactual, variance