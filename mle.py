from scipy.optimize import minimize
from hark import log_likelihood_hark, log_likelihood_harkred
import pickle
import numpy as np
from nmse import nelder_mead_se as nmse
from rhark import log_likelihood_rhark

def mle_hark(log_rv, onv, filename):
    init_params = [0.001, 0.5, 0.5, 0.5, 0.1]
    est = minimize(
        log_likelihood_hark,
        init_params,
        args=(log_rv, onv),
        method='Nelder-Mead',
        options={'xatol': 1e-6, 'fatol': 1e-2, 'maxfev': 2000}
    )

    with open(f'isa_result/{filename}.pickle', 'wb') as file:
        pickle.dump(est, file)

    with open(f'isa_result/{filename}.pickle', 'rb') as file:
        result = pickle.load(file)
    
    est_params = result.x
    ll = - result.fun
    aic = (2 * len(init_params)) - (2 * ll)
    se = nmse(log_likelihood_hark, filename, log_rv=log_rv, onv=onv)

    np.set_printoptions(suppress=True)
    print(result)
    print('Estimated Params: ', est_params, 4)
    print('LL: ', ll)
    print('AIC: ', aic)
    print('SE: ', se)

    return est_params

def mle_harkred(log_rv, filename):
    init_params = [0.001, 0.5, 0.5, 0.5, 0.1, 0.1]
    est = minimize(
        log_likelihood_harkred,
        init_params,
        args=(log_rv),
        method='Nelder-Mead',
        options={'xatol': 1e-6, 'fatol': 1e-2, 'maxfev': 2000}
    )

    with open(f'isa_result/{filename}.pickle', 'wb') as file:
        pickle.dump(est, file)

    with open(f'isa_result/{filename}.pickle', 'rb') as file:
        result = pickle.load(file)
    
    est_params = result.x
    ll = - result.fun
    aic = (2 * len(init_params)) - (2 * ll)
    se = nmse(log_likelihood_hark, filename, log_rv=log_rv)

    np.set_printoptions(suppress=True)
    print(result)
    print('Estimated Params: ', est_params, 4)
    print('LL: ', ll)
    print('AIC: ', aic)
    print('SE: ', se)

    return est_params

def mle_rhark(log_rv, filename):
    init_params = [0.001, 0.5, 0.5, 0.5, 0.1, 0.1] # back - add 0.1
    # print('start mle')
    # est = minimize(
    #     log_likelihood_rhark,
    #     init_params,
    #     args=(log_rv),
    #     method='Nelder-Mead',
    #     options={'xatol': 1e-6, 'fatol': 1e-2, 'maxfev': 2000}
    # )

    # with open(f'isa_result/{filename}.pickle', 'wb') as file:
    #     pickle.dump(est, file)

    with open(f'isa_result/{filename}.pickle', 'rb') as file:
        result = pickle.load(file)
    
    est_params = result.x
    ll = - result.fun
    aic = (2 * len(init_params)) - (2 * ll)
    se = nmse(log_likelihood_rhark, filename, log_rv=log_rv)

    np.set_printoptions(suppress=True)
    print(result)
    print('Estimated Params: ', est_params)
    print('LL: ', ll)
    print('AIC: ', aic)
    print('SE: ', se)

    return est_params