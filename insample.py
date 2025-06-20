from hark import HARK, log_likelihood_harkred, log_likelihood_hark
from hark2 import HARK2, log_likelihood_hark2
from data_processing import load_rv_one, load_rv
import numpy as np
import pickle
import pandas as pd

log_rv = np.log(load_rv('data/SNP500_RV_5min.csv', 'RV'))
rq = (2 / 78) * (np.array(load_rv('data/SP500_RQ_5min.csv', 'RQ')) / np.exp(log_rv) ** 2)

with open(f'premestm_result/HARK_RV_EST.pickle', 'rb') as file:
    result = pickle.load(file)
    print(result)

b0, b1, b2, b3, q = result.x

columns = ['iteration', 'predicted', 'var', 'actual']

# Output file path
output_file = 'isa_result/HARK_RV_FCST.csv'
pd.DataFrame(columns=columns).to_csv(output_file, index=False)

y = HARK(b0, b1, b2, b3, q, r)
# y.construct_z(len(log_rv))
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
        'r': r,
        'predicted': predicted,
        'var': var,
        'actual': actual
    }])

    row.to_csv(output_file, mode='a', index=False, header=False)