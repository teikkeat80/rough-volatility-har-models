from hark import HARK, log_likelihood_harkred, log_likelihood_hark
from hark2 import HARK2, log_likelihood_hark2
from data_processing import load_rv_one, load_rv
import numpy as np
import pickle

def nelder_mead_se(function, result_path, **kwargs):

    with open(result_path, 'rb') as file:
        result = pickle.load(file)
        print(result)

    n = len(result.x)
    p = np.array(result.final_simplex[0])
    ori_coord = np.array(result.final_simplex[1])
    midpoint = (p[:, None, :] + p[None, :, :]) / 2
    new_coord = np.empty((n + 1, n + 1))

    for i in range(n + 1):
        for j in range(n + 1):
            new_coord[i, j] = function(params=midpoint[i, j], rv=kwargs['series'])

    b_mat = np.empty((n, n))

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i == j:
                b_mat[i - 1, j - 1] = 2 * (ori_coord[i] + ori_coord[0] - 2 * new_coord[0, i])
            else:
                b_mat[i - 1, j - 1] = 2 * (new_coord[i, j] + ori_coord[0] - new_coord[0, i] - new_coord[0, j])

    q_mat = (p[1:] - p[0]).T

    print(b_mat)

    try:
        q_inv = np.linalg.inv(q_mat)
        b_inv = np.linalg.inv(b_mat)
    except np.linalg.LinAlgError:
        print('Matrix is singular. Use pseudo-inverse.')
        try:
            b_inv = np.linalg.pinv(b_mat)
        except np.linalg.LinAlgError:
            print('Pseudo-inverse not working. Regularise.')
            b_inv = np.linalg.inv(b_mat + 1e-8 * np.eye(n))

    print(b_inv)

    info_mat = q_inv.T @ b_mat @ q_inv
    cv_mat = q_mat @ b_inv @ q_mat.T

    print(info_mat)
    print(cv_mat)
    print(np.diag(cv_mat))
    se = np.sqrt(np.abs(np.diag(cv_mat)))

    return se

# indices = ['SPX', 'GDAXI', 'FCHI', 'FTSE', 'OMXSPI', 'N225', 'KS11', 'HSI']
# for idx in indices:
log_rv = np.log(load_rv('data/SNP500_RV_5min.csv', 'RV'))
rq = (2 / 78) * (np.array(load_rv('data/SP500_RQ_5min.csv', 'RQ')) / np.exp(log_rv) ** 2)
result_path = f'premestm_result/HARK2QC_RV_EST.pickle'
se = nelder_mead_se(log_likelihood_hark2, result_path, series=log_rv)
# print(idx)
print('------------------------------')
print(se)

# with open(f'estm_result/HARK2_SPX_EST.pickle', 'rb') as file:
#     result = pickle.load(file)
#     print(result)

# p = np.array(result.final_simplex[0])
# ori_coord = np.array(result.final_simplex[1])

# # Compute all pairwise midpoints
# n, d = p.shape
# midpoint = (p[:, None, :] + p[None, :, :]) / 2

# new_coord = np.empty((8, 8))

# for i in range(8):
#     for j in range(8):
#         new_coord[i, j] = log_likelihood_hark2(params=midpoint[i, j], rv=log_rv)

# b_mat = np.empty((7, 7))

# for i in range(1, 8):
#     for j in range(1, 8):
#         if i == j:
#             b_mat[i - 1, j - 1] = 2 * (ori_coord[i] + ori_coord[0] - 2 * new_coord[0, i])
#         else:
#             b_mat[i - 1, j - 1] = 2 * (new_coord[i, j] + ori_coord[0] - new_coord[0, i] - new_coord[0, j])

# q_mat = (p[1:] - p[0]).T

# q_inv = np.linalg.inv(q_mat)
# b_inv = np.linalg.inv(b_mat)

# info_mat = q_inv.T @ b_mat @ q_inv

# cv_mat = q_mat @ b_inv @ q_mat.T
# print(cv_mat)
# print(np.sqrt(np.diag(cv_mat)))