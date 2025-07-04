import numpy as np
import pickle

def nelder_mead_se(function, filename, **kwargs):

    with open(f'isa_result/{filename}.pickle', 'rb') as file:
        result = pickle.load(file)
        print(result)

    n = len(result.x)
    p = np.array(result.final_simplex[0])
    ori_coord = np.array(result.final_simplex[1])
    midpoint = (p[:, None, :] + p[None, :, :]) / 2
    new_coord = np.empty((n + 1, n + 1))

    for i in range(n + 1):
        for j in range(n + 1):
            if 'onv' in kwargs:
                new_coord[i, j] = function(params=midpoint[i, j], rv=kwargs['log_rv'], onv=kwargs['onv'])
            else:
                new_coord[i, j] = function(params=midpoint[i, j], rv=kwargs['log_rv'])

    b_mat = np.empty((n, n))

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i == j:
                b_mat[i - 1, j - 1] = 2 * (ori_coord[i] + ori_coord[0] - 2 * new_coord[0, i])
            else:
                b_mat[i - 1, j - 1] = 2 * (new_coord[i, j] + ori_coord[0] - new_coord[0, i] - new_coord[0, j])

    q_mat = (p[1:] - p[0]).T

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

    info_mat = q_inv.T @ b_mat @ q_inv
    cv_mat = q_mat @ b_inv @ q_mat.T
    se = np.sqrt(np.abs(np.diag(cv_mat)))

    return se