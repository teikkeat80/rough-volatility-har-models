import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

class Ols:
    def __init__(self, dep, indep, w):
        self.dep = dep
        self.indep = sm.add_constant(indep)
        self.w = w
    
    def rols(self):
        rols = RollingOLS(self.dep, self.indep, window=self.w)
        params = rols.fit().params
        return params
    
    def fols(self, nw=False, pr=False):
        fols = sm.OLS(self.dep, self.indep)
        params = fols.fit().params.T
        if pr:
            print(fols.fit(cov_type='HAC', cov_kwds={'maxlags': 20}).summary() if nw else fols.fit().summary())
        return params

    def rol_predict(self):
        params = self.rols()
        return np.sum(params * self.indep, axis=1)
    
    def fol_predict(self):
        params = self.fols()
        return np.sum(params * self.indep, axis=1)
    
# def load_rv_all(path):
#     df_raw = pd.read_csv(path)
#     rv_all = df_raw.iloc[:, 1:].to_dict(orient='list')
#     return rv_all

# def load_rv_one(path, select):
#     rv_all = load_rv_all(path)
#     rv_select = rv_all[select]
#     return rv_select

# rv = load_rv_one('data/rv_dataset.csv', '.SPX')
# har = Har(rv=rv, beg_index=30 * 22 + 21)
# har_y = har.y
# har_x_d = har.xd
# har_x_w = har.xw
# har_x_m = har.xm
# har_indep = np.array([har_x_d, har_x_w, har_x_m]).T

# ols = Ols(har_y, har_indep, 1001)
# ols.fols_summary()
# print(ols.params.shape)
# print(ols.indep.shape)
# pred = ols.predict()
# print(pred.shape)
# predicted = pred[1900:]
# actual = har_y[1900:]
# print(f"rmse: {np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2))}")
