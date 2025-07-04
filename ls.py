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
    
    def fols(self, nw=False, pr=False, r2=False):
        fols = sm.OLS(self.dep, self.indep)
        params = fols.fit().params.T
        if nw:
            mod = fols.fit(cov_type='HAC', cov_kwds={'maxlags': 20})
        else:
            mod = fols.fit()
        if pr:
            print(mod.summary())
            print(f'R squared: {mod.rsquared}')
        return params

    def rol_predict(self):
        params = self.rols()
        return np.sum(params * self.indep, axis=1)
    
    def fol_predict(self):
        params = self.fols()
        return np.sum(params * self.indep, axis=1)
    
    def fol_res_var(self):
        fitted = self.fol_predict()
        actual = self.dep
        res_var = np.sum((actual - fitted) ** 2) / (len(actual) - 4)
        return res_var
    
    def rol_res_var(self):
        res_var = []
        window = self.w - 1
        params = self.rols()[window:]
        for j in range(window):
            res_var.append(np.nan)
        for i in range(0, len(self.dep) - window):
            dep_set = np.array(self.dep[i: window + i])
            indep_set = self.indep[i: window + i]
            params_set = params[i]
            fitted = np.sum(params_set * indep_set, axis=1)
            resi = dep_set - fitted
            res_vari = np.sum(resi ** 2) / (window - 4)
            res_var.append(res_vari)
        return np.array(res_var)