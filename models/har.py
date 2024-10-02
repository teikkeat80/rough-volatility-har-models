import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import pandas as pd

class Har:
    def __init__(self, rv: list, beg_index: int = 22):
        self.log_rv = np.log(rv)
        self.beg_index = beg_index
        self.y = self.log_rv[self.beg_index:]
        self.xd = self.log_rv[self.beg_index - 1: len(self.log_rv) - 1]
        self.xw = self._calc_backwards_ma(5)
        self.xm = self._calc_backwards_ma(22)

    def _calc_backwards_ma(self, k: int):
        output = []
        for i in range(len(self.log_rv) - self.beg_index):
            ma = sum(self.log_rv[self.beg_index - k + i:self.beg_index + i])/k
            output.append(ma)
        return output

class HarOls:
    def __init__(self, df, dep: str, indep: list, rolling_window: int):
        self.dep = dep
        self.indep = indep
        self.df = df[self.indep + [self.dep]]
        self.rolling_window = rolling_window
        self.params = self.rols_params()
        self.df_full = pd.concat([self.df, self.params], axis = 1)

    def _get_dep_and_indep(self, df=None):
        if df is None:
            df = self.df
        dep = df[self.dep]
        indep = sm.add_constant(df[self.indep])
        return dep, indep
    
    def rols_params(self):
        dep, indep = self._get_dep_and_indep()
        rols = RollingOLS(dep, indep, window=self.rolling_window)
        params = rols.fit().params
        return params.add_prefix("params_")
    
    def fols_summary(self):
        dep, indep = self._get_dep_and_indep()
        fols = sm.OLS(dep, indep)
        print(fols.fit().summary())

    def tols_summary(self):
        df_train = self.df.iloc[:self.rolling_window - 1]
        dep, indep = self._get_dep_and_indep(df_train)    
        tols = sm.OLS(dep, indep)
        print(tols.fit().summary())

    def predict(self):
        param_cols = {col[7:]: col for col in self.df_full.columns if col.startswith('params_') and col != "params_const"}
        ind_cols = {col: col for col in self.df_full.columns if (not col.startswith('params_')) and col != self.dep}
        result = self.df_full["params_const"] + sum(self.df_full[param_cols[s]] * self.df_full[ind_cols[s]] for s in ind_cols if s in param_cols)
        return result