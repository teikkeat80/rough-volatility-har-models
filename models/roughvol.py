import numpy as np
import math

class RoughVolatility:
    def __init__(self, series, hurst, error, nu):
        self.h = hurst
        self.series = np.log(series)
        self.err = error
        self.nu = nu
        self.ammse = math.gamma(3 / 2 - hurst) / (math.gamma(hurst + 1 / 2) * math.gamma(2 - 2 * hurst))

    def cond_var(self, delta=1):
        h, ammse = self.h, self.ammse
        return ammse * delta ** (2 * h)

    def back_transform(self, lpred, delta=1):
        cond_var = self.cond_var(delta=delta)
        nu = self.nu
        return np.exp(lpred + 2 * cond_var * nu ** 2)
    
    def lpred(self, delta=1, bt=True, series=None):
        h, err = self.h, self.err
        series = np.asarray(self.series if series is None else series)
        n = len(series)
        idx = np.arange(n)
        rev_series = series[::-1]

        i_plus = idx + err
        denom = (i_plus + delta) * (i_plus ** (h + 1 / 2))

        if h < 1 / 2:
            numer = rev_series
            factor = delta ** (h + 1 / 2)
        elif h > 1 / 2:
            numer = delta * rev_series - (i_plus + delta) * rev_series[0]
            factor = delta ** (h - 1 / 2)
        else:
            raise ValueError("h = 0.5 case is not implemented.")

        summation = np.sum(numer / denom)
        output = (math.cos(h * math.pi) / math.pi) * factor * summation

        return self.back_transform(output, delta) if bt else output

    def moving_window_lpred(self, train_size, delta=1, bt=True):
        series = self.series
        n = len(series)
        output = [
            self.lpred(delta=delta, bt=bt, series=series[i:i + train_size])
            for i in range(n - train_size)
        ]
        return output


    # def lpred(self, delta=1, bt=True):
    #     if self.h < 0.5:
    #         summation = 0 
    #         for i in range(len(self.series)):
    #             numerator = self.series[len(self.series) - i - 1]
    #             denominator = (i + self.err + delta) * ((i + self.err) ** (self.h + 0.5))
    #             summation += numerator / denominator
    #         output = (math.cos(self.h * math.pi) / math.pi) * delta ** (self.h + 0.5) * summation
    #     elif self.h > 0.5:
    #         summation = 0      
    #         for i in range(len(self.series)):
    #             numerator = delta * self.series[len(self.series) - i - 1] - (i + self.err + delta) * self.series[len(self.series) - 1]
    #             denominator = (i + self.err + delta) * ((i + self.err) ** (self.h + 0.5))
    #             summation += numerator / denominator
    #         output = (math.cos(self.h * math.pi) / math.pi) * delta ** (self.h - 0.5) * summation
    #     if bt:
    #         output = self._back_transform(output, delta=delta)
    #     return output

    
    # def forecast_nr(self, delta=1, back_transform : bool = False):
    #     summation = 0      
    #     for i in range(len(self.log_rv)):
    #         numerator = delta * self.log_rv[len(self.log_rv) - i - 1] - (i + self.err + delta) * self.log_rv[len(self.log_rv) - 1]
    #         denominator = (i + self.err + delta) * ((i + self.err) ** (self.h + 0.5))
    #         summation += numerator / denominator
    #     fc = (math.cos(self.h * math.pi) / math.pi) * (delta ** (self.h - 0.5)) * summation
    #     if back_transform:
    #         output = self._back_transform(fc=fc, delta=delta)
    #     else:
    #         output = fc
    #     return output

    # def backwards_average_forecast(self, delta=1, k=1, back_transform : bool = False):
    #     log_rv_cp = self.log_rv.copy()
    #     bwd_sum = 0
    #     for _ in range(k):
    #         r = self.forecast(delta=delta, back_transform=back_transform)
    #         bwd_sum += r
    #         self.log_rv = self.log_rv[:-1]
    #     output = bwd_sum / k
    #     self.log_rv = log_rv_cp
    #     return output
    
    # def moving_window_lpred(self, train_size, delta=1, bt=True):
    #     series_cp = self.series.copy()
    #     output = []
    #     beg = 0
    #     end = train_size
    #     while end < len(self.series):
    #         self.series = self.series[beg:end]
    #         r = self.lpred(delta=delta, bt=bt)
    #         output.append(r)
    #         self.series = series_cp
    #         beg, end = beg + 1, end + 1
    #     return output

        # log_rv_cp = self.log_rv.copy()
        # fc = []
        # log_rv_tr = self.log_rv[:train_size]
        # count = train_size
        # while count < len(log_rv_cp):
        #     self.log_rv = log_rv_tr
        #     if k is None:
        #         r = self.forecast(delta=delta, back_transform=back_transform)
        #     else:
        #         r = self.backwards_average_forecast(delta=delta, k=k, back_transform=back_transform)
        #     fc.append(r)
        #     log_rv_tr = np.append(log_rv_tr, r)
        #     count += 1
        # self.log_rv = log_rv_cp
        # return fc

    # def incremental_forecast(self, train_size, inc=1):
    #     log_rv_cp = self.log_rv.copy()
    #     log_rv_fc = []
    #     d = inc
    #     self.log_rv = self.log_rv[0:train_size]
    #     while d <= len(log_rv_cp) - train_size:
    #         r = self.forecast(delta=d)
    #         log_rv_fc.append(r)
    #         d += 1      
    #     self.log_rv = log_rv_cp
    #     return log_rv_fc
    
    # Not correct
    # def bwd_forecast(self, delta=1):
    #     cons = (math.cos(self.h * math.pi) / math.pi) * pow(delta, self.h + 0.5)
    #     sum_terms = []

    #     for i in range(len(self.log_arr)):
    #         numerator = self.log_arr[len(self.log_arr) - i - 1]
    #         denominator = (i + self.err) * pow(i - delta + self.err, self.h + 0.5)
    #         sum_terms.append(numerator / denominator)
        
    #     output = cons * sum(sum_terms)
    #     return output