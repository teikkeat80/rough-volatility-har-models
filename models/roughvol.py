import numpy as np
import math

class RoughVolatility:
    def __init__(self, rv: list, h: float, err: float, nu: float):
        self.h = h
        self.log_rv = np.log(rv)
        self.err = err
        self.nu = nu

    def _back_transform(self, fc, delta=1):
        c = math.gamma(1.5 - self.h) / (math.gamma(self.h + 0.5) * math.gamma(2 - 2 * self.h))
        return math.exp(fc + (2 * (self.nu ** 2) * c * (delta ** (2 * self.h))))
    
    def forecast(self, delta=1, back_transform : bool = False):
        summation = 0      
        for i in range(len(self.log_rv)):
            numerator = self.log_rv[len(self.log_rv) - i - 1]
            denominator = (i + self.err + delta) * ((i + self.err) ** (self.h + 0.5))
            summation += numerator / denominator
        fc = (math.cos(self.h * math.pi) / math.pi) * (delta ** (self.h + 0.5)) * summation
        if back_transform:
            output = self._back_transform(fc=fc, delta=delta)
        else:
            output = fc
        return output
    
    def forecast_nr(self, delta=1, back_transform : bool = False):
        summation = 0      
        for i in range(len(self.log_rv)):
            numerator = delta * self.log_rv[len(self.log_rv) - i - 1] - (i + self.err + delta) * self.log_rv[len(self.log_rv) - 1]
            denominator = (i + self.err + delta) * ((i + self.err) ** (self.h + 0.5))
            summation += numerator / denominator
        fc = (math.cos(self.h * math.pi) / math.pi) * (delta ** (self.h + 0.5)) * summation
        if back_transform:
            output = self._back_transform(fc=fc, delta=delta)
        else:
            output = fc
        return output

    def backwards_average_forecast(self, delta=1, k=1, back_transform : bool = False):
        log_rv_cp = self.log_rv.copy()
        bwd_sum = 0
        for _ in range(k):
            r = self.forecast(delta=delta, back_transform=back_transform)
            bwd_sum += r
            self.log_rv = self.log_rv[:-1]
        output = bwd_sum / k
        self.log_rv = log_rv_cp
        return output
    
    def moving_window_forecast(self, train_size, delta=1, k=None, back_transform : bool = False):
        log_rv_cp = self.log_rv.copy()
        output = []
        beg = 0
        end = train_size
        while end < len(self.log_rv):
            self.log_rv = self.log_rv[beg:end]
            if k is None:
                if self.h < 0.5:
                    r = self.forecast(delta=delta, back_transform=back_transform)
                else:
                    r = self.forecast_nr(delta=delta, back_transform=back_transform)
            else:
                r = self.backwards_average_forecast(delta=delta, k=k, back_transform=back_transform)
            output.append(r)
            self.log_rv = log_rv_cp
            beg, end = beg + 1, end + 1
        return output

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

    def incremental_forecast(self, train_size, inc=1):
        log_rv_cp = self.log_rv.copy()
        log_rv_fc = []
        d = inc
        self.log_rv = self.log_rv[0:train_size]
        while d <= len(log_rv_cp) - train_size:
            r = self.forecast(delta=d)
            log_rv_fc.append(r)
            d += 1      
        self.log_rv = log_rv_cp
        return log_rv_fc
    
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