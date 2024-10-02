import numpy as np
import math

class RoughVolatility:
    def __init__(self, rv: list, h: float, err: float):
        self.h = h
        self.log_rv = np.log(rv)
        self.err = err
    
    def forecast(self, delta=1):
        summation = 0      
        for i in range(len(self.log_rv)):
            numerator = self.log_rv[len(self.log_rv) - i - 1]
            denominator = (i + self.err + delta) * ((i + self.err) ** (self.h + 0.5))
            summation += numerator / denominator    
        output = (math.cos(self.h * math.pi) / math.pi) * (delta ** (self.h + 0.5)) * summation
        return output

    def backwards_average_forecast(self, delta=1, k=1):
        log_rv_cp = self.log_rv.copy()
        bwd_sum = 0
        for _ in range(k):
            r = self.forecast(delta=delta)
            bwd_sum += r
            self.log_rv = self.log_rv[:-1]
        output = bwd_sum / k
        self.log_rv = log_rv_cp
        return output
    
    def moving_window_forecast(self, train_size, delta=1, k=None):
        log_rv_cp = self.log_rv.copy()
        log_rv_fc = []
        beg = 0
        end = train_size
        while end < len(self.log_rv):
            self.log_rv = self.log_rv[beg:end]
            if k is None:
                r = self.forecast(delta=delta)
            else:
                r = self.backwards_average_forecast(delta=delta, k=k)
            log_rv_fc.append(r)
            self.log_rv = log_rv_cp
            beg, end = beg + 1, end + 1
        return log_rv_fc
    
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