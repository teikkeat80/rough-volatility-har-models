import numpy as np
import math

class RFSV:
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