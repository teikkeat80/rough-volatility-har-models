import numpy as np
import math
from matplotlib import pyplot as plt

class Hurst:
    def __init__(self, series, q_list, max_delta, log=True):
        if log:
            self.series = np.log(series)
        else:
            self.series = series
        self.q = q_list
        self.delta = np.arange(1, max_delta)
        self.log_delta = np.log(self.delta)
        self.n = len(series)

    def _calc_m(self, q):
        m = []
        for d in self.delta:
            diff = np.abs(self.series[d:self.n] - self.series[:self.n - d]) ** q
            m.append(np.mean(diff))
        return m
    
    def _calc_zeta(self):
        zeta = []
        k = []
        for qq in self.q:
            mod = np.polyfit(self.log_delta, np.log(self._calc_m(qq)), 1)
            zeta.append(mod[0])
            k.append(mod[1])       
        return zeta, k
    
    def _scale_zeta_q(self):
        zeta, _ = self._calc_zeta()
        line = np.polyfit(self.q, zeta, 1)
        return line[0], line[1]
    
    def est_h(self):
        h, _ = self._scale_zeta_q()
        return h
    
    def est_nu(self, order=2):
        _, k = self._calc_zeta()
        return math.sqrt(math.exp(k[self.q.index(order)]))

    def plot_scale_m_delta(self):
        zeta, k = self._calc_zeta()

        plt.figure(figsize=(10, 6))
        for i, q in enumerate(self.q):
            plt.plot(self.log_delta, np.log(self._calc_m(q)), 'o', label=str(q))
            plt.plot(self.log_delta, self.log_delta * zeta[i] + k[i])
        plt.title('Scaling of log m with lag')
        plt.xlabel('log delta')
        plt.ylabel('log m')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_scale_zeta_q(self):
        h, c = self._scale_zeta_q()
        zeta, _ = self._calc_zeta()
        x = np.linspace(self.q[0], self.q[-1], 100)
        y = h * x + c

        plt.figure(figsize=(10, 6))
        plt.plot(self.q, zeta, label='zeta to q')
        plt.plot(x, y, label=f'h={round(h, 6)} line')
        plt.title('Scaling of Zeta with q')
        plt.xlabel('q')
        plt.ylabel('zeta')
        plt.legend()
        plt.show()