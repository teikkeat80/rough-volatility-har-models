import numpy as np
import math
import scipy.integrate as integrate
import visualisation as vis
from fbm import FBM

class FBMAPP:
    def __init__(self, h, n):
        self.n = n
        self.j = math.floor(2 * n ** math.log(1 + 0.25) * math.log(n))
        self.ch = math.sqrt((math.pi * h * ((2 * h) - 1)) / (math.gamma(2 - (2 * h)) * math.gamma(h + .5) ** 2 * math.sin(math.pi * (h - .5))))
        self.zeta_ratio = ((self.j ** (4 - 2 * (h + .5))) / (self.j ** ((- 2) * (h + .5)))) ** (1 / self.j)
        self.zetas = [(self.j ** ((- 2) * (h + .5))) * (self.zeta_ratio ** i) for i in range(self.j + 1)]
        self.c = np.array([integrate.quad(lambda x: self.ch * x ** (- h - .5) / math.gamma(.5 - h), self.zetas[i], self.zetas[i + 1])[0] for i in range(self.j)])
        self.kappa = np.array([(1 / self.c[i]) * integrate.quad(lambda x: self.ch * x * x ** (- h - .5) / math.gamma(.5 - h), self.zetas[i], self.zetas[i + 1])[0] for i in range(self.j)])
        self.y = np.zeros(self.j)
    
    def simulate(self):
        coefficient = self.c.reshape(1, self.j)
        sp = []
        sp.append((coefficient @ self.y).item())
        for _ in range(self.n - 1):
            v = np.random.standard_normal()
            self.y = np.exp(- self.kappa) * self.y + np.sqrt((1 - np.exp(- 2 * self.kappa)) / (2 * self.kappa)) * v
            sp.append((coefficient @ self.y).item())
        sp_trim = sp[-int(self.n / 10):]
        vis.plot_series(sp_trim, r'$W^H_t$')
        return sp_trim

class FBMCLS:
    def __init__(self, h, n):
        self.n = n
        self.h = h

    def simulate(self):
        sp = FBM(n=self.n, hurst=self.h, length=self.n, method='cholesky').fbm()
        vis.plot_series(sp, r'$W^H_t$')
        return sp
    
class FBMDH:
    def __init__(self, h, n):
        self.n = n
        self.h = h

    def simulate(self):
        sp = FBM(n=self.n, hurst=self.h, length=self.n, method='daviesharte').fbm()
        vis.plot_series(sp, r'$W^H_t$')
        return sp