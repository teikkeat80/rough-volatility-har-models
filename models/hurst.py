import numpy as np
from matplotlib import pyplot as plt

class Hurst:
    def __init__(self, rv: list, q_list: list, max_delta: int, m_calc_method: str):
        self.log_sqrt_rv = np.log(np.sqrt(rv))
        self.q_list = q_list
        self.delta_list = np.arange(1, max_delta)
        self.log_delta_list = np.log(self.delta_list)
        self.arr_len = len(rv)
        if m_calc_method == 'disjoint':
            self.m_calc = self._calc_m_disjoint_inc
        elif m_calc_method == 'overlap':
            self.m_calc = self._calc_m_overlap_inc
        else:
            raise ValueError("Invalid method to calculate m values. Choose either 'disjoint' or 'overlap'.")

    def _calc_m_disjoint_inc(self, q):      
        m = []
        for d in self.delta_list:
            n = (self.arr_len - 1) // d
            # Optimised with np
            # diff = np.abs(self.log_sqrt_rv[(1 + np.arange(n)) * d] - self.log_sqrt_rv[np.arange(n) * d]) ** q
            # m.append(np.mean(diff))
            # Original
            x = 0
            for k in range(n):
                x += abs(self.log_sqrt_rv[(k + 1) * d] - self.log_sqrt_rv[k * d]) ** q
            m.append((1 / n) * x)
        return m

    def _calc_m_overlap_inc(self, q):
        m = []
        for d in self.delta_list:
            # Optimised
            # diff = np.abs(self.log_sqrt_rv[d:self.arr_len] - self.log_sqrt_rv[:self.arr_len - d]) ** q
            # m.append(np.mean(diff))
            # Original
            x = 0
            n = 0
            for i in range(d, self.arr_len):
                x += abs(self.log_sqrt_rv[i] - self.log_sqrt_rv[i - d]) ** q
                n += 1
            m.append((1 / n) * x)
        return m
    
    def _calc_zeta(self):
        zeta_list = []
        k_list = []

        for q in self.q_list:
            log_m = np.log(self.m_calc(q))
            mod = np.polyfit(self.log_delta_list, log_m, 1)
            zeta_list.append(mod[0])
            k_list.append(mod[1])
        
        return zeta_list, k_list
    
    def _scale_zeta_q(self):
        zeta_list, _ = self._calc_zeta()
        line = np.polyfit(self.q_list, zeta_list, 1)
        return line[0], line[1]
    
    def est_h(self):
        h, _ = self._scale_zeta_q()
        return h
    
    def plot_scale_m_delta(self):
        zeta_list, k_list = self._calc_zeta()

        plt.figure(figsize=(10, 6))

        for i, q in enumerate(self.q_list):
            log_m = np.log(self.m_calc(q))
            plt.plot(self.log_delta_list, log_m, 'o', label=str(q))
            plt.plot(self.log_delta_list, self.log_delta_list * zeta_list[i] + k_list[i])

        plt.title('Scaling of log m with lag')
        plt.xlabel('log delta')
        plt.ylabel('log m')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_scale_zeta_q(self):
        h, c = self._scale_zeta_q()
        zeta_list, _ = self._calc_zeta()

        plt.figure(figsize=(10, 6))

        x = np.linspace(self.q_list[0], self.q_list[-1], 100)
        y = h * x + c

        plt.plot(self.q_list, zeta_list, label='zeta to q')
        plt.plot(x, y, label=f'h={round(h, 6)} line')

        plt.title('Scaling of Zeta with q')
        plt.xlabel('q')
        plt.ylabel('zeta')
        plt.legend()
        plt.show()
    
# # Test
# df_raw = pd.read_csv('SNP500_RV_5min.csv')
# df_sorted = df_raw.sort_values(by='Date', ignore_index=True)
# rv = df_sorted['RV'].tolist()
# q_list = [0.5, 1, 1.5, 2, 3]
# max_delta = 30

# h_instance = Hurst(rv, q_list, max_delta)
# h_values = h_instance.calc_h('overlap')
# print(h_values)
# h = h_instance.h(2, 'overlap')
# print(h)

# h_instance.plot_scale_diagram('overlap')
# h_instance.plot_zeta_diagram(2, 'overlap')

# def calc_m(rv, q, delta:list, t):
#     log_sqrt_rv = np.log(np.sqrt(rv))
#     m = []

#     for d in delta:
#         n = math.floor(t / d)
#         x = []
#         for k in range(n):
#             x.append(abs(log_sqrt_rv[(k + 1) * d] - log_sqrt_rv[k * d]) ** q)
#         m.append((1 / n) * sum(x))
    
#     return m

# def calc_m_alt(rv, q, delta:list, t):
#     log_sqrt_rv = np.log(np.sqrt(rv))[:t]
#     m = []
    
#     for d in delta:
#         x = []
#         for i in range(d, len(log_sqrt_rv)):
#             x.append(abs(log_sqrt_rv[i] - log_sqrt_rv[i - d]) ** q)
#         m.append((1 / len(x)) * sum(x))
    
#     return m

# # Calculation of zeta and H
# zeta_list = []
# k_list = []
# hurst_list = []

# for q in q_list:
#     log_m = np.log(calc_m_alt(rv, q, delta_list, 1000))
#     mod = np.polyfit(log_delta_list, log_m, 1)
#     zeta_list.append(mod[0])
#     k_list.append(mod[1])
#     hurst_list.append(mod[0] / q)

# print(hurst_list)

# hurst_q2 = hurst_list[3]

# # Log m diagram
# plt.figure(figsize=(10, 6))

# for i, q in enumerate(q_list):
#     log_m = np.log(calc_m_alt(rv, q, delta_list, 1000))
#     plt.plot(log_delta_list, log_m, 'o', label=str(q))
#     plt.plot(log_delta_list, log_delta_list * zeta_list[i] + k_list[i])

# plt.title('Scaling of log m with lag')
# plt.xlabel('log delta')
# plt.ylabel('log m')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Zeta diagram
# plt.figure(figsize=(10, 6))
# x = np.linspace(q_list[0], q_list[-1], 100)
# y = hurst_q2 * x

# plt.plot(q_list, zeta_list, label='zeta to q')
# plt.plot(x, y, label='h_line')

# plt.title('zeta with q')
# plt.xlabel('q')
# plt.ylabel('zeta')
# plt.legend()
# plt.show()