import numpy as np

# HARK Class
class HARK:
    def __init__(self, initial_a: np.array, initial_P: np.array, **kwargs) -> None:
        self._a = initial_a
        self._P = initial_P
        self._T = np.vstack((np.array([kwargs.get('beta_d')] + 
                                      [kwargs.get('beta_w')/4] * 4 + 
                                      [kwargs.get('beta_m')/17] * 17),          # why 4 and 17?
                                      np.hstack((np.eye(21), np.zeros((21, 1))))))
        self._c = np.array([kwargs.get('beta_0')] + [0] * (21))
        self._Q = np.zeros((22, 22))
        self._Q[0, 0] = kwargs.get('q')
        self._Z = np.zeros(22)
        self._Z[0] = 1
    
    def predict(self) -> None:
        # a_t+1 = c + T a_t
        a_pred = self._c + self._T.dot(self._a)
        # P_t+1 = T P_t T^T + Q
        P_pred = self._T.dot(self._P).dot(self._T.T) + self._Q

        self._a = a_pred
        self._P = P_pred

    def update(self, rv: np.array, h: np.array) -> None:
        # v_t = RV_t - Z a_t
        v = rv - self._Z.dot(self._a)           # is rv a vector or a single value
        # F_t = Z P_t Z^T + h_t
        F = self._Z.dot(self._P).dot(self._Z.T) + h         # is h a vector or a single value
        # K_t = T P_t Z^T F_t^-1
        K = self._T.dot(self._P).dot(self._Z.T).dot(np.linalg.inv(F))
        # a_t+1 = c + T a_t + K_t v_t
        a_upd = self._c + self._T.dot(self._a) + K.dot(v)
        # P_t+1 = T P_t (T - K_t Z)^T + Q
        P_upd = self.T.dot(self._P).dot((self._T - K.dot(self._Z)).T) + self._Q

        self._a = a_upd
        self._P = P_upd