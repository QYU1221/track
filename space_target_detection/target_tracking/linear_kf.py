"""Linear constant-velocity Kalman filter implementation."""

import numpy as np


class LinearKF:
    """4 维线性 Kalman 滤波器：状态 = [x, y, vx, vy]"""

    def __init__(self, x0, y0, vx0=0.0, vy0=0.0,
                 Q_diag=None, R_diag=None, P_init_diag=None):
        from configuration.values import cv_Q_diag, cv_P_init_diag, R_diag as _R

        self.X = np.array([x0, y0, vx0, vy0], dtype=np.float64)

        self.P = np.diag(P_init_diag if P_init_diag is not None else cv_P_init_diag).astype(np.float64)
        self.Q = np.diag(Q_diag if Q_diag is not None else cv_Q_diag).astype(np.float64)
        self.R = np.diag(R_diag if R_diag is not None else _R).astype(np.float64)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float64)

        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

    def predict(self):
        self.X = self.F @ self.X
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.X.copy(), self.P.copy()

    def update(self, z):
        z = np.asarray(z, dtype=np.float64)
        y_residual = z - self.H @ self.X
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.X = self.X + K @ y_residual
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.X.copy(), self.P.copy()

    def update_jpda(self, measurements, weights):
        nu_weighted = np.zeros(2, dtype=np.float64)
        for z, w in zip(measurements, weights):
            if w > 1e-8:
                nu_weighted += w * (np.asarray(z, dtype=np.float64) - self.H @ self.X)

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        beta_0 = max(0.0, 1.0 - sum(weights))

        self.X = self.X + K @ nu_weighted
        P_correct = (np.eye(4) - K @ self.H) @ self.P
        self.P = beta_0 * self.P + (1.0 - beta_0) * P_correct
        return self.X.copy(), self.P.copy()

    def get_innovation(self, z):
        z = np.asarray(z, dtype=np.float64)
        nu = z - self.H @ self.X
        S = self.H @ self.P @ self.H.T + self.R
        return nu, S

    def set_state(self, X, P):
        self.X = X.copy()
        self.P = P.copy()

    def set_position(self, x, y):
        self.X[0], self.X[1] = x, y
