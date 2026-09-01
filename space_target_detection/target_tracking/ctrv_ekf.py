"""CTRV extended Kalman filter implementation."""

import numpy as np
from configuration.values import Q_diag, R_diag, P_init_diag


class EKFTracker:
    """5 维扩展 Kalman 滤波器：状态 = [x, y, θ, v, ω]"""

    def __init__(self, x0, y0, theta0=0.0, v0=2.0, omega0=0.0):
        self.X = np.array([x0, y0, theta0, v0, omega0], dtype=np.float64)

        self.P = np.diag(P_init_diag).astype(np.float64)
        self.Q = np.diag(Q_diag).astype(np.float64)
        self.R = np.diag(R_diag).astype(np.float64)
        self.H = np.array([[1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0]], dtype=np.float64)

    def predict(self):
        x, y, theta, v, omega = self.X

        sin_t = np.sin(theta)
        cos_t = np.cos(theta)

        x_pred = x + v * cos_t
        y_pred = y + v * sin_t
        theta_pred = theta + omega
        v_pred = v
        omega_pred = omega

        self.X = np.array([x_pred, y_pred, theta_pred, v_pred, omega_pred])

        # 雅可比 F (5×5)
        self.F = np.array([
            [1, 0, -v * sin_t, cos_t, 0],
            [0, 1,  v * cos_t, sin_t, 0],
            [0, 0,  1,         0,     1],
            [0, 0,  0,         1,     0],
            [0, 0,  0,         0,     1],
        ], dtype=np.float64)

        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.X.copy(), self.P.copy()

    def update(self, z):
        z = np.asarray(z, dtype=np.float64)
        y_residual = z - self.H @ self.X
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.X = self.X + K @ y_residual
        self.P = (np.eye(5) - K @ self.H) @ self.P
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
        P_correct = (np.eye(5) - K @ self.H) @ self.P
        self.P = beta_0 * self.P + (1.0 - beta_0) * P_correct
        return self.X.copy(), self.P.copy()

    def set_position(self, x, y):
        self.X[0], self.X[1] = x, y

    def set_state(self, X, P):
        self.X = X.copy()
        self.P = P.copy()
