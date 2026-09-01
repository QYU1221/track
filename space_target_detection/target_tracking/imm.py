"""Interacting multiple-model tracker implementation."""

import numpy as np
from configuration.values import (imm_Pi, imm_mu_init, imm_mu_min,
                     cv_Q_diag, cv_P_init_diag,
                     ctrv_Q_diag, ctrv_P_init_diag)
from .linear_kf import LinearKF
from .ctrv_ekf import EKFTracker


# ═══════════════════════════════════════════════════════════════
# 跨维度状态映射（Jacobian 传播协方差）
# ═══════════════════════════════════════════════════════════════

def _cv_to_ctrv(X_cv, P_cv):
    """CV(4D) → CTRV(5D) 状态映射"""
    x, y, vx, vy = X_cv
    v2 = vx**2 + vy**2
    v = np.sqrt(max(v2, 1e-12))
    theta = np.arctan2(vy, vx)
    X_ctrv = np.array([x, y, theta, v, 0.0], dtype=np.float64)

    J = np.zeros((5, 4), dtype=np.float64)
    J[0, 0] = 1.0
    J[1, 1] = 1.0
    J[2, 2] = -vy / max(v2, 1e-12)
    J[2, 3] =  vx / max(v2, 1e-12)
    J[3, 2] = vx / max(v, 1e-12)
    J[3, 3] = vy / max(v, 1e-12)

    P_ctrv = J @ P_cv @ J.T
    return X_ctrv, P_ctrv


def _ctrv_to_cv(X_ctrv, P_ctrv):
    """CTRV(5D) → CV(4D) 状态映射"""
    x, y, theta, v, _omega = X_ctrv
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    vx = v * cos_t
    vy = v * sin_t
    X_cv = np.array([x, y, vx, vy], dtype=np.float64)

    J = np.zeros((4, 5), dtype=np.float64)
    J[0, 0] = 1.0
    J[1, 1] = 1.0
    J[2, 2] = -v * sin_t
    J[2, 3] = cos_t
    J[3, 2] = v * cos_t
    J[3, 3] = sin_t

    P_cv = J @ P_ctrv @ J.T
    return X_cv, P_cv


def _gaussian_log_likelihood(nu, S):
    """计算高斯对数似然 log N(nu; 0, S)，用于模型概率更新"""
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0:
        return -1e10
    d2 = nu @ np.linalg.solve(S, nu)
    return -0.5 * (d2 + logdet + 2.0 * np.log(2.0 * np.pi))


# ═══════════════════════════════════════════════════════════════
# IMM 主类
# ═══════════════════════════════════════════════════════════════

class IMMTracker:
    """交互式多模型滤波器：CV(4D 线性) + CTRV(5D 非线性)

    对外状态统一为 4D CV 形式 [x, y, vx, vy]。
    模型概率 μ = [μ_cv, μ_ctrv] 自动在直线/转弯模型间切换。
    """

    def __init__(self, x0, y0, vx0=0.0, vy0=0.0):
        # CV 子滤波器
        self.cv = LinearKF(x0, y0, vx0, vy0,
                           Q_diag=cv_Q_diag, P_init_diag=cv_P_init_diag)

        # CTRV 子滤波器
        v0 = np.sqrt(vx0**2 + vy0**2) if (vx0 != 0.0 or vy0 != 0.0) else 2.0
        theta0 = np.arctan2(vy0, vx0) if (vx0 != 0.0 or vy0 != 0.0) else 0.0
        self.ctrv = EKFTracker(x0, y0, theta0=theta0, v0=v0, omega0=0.0)
        # 覆盖 EKF 默认参数为 IMM 专用
        self.ctrv.Q = np.diag(ctrv_Q_diag).astype(np.float64)
        self.ctrv.P = np.diag(ctrv_P_init_diag).astype(np.float64)

        # 模型概率
        self.mu = np.array(imm_mu_init, dtype=np.float64)
        self._cbar = np.ones(2, dtype=np.float64)  # 归一化常数缓存

        # 各模型上一次更新后的状态（用于交互混合）
        self._X_cv = self.cv.X.copy()
        self._X_ctrv = self.ctrv.X.copy()
        self._P_cv = self.cv.P.copy()
        self._P_ctrv = self.ctrv.P.copy()

        # 对外组合状态
        self.X = self._X_cv.copy()
        self.P = self._P_cv.copy()

    # ── 预测：交互混合 → 各自预测 → 组合 ──────────────

    def predict(self):
        self._interact()

        self.cv.predict()
        self.ctrv.predict()

        self._X_cv = self.cv.X.copy()
        self._X_ctrv = self.ctrv.X.copy()
        self._P_cv = self.cv.P.copy()
        self._P_ctrv = self.ctrv.P.copy()

        self._ctrv_to_cv_combined()
        return self.X.copy(), self.P.copy()

    # ── JPDA 软关联更新 ──────────────────────────────

    def update_jpda(self, measurements, weights):
        # 用预测状态和协方差计算各模型似然
        L_cv = self._jpda_likelihood_cv(measurements, weights)
        L_ctrv = self._jpda_likelihood_ctrv(measurements, weights)

        # 各模型独立更新
        self.cv.update_jpda(measurements, weights)
        self.ctrv.update_jpda(measurements, weights)

        self._update_mu(L_cv, L_ctrv)

        self._X_cv = self.cv.X.copy()
        self._X_ctrv = self.ctrv.X.copy()
        self._P_cv = self.cv.P.copy()
        self._P_ctrv = self.ctrv.P.copy()

        self._ctrv_to_cv_combined()
        return self.X.copy(), self.P.copy()

    # ── 直接设位置（fallback 覆盖）─────────────────

    def set_position(self, x, y):
        """直接设位置，同时更新两个子滤波器"""
        self.cv.X[0], self.cv.X[1] = x, y
        self.ctrv.X[0], self.ctrv.X[1] = x, y
        self._X_cv = self.cv.X.copy()
        self._X_ctrv = self.ctrv.X.copy()
        self._ctrv_to_cv_combined()

    # ═══════════════════════════════════════════════════════════
    # 内部方法
    # ═══════════════════════════════════════════════════════════

    def _interact(self):
        """IMM 交互混合：将上一帧各模型估计按 μ_{i|j} 混合后注入各子滤波器"""
        Pi = np.array(imm_Pi, dtype=np.float64)
        mu = np.maximum(self.mu, imm_mu_min)
        mu = mu / mu.sum()

        cbar = Pi.T @ mu  # [c̄_0, c̄_1]
        self._cbar = cbar

        # 混合权重 μ_{i|j} = (1 / c̄_j) * Π_{ij} * μ_i
        mix_w = np.zeros((2, 2), dtype=np.float64)
        for j in range(2):
            for i in range(2):
                mix_w[i, j] = Pi[i, j] * mu[i] / max(cbar[j], 1e-12)

        # 源模型状态 (i) → 目标模型空间 (j) → 加权混合
        src_X = {0: self._X_cv, 1: self._X_ctrv}
        src_P = {0: self._P_cv, 1: self._P_ctrv}

        mappings = {
            (0, 0): lambda x, p: (x.copy(), p.copy()),         # CV → CV (identity)
            (0, 1): _cv_to_ctrv,                                # CV → CTRV
            (1, 0): _ctrv_to_cv,                                # CTRV → CV
            (1, 1): lambda x, p: (x.copy(), p.copy()),         # CTRV → CTRV (identity)
        }

        for j in range(2):
            X_mix = None
            P_mix = None
            mapped_states = []

            for i in range(2):
                if mix_w[i, j] < 1e-12:
                    continue
                X_ij, P_ij = mappings[(i, j)](src_X[i], src_P[i])
                mapped_states.append((mix_w[i, j], X_ij, P_ij))

            if not mapped_states:
                _, X_ij, P_ij = mappings[(j, j)](src_X[j], src_P[j])
                X_mix = X_ij
                P_mix = P_ij
            else:
                X_mix = np.zeros_like(mapped_states[0][1])
                for w, X_ij, _ in mapped_states:
                    X_mix += w * X_ij

                P_mix = np.zeros_like(mapped_states[0][2])
                for w, X_ij, P_ij in mapped_states:
                    dX = (X_ij - X_mix).reshape(-1, 1)
                    P_mix += w * (P_ij + dX @ dX.T)

            # 注入子滤波器，作为预测起点
            if j == 0:
                self.cv.set_state(X_mix, P_mix)
            else:
                self.ctrv.set_state(X_mix, P_mix)

    def _jpda_likelihood_cv(self, measurements, weights):
        """计算 JPDA 加权创新在 CV 模型下的对数似然"""
        nu_w = np.zeros(2, dtype=np.float64)
        for z, w in zip(measurements, weights):
            if w > 1e-8:
                nu_w += w * (np.asarray(z, dtype=np.float64) - self.cv.H @ self.cv.X)
        S = self.cv.H @ self.cv.P @ self.cv.H.T + self.cv.R
        return _gaussian_log_likelihood(nu_w, S)

    def _jpda_likelihood_ctrv(self, measurements, weights):
        """计算 JPDA 加权创新在 CTRV 模型下的对数似然"""
        nu_w = np.zeros(2, dtype=np.float64)
        for z, w in zip(measurements, weights):
            if w > 1e-8:
                nu_w += w * (np.asarray(z, dtype=np.float64) - self.ctrv.H @ self.ctrv.X)
        S = self.ctrv.H @ self.ctrv.P @ self.ctrv.H.T + self.ctrv.R
        return _gaussian_log_likelihood(nu_w, S)

    def _update_mu(self, L_cv, L_ctrv):
        """根据模型似然更新模型概率"""
        L = np.array([L_cv, L_ctrv], dtype=np.float64)
        L_max = L.max()
        L_exp = np.exp(L - L_max)  # log-sum-exp 防下溢
        mu_unnorm = L_exp * self._cbar
        mu_new = mu_unnorm / mu_unnorm.sum()
        self.mu = np.maximum(mu_new, imm_mu_min)
        self.mu = self.mu / self.mu.sum()

    def _ctrv_to_cv_combined(self):
        """将各模型状态映射回 CV 空间并组合为对外状态"""
        X0, P0 = self._X_cv, self._P_cv                         # CV: already 4D
        X1_cv, P1_cv = _ctrv_to_cv(self._X_ctrv, self._P_ctrv)  # CTRV → 4D

        X = self.mu[0] * X0 + self.mu[1] * X1_cv
        dX0 = (X0 - X).reshape(-1, 1)
        dX1 = (X1_cv - X).reshape(-1, 1)
        P = (self.mu[0] * (P0 + dX0 @ dX0.T) +
             self.mu[1] * (P1_cv + dX1 @ dX1.T))

        self.X = X
        self.P = P
