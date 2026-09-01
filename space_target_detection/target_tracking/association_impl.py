"""Track/candidate feature fusion and Sinkhorn assignment."""

import numpy as np
from configuration.values import alpha, beta, gamma


def sinkhorn_soft_assignment(S, num_iters=None, epsilon=None):
    """Sinkhorn-Knopp 最优传输软分配

    将相似度矩阵 S (n×m) 转化为软分配矩阵 P，满足：
      - 每行近似 sum ≤ 1（每条轨迹的关联质量分布到各候选）
      - 每列近似 sum ≤ 1（每个候选的贡献分布到各轨迹）

    通过增广 dustbin 行/列处理轨迹-候选数量不匹配。

    参数
    ----
    S : ndarray (n_tracks, n_cands)
        相似度矩阵，值越大匹配越好
    num_iters : int or None
        Sinkhorn 迭代次数，None 则用 config 默认值
    epsilon : float or None
        熵正则化强度，越小越接近硬匹配，None 则用 config 默认值

    返回
    ----
    P : ndarray (n_tracks, n_cands)
        软分配概率矩阵
    """
    from configuration.values import sinkhorn_epsilon, sinkhorn_iters

    if epsilon is None:
        epsilon = sinkhorn_epsilon
    if num_iters is None:
        num_iters = sinkhorn_iters

    n, m = S.shape
    if n == 0 or m == 0:
        return S.copy()

    # 增广 dustbin —— 吸收无法匹配的质量
    S_aug = np.zeros((n + 1, m + 1))
    S_aug[:n, :m] = S

    # dustbin 亲和度取相似度中位数的 0.3 倍，保守处理未匹配
    valid = S[S > 1e-8]
    dustbin_val = float(np.median(valid)) * 0.3 if len(valid) > 0 else 0.01
    S_aug[:n, m] = dustbin_val
    S_aug[n, :m] = dustbin_val

    # Gibbs 核 K = exp(S / ε)
    K = np.exp(S_aug / epsilon)

    # 目标边缘分布：每行/列 sum = 1，dustbin 吸收剩余质量
    a = np.ones(n + 1)
    a[-1] = m  # dustbin 行提供 m 单位质量源
    b = np.ones(m + 1)
    b[-1] = n  # dustbin 列吸收 n 单位质量

    # Sinkhorn-Knopp 迭代
    v = np.ones(m + 1)
    for _ in range(num_iters):
        u = a / (K @ v + 1e-12)
        v = b / (K.T @ u + 1e-12)

    P_aug = np.diag(u) @ K @ np.diag(v)
    return P_aug[:n, :m]


def build_cost_matrix(tracks, candidates, energies=None, radii=None):
    """
    三特征融合关联代价矩阵（马氏距离 + 速度门控 + 速度一致性）

    φ = α·pos_err + β·E_err + γ·R_err + δ·v_penalty
    其中 pos_err 使用马氏距离，利用 EKF 协方差 P 自适应缩放。

    速度门控：候选距轨迹预测位置超过 max_velocity 像素的直接拒绝。
    速度一致性：候选引起的位移矢量与近期位移矢量偏差越大、惩罚越重，
    防止噪点诱偏 EKF 导致轨迹断裂。

    参数
    ----
    tracks : List[dict]
    candidates : List[(x, y, flux)]
    energies : ndarray or None  候选点能量数组（CSV 输入时预计算）
    radii : ndarray or None     候选点半径数组（CSV 输入时预计算）

    返回行归一化的概率矩阵 Phi，每行 sum = 1。
    """
    from configuration.values import R_diag, max_velocity

    # 速度一致性权重（与 alpha/beta/gamma 同级）
    delta = 0.15

    n_tracks = len(tracks)
    n_cands = len(candidates)

    if n_tracks == 0 or n_cands == 0:
        return np.zeros((n_tracks, n_cands))

    if energies is None:
        energies = np.zeros(n_cands)
    if radii is None:
        radii = np.full(n_cands, 2.0)
    cand_energies = np.asarray(energies)
    cand_radii = np.asarray(radii)
    track_positions = np.array([t['state'][:2] for t in tracks])

    Phi = np.zeros((n_tracks, n_cands))

    for i in range(n_tracks):
        t = tracks[i]
        pred_x, pred_y = track_positions[i]

        E_i = t['energy_hist'][-1] if t['energy_hist'] else 0.0
        R_i = (t['radius_hist'][-1]
               if t['radius_hist'] and not np.isnan(t['radius_hist'][-1])
               else 2.0)

        # EKF 平滑速度（用于速度一致性检查，比原始位置差更稳定）
        state = t['state']
        if len(state) == 4:
            ekf_vx, ekf_vy = float(state[2]), float(state[3])
        elif len(state) >= 5:
            v, theta = float(state[3]), float(state[2])
            ekf_vx, ekf_vy = float(v * np.cos(theta)), float(v * np.sin(theta))
        else:
            ekf_vx, ekf_vy = 0.0, 0.0
        ekf_speed = float(np.sqrt(ekf_vx**2 + ekf_vy**2))

        S_inv = None
        S = t['P'][:2, :2] + np.diag(R_diag)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            pass

        for j in range(n_cands):
            cx, cy, _ = candidates[j]

            # 速度门控：帧间位移超过物理上限直接拒绝
            eucl_dist = np.sqrt((pred_x - cx) ** 2 + (pred_y - cy) ** 2)
            if eucl_dist > max_velocity:
                Phi[i, j] = 0.0
                continue

            if S_inv is not None:
                z_diff = np.array([cx - pred_x, cy - pred_y])
                pos_err = np.sqrt(max(z_diff @ S_inv @ z_diff, 0.0))
            else:
                pos_err = eucl_dist

            E_j = cand_energies[j]
            E_diff = abs(E_i - E_j) / max(abs(E_i) + abs(E_j), 1e-8)

            R_j = cand_radii[j]
            if np.isnan(R_j):
                R_j = R_i
            R_diff = abs(R_i - R_j) / max(abs(R_i) + abs(R_j), 1e-8)

            # 速度一致性惩罚：候选位移与 EKF 平滑速度的偏差
            # 用 EKF 速度而非原始位置差做参考——
            # 交叉时两轨迹滤波速度指向不同方向，能正确区分；
            # 诱偏时噪点位移与 EKF 平滑速度偏差大，同样被惩罚。
            v_penalty = 0.0
            if ekf_speed > 0.5:
                cand_disp = np.array([cx - pred_x, cy - pred_y],
                                     dtype=np.float64)
                cand_mag = float(np.linalg.norm(cand_disp))
                if cand_mag > 0.5:
                    # 方向偏差：cos 相似度
                    cos_sim = np.dot(cand_disp, [ekf_vx, ekf_vy]) / max(ekf_speed * cand_mag, 1e-8)
                    cos_sim = np.clip(float(cos_sim), -1.0, 1.0)
                    angle_penalty = 1.0 - cos_sim  # 同向→0, 反向→2
                    # 速率偏差
                    ratio = max(ekf_speed, cand_mag) / max(min(ekf_speed, cand_mag), 1e-8)
                    speed_penalty = max(0.0, np.log2(float(ratio)))  # 2x→1, 不变→0
                    v_penalty = angle_penalty + speed_penalty

            phi = (alpha * pos_err + beta * E_diff +
                   gamma * R_diff + delta * v_penalty)
            Phi[i, j] = 1.0 / (phi + 1e-8)

    # 行归一化：每行变成一个概率分布（sum_j P(track_i → cand_j) = 1）
    row_sums = Phi.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-12] = 1.0
    Phi = Phi / row_sums

    return Phi
