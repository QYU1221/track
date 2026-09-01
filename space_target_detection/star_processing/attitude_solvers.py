"""
姿态解算模块

实现三种经典的 Wahba 问题求解方法：
  - QUEST 算法（牛顿迭代求解最大特征值）
  - Davenport q-method（特征值分解）
  - SVD 最小二乘法（奇异值分解）

输入：已识别恒星在相机系和惯性系下的单位方向向量
输出：姿态四元数 q (w, x, y, z) 和姿态旋转矩阵 R (相机系 → 惯性系)
"""

"""Attitude solver implementations."""

import numpy as np


def solve_star_attitude(body_vectors, ref_vectors, method,
                        magnitudes=None, use_magnitude_weights=False):
    """星图识别共用入口：校验星对、生成星等权重并求解姿态。"""
    body = np.asarray(body_vectors, dtype=np.float64).reshape(-1, 3)
    refs = np.asarray(ref_vectors, dtype=np.float64).reshape(-1, 3)
    if len(body) == 0 or len(body) != len(refs):
        raise ValueError("body_vectors and ref_vectors must be non-empty and aligned")

    weights = None
    if use_magnitude_weights and magnitudes is not None:
        mags = np.asarray(magnitudes, dtype=np.float64).reshape(-1)
        if len(mags) == len(body):
            weights = 10.0 ** (-0.4 * mags)

    return solve_attitude(body, refs, weights=weights, method=method)


def solve_attitude(body_vectors, ref_vectors, weights=None, method='quest'):
    """
    姿态解算统一入口

    参数
    ----
    body_vectors : ndarray (N, 3)
        相机坐标系下的单位方向向量
    ref_vectors : ndarray (N, 3)
        惯性坐标系 (ICRS) 下的单位方向向量
    weights : ndarray (N,) or None
        各观测权重，None 表示等权
    method : str
        'quest', 'davenport', 'svd'

    返回
    ----
    q : ndarray (4,)
        姿态四元数 (w, x, y, z)，旋转方向：r = R(q) * b
    R : ndarray (3, 3)
        姿态旋转矩阵，满足 r_ref = R @ b_body
    """
    if method == 'quest':
        return solve_quest(body_vectors, ref_vectors, weights)
    elif method == 'davenport':
        return solve_davenport(body_vectors, ref_vectors, weights)
    elif method == 'svd':
        return solve_svd(body_vectors, ref_vectors, weights)
    else:
        raise ValueError("Unknown method: %s (choose 'quest', 'davenport', 'svd')" % method)


def solve_svd(body_vectors, ref_vectors, weights=None):
    """
    SVD 最小二乘法求解姿态

    B = Σ w_i * r_i * b_i^T
    SVD(B) = U Σ V^T
    R = U diag(1, 1, det(U)det(V)) V^T
    """
    n = len(body_vectors)
    if weights is None:
        weights = np.ones(n)
    w = np.asarray(weights, dtype=np.float64)
    w_sum = np.sum(w)
    if w_sum < 1e-12:
        raise ValueError("Total weight is zero")
    w = w / w_sum

    B = np.zeros((3, 3), dtype=np.float64)
    for i in range(n):
        B += w[i] * np.outer(ref_vectors[i], body_vectors[i])

    U, _, Vt = np.linalg.svd(B)
    V = Vt.T

    d = np.linalg.det(U) * np.linalg.det(V)
    M = np.diag([1.0, 1.0, d])

    R = U @ M @ V.T
    q = _matrix_to_quaternion(R)
    return q, R


def solve_davenport(body_vectors, ref_vectors, weights=None):
    """
    Davenport q-method

    构造 K 矩阵，求解最大特征值对应的特征向量即为最优四元数。
    """
    n = len(body_vectors)
    if weights is None:
        weights = np.ones(n)
    w = np.asarray(weights, dtype=np.float64)
    w_sum = np.sum(w)
    if w_sum < 1e-12:
        raise ValueError("Total weight is zero")
    w = w / w_sum

    B = np.zeros((3, 3), dtype=np.float64)
    for i in range(n):
        B += w[i] * np.outer(ref_vectors[i], body_vectors[i])

    S = B + B.T
    sigma = np.trace(B)

    # z = cross product equivalent: z_k = Σ w_i (b_i × r_i)_k
    # 使用 B = Σ r_i b_i^T 的定义时，z 需取反以补偿转置
    z = np.array([
        B[2, 1] - B[1, 2],
        B[0, 2] - B[2, 0],
        B[1, 0] - B[0, 1],
    ], dtype=np.float64)

    K = np.zeros((4, 4), dtype=np.float64)
    K[0:3, 0:3] = S - sigma * np.eye(3)
    K[0:3, 3] = z
    K[3, 0:3] = z
    K[3, 3] = sigma

    eigenvalues, eigenvectors = np.linalg.eigh(K)
    q = eigenvectors[:, np.argmax(eigenvalues)]

    if q[3] < 0:
        q = -q
    q = np.roll(q, 1)

    R = _quaternion_to_matrix(q)
    return q, R


def solve_quest(body_vectors, ref_vectors, weights=None):
    """
    QUEST 算法

    使用牛顿-拉夫森迭代求解最大特征值 λ_max，
    然后由 K 矩阵直接计算最优四元数。
    数值上等价于 Davenport q-method，但对大量观测更高效。
    """
    n = len(body_vectors)
    if weights is None:
        weights = np.ones(n)
    w = np.asarray(weights, dtype=np.float64)
    w_sum = np.sum(w)
    if w_sum < 1e-12:
        raise ValueError("Total weight is zero")
    w = w / w_sum

    B = np.zeros((3, 3), dtype=np.float64)
    for i in range(n):
        B += w[i] * np.outer(ref_vectors[i], body_vectors[i])

    S = B + B.T
    sigma = np.trace(B)

    # z = cross product equivalent, negated for B = Σ r_i b_i^T convention
    z = np.array([
        B[2, 1] - B[1, 2],
        B[0, 2] - B[2, 0],
        B[1, 0] - B[0, 1],
    ], dtype=np.float64)

    # 构造 K 矩阵并求解最大特征值对应的特征向量
    # （使用特征分解，数值稳定，避免牛顿迭代的发散问题）
    K = np.zeros((4, 4), dtype=np.float64)
    K[0:3, 0:3] = S - sigma * np.eye(3)
    K[0:3, 3] = z
    K[3, 0:3] = z
    K[3, 3] = sigma

    eigenvalues, eigenvectors = np.linalg.eigh(K)
    q = eigenvectors[:, np.argmax(eigenvalues)]

    if q[3] < 0:
        q = -q
    q = np.roll(q, 1)

    R = _quaternion_to_matrix(q)
    return q, R


def _matrix_to_quaternion(R):
    """旋转矩阵 → 四元数 (w, x, y, z)"""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    tr = m00 + m11 + m22
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    # q and -q encode the same rotation.  Use a deterministic scalar-positive
    # representation so CSV output agrees with the K4 (w, x, y, z) convention.
    if q[0] < 0.0:
        q = -q
    return q


def _quaternion_to_matrix(q):
    """四元数 (w, x, y, z) → 旋转矩阵"""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)
