"""投影点与观测点的全局一对一匹配。

匈牙利匹配优先；scipy 不可用时自动回退为贪心最近邻。
"""

from typing import List, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


def match_projected_points(
    projected_xy: np.ndarray,
    observed_xy: np.ndarray,
    max_distance: float,
) -> List[Tuple[int, int, float]]:
    """对投影星和观测星执行带距离门控的全局一对一匹配。

    参数
    ----
    projected_xy : (N, 2)  投影点像素坐标
    observed_xy  : (M, 2)  观测点像素坐标
    max_distance : float   匹配距离上限（像素），超过此距离的配对不可匹配

    返回
    ----
    List[(projected_row, observed_col, distance)]  匹配对列表
    """
    projected = np.asarray(projected_xy, dtype=np.float64).reshape(-1, 2)
    observed = np.asarray(observed_xy, dtype=np.float64).reshape(-1, 2)
    if len(projected) == 0 or len(observed) == 0:
        return []
    distances = np.linalg.norm(
        projected[:, None, :] - observed[None, :, :], axis=2)
    cost = distances.copy()
    cost[cost > max_distance] = 1e9
    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(cost)
    else:
        pairs = sorted(
            (float(cost[row, col]), row, col)
            for row in range(cost.shape[0]) for col in range(cost.shape[1]))
        rows, cols, used_rows, used_cols = [], [], set(), set()
        for _, row, col in pairs:
            if row in used_rows or col in used_cols:
                continue
            rows.append(row)
            cols.append(col)
            used_rows.add(row)
            used_cols.add(col)
    return [
        (int(row), int(col), float(distances[row, col]))
        for row, col in zip(rows, cols)
        if distances[row, col] <= max_distance
    ]
