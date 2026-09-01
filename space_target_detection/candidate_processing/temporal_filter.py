"""
候选点时序过滤模块

提供候选点帧间时序一致性检查，分三路放行：
  1. 信任通道——候选靠近活跃轨迹预测位置直接放行
  2. 边界新生通道——候选靠近图像边界时暂时放行，供飞出再入重连形成短轨迹
  3. 时序通道——新候选必须在上一帧有邻近匹配才放行
"""

# Domain implementation: temporal candidate evidence.
import numpy as np

try:
    from configuration.values import (FRAME_W, FRAME_H, enable_exit_reconnect,
                        exit_reconnect_border_margin)
except Exception:
    FRAME_W, FRAME_H = 1024, 1024
    enable_exit_reconnect = True
    exit_reconnect_border_margin = 80.0


class CandidateTemporalFilter:
    """
    候选点时序过滤器

    三路放行：
    1. 信任通道——候选靠近活跃轨迹预测位置（已确认目标）直接放行。
    2. 边界新生通道——靠近边界的新点暂时放行，避免飞出后再入第一帧被过滤掉。
       注意：这里只是放行，是否真重连仍由 trajectory_manager 的短轨迹评分确认。
    3. 时序通道——其他新候选必须在上一帧有邻近匹配，用于过滤差分残影。
    """

    def __init__(self, match_radius=10.0, trusted_radius=15.0):
        self.match_radius = match_radius
        self.trusted_radius = trusted_radius
        self._prev_cands = []  # List[(x, y)]
        self._prev_provisional = []  # 上帧位移异常候选，用于两帧确认

    @staticmethod
    def _distance_to_border(x, y):
        return float(min(float(x), FRAME_W - 1 - float(x),
                         float(y), FRAME_H - 1 - float(y)))

    def _is_near_border(self, x, y):
        if not enable_exit_reconnect:
            return False
        return self._distance_to_border(x, y) <= exit_reconnect_border_margin

    def filter(self, candidates, M_prev_to_curr=None, trusted_positions=None,
               provisional_candidates=None, trusted_candidates=None):
        """
        参数
        ----
        candidates : List[(x, y, flux)]
        M_prev_to_curr : ndarray (2,3) or None
        trusted_positions : List[(x, y)] or None
            活跃轨迹的预测位置，附近的候选无条件放行。

        返回
        ----
        filtered : List[(x, y, flux)]
        """
        if trusted_positions is None:
            trusted_positions = []
        if provisional_candidates is None:
            provisional_candidates = []
        if trusted_candidates is None:
            trusted_candidates = []

        trusted = []
        edge_birth = []
        remaining = []
        trusted_coords = (np.array(trusted_positions, dtype=np.float32)
                          if trusted_positions else np.zeros((0, 2), dtype=np.float32))

        # ── 第一路：信任通道 ──
        # ── 第二路：边界新生通道 ──
        for cx, cy, cflux in candidates:
            if len(trusted_coords) > 0:
                dx_t = trusted_coords[:, 0] - cx
                dy_t = trusted_coords[:, 1] - cy
                if np.any(np.sqrt(dx_t * dx_t + dy_t * dy_t) <= self.trusted_radius):
                    trusted.append((cx, cy, cflux))
                    continue
            if self._is_near_border(cx, cy):
                edge_birth.append((cx, cy, cflux))
                continue
            remaining.append((cx, cy, cflux))

        # ── 第三路：时序通道（仅对非信任、非边界新生候选） ──
        temporal = []
        if remaining and self._prev_cands:
            prev_coords = np.array(self._prev_cands, dtype=np.float32)
            if M_prev_to_curr is None:
                projected = prev_coords
            else:
                M = M_prev_to_curr
                projected = np.column_stack([
                    M[0, 0] * prev_coords[:, 0] + M[0, 1] * prev_coords[:, 1] + M[0, 2],
                    M[1, 0] * prev_coords[:, 0] + M[1, 1] * prev_coords[:, 1] + M[1, 2],
                ])
            used_prev = set()
            for cx, cy, cflux in remaining:
                dx = projected[:, 0] - cx
                dy = projected[:, 1] - cy
                dists = np.sqrt(dx * dx + dy * dy)
                min_idx = int(np.argmin(dists))
                min_dist = float(dists[min_idx])
                if min_dist <= self.match_radius and min_idx not in used_prev:
                    temporal.append((cx, cy, cflux))
                    used_prev.add(min_idx)
        elif not self._prev_cands:
            # 首帧或无历史时，剩余候选正常放行
            temporal = remaining

        filtered = trusted + edge_birth + temporal

        # 仅由帧间位移异常产生的候选不立即建轨：靠近边界、
        # 靠近已有轨迹，或与上帧位移异常候选连续时才放行。
        confirmed_provisional = []
        prev_provisional = (np.array(self._prev_provisional, dtype=np.float32)
                            if self._prev_provisional
                            else np.zeros((0, 2), dtype=np.float32))
        for cx, cy, cflux in provisional_candidates:
            confirmed = self._is_near_border(cx, cy)
            if not confirmed and len(trusted_coords) > 0:
                d = np.linalg.norm(trusted_coords - np.array([cx, cy]), axis=1)
                confirmed = bool(np.any(d <= self.trusted_radius))
            if not confirmed and len(prev_provisional) > 0:
                d = np.linalg.norm(prev_provisional - np.array([cx, cy]), axis=1)
                confirmed = bool(np.any(d <= self.match_radius))
            if confirmed:
                confirmed_provisional.append((cx, cy, cflux))

        # 保存原始位移异常候选，而非仅保存已放行项，避免首帧候选
        # 因未确认而永远失去第二帧确认机会。
        self._prev_provisional = [(x, y) for x, y, _ in provisional_candidates]

        # 可靠轨迹命中的候选无条件放行；最后按 0.1 px 坐标去重。
        result = []
        seen = set()
        for cand in filtered + confirmed_provisional + list(trusted_candidates):
            key = (round(cand[0], 1), round(cand[1], 1))
            if key not in seen:
                result.append(cand)
                seen.add(key)
        # 下帧连续性应基于本帧所有最终放行点，包括高置信评分目标、
        # 已确认 provisional 和可信轨迹直通点。
        self._prev_cands = [(x, y) for x, y, _ in result]
        return result
