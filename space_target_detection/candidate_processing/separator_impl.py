"""恒星背景与空间目标候选的两阶段分离模块。

本模块位于检测 CSV 加载之后、轨迹关联之前，负责把逐帧点源检测分成
“用于星图识别的点”和“送入多目标跟踪器的未知目标候选”。处理分两阶段：

1. :meth:`CandidateSeparator.pre_separate` 在姿态未知时做保守粗分离：
   可信目标轨迹附近的点直接保护，帧间位移异常点暂列为目标，其余点仍送
   星图识别。这里宁可多保留恒星，也不让未确认轨迹污染星图识别输入。
2. :meth:`CandidateSeparator.refine` 在姿态求解后做精分离：利用投影星表建立
   背景模型，再结合姿态运动、轨迹类别证据和跨帧连续性生成最终目标候选。

候选点统一使用 ``(x, y, flux)`` 三元组；轨迹使用
``trajectory_manager`` 定义的字典结构。每条序列由独立的
:class:`CandidateSeparator` 实例维护状态。
"""

# Domain implementation: candidate processing.
import numpy as np
from configuration.values import (motion_separation_match_radius, max_gap_frames,
                    displacement_mad_multiplier,
                    pre_separation_min_track_hits,
                    pre_separation_min_track_confidence,
                    camera_focal_length_m, camera_pixel_size_m,
                    camera_cx, camera_cy, camera_projection_model,
                    motion_separation_consensus_threshold,
                    track_star_evidence_frames,
                    track_nonstar_evidence_frames,
                    track_nonstar_residual_threshold,
                    candidate_temporal_filter,
                    candidate_temporal_radius,
                    candidate_trusted_radius,
                    background_subtraction_match_radius,
                    background_subtraction_missed_radius,
                    enable_background_subtraction,
                    candidate_geometry_debug,
                    candidate_vector_motion_debug,
                    candidate_vector_motion_min_matches,
                    candidate_vector_motion_max_mad,
                    candidate_vector_motion_min_scale,
                    candidate_star_reject_threshold,
                    candidate_target_pass_threshold,
                    candidate_no_attitude_motion_threshold,
                    candidate_no_attitude_temporal_threshold,
                    candidate_catalog_quality_min_matches,
                    candidate_catalog_quality_min_ratio,
                    candidate_catalog_quality_max_median_residual,
                    triangle_star_evidence_frames,
                    triangle_debug,
                    candidate_debug)
from .temporal_filter import CandidateTemporalFilter


def _match_points_one_to_one(candidates, prev_pts, match_radius):
    """在距离门控内用全局距离升序贪心建立一对一帧间匹配。

    所有可行的“当前点—上一帧点”组合按欧氏距离从小到大排序，每次只接纳
    两端都尚未使用的组合。因此同一上一帧点不会被多个当前点重复认领。

    返回 ``(current_index, previous_index, distance)`` 列表。
    """
    if not candidates or not prev_pts:
        return []

    current_xy = np.asarray([cand[:2] for cand in candidates], dtype=np.float64)
    previous_xy = np.asarray(prev_pts, dtype=np.float64)
    pair_distances = np.linalg.norm(
        current_xy[:, None, :] - previous_xy[None, :, :], axis=2)

    feasible_pairs = []
    current_indices, previous_indices = np.where(
        pair_distances <= match_radius)
    for current_idx, previous_idx in zip(current_indices, previous_indices):
        feasible_pairs.append((
            float(pair_distances[current_idx, previous_idx]),
            int(current_idx),
            int(previous_idx),
        ))
    feasible_pairs.sort(key=lambda item: item[0])

    used_current = set()
    used_previous = set()
    matches = []
    for distance, current_idx, previous_idx in feasible_pairs:
        if current_idx in used_current or previous_idx in used_previous:
            continue
        used_current.add(current_idx)
        used_previous.add(previous_idx)
        matches.append((current_idx, previous_idx, distance))
    return matches


def _vector_motion_separate(candidates, prev_pts):
    """按相对二维背景运动的矢量残差筛选异常目标。

    先建立门控后的一对一帧间匹配，再用所有可靠匹配位移向量的分量中位数
    估计整体背景运动 ``[median(dx), median(dy)]``。每个匹配点的运动证据为
    它自身位移向量与背景运动向量之差的模长。

    异常阈值为 ``残差中位数 + displacement_mad_multiplier × 残差 MAD``。
    未匹配点、匹配数量不足或背景残差 MAD 过大时均采用保守策略：候选继续
    进入星图识别，不直接输出为异常运动目标。
    """
    matches = _match_points_one_to_one(
        candidates, prev_pts, motion_separation_match_radius)
    n_matches = len(matches)
    if n_matches < candidate_vector_motion_min_matches:
        if candidate_vector_motion_debug:
            print('[candidate-vector-motion] matches=%d insufficient' % n_matches)
        return list(candidates), []

    current_xy = np.asarray([cand[:2] for cand in candidates], dtype=np.float64)
    previous_xy = np.asarray(prev_pts, dtype=np.float64)
    motion_vectors = np.asarray([
        current_xy[current_idx] - previous_xy[previous_idx]
        for current_idx, previous_idx, _ in matches
    ], dtype=np.float64)

    background_motion = np.median(motion_vectors, axis=0)
    motion_residuals = np.linalg.norm(
        motion_vectors - background_motion, axis=1)
    residual_median = float(np.median(motion_residuals))
    residual_mad = float(np.median(np.abs(
        motion_residuals - residual_median)))

    if (not np.isfinite(residual_mad) or
            residual_mad > candidate_vector_motion_max_mad):
        if candidate_vector_motion_debug:
            print('[candidate-vector-motion] matches=%d bg=(%.3f,%.3f) '
                  'mad=%.3f unstable' % (
                      n_matches, background_motion[0], background_motion[1],
                      residual_mad))
        return list(candidates), []

    robust_scale = max(residual_mad, candidate_vector_motion_min_scale)
    threshold = (residual_median +
                 displacement_mad_multiplier * robust_scale)
    residual_by_current = {
        current_idx: float(residual)
        for (current_idx, _, _), residual in zip(matches, motion_residuals)
    }

    star_candidates = []
    motion_targets = []
    for current_idx, candidate in enumerate(candidates):
        residual = residual_by_current.get(current_idx)
        if residual is not None and residual > threshold:
            motion_targets.append(candidate)
        else:
            # 未匹配点属于运动证据不足，不在粗分离阶段直接判成目标。
            star_candidates.append(candidate)

    if candidate_vector_motion_debug:
        print('[candidate-vector-motion] matches=%d bg=(%.3f,%.3f) '
              'median=%.3f mad=%.3f threshold=%.3f anomalies=%d' % (
                  n_matches, background_motion[0], background_motion[1],
                  residual_median, residual_mad, threshold,
                  len(motion_targets)))

    return star_candidates, motion_targets


def _displacement_separate(candidates, prev_pts):
    """使用一对一匹配后的二维背景运动矢量残差做粗分离。"""
    return _vector_motion_separate(candidates, prev_pts)


def _candidate_key(candidate):
    """生成仅供单帧证据表使用的稳定候选坐标键。"""
    return round(float(candidate[0]), 3), round(float(candidate[1]), 3)


def _relative_motion_scores(candidates, prev_pts):
    """计算相对背景二维运动的连续目标证据，范围为 0 到 1。"""
    matches = _match_points_one_to_one(
        candidates, prev_pts, motion_separation_match_radius)
    if len(matches) < candidate_vector_motion_min_matches:
        return {}

    current_xy = np.asarray([cand[:2] for cand in candidates], dtype=np.float64)
    previous_xy = np.asarray(prev_pts, dtype=np.float64)
    vectors = np.asarray([
        current_xy[current_idx] - previous_xy[previous_idx]
        for current_idx, previous_idx, _ in matches
    ], dtype=np.float64)
    background = np.median(vectors, axis=0)
    residuals = np.linalg.norm(vectors - background, axis=1)
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    if not np.isfinite(mad) or mad > candidate_vector_motion_max_mad:
        return {}

    robust_scale = max(mad, candidate_vector_motion_min_scale)
    scale = max(median + displacement_mad_multiplier * robust_scale, 1e-6)
    return {
        _candidate_key(candidates[current_idx]): float(np.clip(
            residual / scale, 0.0, 1.0))
        for (current_idx, _, _), residual in zip(matches, residuals)
    }


def _get_trusted_track_positions(tracks):
    """决定哪些轨迹有资格认领检测点。
    从已有轨迹中选出可以直接认领当前检测点的目标位置。

    轨迹必须同时满足以下条件：

    1. 已经由 tentative 转为 active；
    2. 历史命中次数达到最低要求；
    3. 综合置信度达到最低要求；
    4. 多帧分类结果明确为 target。

    只有满足全部条件的轨迹才能在星图识别前保护附近检测点。这样可以避免
    恒星或噪声形成的临时轨迹提前认领检测点，造成后续星图识别缺少恒星。

    返回
    ----
    list[tuple[float, float]]
        可信目标轨迹经过滤波器预测后的当前帧 ``(x, y)`` 位置。
    """
    trusted_positions = []

    for track in tracks:
        # tentative、漏检等待和已结束轨迹都不具备提前认领检测点的资格。
        is_active = track.get('status') == 'active'
        if not is_active:
            continue

        # 命中次数过少时，轨迹可能只是由恒星或瞬时噪声误建立的短轨迹。
        hit_count = track.get('hit_count', 0)
        has_enough_hits = hit_count >= pre_separation_min_track_hits
        if not has_enough_hits:
            continue

        # 置信度综合反映关联、运动和光度稳定性，低质量轨迹不能走保护通道。
        confidence = track.get('confidence', 0.0)
        has_enough_confidence = (
            confidence >= pre_separation_min_track_confidence)
        if not has_enough_confidence:
            continue

        # 只有明确分类为目标的轨迹才被信任；unknown 和 star 继续参与恒星判断。
        object_class = track.get('object_class', 'unknown')
        is_confirmed_target = object_class == 'target'
        if not is_confirmed_target:
            continue

        # main.py 已在粗分离前执行滤波预测，因此这里读取的是当前帧预测位置。
        state = track['state']
        predicted_x = float(state[0])
        predicted_y = float(state[1])
        trusted_positions.append((predicted_x, predicted_y))

    return trusted_positions


def _split_by_trusted_tracks(candidates, trusted_positions, match_radius):
    """决定哪些检测点位于这些轨迹附近
    按可信轨迹邻域将检测点拆为“受保护目标”和“剩余点”。

    参数
    ----
    candidates : sequence[(float, float, float)]
        当前帧检测点。
    trusted_positions : sequence[(float, float)]
        高可信目标轨迹的当前预测位置。
    match_radius : float
        检测点到任一可信轨迹的最大匹配距离，单位为像素。

    返回
    ----
    tuple[list, list]
        ``(tracked_targets, remaining_candidates)``。
    """
    if not trusted_positions:
        return [], list(candidates)

    trusted = np.asarray(trusted_positions, dtype=np.float64)
    tracked_targets, remaining = [], []
    for cand in candidates:
        point = np.asarray(cand[:2], dtype=np.float64)
        nearest = float(np.min(np.linalg.norm(trusted - point, axis=1)))
        if nearest <= match_radius:
            tracked_targets.append(cand)
        else:
            remaining.append(cand)
    return tracked_targets, remaining


def _extract_projected_points(entries):
    """从投影星表或背景模型条目中提取有效的二维像面坐标。"""
    points = []
    for entry in entries or []:
        if (entry.get('projected_x') is None or
                entry.get('projected_y') is None):
            continue
        point = (float(entry['projected_x']), float(entry['projected_y']))
        if np.all(np.isfinite(point)):
            points.append(point)
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def _nearest_projected_distance(point, projected_points):
    """返回检测点到最近投影恒星的像素距离；无投影星时返回无穷大。"""
    if projected_points.size == 0:
        return float('inf')
    point = np.asarray(point, dtype=np.float64)
    return float(np.min(np.linalg.norm(projected_points - point, axis=1)))


def _pixel_to_camera_unit_vector(point, focal_length_px, principal_point):
    """将图像像素转换为 y-up 相机坐标系中的单位视线。"""
    x, y = float(point[0]), float(point[1])
    cx, cy = float(principal_point[0]), float(principal_point[1])
    if camera_projection_model == 'k4':
        camera_vector = np.array([(y - cy) / focal_length_px,
                                  -(x - cx) / focal_length_px, 1.0])
    else:
        camera_vector = np.array([(x - cx) / focal_length_px,
                                  -(y - cy) / focal_length_px, 1.0])
    return camera_vector / np.linalg.norm(camera_vector)


def _camera_unit_vector_to_pixel(camera_vector, focal_length_px,
                                 principal_point):
    """将 y-up 相机视线投影回 y-down 图像像素坐标。"""
    vector = np.asarray(camera_vector, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError("camera_vector must contain exactly three elements")
    if vector[2] <= 1e-12:
        return None

    cx, cy = float(principal_point[0]), float(principal_point[1])
    if camera_projection_model == 'k4':
        pixel_x = cx - focal_length_px * vector[1] / vector[2]
        pixel_y = cy + focal_length_px * vector[0] / vector[2]
    else:
        pixel_x = cx + focal_length_px * vector[0] / vector[2]
        pixel_y = cy - focal_length_px * vector[1] / vector[2]
    return float(pixel_x), float(pixel_y)


def _predict_point_from_attitude(point, relative_rotation, focal_length_px,
                                 principal_point):
    """预测上一帧固定天球点在当前帧的理论像素位置。

    项目姿态矩阵满足 ``inertial = R @ camera``。调用方传入
    ``R_current.T @ R_previous``，本函数统一执行像素到相机视线、相对旋转、
    相机视线到像素的完整转换。
    """
    previous_camera_vector = _pixel_to_camera_unit_vector(
        point, focal_length_px, principal_point)
    current_camera_vector = relative_rotation @ previous_camera_vector
    predicted = _camera_unit_vector_to_pixel(
        current_camera_vector, focal_length_px, principal_point)

    if candidate_geometry_debug:
        print('[candidate-geometry] prev=(%.3f, %.3f) pred=%s' % (
            float(point[0]), float(point[1]),
            'None' if predicted is None else '(%.3f, %.3f)' % predicted))

    return predicted


class CandidateSeparator:
    """统一管理一条图像序列中的候选点分离状态。

    每条独立序列应创建一个实例。实例同时保存：上一有效帧检测点、上一姿态
    有效帧检测点与旋转矩阵，以及可选的候选时序过滤器。实例之间状态隔离，
    可以连续或并行处理不同数据集而不互相污染。

    典型调用顺序为：每帧先调用 :meth:`pre_separate`，完成星图识别和姿态
    解算后再调用 :meth:`refine`。
    """

    def __init__(self):
        """按全局配置创建一个空的序列分离器。"""
        # 粗分离状态：最近一个有效检测帧的全部点及帧号。
        self._prev_points = None
        self._prev_valid_frame_idx = None

        # 姿态运动状态：最近一个姿态有效帧的检测点和惯性系到相机系旋转。
        self._prev_attitude_points = None
        self._prev_attitude_R = None
        # 仅保留上一帧目标倾向点，避免恒星/噪声为当前点制造虚假时序证据。
        self._previous_scoring_points = []
        self._last_motion_scores = {}
        self._candidate_evidence = []
        self._provisional_candidates = []
        self._catalog_diagnostics = {}

        # 时序过滤关闭时保留 None，后续仅做三路候选合并和去重。
        self._temporal_filter = (
            CandidateTemporalFilter(
                match_radius=candidate_temporal_radius,
                trusted_radius=candidate_trusted_radius,
            ) if candidate_temporal_filter else None
        )

    @property
    def last_valid_frame_idx(self):
        """最近一次完成粗分离的有效帧号；冷启动时为 ``None``。"""
        return self._prev_valid_frame_idx

    @property
    def candidate_evidence(self):
        """当前帧全部候选的三分类证据记录只读副本。"""
        return list(self._candidate_evidence)

    @property
    def provisional_candidates(self):
        """当前帧证据冲突或置信度不足的候选三元组。"""
        return list(self._provisional_candidates)

    @property
    def catalog_diagnostics(self):
        """当前帧投影星表质量门控统计。"""
        return dict(self._catalog_diagnostics)

    def reset(self):
        """清空全部帧间状态，使实例可以安全地处理一条新序列。"""
        self._prev_points = None
        self._prev_valid_frame_idx = None
        self._prev_attitude_points = None
        self._prev_attitude_R = None
        self._previous_scoring_points = []
        self._last_motion_scores = {}
        self._candidate_evidence = []
        self._provisional_candidates = []
        self._catalog_diagnostics = {}
        if self._temporal_filter is not None:
            self._temporal_filter = CandidateTemporalFilter(
                match_radius=candidate_temporal_radius,
                trusted_radius=candidate_trusted_radius,
            )

    def pre_separate(self, candidates, frame_idx, tracks):
        """在星图识别前保守地分离已知目标和异常运动点。

        参数
        ----
        candidates : sequence[(float, float, float)]
            当前帧全部检测点。
        frame_idx : int
            当前帧序号。
        tracks : sequence[dict]
            跟踪器维护的轨迹。只有 active、命中数和置信度达标、且已分类为
            target 的轨迹才可提前保护附近检测点。

        返回
        ----
        tuple[list, list, list]
            ``(star_candidates, tracked_targets, displacement_targets)``：分别为
            星图识别输入、可信轨迹命中点和位移异常候选。

        冷启动或有效帧间隔超过 ``max_gap_frames`` 时，无法可靠计算位移，
        因而把所有点送入星图识别并重新建立历史。
        """
        if not candidates:
            return [], [], []

        if (self._prev_valid_frame_idx is None or
                frame_idx - self._prev_valid_frame_idx > max_gap_frames):
            self._last_motion_scores = {}
            self._prev_points = [(c[0], c[1]) for c in candidates]
            self._prev_valid_frame_idx = frame_idx
            return list(candidates), [], []

        gap = frame_idx - self._prev_valid_frame_idx
        # 只保护高可信目标，避免恒星形成 tentative 伪轨迹后反向污染粗分离。
        tracked_positions = _get_trusted_track_positions(tracks)
        matched_targets, remaining = _split_by_trusted_tracks(
            candidates, tracked_positions, motion_separation_match_radius)

        # 位移统计只在严格相邻帧上计算；跨帧速度尺度不同，不直接套同一阈值。
        if gap == 1 and remaining and self._prev_points:
            self._last_motion_scores = _relative_motion_scores(
                remaining, self._prev_points)
            star_cands, new_targets = _displacement_separate(
                remaining, self._prev_points)
        else:
            self._last_motion_scores = {}
            star_cands, new_targets = list(remaining), []

        self._prev_points = [(c[0], c[1]) for c in candidates]
        self._prev_valid_frame_idx = frame_idx
        return star_cands, matched_targets, new_targets

    def _score_candidates(self, candidates, tracks, background, attitude_R,
                          attitude_ok, projected_catalog):
        """融合轻量连续证据，输出高置信目标、恒星和模糊候选。"""
        projected_points = _extract_projected_points(projected_catalog)
        attitude_matrix = (None if attitude_R is None else
                           np.asarray(attitude_R, dtype=np.float64))
        attitude_matrix_valid = (
            attitude_matrix is not None and
            attitude_matrix.shape == (3, 3) and
            np.all(np.isfinite(attitude_matrix)))
        catalog_available = bool(
            attitude_ok and attitude_matrix_valid and projected_points.size > 0)
        background_keys = self._background_keys(background)
        background_residuals = [
            float(entry['residual']) for entry in (background or [])
            if (entry.get('matched_x') is not None and
                entry.get('matched_y') is not None and
                entry.get('residual') is not None and
                np.isfinite(float(entry['residual'])))
        ]
        catalog_match_ratio = (
            len(background_residuals) / max(len(candidates), 1)
            if catalog_available else 0.0)
        catalog_median_residual = (
            float(np.median(background_residuals))
            if background_residuals else float('inf'))
        catalog_target_available = bool(
            catalog_available and
            len(background_residuals) >= candidate_catalog_quality_min_matches and
            catalog_match_ratio >= candidate_catalog_quality_min_ratio and
            catalog_median_residual <=
            candidate_catalog_quality_max_median_residual)
        self._catalog_diagnostics = {
            'available': catalog_available,
            'target_available': catalog_target_available,
            'matches': len(background_residuals),
            'match_ratio': catalog_match_ratio,
            'projected_stars': len(projected_points),
            'detections': len(candidates),
            'median_residual': catalog_median_residual,
        }
        track_points = np.asarray([
            track['state'][:2] for track in tracks
            if (track.get('status') in ('active', 'tentative', 'lost_in_fov') and
                track.get('object_class') == 'target')
        ], dtype=np.float64).reshape(-1, 2)
        temporal_points = np.asarray(
            self._previous_scoring_points, dtype=np.float64).reshape(-1, 2)

        relative_rotation = None
        previous_attitude_points = np.asarray(
            self._prev_attitude_points or [], dtype=np.float64).reshape(-1, 2)
        if (attitude_ok and attitude_matrix_valid and
                self._prev_attitude_R is not None):
            relative_rotation = attitude_matrix.T @ self._prev_attitude_R
        focal_length_px = camera_focal_length_m / camera_pixel_size_m
        pp = (camera_cx, camera_cy)

        # 先把上一帧所有点按姿态投影到当前帧，再做一对一匹配。
        # 原先的”先找原始近邻、再投影”在转动较大或密集星场中容易选错恒星。
        attitude_residual_by_current = {}
        if relative_rotation is not None and previous_attitude_points.size > 0:
            projected_previous = []
            for previous_point in previous_attitude_points:
                predicted = _predict_point_from_attitude(
                    previous_point, relative_rotation, focal_length_px, pp)
                if predicted is not None and np.all(np.isfinite(predicted)):
                    projected_previous.append(predicted)
            if projected_previous:
                attitude_matches = _match_points_one_to_one(
                    candidates, projected_previous,
                    motion_separation_match_radius)
                attitude_residual_by_current = {
                    current_idx: float(distance)
                    for current_idx, _, distance in attitude_matches
                }

        records = []
        high_stars = []
        high_targets = []
        provisional = []
        reason_counts = {}
        no_catalog_blocked = 0

        for current_idx, candidate in enumerate(candidates):
            point = np.asarray(candidate[:2], dtype=np.float64)
            key = _candidate_key(candidate)
            flags = []

            catalog_distance = float('inf')
            catalog_star_score = 0.0
            catalog_target_score = 0.0
            if catalog_available:
                catalog_distance = _nearest_projected_distance(
                    point, projected_points)
                catalog_star_score = (0.0 if not np.isfinite(catalog_distance)
                                      else float(np.exp(
                                          -catalog_distance /
                                          max(background_subtraction_match_radius,
                                              1e-6))))
                if key in background_keys:
                    catalog_star_score = 1.0
                    flags.append('catalog_matched')
                elif catalog_distance > background_subtraction_missed_radius:
                    flags.append('catalog_excluded')
                # 只有投影星表质量也通过门控时，“远离所有恒星”才支持目标。
                if catalog_target_available:
                    catalog_target_score = 1.0 - catalog_star_score
                else:
                    flags.append('catalog_quality_low')
            else:
                flags.append('catalog_unavailable')

            attitude_residual = None
            attitude_star_score = 0.0
            attitude_residual = attitude_residual_by_current.get(current_idx)
            if attitude_residual is not None:
                attitude_star_score = float(np.exp(
                    -attitude_residual /
                    max(motion_separation_consensus_threshold, 1e-6)))
                flags.append('attitude_consistent' if
                             attitude_star_score >= 0.5 else
                             'attitude_divergent')

            motion_score = float(self._last_motion_scores.get(key, 0.0))
            if motion_score >= 0.5:
                flags.append('relative_motion')

            track_distance = _nearest_projected_distance(point, track_points)
            track_score = (0.0 if not np.isfinite(track_distance)
                           else float(np.exp(
                               -track_distance /
                               max(candidate_trusted_radius, 1e-6))))
            if track_score >= 0.5:
                flags.append('track_near')

            temporal_distance = _nearest_projected_distance(
                point, temporal_points)
            temporal_score = (0.0 if not np.isfinite(temporal_distance)
                              else float(np.exp(
                                  -temporal_distance /
                                  max(candidate_temporal_radius, 1e-6))))
            if temporal_score >= 0.5:
                flags.append('temporal')

            raw_star = 0.1 + 1.4 * catalog_star_score + 0.8 * attitude_star_score
            raw_target = (0.1 + 1.2 * catalog_target_score +
                          0.9 * motion_score + 0.8 * track_score +
                          0.5 * temporal_score)
            raw_noise = (0.1 + 0.8 * (1.0 - temporal_score) *
                         (1.0 - track_score) *
                         (1.0 - catalog_star_score))
            logits = np.asarray([raw_star, raw_target, raw_noise], dtype=np.float64)
            probabilities = np.exp(logits - np.max(logits))
            probabilities /= np.sum(probabilities)
            p_star, p_target, p_noise = [float(value) for value in probabilities]

            conflict = (
                catalog_star_score >= 0.65 and
                (track_score >= 0.65 or motion_score >= 0.65))
            if conflict:
                flags.append('evidence_conflict')

            no_catalog_motion_confirmed = (
                motion_score >= candidate_no_attitude_motion_threshold and
                temporal_score >= candidate_no_attitude_temporal_threshold)

            if p_star >= candidate_star_reject_threshold and not conflict:
                priority = 'star'
                high_stars.append(candidate)
            elif (p_target >= candidate_target_pass_threshold and not conflict and
                  (catalog_target_available or no_catalog_motion_confirmed)):
                priority = 'target'
                high_targets.append(candidate)
            else:
                priority = 'provisional'
                provisional.append(candidate)
                if (not catalog_target_available and
                        p_target >= candidate_target_pass_threshold and
                        not no_catalog_motion_confirmed):
                    flags.append('no_catalog_conservative_block')
                    no_catalog_blocked += 1

            primary_reason = flags[0] if flags else 'weak_evidence'
            reason_counts[primary_reason] = reason_counts.get(primary_reason, 0) + 1
            records.append({
                'candidate': candidate,
                'catalog_available': catalog_available,
                'catalog_target_available': catalog_target_available,
                'catalog_match_ratio': catalog_match_ratio,
                'catalog_median_residual': catalog_median_residual,
                'catalog_distance': catalog_distance,
                'attitude_motion_residual': attitude_residual,
                'relative_background_motion_score': motion_score,
                'track_distance': track_distance,
                'temporal_score': temporal_score,
                'p_star': p_star,
                'p_target': p_target,
                'p_noise': p_noise,
                'source_flags': flags,
                'priority': priority,
            })

        self._candidate_evidence = records
        self._provisional_candidates = list(provisional)
        self._previous_scoring_points = [
            record['candidate'][:2] for record in records
            if (record['priority'] == 'target' or
                (record['priority'] == 'provisional' and
                 record['p_target'] >= record['p_star'] and
                 record['p_target'] >= record['p_noise']))
        ]

        if candidate_debug:
            reasons = ','.join(
                '%s:%d' % item for item in sorted(reason_counts.items()))
            print('[candidate-score] catalog_available=%s catalog_target=%s '
                  'matches=%d ratio=%.3f median=%.3f star=%d target=%d '
                  'provisional=%d no_catalog_blocked=%d reasons=%s' % (
                      catalog_available, catalog_target_available,
                      len(background_residuals), catalog_match_ratio,
                      catalog_median_residual, len(high_stars), len(high_targets),
                      len(provisional), no_catalog_blocked, reasons))

        return high_targets, high_stars, provisional

    def refine(self, candidates, tracks, attitude_R, attitude_ok,
               projected_catalog=None, recognized_stars=None,
               tracked_targets=None, displacement_targets=None,
               triangle_background=None):
        """利用姿态和多帧证据生成最终目标候选。

        参数
        ----
        candidates : sequence[(float, float, float)]
            当前帧全部检测点。
        tracks : sequence[dict]
            当前轨迹集合；本方法会原地更新其星/目标证据和 ``object_class``。
        attitude_R : ndarray, shape (3, 3) or None
            当前帧姿态旋转矩阵。
        attitude_ok : bool
            姿态是否可信。为假时不执行星表背景和姿态运动判断。
        projected_catalog : sequence[dict] or None
            按当前姿态投影到像面的可见星表，条目包含
            ``projected_x/projected_y/catalog_id``。
        recognized_stars : sequence[dict] or None
            当前帧已经由星图识别器确认的恒星。
        tracked_targets : sequence or None
            粗分离阶段命中可信目标轨迹的检测点。
        displacement_targets : sequence or None
            粗分离阶段得到的异常位移候选。
        triangle_background : dict or None
            相邻帧三角拓扑得到的背景仿射内点和质量；不可用时为 ``None``。

        返回
        ----
        target_cands : list
            去星、时序筛选和去重后的最终跟踪候选。
        background : list[dict]
            投影星表与实际检测的匹配背景模型。
        tracked_targets : list
            经背景恒星否决后的可信轨迹候选。
        n_motion_removed : int
            为兼容旧调用接口保留，当前固定为 0；姿态一致性已纳入统一评分。

        处理顺序很重要：先用星表建立背景并完成统一评分，再合并目标倾向
        暂定点和位移异常点，然后用三角背景复核三路候选，最后直通高置信目标
        并对统一暂定通道执行跨帧确认。
        """
        tracked_targets = list(tracked_targets or [])
        displacement_targets = list(displacement_targets or [])
        background = []
        target_cands = []

        if enable_background_subtraction and attitude_ok:
            from star_processing.descriptor_identifier import StarMapRecognizer
            background = StarMapRecognizer.build_background_model(
                projected_catalog or [], candidates, recognized_stars or [],
                match_radius=background_subtraction_match_radius)

        scored_targets, _, scored_provisional = self._score_candidates(
            candidates, tracks, background, attitude_R,
            attitude_ok, projected_catalog)
        target_cands = list(scored_targets)

        # 统一 provisional 通道：只将“目标概率占优”的评分暂定点
        # 与位移异常点合并，交给同一个两帧确认通道。
        evidence_by_key = {
            _candidate_key(record['candidate']): record
            for record in self._candidate_evidence
        }
        target_leaning_provisional = [
            candidate for candidate in scored_provisional
            if (evidence_by_key[_candidate_key(candidate)]['p_target'] >=
                evidence_by_key[_candidate_key(candidate)]['p_star'] and
                evidence_by_key[_candidate_key(candidate)]['p_target'] >=
                evidence_by_key[_candidate_key(candidate)]['p_noise'])
        ]
        provisional_by_key = {
            _candidate_key(candidate): candidate
            for candidate in (list(displacement_targets) +
                              target_leaning_provisional)
        }
        displacement_targets = list(provisional_by_key.values())

        if attitude_ok:
            self._update_track_object_evidence(
                tracks, candidates, background, attitude_R)
            tracked_targets = self._veto_background_stars_from_tracked(
                tracked_targets, tracks, background)

            # 位移异常只是运动证据。若当前点已经被星表背景明确识别为恒星，
            # 禁止它通过位移异常通道绕回跟踪器。
            background_keys = self._background_keys(background)
            displacement_targets = [
                candidate for candidate in displacement_targets
                if _candidate_key(candidate) not in background_keys
            ]

        # 姿态一致性已在 _score_candidates() 中统一计分，不再用另一套
        # 近邻逻辑对 scored_targets 二次硬删除。这里只更新下帧姿态历史。
        n_motion_removed = 0
        if attitude_ok and attitude_R is not None:
            self._prev_attitude_points = [(c[0], c[1]) for c in candidates]
            self._prev_attitude_R = np.asarray(
                attitude_R, dtype=np.float64).copy()
        else:
            self._prev_attitude_points = None
            self._prev_attitude_R = None

        target_cands, displacement_targets, tracked_targets = \
            self._triangle_background_filter(
                target_cands, displacement_targets, tracked_targets,
                tracks, triangle_background)

        target_cands = self._apply_temporal_filter(
            target_cands, tracks, displacement_targets, tracked_targets)
        return target_cands, background, tracked_targets, n_motion_removed

    def _triangle_background_filter(self, target_cands,
                                    displacement_targets, tracked_targets,
                                    tracks, triangle_background):
        """在姿态反馈处理之后，用三角仿射背景内点二次剔除恒星。

        三角模型不可用或候选未命中背景内点时保持原结果。命中背景内点的
        普通候选被剔除；靠近已确认目标轨迹的候选继续保护；同时具有强相对
        背景运动证据的冲突点只进入 ``provisional_candidates``，不送跟踪器。
        """
        if not triangle_background or not triangle_background.get('available'):
            return (list(target_cands), list(displacement_targets),
                    list(tracked_targets))

        matched_points = np.asarray(
            triangle_background.get('matched_points') or [],
            dtype=np.float64).reshape(-1, 2)
        if matched_points.size == 0:
            return (list(target_cands), list(displacement_targets),
                    list(tracked_targets))

        residual_by_point = triangle_background.get('residual_by_point', {})
        target_positions = np.asarray([
            track['state'][:2] for track in tracks
            if (track.get('status') in ('active', 'tentative', 'lost_in_fov') and
                track.get('object_class') == 'target')
        ], dtype=np.float64).reshape(-1, 2)
        evidence_by_key = {
            _candidate_key(record['candidate']): record
            for record in self._candidate_evidence
        }
        provisional = list(self._provisional_candidates)
        provisional_keys = {_candidate_key(candidate) for candidate in provisional}
        triangle_conflict_candidates = []
        triangle_conflict_keys = set()
        counts = {'rejected': 0, 'conflict': 0, 'protected': 0}
        eligible_tracks = [
            track for track in tracks
            if track.get('status') in ('active', 'tentative', 'lost_in_fov')
        ]
        triangle_evidence_track_ids = set()

        def update_nearby_track(point, is_star_evidence):
            if not eligible_tracks:
                return
            distances = np.asarray([
                np.linalg.norm(np.asarray(track['state'][:2]) - point)
                for track in eligible_tracks
            ], dtype=np.float64)
            nearest_idx = int(np.argmin(distances))
            if float(distances[nearest_idx]) > candidate_trusted_radius:
                return
            track = eligible_tracks[nearest_idx]
            track_id = track.get('id')
            if track_id in triangle_evidence_track_ids:
                return
            triangle_evidence_track_ids.add(track_id)
            track.setdefault('triangle_star_evidence_count', 0)
            if is_star_evidence:
                track['triangle_star_evidence_count'] += 1
                if (track['triangle_star_evidence_count'] >=
                        triangle_star_evidence_frames):
                    track['object_class'] = 'star'
            else:
                track['triangle_star_evidence_count'] = max(
                    0, track['triangle_star_evidence_count'] - 1)

        def filter_channel(channel):
            kept = []
            for candidate in channel:
                point = np.asarray(candidate[:2], dtype=np.float64)
                distances = np.linalg.norm(matched_points - point, axis=1)
                match_idx = int(np.argmin(distances))
                # 内点坐标来自当前检测本身；小容差只吸收 float32 转换误差。
                if float(distances[match_idx]) > 0.75:
                    kept.append(candidate)
                    continue

                key = _candidate_key(candidate)
                triangle_key = _candidate_key(matched_points[match_idx])
                residual = residual_by_point.get(triangle_key)
                record = evidence_by_key.get(key)
                if record is not None:
                    record['triangle_background_available'] = True
                    record['triangle_background_residual'] = residual

                protected = False
                if target_positions.size > 0:
                    protected = bool(np.min(np.linalg.norm(
                        target_positions - point, axis=1)) <= candidate_trusted_radius)
                # 高质量星表已明确说明该位置没有恒星时，通用三角
                # 背景不能反向删除已通过统一评分的高置信目标。
                catalog_excluded_target = bool(
                    record is not None and
                    record.get('priority') == 'target' and
                    record.get('catalog_target_available') and
                    record.get('catalog_distance', float('inf')) >
                    background_subtraction_missed_radius)
                protected = protected or catalog_excluded_target
                motion_score = float(self._last_motion_scores.get(key, 0.0))

                if protected:
                    kept.append(candidate)
                    counts['protected'] += 1
                    if record is not None:
                        record['source_flags'].append('triangle_target_protected')
                    continue

                if motion_score >= candidate_no_attitude_motion_threshold:
                    update_nearby_track(point, is_star_evidence=False)
                    if key not in provisional_keys:
                        provisional.append(candidate)
                        provisional_keys.add(key)
                    if key not in triangle_conflict_keys:
                        triangle_conflict_candidates.append(candidate)
                        triangle_conflict_keys.add(key)
                    counts['conflict'] += 1
                    if record is not None:
                        record['source_flags'].append('triangle_motion_conflict')
                        record['priority'] = 'provisional'
                    continue

                counts['rejected'] += 1
                update_nearby_track(point, is_star_evidence=True)
                if record is not None:
                    record['source_flags'].append('triangle_background_rejected')
                    record['priority'] = 'star'
            return kept

        filtered_targets = filter_channel(list(target_cands))
        filtered_displacement = filter_channel(list(displacement_targets))
        filtered_tracked = filter_channel(list(tracked_targets))
        # 三角背景与强运动冲突的点不在当帧直通，但必须真正
        # 传给统一 provisional 通道，使其可在下一帧完成连续性确认。
        filtered_keys = {_candidate_key(candidate)
                         for candidate in filtered_displacement}
        filtered_displacement.extend(
            candidate for candidate in triangle_conflict_candidates
            if _candidate_key(candidate) not in filtered_keys)
        self._provisional_candidates = provisional
        triangle_background['rejected'] = counts['rejected']
        triangle_background['conflicts'] = counts['conflict']
        triangle_background['protected'] = counts['protected']

        if candidate_debug or triangle_debug:
            print('[triangle-filter] rejected=%d conflict=%d protected=%d' % (
                counts['rejected'], counts['conflict'], counts['protected']))
        return filtered_targets, filtered_displacement, filtered_tracked

    def _apply_temporal_filter(self, target_cands, tracks,
                               displacement_targets, tracked_targets):
        """合并三类目标证据并执行可选的跨帧连续性检查。

        ``target_cands`` 是星表背景排除后的未知点；``displacement_targets``
        是尚需连续帧确认的位移异常点；``tracked_targets`` 是可信轨迹直接命中
        的点，拥有最高放行优先级。时序过滤关闭时仅合并三路并按 0.1 像素
        坐标去重。

        返回最终候选列表，不修改输入列表。
        """
        if self._temporal_filter is not None:
            # target 类轨迹的预测位置作为信任区域；unknown/star 不享受保护。
            trusted = [
                (t['state'][0], t['state'][1]) for t in tracks
                if (t['status'] in ('active', 'tentative', 'lost_in_fov')
                    and t.get('object_class') == 'target')
            ]
            return self._temporal_filter.filter(
                [], M_prev_to_curr=None,
                trusted_positions=trusted,
                provisional_candidates=displacement_targets,
                trusted_candidates=(list(target_cands) +
                                    list(tracked_targets)))

        # 配置关闭时仍需统一去重，否则同一检测点可能由多条证据重复加入。
        result, seen = [], set()
        for cand in list(target_cands) + list(displacement_targets) + list(tracked_targets):
            key = (round(cand[0], 1), round(cand[1], 1))
            if key not in seen:
                result.append(cand)
                seen.add(key)
        return result

    def _update_track_object_evidence(self, tracks, candidates,
                                      background, attitude_R):
        """累积轨迹的恒星/目标证据并更新 ``object_class``。

        对每条可用轨迹寻找当前帧最近检测点，证据优先级为：

        1. 检测点被背景模型明确匹配，记恒星证据；
        2. 检测点远离所有投影恒星，记非恒星证据；
        3. 位置模糊时，再比较姿态预测恒星位置与实际检测的残差。

        证据采用“命中一类、衰减另一类”的多帧累计方式，达到配置阈值后
        才输出 ``star`` 或 ``target``，否则保持 ``unknown``。轨迹字典会被
        原地修改，本方法无返回值。
        """
        if not tracks or not candidates:
            return
        cand_xy = np.asarray([c[:2] for c in candidates], dtype=np.float64)
        background_keys = self._background_keys(background)
        projected_xy = _extract_projected_points(background)
        can_predict = attitude_R is not None and self._prev_attitude_R is not None
        if can_predict:
            f_px = camera_focal_length_m / camera_pixel_size_m
            pp_search = (camera_cx, camera_cy)
            # 与姿态运动过滤使用完全相同的相对旋转和统一几何入口。
            R_rel = attitude_R.T @ self._prev_attitude_R

        for track in tracks:
            # 兼容尚未包含分类字段的旧轨迹数据。
            track.setdefault('star_evidence_count', 0)
            track.setdefault('nonstar_evidence_count', 0)
            track.setdefault('object_class', 'unknown')
            if track.get('status') not in ('active', 'tentative', 'lost_in_fov'):
                continue
            dists = np.linalg.norm(cand_xy - np.asarray(track['state'][:2]), axis=1)
            idx = int(np.argmin(dists))
            if dists[idx] > motion_separation_match_radius:
                # 附近无观测时不制造正负证据，由轨迹管理器单独处理漏检。
                continue
            cand = candidates[idx]
            is_star = (round(float(cand[0]), 3), round(float(cand[1]), 3)) in background_keys
            is_nonstar = False
            catalog_distance = _nearest_projected_distance(
                cand[:2], projected_xy)
            catalog_excluded = (
                projected_xy.size > 0 and
                catalog_distance > background_subtraction_missed_radius)
            if not is_star and catalog_excluded:
                # 多帧持续且远离所有投影恒星，是比运动相似度更强的目标证据。
                is_nonstar = True
            elif not is_star and can_predict and track.get('positions'):
                prev_x, prev_y = track['positions'][-1]
                predicted = _predict_point_from_attitude(
                    (prev_x, prev_y), R_rel, f_px, pp_search)
                if predicted is not None:
                    pred_x, pred_y = predicted
                    residual = float(np.hypot(cand[0] - pred_x, cand[1] - pred_y))
                    is_star = residual <= motion_separation_consensus_threshold
                    is_nonstar = residual >= track_nonstar_residual_threshold
            if is_star:
                # 相反证据逐步衰减，避免一次误匹配造成类别长期锁死。
                track['star_evidence_count'] += 1
                track['nonstar_evidence_count'] = max(0, track['nonstar_evidence_count'] - 1)
            elif is_nonstar:
                track['nonstar_evidence_count'] += 1
                track['star_evidence_count'] = max(0, track['star_evidence_count'] - 1)
            if (track['star_evidence_count'] >= track_star_evidence_frames or
                    track.get('triangle_star_evidence_count', 0) >=
                    triangle_star_evidence_frames):
                track['object_class'] = 'star'
            elif track['nonstar_evidence_count'] >= track_nonstar_evidence_frames:
                track['object_class'] = 'target'
            else:
                track['object_class'] = 'unknown'

    def _veto_background_stars_from_tracked(self, tracked, tracks, background):
        """阻止背景恒星经“可信轨迹保护通道”重新混入目标候选。

        只有未被背景模型匹配、且仍靠近一条已分类 target 轨迹的检测点才被
        保留。该二次核验用于切断“恒星误建轨后持续自我保护”的正反馈。
        """
        background_keys = self._background_keys(background)
        target_xy = np.asarray([
            t['state'][:2] for t in tracks if t.get('object_class') == 'target'
        ], dtype=np.float64)
        protected = []
        for cand in tracked:
            key = (round(float(cand[0]), 3), round(float(cand[1]), 3))
            if key in background_keys or target_xy.size == 0:
                continue
            if np.min(np.linalg.norm(target_xy - np.asarray(cand[:2]), axis=1)) <= motion_separation_match_radius:
                protected.append(cand)
        return protected

    @staticmethod
    def _background_keys(background):
        """提取背景模型中已匹配检测点的三位小数坐标键集合。"""
        return {
            (round(float(entry['matched_x']), 3),
             round(float(entry['matched_y']), 3))
            for entry in (background or [])
            if entry.get('matched_x') is not None and entry.get('matched_y') is not None
        }
