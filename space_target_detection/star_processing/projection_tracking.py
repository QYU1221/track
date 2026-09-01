"""基于姿态先验的两阶段星表投影跟踪。

第一阶段用上一帧姿态建立初始星对并解算当前姿态；第二阶段用当前
姿态重新投影完整星表，以更小的门限重新匹配，从而补充新入视场或
第一阶段漏配的恒星。统一质量判定由主流程负责。
"""
import time

import numpy as np

from utils.point_matching import match_projected_points
from .projection_geometry import image_point_to_body_vector
from .attitude_solvers import solve_star_attitude


def _match_points(points, visible, match_radius_px):
    """投影点与观测点一对一匹配（匈牙利优先，自动回退贪心）。"""
    projected_xy = np.asarray([
        [entry['projected_x'], entry['projected_y']] for entry in visible
    ], dtype=np.float64)
    observed_xy = np.asarray([point[:2] for point in points], dtype=np.float64)
    raw = match_projected_points(projected_xy, observed_xy, match_radius_px)
    return [(dist, row, col) for row, col, dist in raw]


def _visible_catalog(projected_catalog, frame_width, frame_height, margin):
    """保留图像内及边缘缓冲区内的投影星。"""
    return [
        entry for entry in projected_catalog
        if -margin <= entry['projected_x'] < frame_width + margin
        and -margin <= entry['projected_y'] < frame_height + margin
    ]


def _solve_from_matches(points, visible, matches, frame_width, frame_height,
                        focal_length_px, principal_point, attitude_method,
                        match_radius_px):
    """由已建立的星对构造方向向量并解算姿态。"""
    body_vectors = np.asarray([
        image_point_to_body_vector(
            points[col][0], points[col][1], frame_width, frame_height,
            focal_length=focal_length_px, principal_point=principal_point)
        for _, _, col in matches
    ], dtype=np.float64)
    ref_vectors = np.asarray([
        visible[row]['inertial_vector'] for _, row, _ in matches
    ], dtype=np.float64)
    try:
        attitude_q, attitude_R = solve_star_attitude(
            body_vectors, ref_vectors, method=attitude_method)
    except Exception:
        return None

    result = {
        'identified': {
            col: visible[row]['catalog_id'] for _, row, col in matches
        },
        'body_vectors': body_vectors,
        'ref_vectors': ref_vectors,
        'magnitudes': np.zeros(len(matches), dtype=np.float64),
        'n_votes': len(matches),
        'n_identified': len(matches),
    }
    recognized_stars = [{
        'x': float(points[col][0]),
        'y': float(points[col][1]),
        'flux': float(points[col][2]),
        'catalog_id': visible[row]['catalog_id'],
        'confidence': float(np.exp(-residual / max(match_radius_px, 1e-6))),
    } for residual, row, col in matches]
    return {
        'star_id_result': result,
        'attitude_q': attitude_q,
        'attitude_R': attitude_R,
        'recognized_stars': recognized_stars,
        'match_count': len(matches),
    }


def track_stars_by_attitude_projection(
        points, projected_catalog, frame_width, frame_height,
        focal_length_px, principal_point, match_radius_px,
        min_matches, attitude_method, reproject_catalog=None,
        refinement_enabled=True, refinement_match_radius_px=3.0):
    """执行上一帧姿态初配和当前帧姿态重投影细化。

    ``reproject_catalog`` 接收第一阶段解算的姿态矩阵，并返回完整星表
    在该姿态下的投影。返回值包含 ``initial`` 和可选的 ``refined``，
    由主流程使用统一质量指标决定最终采用哪一个。
    """
    start = time.time()

    # 边缘缓冲区允许上一帧预测在视场外、当前帧刚进入视场的恒星初配。
    initial_visible = _visible_catalog(
        projected_catalog, frame_width, frame_height, match_radius_px)
    initial_matches = _match_points(
        points, initial_visible, match_radius_px)
    if len(initial_matches) < min_matches:
        return None
    initial = _solve_from_matches(
        points, initial_visible, initial_matches, frame_width, frame_height,
        focal_length_px, principal_point, attitude_method, match_radius_px)
    if initial is None:
        return None

    refined = None
    if refinement_enabled and reproject_catalog is not None:
        current_projection = reproject_catalog(initial['attitude_R'])
        refined_visible = _visible_catalog(
            current_projection, frame_width, frame_height, 0.0)
        refined_matches = _match_points(
            points, refined_visible, refinement_match_radius_px)
        if len(refined_matches) >= min_matches:
            refined = _solve_from_matches(
                points, refined_visible, refined_matches,
                frame_width, frame_height, focal_length_px, principal_point,
                attitude_method, refinement_match_radius_px)

    return {
        'initial': initial,
        'refined': refined,
        'time_ms': (time.time() - start) * 1000.0,
    }
