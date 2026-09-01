"""Behavior-compatible complete sequence implementation.

This module is the proven implementation behind the thin public entry point.

空间多目标在轨跟踪 —— 主流程（CSV 检测输入）

输入：检测后 CSV 文件（measurements/ 目录下的逐帧点源检测结果）
Stage 0：星图识别与姿态解算（Delaunay 三角 → 星对角距匹配 → 姿态求解）
Stage 0.5：姿态反馈恒星背景建模 → 剔除星表匹配点 → 提取未知目标候选
Stage 1：多目标轨迹跟踪（EKF/IMM 预测 + 三特征融合关联 + Sinkhorn/JPDA 轨迹管理）

不再包含星点提取、图像预处理等底层检测环节。
"""

import os, sys, time, csv
import numpy as np

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_DIR)

from configuration.values import (FRAME_W, FRAME_H, prune_len,
                    enable_star_id, enable_trajectory_association, enable_attitude,
                    star_id_method,
                    camera_focal_length_m, camera_pixel_size_m, camera_cx, camera_cy,
                    star_catalog_file, star_catalog_mag_limit,
                    star_id_angle_tolerance, star_id_vote_threshold, star_id_min_matched,
                    star_id_use_triangle_edges,
                    star_id_delaunay_cache_enabled,
                    star_id_delaunay_cache_rebuild,
                    star_id_delaunay_max_triangle_candidates,
                    star_id_delaunay_match_pixel_threshold,
                    star_id_delaunay_min_hypotheses,
                    star_id_adaptive_sparse_max_points,
                    star_id_adaptive_dense_min_points,
                    star_id_adaptive_min_matches,
                    star_id_adaptive_max_median_residual,
                    star_id_adaptive_max_residual,
                    star_id_adaptive_min_coverage,
                    star_id_adaptive_fail_frames,
                    star_id_adaptive_fallback_enabled,
                    star_id_adaptive_fallback_cooldown,
                    star_id_adaptive_delaunay_max_triangles,
                    star_id_adaptive_prefer_previous,
                    star_id_adaptive_debug,
                    star_id_projection_refinement_enabled,
                    star_id_projection_refinement_radius,
                    star_id_descriptor_k, star_id_descriptor_top_k,
                    star_id_descriptor_ransac_iters, star_id_descriptor_residual,
                    star_id_descriptor_min_conf,
                    star_id_descriptor_catalog_neighbors,
                    star_id_descriptor_min_support,
                    star_id_descriptor_trimmed_matches,
                    star_id_descriptor_radial_tolerance,
                    star_id_descriptor_pair_tolerance,
                    star_id_descriptor_max_observations,
                    star_id_descriptor_debug,
                    star_id_descriptor_max_top_k,
                    star_id_descriptor_candidate_score_margin,
                    star_id_descriptor_min_pair_support_ratio,
                    star_id_descriptor_triple_min_separation,
                    star_id_descriptor_triple_min_shape,
                    star_id_descriptor_sampling_grid,
                    star_id_descriptor_max_per_cell,
                    attitude_method, attitude_use_mag_weight,
                    max_gap_frames, triangle_background_enabled,
                    triangle_target_exclusion_radius, triangle_debug)
from configuration.camera import external_quaternion_to_wxyz
from data_io import DetectionLoader
from candidate_processing.separation import CandidateSeparator
from target_tracking.lifecycle import get_active_tracks
from visualization import (
    draw_measurements_and_tracks,
    draw_star_recognition_result,
    draw_tracks_on_frame,
    save_image,
)
from star_processing.attitude import (solve_star_attitude,
                                      track_stars_by_attitude_projection)
from candidate_processing.stage import CandidateProcessingStage
from star_processing import StarProcessingRuntime, StarProcessingStage
from target_tracking.stage import FrameTrackingStage, TrackPredictionStage
from track_reconnection import TrackReconnectionStage
from pipeline import save_stage_snapshot
from reporting.core import (setup_output_dirs as create_run_directory,
                            save_reconnect_events_csv,
                            validate_track_sequences)
from reporting.artifacts import (
    print_attitude_evaluation, print_evaluation_summary, print_mota_motp,
    print_star_removal_summary, save_attitude, save_evaluation,
    save_frame_statistics, save_track_events, save_track_overlay, save_tracks,
)

# 延迟导入星图识别与姿态解算模块（避免未启用时加载星表耗时）
_star_resources = StarProcessingRuntime()
_star_id_runtime = _star_resources.adaptive


def _scan_datasets(datasets_root):
    """
    扫描 datasets_root 下的所有数据集，返回有效数据集列表。

    支持两种目录结构：
      结构 A: datasets/dataset_name/measurements/*.csv
      结构 B: datasets/dataset_name/pure/frame_*.png（兼容旧格式）

    返回
    ----
    List[dict]  每项包含:
        name          : 数据集名称
        measurements_dir : measurements CSV 目录
        truth_dir     : truth 目录路径
        n_frames      : 检测到的帧数
        has_truth     : 是否有 ground truth
    """
    if not os.path.isdir(datasets_root):
        return []

    datasets = []
    for name in sorted(os.listdir(datasets_root)):
        dataset_path = os.path.join(datasets_root, name)
        if not os.path.isdir(dataset_path):
            continue

        measurements_dir = os.path.join(dataset_path, "measurements")
        pure_dir = os.path.join(dataset_path, "pure")

        n = 0
        input_dir = None

        if os.path.isdir(measurements_dir):
            csv_files = [f for f in os.listdir(measurements_dir)
                         if f.lower().endswith('.csv')]
            if csv_files:
                input_dir = measurements_dir
                n = len(csv_files)
        elif os.path.isdir(pure_dir) and os.path.exists(os.path.join(pure_dir, "frame_0000.png")):
            input_dir = pure_dir
            n = 0
            while os.path.exists(os.path.join(pure_dir, "frame_%04d.png" % n)):
                n += 1

        if input_dir is None or n == 0:
            continue

        truth_dir = os.path.join(dataset_path, "truth")
        has_truth = os.path.isdir(truth_dir)

        datasets.append({
            'name': name,
            'measurements_dir': input_dir,
            'truth_dir': truth_dir,
            'n_frames': n,
            'has_truth': has_truth,
        })

    return datasets


def _select_dataset_interactive(datasets):
    """交互式列表选择数据集，返回选中的 dataset dict。"""
    print("")
    print("=" * 60)
    print("  可用数据集列表")
    print("=" * 60)
    for idx, ds in enumerate(datasets):
        truth_tag = " [含truth]" if ds['has_truth'] else ""
        print("  [%d] %-30s  %d 帧%s" % (idx + 1, ds['name'], ds['n_frames'], truth_tag))
    print("=" * 60)

    while True:
        try:
            raw = input("请输入数据集编号 (1-%d)，或直接粘贴目录路径: " % len(datasets)).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已取消。")
            sys.exit(0)

        if os.sep in raw or raw.startswith(".") or raw.startswith("/"):
            if os.path.isdir(raw):
                return {'name': os.path.basename(raw.rstrip("/\\")),
                        'measurements_dir': raw, 'truth_dir': None,
                        'n_frames': -1, 'has_truth': False}
            print("  路径不存在，请重新输入。")
            continue

        if raw.isdigit():
            choice = int(raw)
            if 1 <= choice <= len(datasets):
                return datasets[choice - 1]
            print("  编号超出范围，请输入 1 到 %d。" % len(datasets))
        else:
            print("  无效输入，请输入编号或目录路径。")


def _init_shared_catalog():
    """加载一次星表基础数据，供 Delaunay 和描述子共用。"""
    if _star_resources.catalog_data is not None:
        return
    from utils.catalog_loader import load_star_catalog
    script_dir = PACKAGE_DIR
    catalog_path = os.path.join(script_dir, star_catalog_file)
    _star_resources.catalog_data = load_star_catalog(
        catalog_path, mag_limit=star_catalog_mag_limit)


def _init_star_id_delaunay():

    from star_processing.delaunay_identifier import StarIdentifier, DelaunayCatalogIndex

    _init_shared_catalog()

    script_dir = PACKAGE_DIR
    catalog_path = os.path.join(script_dir, star_catalog_file)

    _star_resources.delaunay_index = DelaunayCatalogIndex(
        _star_resources.catalog_data,
        cache_enabled=star_id_delaunay_cache_enabled,
        cache_rebuild=star_id_delaunay_cache_rebuild,
        catalog_path=catalog_path,
    )

    _star_resources.delaunay_identifier = StarIdentifier(
        _star_resources.delaunay_index, _star_resources.camera_matrix,
        angle_tolerance=star_id_angle_tolerance,
        vote_threshold=star_id_vote_threshold,
        max_triangle_candidates=star_id_delaunay_max_triangle_candidates,
        match_pixel_threshold=star_id_delaunay_match_pixel_threshold,
        frame_width=FRAME_W,
        frame_height=FRAME_H,
        attitude_method=attitude_method,
        min_hypotheses=star_id_delaunay_min_hypotheses,
    )


def _init_star_id_descriptor():

    from star_processing.descriptor_identifier import StarCatalog as DescCatalog, StarMapRecognizer

    _init_shared_catalog()

    f_px = camera_focal_length_m / camera_pixel_size_m
    cx, cy = (camera_cx, camera_cy)

    _star_resources.descriptor_catalog = DescCatalog.from_catalog_data(
        _star_resources.catalog_data,
        k_neighbors=star_id_descriptor_catalog_neighbors,
    )

    _star_resources.descriptor_recognizer = StarMapRecognizer(
        _star_resources.descriptor_catalog,
        k_neighbors=star_id_descriptor_k,
        top_k=star_id_descriptor_top_k,
        ransac_iters=star_id_descriptor_ransac_iters,
        residual_threshold=star_id_descriptor_residual,
        min_confidence=star_id_descriptor_min_conf,
        focal_length_px=f_px,
        principal_point=(cx, cy),
        min_support=star_id_descriptor_min_support,
        trimmed_matches=star_id_descriptor_trimmed_matches,
        radial_tolerance=star_id_descriptor_radial_tolerance,
        pair_tolerance=star_id_descriptor_pair_tolerance,
        max_observations=star_id_descriptor_max_observations,
        max_top_k=star_id_descriptor_max_top_k,
        candidate_score_margin=star_id_descriptor_candidate_score_margin,
        min_pair_support_ratio=star_id_descriptor_min_pair_support_ratio,
        triple_min_separation=star_id_descriptor_triple_min_separation,
        triple_min_shape=star_id_descriptor_triple_min_shape,
        sampling_grid=star_id_descriptor_sampling_grid,
        max_per_cell=star_id_descriptor_max_per_cell,
        attitude_method=attitude_method,
        debug=star_id_descriptor_debug,
    )

def _init_star_id(method=None):
    """按需初始化指定识别器；adaptive 模式下两套对象互不覆盖。"""
    method = method or star_id_method
    if method == "adaptive":
        method = "descriptor"
    if method == "descriptor" and _star_resources.descriptor_recognizer is not None:
        return
    if method == "delaunay" and _star_resources.delaunay_identifier is not None:
        return

    if method not in ("descriptor", "delaunay"):
        raise ValueError("星图识别方法必须是 'descriptor' 或 'delaunay'")

    if _star_resources.camera_matrix is None:
        f_px = camera_focal_length_m / camera_pixel_size_m
        cx, cy = (camera_cx, camera_cy)
        _star_resources.camera_matrix = np.array([
            [f_px, 0, cx],
            [0, f_px, cy],
            [0, 0, 1],
        ], dtype=np.float64)

    if method == "descriptor":
        _init_star_id_descriptor()
    else:
        _init_star_id_delaunay()


def _recognition_quality(result, attitude_R, points):
    """用两种识别器共有的视线重投影残差评价结果。"""
    metrics = {
        'valid': False, 'matches': 0, 'median_residual': float('inf'),
        'max_residual': float('inf'), 'coverage': 0.0, 'score': 0.0,
    }
    if result is None or attitude_R is None:
        return metrics
    body = np.asarray(result.get('body_vectors', []), dtype=np.float64).reshape(-1, 3)
    refs = np.asarray(result.get('ref_vectors', []), dtype=np.float64).reshape(-1, 3)
    count = min(len(body), len(refs))
    if count == 0:
        return metrics
    predicted = body[:count] @ np.asarray(attitude_R, dtype=np.float64).T
    cosines = np.sum(predicted * refs[:count], axis=1)
    f_px = camera_focal_length_m / camera_pixel_size_m
    residuals = np.arccos(np.clip(cosines, -1.0, 1.0)) * f_px
    median_residual = float(np.median(residuals))
    max_residual = float(np.max(residuals))

    identified_indices = [
        int(index) for index in result.get('identified', {}).keys()
        if 0 <= int(index) < len(points)
    ]
    coverage = 0.0
    if len(identified_indices) >= 2:
        xy = np.asarray([points[index][:2] for index in identified_indices], dtype=np.float64)
        width = float(np.max(xy[:, 0]) - np.min(xy[:, 0])) / max(FRAME_W, 1)
        height = float(np.max(xy[:, 1]) - np.min(xy[:, 1])) / max(FRAME_H, 1)
        coverage = float(np.sqrt(max(width * height, 0.0)))

    min_matches = max(star_id_min_matched, star_id_adaptive_min_matches)
    valid = bool(
        count >= min_matches and
        median_residual <= star_id_adaptive_max_median_residual and
        max_residual <= star_id_adaptive_max_residual and
        coverage >= star_id_adaptive_min_coverage)
    match_score = min(count / max(min_matches + 2.0, 1.0), 1.0)
    residual_score = float(np.exp(
        -median_residual / max(star_id_adaptive_max_median_residual, 1e-6)))
    score = 0.55 * match_score + 0.30 * residual_score + 0.15 * min(coverage, 1.0)
    metrics.update({
        'valid': valid, 'matches': count,
        'median_residual': median_residual, 'max_residual': max_residual,
        'coverage': coverage, 'score': float(score),
    })
    return metrics


def _identify_descriptor(points, meta):
    """描述子方法：局部描述子匹配 + RANSAC 几何一致性 + 姿态解算。

    返回 (star_id_result, attitude_q, attitude_R, attitude_ok)，与 delaunay 方法
    保持一致的输出格式，以便主循环无需区分方法。
    """
    from star_processing.projection_geometry import image_point_to_body_vector

    result = _star_resources.descriptor_recognizer.recognize_frame(points, meta)
    _star_resources.descriptor_result = result

    # 构建候选索引 → 坐标的快速查找
    cand_index = {}
    for idx, (cx, cy, _) in enumerate(points):
        key = (round(cx, 2), round(cy, 2))
        cand_index[key] = idx

    identified = {}
    body_vecs = []
    ref_vecs = []
    magnitudes = []

    f_px = camera_focal_length_m / camera_pixel_size_m
    pp = (camera_cx, camera_cy)

    for star in result.recognized_stars:
        sx, sy = star['x'], star['y']
        key = (round(sx, 2), round(sy, 2))
        idx = cand_index.get(key)
        if idx is None:
            # 精确匹配失败时做最近邻匹配
            best_i, best_d = None, float('inf')
            for i, (cx, cy, _) in enumerate(points):
                d = (cx - sx) ** 2 + (cy - sy) ** 2
                if d < best_d:
                    best_d, best_i = d, i
            if best_d < 4.0:
                idx = best_i

        if idx is None:
            continue

        inertial = star.get('catalog_inertial_vector')
        if inertial is None:
            inertial = star.get('inertial_vector')
        if inertial is None:
            continue

        body = image_point_to_body_vector(
            sx, sy, FRAME_W, FRAME_H,
            focal_length=f_px, principal_point=pp)

        identified[idx] = star.get('catalog_id', idx)
        body_vecs.append(body)
        ref_vecs.append(np.asarray(inertial, dtype=np.float64))
        magnitudes.append(star.get('magnitude', 0.0))

    star_id_result = {
        'identified': identified,
        'body_vectors': np.array(body_vecs) if body_vecs else np.zeros((0, 3)),
        'ref_vectors': np.array(ref_vecs) if ref_vecs else np.zeros((0, 3)),
        'magnitudes': np.array(magnitudes),
        'n_votes': len(body_vecs),
        'n_identified': len(body_vecs),
    }

    attitude_q = None
    attitude_R = None
    attitude_ok = False

    if enable_attitude and star_id_result['n_identified'] >= star_id_min_matched:
        try:
            attitude_q, attitude_R = solve_star_attitude(
                star_id_result['body_vectors'], star_id_result['ref_vectors'],
                method=attitude_method,
                magnitudes=star_id_result['magnitudes'],
                use_magnitude_weights=attitude_use_mag_weight)
            attitude_ok = True
        except Exception:
            attitude_q, attitude_R = None, None

    return star_id_result, attitude_q, attitude_R, attitude_ok


def _identify_delaunay(points, triangles):
    """执行 Delaunay 识别并转换为统一姿态输出。"""
    triangles_for_id = triangles if star_id_use_triangle_edges else None
    result = _star_resources.delaunay_identifier.identify(
        points, triangles=triangles_for_id)
    attitude_q = None
    attitude_R = None
    attitude_ok = False
    if enable_attitude and result['n_identified'] >= star_id_min_matched:
        try:
            attitude_q, attitude_R = solve_star_attitude(
                result['body_vectors'], result['ref_vectors'],
                method=attitude_method,
                magnitudes=result['magnitudes'],
                use_magnitude_weights=attitude_use_mag_weight)
            attitude_ok = True
        except Exception:
            attitude_q, attitude_R = None, None
    return result, attitude_q, attitude_R, attitude_ok


def _run_star_id_method(method, points, triangles, meta):
    start = time.time()
    _init_star_id(method)
    if method == 'descriptor':
        result = _identify_descriptor(points, meta)
    else:
        result = _identify_delaunay(points, triangles)
    star_result, attitude_q, attitude_R, attitude_ok = result
    quality = _recognition_quality(star_result, attitude_R, points)
    if not quality['valid']:
        attitude_ok = False
    return {
        'method': method,
        'star_id_result': star_result,
        'attitude_q': attitude_q,
        'attitude_R': attitude_R,
        'attitude_ok': bool(attitude_ok and quality['valid']),
        'quality': quality,
        'time_ms': (time.time() - start) * 1000.0,
        'recognized_stars': (_star_resources.descriptor_result.recognized_stars
                             if method == 'descriptor' and
                             _star_resources.descriptor_result is not None else None),
    }


def _get_projected_catalog(attitude_R, method=None):
    """根据姿态将星表恒星投影到当前图像平面。

    基于共享 CatalogData，不再区分 Delaunay / 描述子星表来源。
    method 参数保留用于兼容旧调用方，已不再用于星表选择。
    """
    from star_processing.projection_geometry import project_catalog_vectors

    f_px = camera_focal_length_m / camera_pixel_size_m
    pp = (camera_cx, camera_cy)

    return project_catalog_vectors(
        _star_resources.catalog_data.unit_vectors,
        _star_resources.catalog_data.catalog_ids,
        attitude_R,
        frame_w=FRAME_W, frame_h=FRAME_H,
        focal_length=f_px, principal_point=pp,
        magnitudes=_star_resources.catalog_data.magnitudes,
    )


def _try_projection_tracking(points):
    """用上一帧姿态投影星表并一对一匹配，作为稳定姿态下的快速通道。"""
    previous_R = _star_id_runtime.get('previous_attitude_R')
    method = _star_id_runtime.get('previous_method')
    if previous_R is None or method not in ('delaunay', 'descriptor'):
        return None
    projected = _get_projected_catalog(previous_R, method)
    f_px = camera_focal_length_m / camera_pixel_size_m
    pp = (camera_cx, camera_cy)
    tracked = track_stars_by_attitude_projection(
        points=points,
        projected_catalog=projected,
        frame_width=FRAME_W,
        frame_height=FRAME_H,
        focal_length_px=f_px,
        principal_point=pp,
        match_radius_px=star_id_adaptive_max_residual,
        min_matches=max(star_id_min_matched, star_id_adaptive_min_matches),
        attitude_method=attitude_method,
        reproject_catalog=lambda current_R: _get_projected_catalog(
            current_R, method),
        refinement_enabled=star_id_projection_refinement_enabled,
        refinement_match_radius_px=star_id_projection_refinement_radius,
    )
    if tracked is None:
        return None

    initial = tracked['initial']
    initial_quality = _recognition_quality(
        initial['star_id_result'], initial['attitude_R'], points)
    chosen = initial
    quality = initial_quality
    refinement_used = False
    refined = tracked.get('refined')
    if refined is not None:
        refined_quality = _recognition_quality(
            refined['star_id_result'], refined['attitude_R'], points)
        # 细化结果必须通过统一质量门限，且不能减少已识别恒星数量。
        if (refined_quality['valid'] and
                refined_quality['matches'] >= initial_quality['matches']):
            chosen = refined
            quality = refined_quality
            refinement_used = True
    return {
        'method': method, 'star_id_result': chosen['star_id_result'],
        'attitude_q': chosen['attitude_q'], 'attitude_R': chosen['attitude_R'],
        'attitude_ok': bool(quality['valid']), 'quality': quality,
        'time_ms': tracked['time_ms'],
        'recognized_stars': chosen['recognized_stars'], 'tracking_used': True,
        'refinement_used': refinement_used,
        'initial_matches': initial_quality['matches'],
        'refined_matches': (refined['match_count'] if refined else 0),
    }


def _choose_primary_star_id_method(n_points, n_triangles):
    previous = _star_id_runtime.get('previous_method')
    if n_points <= star_id_adaptive_sparse_max_points:
        return 'delaunay', 'sparse_points'
    if n_points >= star_id_adaptive_dense_min_points:
        return 'descriptor', 'dense_points'
    if (star_id_adaptive_prefer_previous and
            previous in ('delaunay', 'descriptor')):
        return previous, 'hysteresis_previous'
    if n_triangles <= star_id_adaptive_delaunay_max_triangles:
        return 'delaunay', 'middle_low_triangles'
    return 'descriptor', 'middle_high_triangles'


def _adaptive_star_identification(points, triangles, meta, frame_idx):
    """选择主识别器，并在质量失败、预算允许时执行一次备用回退。"""
    tracking_result = _try_projection_tracking(points)
    if tracking_result is not None and tracking_result['attitude_ok']:
        _star_id_runtime['previous_attitude_R'] = tracking_result['attitude_R']
        _star_id_runtime['consecutive_failures'] = 0
        diagnostics = {
            'selected_method': tracking_result['method'],
            'primary_method': 'projection_tracking', 'fallback_used': 0,
            'switch_reason': 'previous_attitude_tracking',
            'primary_time_ms': tracking_result['time_ms'],
            'fallback_time_ms': 0.0,
            'quality_score': tracking_result['quality']['score'],
            'quality_matches': tracking_result['quality']['matches'],
            'quality_median_residual': tracking_result['quality']['median_residual'],
            'quality_max_residual': tracking_result['quality']['max_residual'],
            'quality_coverage': tracking_result['quality']['coverage'],
            'consecutive_failures': 0,
        }
        return tracking_result, diagnostics

    primary, reason = _choose_primary_star_id_method(len(points), len(triangles))
    if (primary == 'delaunay' and
            len(triangles) > star_id_adaptive_delaunay_max_triangles):
        primary, reason = 'descriptor', 'delaunay_triangle_budget'

    primary_result = _run_star_id_method(primary, points, triangles, meta)
    selected = primary_result
    fallback_result = None
    fallback_used = False

    cooldown_ok = (
        int(frame_idx) - int(_star_id_runtime.get('last_fallback_frame', -10 ** 9)) >=
        star_id_adaptive_fallback_cooldown)
    alternate = 'descriptor' if primary == 'delaunay' else 'delaunay'
    alternate_feasible = not (
        alternate == 'delaunay' and
        len(triangles) > star_id_adaptive_delaunay_max_triangles)
    should_fallback = bool(
        star_id_adaptive_fallback_enabled and cooldown_ok and
        alternate_feasible and not primary_result['attitude_ok'] and
        (_star_id_runtime.get('consecutive_failures', 0) + 1 >=
         star_id_adaptive_fail_frames))
    if should_fallback:
        fallback_used = True
        _star_id_runtime['last_fallback_frame'] = int(frame_idx)
        fallback_result = _run_star_id_method(alternate, points, triangles, meta)
        if (fallback_result['attitude_ok'] and
                (not primary_result['attitude_ok'] or
                 fallback_result['quality']['score'] >
                 primary_result['quality']['score'])):
            selected = fallback_result

    if selected['attitude_ok']:
        _star_id_runtime['previous_method'] = selected['method']
        _star_id_runtime['previous_attitude_R'] = selected['attitude_R']
        _star_id_runtime['consecutive_failures'] = 0
    else:
        _star_id_runtime['consecutive_failures'] += 1

    diagnostics = {
        'selected_method': selected['method'],
        'primary_method': primary,
        'fallback_used': int(fallback_used),
        'switch_reason': reason,
        'primary_time_ms': primary_result['time_ms'],
        'fallback_time_ms': fallback_result['time_ms'] if fallback_result else 0.0,
        'quality_score': selected['quality']['score'],
        'quality_matches': selected['quality']['matches'],
        'quality_median_residual': selected['quality']['median_residual'],
        'quality_max_residual': selected['quality']['max_residual'],
        'quality_coverage': selected['quality']['coverage'],
        'consecutive_failures': _star_id_runtime['consecutive_failures'],
    }
    if star_id_adaptive_debug:
        print('[star-id-adaptive] frame=%d primary=%s selected=%s fallback=%s '
              'reason=%s matches=%d median=%.3f coverage=%.3f valid=%s' % (
                  frame_idx, primary, selected['method'], fallback_used, reason,
                  selected['quality']['matches'],
                  selected['quality']['median_residual'],
                  selected['quality']['coverage'], selected['attitude_ok']))
    return selected, diagnostics


def setup_output_dirs(base_output, dataset_name=None):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if dataset_name:
        run_dir = os.path.join(base_output, "%s_%s" % (timestamp, dataset_name))
    else:
        run_dir = os.path.join(base_output, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _get_recognized_stars_for_background(star_id_result, star_points,
                                         method=None):
    """从星图识别结果构建 recognized_stars 列表，供 build_background_model 使用。

    descriptor 方法直接复用 RecognitionResult.recognized_stars；
    delaunay 方法从 identified 字典转换。
    """
    method = method or star_id_method
    if method == "descriptor" and _star_resources.descriptor_result is not None:
        return _star_resources.descriptor_result.recognized_stars

    recognized = []
    for obs_idx, cat_id in star_id_result.get('identified', {}).items():
        obs_idx = int(obs_idx)
        if obs_idx < len(star_points):
            cx, cy, flux = star_points[obs_idx]
            background_cat_id = int(cat_id)
            recognized.append({
                'x': cx,
                'y': cy,
                'flux': flux,
                'catalog_id': background_cat_id,
                'confidence': 0.8,
            })
    return recognized


def _save_reconnect_events_csv(run_dir, reconnect_events):
    """保存飞出视场后再入重连事件。即使没有事件也输出表头。"""
    path = os.path.join(run_dir, "reconnect_events.csv")
    fields = [
        'old_id', 'temp_id', 'exit_frame', 'entry_frame', 'reconnect_frame',
        'score', 'mode', 'exit_edge', 'entry_edge', 'gap',
        'outside_path_distance', 'required_speed', 'speed_limit',
        'new_measured_hits', 'new_measured_ratio', 'new_max_observation_gap'
    ]
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for ev in reconnect_events or []:
            row = dict(ev)
            if row.get('gap') is None:
                try:
                    row['gap'] = int(row.get('entry_frame')) - int(row.get('exit_frame'))
                except Exception:
                    row['gap'] = ''
            w.writerow(row)

def _validate_track_sequences(tracks):
    """检查轨迹序列一致性。只打印 warning，不中断程序。

    重点保证 positions / frames / point_sources 三个列表长度一致，
    并尽量修复 point_sources 缺失问题。frames 与 positions 长度不一致时，
    不在这里强行补帧，避免把飞出再入的非连续轨迹错误补成连续轨迹。
    """
    valid_sources = {
        'measured', 'predicted', 'recovered', 'reconnected',
        'exit_predicted', 'extrapolated_exit'
    }
    for t in tracks:
        tid = t.get('id', 'unknown')
        positions = t.get('positions', []) or []
        frames = t.get('frames', None)
        sources = t.get('point_sources', None)

        if frames is None:
            print("Warning: track %s has no frames field" % tid)
            continue

        if len(frames) != len(positions):
            print("Warning: track %s length mismatch: positions=%d frames=%d" %
                  (tid, len(positions), len(frames)))

        if sources is None:
            t['point_sources'] = ['measured'] * len(positions)
            sources = t['point_sources']
        elif len(sources) < len(positions):
            print("Warning: track %s point_sources shorter than positions: %d < %d, auto filled" %
                  (tid, len(sources), len(positions)))
            sources.extend(['measured'] * (len(positions) - len(sources)))
        elif len(sources) > len(positions):
            print("Warning: track %s point_sources longer than positions: %d > %d, truncated" %
                  (tid, len(sources), len(positions)))
            del sources[len(positions):]

        usable = min(len(frames), len(positions))
        last_frame = None
        seen = set()
        for i in range(usable):
            fid = frames[i]
            if fid is None:
                continue
            try:
                fid_int = int(fid)
            except (TypeError, ValueError):
                print("Warning: track %s has invalid frame value at index %d: %r" %
                      (tid, i, fid))
                continue
            if last_frame is not None and fid_int < last_frame:
                print("Warning: track %s frames are not non-decreasing near index %d: %s -> %s" %
                      (tid, i, last_frame, fid_int))
            if fid_int in seen:
                print("Warning: track %s has duplicate frame %s" % (tid, fid_int))
            seen.add(fid_int)
            last_frame = fid_int

        for src in sources:
            if src not in valid_sources:
                print("Warning: track %s has unknown point source: %r" % (tid, src))


def _save_stats_csv(run_dir, stats_rows):
    path = os.path.join(run_dir, "per_frame_stats.csv")
    fields = ['frame', 'n_stars', 'n_tracked_targets_pre',
              'n_displacement_targets_pre', 'n_stars_predicted', 'n_triangles',
              'n_triangle_background_triangles', 'n_matched_pairs',
              'triangle_available', 'triangle_reason', 'triangle_model',
              'triangle_point_matches', 'triangle_inliers',
              'triangle_rejected', 'triangle_conflicts',
              'triangle_protected', 'triangle_median_residual',
              'catalog_available', 'catalog_target_available',
              'catalog_matches', 'catalog_match_ratio',
              'catalog_median_residual',
              'n_candidates', 'n_targets',
              'n_active_tracks', 'n_star_class_tracks',
              'n_target_class_tracks', 'n_unknown_class_tracks',
              'n_exit_memory', 'n_reconnect_events', 'match_ratio', 'inlier_ratio',
              'edge_preserve',
              'n_identified_stars',
              'star_id_selected_method', 'star_id_primary_method',
              'star_id_fallback_used', 'star_id_switch_reason',
              'star_id_primary_time_ms', 'star_id_fallback_time_ms',
              'star_id_quality_score', 'star_id_quality_matches',
              'star_id_quality_median_residual',
              'star_id_quality_max_residual', 'star_id_quality_coverage',
              'star_id_consecutive_failures',
              'attitude_ok', 'q_w', 'q_x', 'q_y', 'q_z',
              'time_ms']
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(stats_rows)


def _quat_to_euler(w, x, y, z):
    """四元数 (w, x, y, z) → 欧拉角 (roll, pitch, yaw) 度，ZYX 旋转顺序"""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.degrees(np.arctan2(sinr_cosp, cosr_cosp))

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = np.degrees(np.arcsin(sinp))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.degrees(np.arctan2(siny_cosp, cosy_cosp))

    return roll, pitch, yaw


def _save_attitude_csv(run_dir, stats_rows, input_dir=None):
    """保存姿态解算结果 CSV：四元数 + 欧拉角 + 误差"""
    path = os.path.join(run_dir, "attitude.csv")
    rows_with_att = [r for r in stats_rows if r.get('attitude_ok')]

    q_gt_all = None
    if input_dir:
        truth_dir = os.path.join(os.path.dirname(os.path.normpath(input_dir)), "truth")
        q_gt_path = os.path.join(truth_dir, "quaternion_history.npy")
        if os.path.exists(q_gt_path):
            q_gt_all = np.load(q_gt_path)

    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        header = ['frame', 'q_w', 'q_x', 'q_y', 'q_z',
                  'roll_deg', 'pitch_deg', 'yaw_deg',
                  'n_identified']
        if q_gt_all is not None:
            header.append('error_vs_gt_deg')
        w.writerow(header)

        for row in stats_rows:
            fid = row['frame']
            if not row.get('attitude_ok'):
                continue

            qw, qx, qy, qz = row['q_w'], row['q_x'], row['q_y'], row['q_z']
            # q and -q are the same attitude; serialize the canonical w >= 0
            # form to keep the sign stable between solver implementations.
            if qw < 0.0:
                qw, qx, qy, qz = -qw, -qx, -qy, -qz
            roll, pitch, yaw = _quat_to_euler(qw, qx, qy, qz)

            out = [fid,
                   "%.8f" % qw, "%.8f" % qx, "%.8f" % qy, "%.8f" % qz,
                   "%.4f" % roll, "%.4f" % pitch, "%.4f" % yaw,
                   row['n_identified_stars']]

            if q_gt_all is not None and fid < len(q_gt_all):
                # Input is (x, y, z, w); use its conjugate/inverse in internal
                # scalar-first (w, x, y, z) convention.
                q_gt = external_quaternion_to_wxyz(q_gt_all[fid])
                q_est = np.array([qw, qx, qy, qz])
                dot = np.abs(np.dot(q_est, q_gt))
                dot = min(dot, 1.0)
                err_deg = np.degrees(2.0 * np.arccos(dot))
                out.append("%.6f" % err_deg)

            w.writerow(out)

    if rows_with_att:
        frame_list = sorted(set(r['frame'] for r in rows_with_att))
        print("Attitude saved: %s  (%d frames)" % (path, len(frame_list)))


def _save_tracks_csv(run_dir, tracks):
    path = os.path.join(run_dir, "trajectories.csv")
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['track_id', 'frame', 'x', 'y', 'status', 'source'])
        for t in tracks:
            sf = t.get('start_frame', 0)
            frames = t.get('frames')
            use_frames = (frames is not None and len(frames) == len(t['positions']) and
                          all(f is not None for f in frames))
            sources = t.get('point_sources', [])
            for i, (x, y) in enumerate(t['positions']):
                frame_id = int(frames[i]) if use_frames else sf + i + 1
                source = sources[i] if i < len(sources) else 'measured'
                w.writerow([t['id'], frame_id, "%.2f" % x, "%.2f" % y,
                            t['status'], source])


def _save_track_events_csv(run_dir, tracks):
    path = os.path.join(run_dir, "track_events.csv")
    rows = []
    for t in tracks:
        for ev in t.get('event_hist', []):
            rows.append({
                'track_id': t.get('id'),
                'frame': ev.get('frame'),
                'event': ev.get('type'),
                'x': ev.get('x'),
                'y': ev.get('y'),
                'reason': ev.get('reason', ''),
                'score': ev.get('score', ''),
            })
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        fields = ['track_id', 'frame', 'event', 'x', 'y', 'reason', 'score']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _save_track_overlay(run_dir, tracks):
    """将所有轨迹绘制在黑色背景上，输出轨迹合成图。"""
    import numpy as np
    tracks = [t for t in tracks if len(t['positions']) >= prune_len]
    canvas = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    if tracks:
        img = draw_tracks_on_frame(canvas, tracks, show_ids=True)
    else:
        img = canvas
    path = os.path.join(run_dir, "tracks_overlay.png")
    save_image(img, path)
    print("Tracks overlay saved: " + path)


def _save_eval_csv(run_dir, star_rows, target_rows, star_removal_rows=None):
    """保存逐帧评估结果 CSV。"""
    if star_rows:
        path = os.path.join(run_dir, "eval_stars.csv")
        fields = ['frame', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1',
                  'position_error', 'n_gt', 'n_identified',
                  'n_id_correct', 'n_id_wrong', 'n_gt_unpresentable']
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(star_rows)

    if target_rows:
        path = os.path.join(run_dir, "eval_targets.csv")
        fields = ['frame', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1',
                  'position_error', 'n_gt', 'n_tracked']
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(target_rows)

    if star_removal_rows:
        path = os.path.join(run_dir, "eval_star_removal.csv")
        fields = ['frame', 'n_gt', 'n_detected_stars', 'n_gt_undetected',
                  'n_correctly_removed', 'n_leaked_stars',
                  'removal_recall', 'leakage_rate']
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(star_removal_rows)


def _print_star_removal_summary(rows):
    if not rows:
        return
    n_gt = sum(int(r['n_gt']) for r in rows)
    detected = sum(int(r['n_detected_stars']) for r in rows)
    undetected = sum(int(r['n_gt_undetected']) for r in rows)
    removed = sum(int(r['n_correctly_removed']) for r in rows)
    leaked = sum(int(r['n_leaked_stars']) for r in rows)
    print("")
    print("  [Star Removal]")
    print("    GT stars total       : %d" % n_gt)
    print("    Detected GT stars    : %d" % detected)
    print("    Frontend undetected  : %d" % undetected)
    print("    Correctly removed    : %d" % removed)
    print("    Leaked to targets    : %d" % leaked)
    print("    Removal recall       : %.4f" %
          (removed / max(detected, 1)))
    print("    Leakage rate         : %.4f" %
          (leaked / max(detected, 1)))


def _print_eval_summary(summary):
    """打印评估汇总。"""
    print("")
    print("=" * 60)
    print("  Evaluation Summary")
    print("=" * 60)

    s = summary.get('star', {})
    if s:
        print("  [Star Identification]")
        print("    Frames evaluated : %d" % s.get('n_frames_evaluated', 0))
        print("    GT stars total    : %d" % s.get('n_gt_total', 0))
        print("    GT not in input   : %d  (pre-separation loss)" % s.get('n_gt_unpresentable', 0))
        print("    TP / FP / FN      : %d / %d / %d" % (s['tp'], s['fp'], s['fn']))
        print("    Precision         : %.4f" % s['precision'])
        print("    Recall            : %.4f" % s['recall'])
        print("    F1                : %.4f" % s['f1'])
        print("    Position Error    : %.2f px" % s['position_error'])
        n_id_ok = s.get('n_id_correct', 0)
        n_id_bad = s.get('n_id_wrong', 0)
        n_gt = s.get('n_gt_total', 0)
        if n_gt > 0:
            print("    ID correct / wrong: %d / %d  (rate=%.1f%% of GT, pos-match=%.1f%%)" % (
                n_id_ok, n_id_bad,
                100.0 * n_id_ok / max(n_gt, 1),
                100.0 * n_id_ok / max(n_id_ok + n_id_bad, 1)))

    t = summary.get('target', {})
    if t:
        print("  [Target Tracking]")
        print("    Frames evaluated : %d" % t.get('n_frames_evaluated', 0))
        print("    GT targets total  : %d" % t.get('n_gt_total', 0))
        print("    TP / FP / FN      : %d / %d / %d" % (t['tp'], t['fp'], t['fn']))
        print("    Precision         : %.4f" % t['precision'])
        print("    Recall            : %.4f" % t['recall'])
        print("    F1                : %.4f" % t['f1'])
        print("    Position Error    : %.2f px" % t['position_error'])
    print("=" * 60)



def _print_mota_motp(input_dir, tracks, num_frames):
    from evaluation.metrics import calc_mota_motp
    from configuration.values import dist_thresh

    truth_dir = os.path.join(os.path.dirname(os.path.normpath(input_dir)), "truth")
    if not os.path.isdir(truth_dir):
        return

    gt_targets = []
    for fname in sorted(os.listdir(truth_dir)):
        if fname.startswith("target_") and fname.endswith("_trajectory.npy"):
            gt_targets.append(np.load(os.path.join(truth_dir, fname)))
    if not gt_targets:
        return

    gt = {}
    for fid in range(num_frames):
        gt[fid] = {}
        for t_idx, traj in enumerate(gt_targets):
            if fid < len(traj):
                tx, ty = traj[fid]
                if 0 <= tx < FRAME_W and 0 <= ty < FRAME_H:
                    gt[fid][t_idx] = (tx, ty)

    est = {}
    for t in tracks:
        sf = t.get('start_frame', 0)
        frames = t.get('frames')
        use_frames = (frames is not None and len(frames) == len(t['positions']) and
                      all(f is not None for f in frames))
        for i, (px, py) in enumerate(t['positions']):
            frame_id = int(frames[i]) if use_frames else sf + i
            est.setdefault(frame_id, {})[t['id']] = (px, py)

    MOTA, MOTP, stats = calc_mota_motp(est, gt, dist_threshold=dist_thresh)

    if stats['total_gt'] == 0:
        return

    print("MOTA: %.2f%%  MOTP: %.2f%%  (GT=%d  FN=%d  FP=%d  err=%.2fpx)" % (
        MOTA * 100, MOTP * 100, stats['total_gt'], stats['FN'], stats['FP'],
        stats['dist_sum'] / max(stats['N_matched'], 1)))


def _print_attitude_eval(input_dir, stats_rows):
    """对比姿态解算结果与 ground truth 四元数，输出角度误差统计"""
    truth_dir = os.path.join(os.path.dirname(os.path.normpath(input_dir)), "truth")
    if not os.path.isdir(truth_dir):
        return

    q_gt_path = os.path.join(truth_dir, "quaternion_history.npy")
    if not os.path.exists(q_gt_path):
        return

    q_gt_all = np.load(q_gt_path)

    errors_deg = []
    for row in stats_rows:
        fid = row['frame']
        if row['attitude_ok'] and fid < len(q_gt_all):
            q_est = np.array([row['q_w'], row['q_x'], row['q_y'], row['q_z']])
            # Input is (x, y, z, w); compare after conjugating and converting
            # to the internal scalar-first (w, x, y, z) convention.
            q_gt = external_quaternion_to_wxyz(q_gt_all[fid])
            dot = np.abs(np.dot(q_est, q_gt))
            dot = min(dot, 1.0)
            err_rad = 2.0 * np.arccos(dot)
            err_deg = np.degrees(err_rad)
            errors_deg.append(err_deg)

    if not errors_deg:
        return

    errors = np.array(errors_deg)
    print("Attitude Error vs GT: mean=%.4f°  median=%.4f°  max=%.4f°  (N=%d/%d frames)" % (
        np.mean(errors), np.median(errors), np.max(errors),
        len(errors), len(stats_rows)))
def process_sequence(measurements_dir, output_dir=None, end_frame=None,
                     verbose=True, dataset_name=None, snapshot_dir=None):
    separator = CandidateSeparator()
    candidate_stage = CandidateProcessingStage(separator)
    star_stage = StarProcessingStage(
        adaptive_identifier=_adaptive_star_identification,
        fixed_identifier=_run_star_id_method,
        catalog_projector=_get_projected_catalog,
        recognized_star_resolver=_get_recognized_stars_for_background,
        runtime_state=_star_id_runtime,
    )

    def record_stage_input(stage_name, frame_idx, payload):
        if snapshot_dir is None:
            return
        filename = 'frame_%04d_%s.pkl' % (frame_idx, stage_name)
        save_stage_snapshot(
            os.path.join(snapshot_dir, filename), stage_name, payload)

    loader = DetectionLoader(measurements_dir)
    n_frames = loader.n_frames
    if n_frames == 0:
        raise FileNotFoundError("No CSV files in " + measurements_dir)
    if end_frame is not None:
        n_frames = min(n_frames, end_frame)

    if output_dir is None:
        script_dir = PACKAGE_DIR
        output_dir = os.path.join(os.path.dirname(script_dir), "output")
    run_dir = create_run_directory(output_dir, dataset_name)
    star_frame_dir = os.path.join(run_dir, "star_recognition_frames")
    track_frame_dir = os.path.join(run_dir, "track_frames")
    os.makedirs(star_frame_dir, exist_ok=True)
    os.makedirs(track_frame_dir, exist_ok=True)

    if verbose:
        print("Input: %d frames (CSV)" % n_frames)
        print("Output: " + run_dir)

    # ── 检测 ground truth ─────────────────────────────────────────
    dataset_dir = os.path.dirname(os.path.normpath(measurements_dir))
    # 格式 A: 单文件 stars.csv / ground_truth/ground_truth.csv
    stars_gt_path = os.path.join(dataset_dir, "stars.csv")
    if not os.path.exists(stars_gt_path):
        # 格式 B: 目录 stars/
        stars_gt_path = os.path.join(dataset_dir, "stars")
    targets_gt_path = os.path.join(dataset_dir, "ground_truth", "ground_truth.csv")
    if not os.path.exists(targets_gt_path):
        targets_gt_path = os.path.join(dataset_dir, "ground_truth")
    has_stars_gt = os.path.exists(stars_gt_path)
    has_targets_gt = os.path.exists(targets_gt_path)

    if verbose and (has_stars_gt or has_targets_gt):
        tags = []
        if has_stars_gt:
            tags.append("stars")
        if has_targets_gt:
            tags.append("targets")
        print("Ground truth found: " + ", ".join(tags))

    tracks = []
    archived_tracks = []
    trajectory_memory = []
    pending_reconnect = {}
    reconnect_events = []
    stats_rows = []
    star_eval_rows = []
    star_removal_rows = []
    target_eval_rows = []
    previous_background_triangles = None

    for frame_idx in range(n_frames):
        t_start = time.time()

        candidates, meta = loader.load_frame(frame_idx)

        # ── IMM 预测（提前到粗分离之前，利用运动模型推演缺帧位置）────
        prediction_input = {
            'tracks': tracks,
            'frame_idx': frame_idx,
            'last_valid_frame_idx': separator.last_valid_frame_idx,
            'max_gap_frames': max_gap_frames,
            'enabled': enable_trajectory_association,
        }
        record_stage_input('prediction', frame_idx, prediction_input)
        TrackPredictionStage.run(
            **prediction_input
        )

        record_stage_input('candidate_prepare', frame_idx, {
            'separator': separator,
            'candidates': candidates,
            'frame_idx': frame_idx,
            'tracks': tracks,
            'previous_triangles': previous_background_triangles,
            'association_enabled': enable_trajectory_association,
            'triangle_enabled': triangle_background_enabled,
            'target_exclusion_radius': triangle_target_exclusion_radius,
            'triangle_debug': triangle_debug,
        })

        # ── 帧间位移粗分离：IMM + 位移两级分离 ────
        if enable_trajectory_association:
            star_points, tracked_targets, displacement_targets = \
                separator.pre_separate(candidates, frame_idx, tracks)
        else:
            # 仅星图识别模式：全部检测点直接作为星图识别输入，
            # 不执行任何依赖轨迹历史的候选预分离。
            star_points = list(candidates)
            tracked_targets = []
            displacement_targets = []
        # 三角背景网使用“全部检测减去已确认目标邻域”，避免位移异常通道中的
        # 漏检恒星因不在 star_points 中而天然无法接受三角背景复核。
        preprocessing = candidate_stage.prepare_geometry(
            candidates=candidates,
            frame_idx=frame_idx,
            tracks=tracks,
            star_points=star_points,
            tracked_targets=tracked_targets,
            displacement_targets=displacement_targets,
            previous_triangles=previous_background_triangles,
            triangle_background_enabled=triangle_background_enabled,
            target_exclusion_radius=triangle_target_exclusion_radius,
            triangle_debug=triangle_debug,
        )
        tri_curr = preprocessing.star_triangles
        triangle_tri_curr = preprocessing.background_triangles

        # 相邻帧三角拓扑只提供辅助恒星证据，不替代星表识别和姿态求解。
        triangle_background = preprocessing.triangle_background
        previous_background_triangles = preprocessing.next_background_triangles

        # ── Stage 0: 星图识别与姿态解算 ─────────────────────────────
        star_input = {
            'star_points': star_points,
            'triangles': tri_curr,
            'meta': meta,
            'frame_idx': frame_idx,
            'enabled': enable_star_id,
            'method': star_id_method,
            'verbose': verbose,
            'runtime_state': _star_id_runtime,
        }
        record_stage_input('star', frame_idx, star_input)
        star_input.pop('runtime_state')
        star_processing_result = star_stage.process(
            **star_input
        )

        # 保留内部方法名为 descriptor/delaunay，避免影响后续星表选择；
        # 仅在日志和统计中标明本帧实际走了上一帧姿态投影跟踪。
        star_id_result = star_processing_result.identification
        attitude_q = star_processing_result.attitude_quaternion
        attitude_R = star_processing_result.attitude_matrix
        attitude_ok = star_processing_result.attitude_valid
        selected_star_id_method = star_processing_result.selected_method
        displayed_star_id_method = star_processing_result.displayed_method
        star_id_diagnostics = star_processing_result.diagnostics
        projected_catalog = star_processing_result.projected_catalog
        recognized_stars = star_processing_result.recognized_stars
        classification_input = {
            'separator': separator,
            'candidates': candidates,
            'tracks': tracks,
            'star_result': star_processing_result,
            'preprocessing': preprocessing,
            'association_enabled': enable_trajectory_association,
        }
        record_stage_input(
            'candidate_classify', frame_idx, classification_input)
        classification_input.pop('separator')
        classification_result = candidate_stage.classify(
            **classification_input
        )
        target_cands = classification_result.target_candidates
        background = classification_result.background_candidates
        tracked_targets = classification_result.tracked_targets
        n_motion_removed = classification_result.motion_removed_count

        # ── 构建代价矩阵（使用 CSV 元数据中的能量/半径）────────────
        tracking_input = {
            'tracks': tracks,
            'target_candidates': classification_result.target_candidates,
            'measurements_dir': measurements_dir,
            'frame_idx': frame_idx,
            'enabled': enable_trajectory_association,
        }
        record_stage_input('tracking', frame_idx, tracking_input)
        tracking_input.pop('measurements_dir')
        tracking_result = FrameTrackingStage.run(
            loader=loader,
            **tracking_input
        )
        tracks = tracking_result.tracks
        pruned = tracking_result.pruned_tracks
        Phi = tracking_result.cost_matrix
        cand_energies = tracking_result.candidate_energies
        cand_radii = tracking_result.candidate_radii

        # ── Sinkhorn + JPDA 更新 ──────────────────────────────────
        reconnection_input = {
            'tracks': tracks,
            'pruned_tracks': pruned,
            'archived_tracks': archived_tracks,
            'trajectory_memory': trajectory_memory,
            'pending_reconnect': pending_reconnect,
            'frame_idx': frame_idx,
            'enabled': enable_trajectory_association,
        }
        record_stage_input('reconnection', frame_idx, reconnection_input)
        reconnection_result = TrackReconnectionStage.run(
            **reconnection_input
        )
        tracks = reconnection_result.tracks
        archived_tracks = reconnection_result.archived_tracks
        trajectory_memory = reconnection_result.trajectory_memory
        pending_reconnect = reconnection_result.pending_reconnect
        frame_reconnect_events = reconnection_result.frame_events
        if frame_reconnect_events:
            reconnect_events.extend(frame_reconnect_events)
        n_active = len(get_active_tracks(tracks))

        # ── 逐帧评估（对比 ground truth）───────────────────────────
        eval_star = None
        eval_target = None
        stars_gt = []
        if has_stars_gt or has_targets_gt:
            from evaluation import (load_stars_gt, load_targets_gt,
                                    evaluate_star_id_frame,
                                    evaluate_star_removal_frame,
                                    evaluate_targets_frame)

            if has_stars_gt:
                stars_gt = load_stars_gt(stars_gt_path, frame_idx)
                eval_star = evaluate_star_id_frame(star_id_result, star_points, stars_gt)
                eval_star['frame'] = frame_idx
                star_eval_rows.append(eval_star)
                removal_eval = evaluate_star_removal_frame(
                    candidates, target_cands, stars_gt)
                removal_eval['frame'] = frame_idx
                star_removal_rows.append(removal_eval)

            if has_targets_gt:
                targets_gt = load_targets_gt(targets_gt_path, frame_idx)
                eval_target = evaluate_targets_frame(tracks, frame_idx, targets_gt)
                eval_target['frame'] = frame_idx
                target_eval_rows.append(eval_target)

        # ── 逐帧可视化 ─────────────────────────────────────────────
        star_image = draw_star_recognition_result(
            FRAME_W, FRAME_H, candidates, stars_gt,
            star_points, star_id_result)
        save_image(star_image, os.path.join(
            star_frame_dir, "frame_%04d.png" % frame_idx))

        frame_tracks = []
        seen_track_ids = set()
        for track in archived_tracks + tracks:
            track_id = track.get('id')
            if track_id in seen_track_ids:
                continue
            seen_track_ids.add(track_id)
            frame_tracks.append(track)
        track_image = draw_measurements_and_tracks(
            FRAME_W, FRAME_H, candidates, frame_tracks)
        save_image(track_image, os.path.join(
            track_frame_dir, "frame_%04d.png" % frame_idx))

        # ── 统计 ──────────────────────────────────────────────────
        stats_rows.append({
            'frame': frame_idx,
            'n_stars': len(star_points),
            'n_tracked_targets_pre': len(tracked_targets),
            'n_displacement_targets_pre': len(displacement_targets),
            'n_stars_predicted': 0,
            'n_triangles': len(tri_curr),
            'n_triangle_background_triangles': len(triangle_tri_curr),
            'n_matched_pairs': int((triangle_background or {}).get(
                'matched_triangles', 0)),
            'n_candidates': len(target_cands),
            'n_targets': len(target_cands),
            'n_active_tracks': n_active,
            'n_star_class_tracks': sum(
                1 for t in tracks if t.get('object_class') == 'star'),
            'n_target_class_tracks': sum(
                1 for t in tracks if t.get('object_class') == 'target'),
            'n_unknown_class_tracks': sum(
                1 for t in tracks if t.get('object_class', 'unknown') == 'unknown'),
            'n_exit_memory': len(trajectory_memory),
            'n_reconnect_events': len(frame_reconnect_events) if 'frame_reconnect_events' in locals() else 0,
            'triangle_available': int(bool(
                (triangle_background or {}).get('available', False))),
            'triangle_reason': (triangle_background or {}).get('reason', ''),
            'triangle_model': (triangle_background or {}).get('model', ''),
            'triangle_point_matches': int((triangle_background or {}).get(
                'point_matches', 0)),
            'triangle_inliers': int((triangle_background or {}).get('inliers', 0)),
            'triangle_rejected': int((triangle_background or {}).get('rejected', 0)),
            'triangle_conflicts': int((triangle_background or {}).get('conflicts', 0)),
            'triangle_protected': int((triangle_background or {}).get('protected', 0)),
            'match_ratio': (triangle_background or {}).get('match_ratio'),
            'inlier_ratio': (triangle_background or {}).get('inlier_ratio'),
            'edge_preserve': (triangle_background or {}).get('edge_preserve'),
            'triangle_median_residual': (triangle_background or {}).get(
                'median_residual'),
            'catalog_available': int(bool(
                separator.catalog_diagnostics.get('available', False))),
            'catalog_target_available': int(bool(
                separator.catalog_diagnostics.get('target_available', False))),
            'catalog_matches': int(separator.catalog_diagnostics.get('matches', 0)),
            'catalog_match_ratio': separator.catalog_diagnostics.get('match_ratio'),
            'catalog_median_residual': separator.catalog_diagnostics.get(
                'median_residual'),
            'n_identified_stars': star_id_result['n_identified'] if star_id_result else 0,
            'star_id_selected_method': displayed_star_id_method or '',
            'star_id_primary_method': star_id_diagnostics['primary_method'],
            'star_id_fallback_used': star_id_diagnostics['fallback_used'],
            'star_id_switch_reason': star_id_diagnostics['switch_reason'],
            'star_id_primary_time_ms': star_id_diagnostics['primary_time_ms'],
            'star_id_fallback_time_ms': star_id_diagnostics['fallback_time_ms'],
            'star_id_quality_score': star_id_diagnostics['quality_score'],
            'star_id_quality_matches': star_id_diagnostics['quality_matches'],
            'star_id_quality_median_residual': star_id_diagnostics['quality_median_residual'],
            'star_id_quality_max_residual': star_id_diagnostics['quality_max_residual'],
            'star_id_quality_coverage': star_id_diagnostics['quality_coverage'],
            'star_id_consecutive_failures': star_id_diagnostics['consecutive_failures'],
            'attitude_ok': 1 if attitude_ok else 0,
            'q_w': float(attitude_q[0]) if attitude_ok else None,
            'q_x': float(attitude_q[1]) if attitude_ok else None,
            'q_y': float(attitude_q[2]) if attitude_ok else None,
            'q_z': float(attitude_q[3]) if attitude_ok else None,
            'time_ms': (time.time() - t_start) * 1000,
        })

        if verbose and (
                not enable_trajectory_association or
                frame_idx % 10 == 0 or len(target_cands) > 0 or n_active > 0):
            n_sid = star_id_result['n_identified'] if star_id_result else 0
            att_str = "  +att" if attitude_ok else ""
            eval_str = ""
            if eval_star:
                eval_str += "  star[P=%.2f R=%.2f]" % (
                    eval_star['precision'], eval_star['recall'])
            if eval_target:
                eval_str += "  tgt[P=%.2f R=%.2f err=%.1fpx]" % (
                    eval_target['precision'], eval_target['recall'],
                    eval_target['position_error'])
            mot_str = "  mot=%d" % n_motion_removed if n_motion_removed > 0 else ""
            reid_str = "  reID=%d" % len(frame_reconnect_events) if frame_reconnect_events else ""
            mem_str = "  mem=%d" % len(trajectory_memory) if trajectory_memory else ""
            method_str = ("  id=%s" % displayed_star_id_method
                          if displayed_star_id_method else "")
            print("[%04d]  pts=%d  sid=%d  cand=%d  tracks=%d  %dms%s%s%s%s%s%s" % (
                frame_idx, len(candidates), n_sid,
                len(target_cands), n_active,
                (time.time() - t_start) * 1000, method_str, att_str,
                eval_str, mot_str, reid_str, mem_str))

    # ── 收尾 ──────────────────────────────────────────────────────
    active_tracks = get_active_tracks(tracks)
    all_tracks = [t for t in (active_tracks + archived_tracks) if len(t['positions']) >= 2]
    seen = set()
    all_dedup = []
    for t in all_tracks:
        if t['id'] not in seen:
            seen.add(t['id'])
            all_dedup.append(t)
    all_tracks = sorted(all_dedup, key=lambda t: t['id'])

    validate_track_sequences(all_tracks)
    save_frame_statistics(run_dir, stats_rows)
    save_tracks(run_dir, all_tracks)
    save_track_events(run_dir, all_tracks)
    save_reconnect_events_csv(run_dir, reconnect_events)
    save_attitude(run_dir, stats_rows, measurements_dir)
    save_track_overlay(run_dir, all_tracks)
    print_mota_motp(measurements_dir, all_tracks, n_frames)
    print_attitude_evaluation(measurements_dir, stats_rows)

    # ── 评估汇总 ──────────────────────────────────────────────────
    if star_eval_rows or target_eval_rows:
        from evaluation import summarize
        summary = summarize(star_eval_rows, target_eval_rows)
        save_evaluation(run_dir, star_eval_rows, target_eval_rows,
                        star_removal_rows)
        print_star_removal_summary(star_removal_rows)
        print_evaluation_summary(summary)

    if verbose:
        total_time = sum(r['time_ms'] for r in stats_rows) / 1000
        print("")
        print("Done: %d frames, %.1fs (%.0fms/frame)" % (
            n_frames, total_time, total_time / max(n_frames, 1) * 1000))
        print("Tracks: %d" % len(all_tracks))
        if reconnect_events:
            gaps = [e.get('gap', 0) for e in reconnect_events if e.get('gap') is not None]
            scores = [e.get('score', 0.0) for e in reconnect_events]
            print("Reconnects: %d  mean_gap=%.1f  mean_score=%.3f" % (
                len(reconnect_events), float(np.mean(gaps)) if gaps else 0.0,
                float(np.mean(scores)) if scores else 0.0))
        for t in all_tracks:
            p = t['positions']
            tag = "[%s]" % t['status']
            print("  %s ID=%d: %dpts (%.0f,%.0f)->(%.0f,%.0f)" % (
                tag, t['id'], len(p), p[0][0], p[0][1], p[-1][0], p[-1][1]))
        print("Output: " + run_dir)

    return tracks, stats_rows, run_dir


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Space Multi-Target Tracking (CSV Detection Input)',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "示例:\n"
            "  python main.py                              # 交互式从 datasets/ 选择数据集\n"
            "  python main.py -d run_20260630_184218       # 按名称直接指定数据集\n"
            "  python main.py -i /path/to/measurements     # 直接指定 measurements 目录\n"
            "  python main.py --list                       # 列出所有可用数据集\n"
        )
    )
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='直接指定 measurements CSV 目录（跳过数据集选择）')
    parser.add_argument('--dataset', '-d', type=str, default=None,
                        help='按名称指定数据集（在 datasets/ 下查找）')
    parser.add_argument('--datasets-root', type=str, default=None,
                        help='数据集根目录，默认为脚本同级的 datasets/ 文件夹')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出目录，默认为上级目录下的 output/')
    parser.add_argument('--frames', '-n', type=int, default=None,
                        help='只处理前 N 帧')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='静默模式')
    parser.add_argument('--snapshots', type=str, default=None,
                        help='可选：保存每帧各模块输入快照的目录')
    parser.add_argument('--list', '-l', action='store_true',
                        help='仅列出所有可用数据集后退出')

    args = parser.parse_args()

    script_dir = PACKAGE_DIR
    project_root = os.path.dirname(script_dir)

    # ── 确定 datasets 根目录 ──────────────────────────────────────
    if args.datasets_root:
        datasets_root = args.datasets_root
    else:
        candidate_same = os.path.join(script_dir, "datasets")
        candidate_up = os.path.join(project_root, "datasets")
        if os.path.isdir(candidate_same):
            datasets_root = candidate_same
        elif os.path.isdir(candidate_up):
            datasets_root = candidate_up
        else:
            datasets_root = candidate_same

    if args.list:
        datasets = _scan_datasets(datasets_root)
        if not datasets:
            print("未找到任何数据集（datasets 根目录: %s）" % datasets_root)
        else:
            print("datasets 根目录: %s" % datasets_root)
            for ds in datasets:
                truth_tag = " [含truth]" if ds['has_truth'] else ""
                print("  %-30s  %d 帧%s" % (ds['name'], ds['n_frames'], truth_tag))
        sys.exit(0)

    # ── 确定输入目录 ──────────────────────────────────────────────
    if args.input:
        input_dir = args.input
        if not os.path.isdir(input_dir):
            print("Error: 指定的输入目录不存在: " + input_dir)
            sys.exit(1)
        dataset_name = os.path.basename(os.path.normpath(input_dir))
        if dataset_name == "measurements":
            dataset_name = os.path.basename(os.path.dirname(os.path.normpath(input_dir)))

    elif args.dataset:
        datasets = _scan_datasets(datasets_root)
        matched = [ds for ds in datasets if ds['name'] == args.dataset]
        if not matched:
            print("Error: 在 %s 中未找到名为 '%s' 的数据集" % (datasets_root, args.dataset))
            if datasets:
                print("可用数据集: " + ", ".join(ds['name'] for ds in datasets))
            sys.exit(1)
        input_dir = matched[0]['measurements_dir']
        dataset_name = matched[0]['name']
        if not args.quiet:
            print("数据集: %s  (%d 帧)" % (matched[0]['name'], matched[0]['n_frames']))

    else:
        datasets = _scan_datasets(datasets_root)
        if not datasets:
            print("Error: 未找到任何数据集。")
            print("请在以下位置创建 datasets/ 目录并放入数据集子文件夹:")
            print("  " + datasets_root)
            print("")
            print("目录结构示例:")
            print("  datasets/")
            print("  └── my_dataset/")
            print("      ├── measurements/  ← 存放 measurements_0000.csv, ...")
            print("      └── truth/         ← 可选，存放 target_*_trajectory.npy")
            sys.exit(1)
        else:
            selected = _select_dataset_interactive(datasets)
            input_dir = selected['measurements_dir']
            dataset_name = selected['name']
            if not args.quiet:
                print("已选择: %s  (%s)" % (selected['name'], input_dir))

    if not os.path.isdir(input_dir):
        print("Error: 输入目录不存在: " + input_dir)
        sys.exit(1)

    process_sequence(input_dir, output_dir=args.output, end_frame=args.frames,
                     verbose=not args.quiet, dataset_name=dataset_name,
                     snapshot_dir=args.snapshots)


if __name__ == '__main__':
    main()
