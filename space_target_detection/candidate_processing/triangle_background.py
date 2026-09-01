# Domain implementation: triangle background evidence.
import numpy as np
import cv2
from configuration.values import (triangle_feat_weights, triangle_feat_weights_v2,
                    triangle_match_epsilon, triangle_min_matched_triangles,
                    triangle_ransac_threshold, triangle_use_kdtree_refine,
                    triangle_match_radius, triangle_min_inliers,
                    triangle_min_inlier_ratio, triangle_min_edge_preserve,
                    triangle_max_median_residual, triangle_use_fusion_v2,
                    triangle_target_exclusion_radius, triangle_debug,
                    FRAME_W, FRAME_H)


def calc_fusion_features(triangles):
    """计算每个三角形的融合特征 F = w1*A + w2*P + w3*S"""
    w1, w2, w3 = triangle_feat_weights
    for tri in triangles:
        tri['fusion_feat'] = (w1 * tri['aspect_ratio'] +
                              w2 * tri['perimeter'] +
                              w3 * tri['area'])
        tri['match_vector'] = np.asarray([
            tri['alpha_min'] / np.pi,
            tri['r1_inv'],
            1.0 / max(tri['r2'], 1e-12),
            np.log(max(tri['area'], 1e-12)),
        ], dtype=np.float64)
    return triangles


def calc_fusion_features_v2(triangles):
    """计算融合特征 V2：F = w₁·α_min + w₂·r₁ + w₃·S + w₄·AR

    α_min  — 最小内角（Delaunay 三角形关键区分量，尺度不变）
    r₁=a/b — 边长比（补充形状细节，尺度不变）
    S      — 面积（提供绝对尺度信息）
    AR     — 宽高比 c/a（形状粗筛，尺度不变）

    混合尺度不变与尺度相关特征，在相机焦距变化或平台机动缩放场景下
    匹配鲁棒性更强。Z-Score 分组与匹配逻辑与原版完全一致。
    """
    w1, w2, w3, w4 = triangle_feat_weights_v2
    for tri in triangles:
        tri['fusion_feat'] = (w1 * tri['alpha_min'] +
                              w2 * tri['r1_inv'] +
                              w3 * tri['area'] +
                              w4 * tri['aspect_ratio'])
        # 匹配保留多维形状信息，不再把不同三角形压缩成一个标量比较。
        tri['match_vector'] = np.asarray([
            tri['alpha_min'] / np.pi,
            tri['r1_inv'],
            1.0 / max(tri['r2'], 1e-12),
            np.log(max(tri['area'], 1e-12)),
        ], dtype=np.float64)
    return triangles


def triangle_match(tri_prev, tri_curr):
    """用多维形状特征的互为最近邻匹配两帧三角形。

    返回
    ----
    matched_pairs : List[(triangle_prev, triangle_curr)]
        匹配成功的三角形对
    """
    if len(tri_prev) < 3 or len(tri_curr) < 3:
        return []

    if triangle_use_fusion_v2:
        calc_fusion_features_v2(tri_prev)
        calc_fusion_features_v2(tri_curr)
    else:
        calc_fusion_features(tri_prev)
        calc_fusion_features(tri_curr)

    distances = np.zeros((len(tri_prev), len(tri_curr)), dtype=np.float64)
    for i, previous in enumerate(tri_prev):
        vp = previous['match_vector']
        for j, current in enumerate(tri_curr):
            vc = current['match_vector']
            shape_distance = float(np.mean(np.abs(vp[:3] - vc[:3])))
            scale_distance = float(abs(vp[3] - vc[3]))
            distances[i, j] = 0.8 * shape_distance + 0.2 * scale_distance

    nearest_curr = np.argmin(distances, axis=1)
    nearest_prev = np.argmin(distances, axis=0)
    matches = []
    for previous_idx, current_idx in enumerate(nearest_curr):
        current_idx = int(current_idx)
        if int(nearest_prev[current_idx]) != previous_idx:
            continue
        distance = float(distances[previous_idx, current_idx])
        if distance <= triangle_match_epsilon:
            matches.append((distance, previous_idx, current_idx))
    matches.sort(key=lambda item: item[0])
    return [
        (tri_prev[previous_idx], tri_curr[current_idx])
        for _, previous_idx, current_idx in matches
    ]


def extract_match_points(matched_pairs):
    """从匹配三角形对中提取对应顶点用于仿射估计

    使用三角形重心作为匹配点对
    """
    pts_src = []
    pts_dst = []

    for tri_prev, tri_curr in matched_pairs:
        # 使用三角形重心
        vp = np.array(tri_prev['vertices'])
        vc = np.array(tri_curr['vertices'])
        centroid_prev = vp.mean(axis=0)
        centroid_curr = vc.mean(axis=0)
        pts_src.append(centroid_prev)
        pts_dst.append(centroid_curr)

        # 同时加入三个顶点
        for k in range(3):
            pts_src.append(vp[k])
            pts_dst.append(vc[k])

    return np.array(pts_src, dtype=np.float32), np.array(pts_dst, dtype=np.float32)


def _collect_star_positions(triangles):
    """从三角形列表中收集去重后的星点坐标"""
    seen = set()
    positions = []
    for tri in triangles:
        for vx, vy in tri['vertices']:
            key = (round(vx, 1), round(vy, 1))
            if key not in seen:
                seen.add(key)
                positions.append((vx, vy))
    return positions


def _match_projected_points_one_to_one(projected, current, radius):
    """在仿射预测位置和当前点之间建立门控后一对一贪心匹配。"""
    if len(projected) == 0 or len(current) == 0:
        return []
    distances = np.linalg.norm(
        projected[:, None, :] - current[None, :, :], axis=2)
    prev_indices, curr_indices = np.where(distances <= radius)
    pairs = sorted(
        (float(distances[i, j]), int(i), int(j))
        for i, j in zip(prev_indices, curr_indices))
    used_prev, used_curr, matches = set(), set(), []
    for distance, prev_idx, curr_idx in pairs:
        if prev_idx in used_prev or curr_idx in used_curr:
            continue
        used_prev.add(prev_idx)
        used_curr.add(curr_idx)
        matches.append((prev_idx, curr_idx, distance))
    return matches


def _transform_points(points, transform, model):
    """按单应或仿射模型投影二维点。"""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if model == 'homography':
        return cv2.perspectiveTransform(points.reshape(1, -1, 2), transform)[0]
    return cv2.transform(points.reshape(1, -1, 2), transform)[0]


def _estimate_background_transform(src, dst, threshold):
    """优先估计旋转星场更适用的单应模型，失败时回退二维仿射。"""
    src = np.asarray(src, dtype=np.float32).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float32).reshape(-1, 2)
    if len(src) >= 4:
        transform, mask = cv2.findHomography(
            src, dst, method=cv2.RANSAC,
            ransacReprojThreshold=threshold)
        if transform is not None and mask is not None and np.all(np.isfinite(transform)):
            return transform, mask, 'homography'
    if len(src) >= 3:
        transform, mask = cv2.estimateAffine2D(
            src, dst, method=cv2.RANSAC,
            ransacReprojThreshold=threshold)
        if transform is not None and mask is not None and np.all(np.isfinite(transform)):
            return transform, mask, 'affine'
    return None, None, None


def match_background_points(tri_prev, tri_curr, tracks=None):
    """仅利用检测点三角拓扑估计相邻帧背景变换和当前恒星内点。

    本入口不读取图像。返回字典中的 ``available`` 只有在三角匹配数量、
    RANSAC 内点率、边保持率和残差均通过门限时才为真；失败表示证据未知，
    调用方不得把未匹配点自动解释成目标。
    """
    result = {
        'available': False,
        'transform': None,
        'model': None,
        'matched_points': [],
        'residual_by_point': {},
        'matched_triangles': 0,
        'match_ratio': 0.0,
        'point_matches': 0,
        'inliers': 0,
        'inlier_ratio': 0.0,
        'edge_preserve': 0.0,
        'median_residual': float('inf'),
        'reason': '',
    }
    if not tri_prev or not tri_curr:
        result['reason'] = 'missing_triangles'
        return result

    matched_pairs = triangle_match(tri_prev, tri_curr)
    result['matched_triangles'] = len(matched_pairs)
    result['match_ratio'] = (
        len(matched_pairs) / max(min(len(tri_prev), len(tri_curr)), 1))
    if len(matched_pairs) < triangle_min_matched_triangles:
        result['reason'] = 'insufficient_triangle_matches'
        return result

    centroids_prev = np.asarray([
        np.mean(np.asarray(prev['vertices'], dtype=np.float32), axis=0)
        for prev, _ in matched_pairs
    ], dtype=np.float32)
    centroids_curr = np.asarray([
        np.mean(np.asarray(curr['vertices'], dtype=np.float32), axis=0)
        for _, curr in matched_pairs
    ], dtype=np.float32)
    coarse, _, coarse_model = _estimate_background_transform(
        centroids_prev, centroids_curr, triangle_ransac_threshold * 2.0)
    if coarse is None:
        result['reason'] = 'coarse_transform_failed'
        return result

    previous = np.asarray(_collect_star_positions(tri_prev), dtype=np.float32)
    current = np.asarray(_collect_star_positions(tri_curr), dtype=np.float32)
    projected = _transform_points(previous, coarse, coarse_model)
    matches = _match_projected_points_one_to_one(
        projected, current, triangle_match_radius)
    result['point_matches'] = len(matches)
    if len(matches) < 3:
        result['reason'] = 'insufficient_point_matches'
        return result

    src = np.asarray([previous[i] for i, _, _ in matches], dtype=np.float32)
    dst = np.asarray([current[j] for _, j, _ in matches], dtype=np.float32)
    transform, inlier_mask, model = _estimate_background_transform(
        src, dst, triangle_ransac_threshold)
    if transform is None or inlier_mask is None:
        result['reason'] = 'transform_ransac_failed'
        return result

    mask = inlier_mask.ravel().astype(bool)
    src_inliers = src[mask]
    dst_inliers = dst[mask]
    projected_inliers = _transform_points(src_inliers, transform, model)
    residuals = np.linalg.norm(projected_inliers - dst_inliers, axis=1)

    # 当前已确认目标附近的点不能被三角背景模型反向标成恒星。
    target_positions = np.asarray([
        track['state'][:2] for track in (tracks or [])
        if track.get('object_class') == 'target'
    ], dtype=np.float64).reshape(-1, 2)
    keep = np.ones(len(dst_inliers), dtype=bool)
    if target_positions.size > 0 and len(dst_inliers) > 0:
        distances = np.linalg.norm(
            dst_inliers[:, None, :] - target_positions[None, :, :], axis=2)
        keep = np.min(distances, axis=1) > triangle_target_exclusion_radius
    dst_inliers = dst_inliers[keep]
    residuals = residuals[keep]

    inlier_count = len(dst_inliers)
    inlier_ratio = float(np.sum(mask)) / max(len(matches), 1)
    edge_preserve = float(_compute_edge_preservation(tri_curr, matched_pairs))
    median_residual = (float(np.median(residuals)) if len(residuals) > 0
                       else float('inf'))
    result.update({
        'transform': transform,
        'model': model,
        'matched_points': [tuple(map(float, point)) for point in dst_inliers],
        'residual_by_point': {
            (round(float(point[0]), 3), round(float(point[1]), 3)): float(residual)
            for point, residual in zip(dst_inliers, residuals)
        },
        'inliers': inlier_count,
        'inlier_ratio': inlier_ratio,
        'edge_preserve': edge_preserve,
        'median_residual': median_residual,
    })
    checks = (
        inlier_count >= triangle_min_inliers,
        inlier_ratio >= triangle_min_inlier_ratio,
        edge_preserve >= triangle_min_edge_preserve,
        median_residual <= triangle_max_median_residual,
    )
    result['available'] = bool(all(checks))
    result['reason'] = 'ok' if result['available'] else 'quality_gate_failed'
    if triangle_debug:
        print('[triangle-background] triangles=%d model=%s inliers=%d '
              'ratio=%.3f edge=%.3f median=%.3f available=%s reason=%s' % (
                  len(matched_pairs), model, inlier_count, inlier_ratio,
                  edge_preserve, median_residual, result['available'],
                  result['reason']))
    return result


def _filter_contaminated_seeds(seed_prev, seed_set, tracks):
    """过滤被目标污染的种子点

    若种子点与已知目标位置距离小于 ``triangle_target_exclusion_radius``，
    视为被目标运动污染，从种子集中移除。
    """
    contaminated = set()
    for t in tracks:
        tx, ty = t['state'][0], t['state'][1]
        for key in seed_set:
            vx, vy = key
            if (np.sqrt((vx - tx) ** 2 + (vy - ty) ** 2) <
                    triangle_target_exclusion_radius):
                contaminated.add(key)

    if contaminated:
        clean_prev = [v for v in seed_prev
                      if (round(v[0], 1), round(v[1], 1)) not in contaminated]
        clean_set = {k for k in seed_set if k not in contaminated}
        return clean_prev, clean_set
    return seed_prev, seed_set


def refine_with_kdtree(tri_prev, tri_curr, matched_pairs, tracks=None):
    """三阶段级联对应点提取：拓扑筛选 → 局部精匹配 → 全局扩展

    Stage 1（拓扑筛选）: 从拓扑一致的匹配三角形提取唯一顶点作为
              高可信度恒星种子点，并用 EKF 预测位置过滤被目标
              污染的顶点，避免目标运动混入仿射估计。
    Stage 2（局部精匹配）: 用匹配三角形重心建立初始对应并估计
              粗仿射，将种子点投影到当前帧坐标系后通过 KDTree
              最近邻搜索建立精确的局部对应关系，得到局部精仿射
              M_local。
    Stage 3（全局扩展）: 用 M_local 将前一帧全部恒星投影到当前
              帧，在全局范围内 KDTree 搜索可靠对应星点，结合
              RANSAC 剔除误匹配，最终返回稳定的全局对应点集。

    tracks : List[track_dict] or None
        活跃轨迹列表，传入时 t['state'] 应为前一帧位置。

    返回 (pts_src, pts_dst) — 全局 RANSAC 内点，用于最终仿射估计。
    """
    from scipy.spatial import KDTree

    # ═══════════════════════════════════════════════════════════
    # Stage 1: 拓扑筛选 — 提取高可信度种子点并过滤污染顶点
    # ═══════════════════════════════════════════════════════════
    seed_set = set()
    seed_prev = []
    for tri_p, tri_c in matched_pairs:
        for vx, vy in tri_p['vertices']:
            key = (round(vx, 1), round(vy, 1))
            if key not in seed_set:
                seed_set.add(key)
                seed_prev.append((vx, vy))

    # 过滤被目标污染的种子点
    if tracks:
        seed_prev, seed_set = _filter_contaminated_seeds(seed_prev, seed_set, tracks)

    if len(seed_prev) < 3:
        return (np.array(seed_prev, dtype=np.float32),
                np.array(seed_prev, dtype=np.float32))

    seed_prev_np = np.array(seed_prev, dtype=np.float32)
    all_curr = _collect_star_positions(tri_curr)
    all_curr_np = np.array(all_curr, dtype=np.float32)

    # ═══════════════════════════════════════════════════════════
    # Stage 2: 局部精匹配 — 种子点建立初始对应 → 粗仿射 → 局部精仿射
    # ═══════════════════════════════════════════════════════════
    # 2a. 匹配三角形重心 → 初始粗对齐
    centroids_prev = []
    centroids_curr = []
    for tri_p, tri_c in matched_pairs:
        vp = np.array(tri_p['vertices'])
        vc = np.array(tri_c['vertices'])
        centroids_prev.append(vp.mean(axis=0))
        centroids_curr.append(vc.mean(axis=0))

    centroids_prev_np = np.array(centroids_prev, dtype=np.float32)
    centroids_curr_np = np.array(centroids_curr, dtype=np.float32)

    result = cv2.estimateAffine2D(centroids_prev_np, centroids_curr_np,
                                   method=cv2.RANSAC,
                                   ransacReprojThreshold=triangle_ransac_threshold * 2)
    M_coarse = result[0]

    # 2b. 粗对齐种子点 → KDTree 局部精确匹配
    if M_coarse is not None:
        ones_s = np.ones((len(seed_prev_np), 1), dtype=np.float32)
        query_seeds = (M_coarse @ np.hstack([seed_prev_np, ones_s]).T).T
    else:
        query_seeds = seed_prev_np

    tree_local = KDTree(all_curr_np)
    dist_seeds, idx_seeds = tree_local.query(query_seeds, k=1)

    local_src = []
    local_dst = []
    for i, (d, j) in enumerate(zip(dist_seeds, idx_seeds)):
        if d < triangle_match_radius:
            local_src.append(seed_prev_np[i])
            local_dst.append(all_curr_np[j])

    if len(local_src) < 3:
        local_src = [seed_prev_np[i] for i in range(len(seed_prev_np))]
        local_dst = [all_curr_np[idx_seeds[i]] for i in range(len(seed_prev_np))]

    local_src_np = np.array(local_src, dtype=np.float32)
    local_dst_np = np.array(local_dst, dtype=np.float32)

    # 2c. 局部对应 → 粗仿射 M_local
    result2 = cv2.estimateAffine2D(local_src_np, local_dst_np,
                                    method=cv2.RANSAC,
                                    ransacReprojThreshold=triangle_ransac_threshold)
    M_local = result2[0]

    if M_local is None:
        return local_src_np, local_dst_np

    # ═══════════════════════════════════════════════════════════
    # Stage 3: 全局扩展 — 全帧恒星投影 → 全局 KDTree → RANSAC 精化
    # ═══════════════════════════════════════════════════════════
    # 3a. 收集全部前帧恒星，用 M_local 投影到当前帧
    all_prev = _collect_star_positions(tri_prev)
    all_prev_np = np.array(all_prev, dtype=np.float32)

    ones_all = np.ones((len(all_prev_np), 1), dtype=np.float32)
    all_transformed = (M_local @ np.hstack([all_prev_np, ones_all]).T).T

    # 3b. 全局 KDTree 搜索 — 在全部当前帧星点中找对应
    tree_global = KDTree(all_curr_np)
    dist_global, idx_global = tree_global.query(all_transformed, k=1)

    # 收集全部通过阈值筛选的对应点对
    global_src = []
    global_dst = []
    for i, (d, j) in enumerate(zip(dist_global, idx_global)):
        # 种子点已参与局部匹配，直接从全局结果加入
        if d < triangle_match_radius:
            global_src.append(all_prev_np[i])
            global_dst.append(all_curr_np[j])

    if len(global_src) < 3:
        return local_src_np, local_dst_np

    global_src_np = np.array(global_src, dtype=np.float32)
    global_dst_np = np.array(global_dst, dtype=np.float32)

    # 3c. RANSAC 剔除误匹配 → 稳定全局内点集
    result3 = cv2.estimateAffine2D(global_src_np, global_dst_np,
                                    method=cv2.RANSAC,
                                    ransacReprojThreshold=triangle_ransac_threshold)
    M_global = result3[0]
    inliers_global = result3[1]

    if M_global is not None and inliers_global is not None:
        mask = inliers_global.ravel().astype(bool)
        pts_src = global_src_np[mask]
        pts_dst = global_dst_np[mask]
    else:
        pts_src, pts_dst = local_src_np, local_dst_np

    if len(pts_src) < 3:
        pts_src, pts_dst = local_src_np, local_dst_np

    return pts_src, pts_dst


def match_and_diff(frame_prev, frame_cur, tri_prev, tri_curr, tracks=None):
    """
    三角匹配 + 仿射配准 + 帧间差分

    tracks : List[track_dict] or None
        活跃轨迹列表，用于过滤被目标污染的三角形顶点

    返回
    ----
    diff_img, matched_pairs, M, star_mask, topo
    """
    matched_pairs = triangle_match(tri_prev, tri_curr)

    if len(matched_pairs) < triangle_min_matched_triangles:
        return None, matched_pairs, None, None, None

    if triangle_use_kdtree_refine:
        pts_src, pts_dst = refine_with_kdtree(tri_prev, tri_curr, matched_pairs,
                                               tracks=tracks)
    else:
        pts_src, pts_dst = extract_match_points(matched_pairs)

    if len(pts_src) < 3:
        return None, matched_pairs, None, None, None

    # 仿射变换估计
    result = cv2.estimateAffine2D(pts_src, pts_dst, method=cv2.RANSAC,
                                  ransacReprojThreshold=triangle_ransac_threshold)
    M = result[0]
    inliers = result[1]  # (N,1) uint8, 1=inlier

    if M is None:
        return None, matched_pairs, None, None, None

    # 拓扑一致性验证
    topo = check_topology_consistency(tri_prev, tri_curr, matched_pairs, inliers)

    # 生成恒星掩膜：将前一帧星点变换到当前帧坐标，标记为背景区域
    # margin 根据亮度自适应调整
    star_mask = _build_star_mask_adaptive(frame_prev.shape, tri_prev, M)

    # 配准并差分
    aligned = cv2.warpAffine(frame_prev, M, (FRAME_W, FRAME_H))
    diff = cv2.absdiff(frame_cur.astype(np.float32), aligned.astype(np.float32))

    # 用掩膜抑制差分图中的恒星残差
    diff_masked = diff.copy()
    diff_masked[star_mask > 0] = 0

    return diff_masked, matched_pairs, M, star_mask, topo


def check_topology_consistency(tri_prev, tri_curr, matched_pairs, inliers):
    """验证两帧星场拓扑一致性，返回三个指标

    指标
    ----
    match_ratio    : 三角形匹配率 = N_matched / min(N_prev, N_curr)
                     高值 → 全局拓扑一致；低值 → 大量新生/消失的三角形
    inlier_ratio   : RANSAC 内点率 = N_inliers / N_total
                     高值 → 匹配点对符合单一刚体/仿射模型，几何一致
    edge_preserve  : Delaunay 边保持率 = 映射后仍为邻边的比例
                     高值 → 星点邻接关系跨帧稳定，局部拓扑一致

    三者同时高 → 星场拓扑高度一致，三角匹配可靠
    任一指标偏低 → 该帧可能存在视场突变、遮挡或星点提取不稳定
    """
    n_prev = len(tri_prev)
    n_curr = len(tri_curr)

    # 指标 1：三角形匹配率
    match_ratio = len(matched_pairs) / max(min(n_prev, n_curr), 1)

    # 指标 2：RANSAC 内点率
    if inliers is not None and len(inliers) > 0:
        inlier_ratio = float(inliers.sum()) / len(inliers)
    else:
        inlier_ratio = 0.0

    # 指标 3：Delaunay 边保持率
    # 从匹配三角形对中提取顶点对应关系，检查相邻顶点对在映射后是否仍为 Delaunay 邻居
    edge_preserve = _compute_edge_preservation(tri_curr, matched_pairs)

    return {
        'match_ratio': round(match_ratio, 4),
        'inlier_ratio': round(inlier_ratio, 4),
        'edge_preserve': round(edge_preserve, 4),
    }


def _compute_edge_preservation(tri_curr, matched_pairs):
    """计算 Delaunay 边跨帧保持率

    从匹配三角形对中提取前帧的边，检查对应顶点在当前帧是否仍共享三角形
    （即仍为 Delaunay 邻居）。
    """
    if not matched_pairs:
        return 0.0

    # 构建当前帧的邻接表：vertex → set of neighbor vertices
    adj_curr = {}
    for tri in tri_curr:
        vs = [tuple(round(c, 1) for c in v) for v in tri['vertices']]
        for i in range(3):
            a, b = vs[i], vs[(i + 1) % 3]
            adj_curr.setdefault(a, set()).add(b)
            adj_curr.setdefault(b, set()).add(a)

    # 从匹配三角形对中收集前帧边，映射到当前帧后检查是否仍邻接
    total_edges = 0
    preserved_edges = 0

    for tri_p, tri_c in matched_pairs:
        vp = [tuple(round(c, 1) for c in v) for v in tri_p['vertices']]
        vc = [tuple(round(c, 1) for c in v) for v in tri_c['vertices']]
        mapping = dict(zip(vp, vc))

        for i in range(3):
            a_p, b_p = vp[i], vp[(i + 1) % 3]
            a_c, b_c = mapping[a_p], mapping[b_p]
            total_edges += 1
            # 检查映射后的两个顶点在当前帧是否仍邻接
            if a_c in adj_curr and b_c in adj_curr[a_c]:
                preserved_edges += 1

    return preserved_edges / max(total_edges, 1)


def _build_star_mask_adaptive(shape, tri_prev, M, base_margin=8):
    """根据三角剖分顶点和亮度，生成自适应大小的恒星掩膜"""
    H, W = shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    seen = set()
    for tri in tri_prev:
        for vx, vy in tri['vertices']:
            key = (round(vx, 1), round(vy, 1))
            if key not in seen:
                seen.add(key)
                pt = np.array([[vx, vy]], dtype=np.float32)
                pt_t = cv2.transform(pt.reshape(1, 1, 2), M)[0][0]
                tx, ty = int(round(pt_t[0])), int(round(pt_t[1]))
                margin = base_margin + 4  # 12px for all stars
                cv2.circle(mask, (tx, ty), margin, 1, -1)

    return mask


def extract_target_candidates_by_exclusion(frame_cur, tri_prev, M,
                                            k=3.0, area_min_val=2, area_max_val=200,
                                            all_spots=None):
    """
    排除法提取目标候选点：
    预测背景星位置 → 排除重合亮斑 → 剩余即为目标候选

    all_spots : List[(x, y, flux)] or None
        预提取的亮斑列表（与恒星提取同源，避免 k 值不一致导致 PSF 合并）。
        None 则内部调用 _extract_bright_spots 提取。
    """
    if frame_cur is None or M is None:
        return []

    if all_spots is None:
        all_spots = _extract_bright_spots(frame_cur, k=k,
                                           area_min_val=area_min_val,
                                           area_max_val=area_max_val)

    predicted_stars = set()
    seen = set()
    for tri in tri_prev:
        for vx, vy in tri['vertices']:
            key = (round(vx, 1), round(vy, 1))
            if key not in seen:
                seen.add(key)
                pt = np.array([[vx, vy]], dtype=np.float32)
                pt_t = cv2.transform(pt.reshape(1, 1, 2), M)[0][0]
                predicted_stars.add((round(pt_t[0]), round(pt_t[1])))

    candidates = []
    for cx, cy, flux in all_spots:
        is_star = False
        for sx, sy in predicted_stars:
            if abs(cx - sx) <= 1.5 and abs(cy - sy) <= 1.5:
                is_star = True
                break
        if not is_star:
            candidates.append((cx, cy, flux))

    return candidates


def _extract_bright_spots(frame, k=3.0, area_min_val=2, area_max_val=200):
    """提取图像中所有亮斑"""
    f32 = frame.astype(np.float32)
    mean_val = f32.mean()
    std_val = f32.std()
    threshold = mean_val + k * std_val
    binary = (f32 > threshold).astype(np.uint8) * 255

    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)

    spots = []
    for i in range(1, n):
        area_i = stats[i, cv2.CC_STAT_AREA]
        if area_min_val <= area_i <= area_max_val:
            x, y = cents[i]
            mask = (labels == i)
            flux = float(f32[mask].sum())
            spots.append((x, y, flux))

    return spots


