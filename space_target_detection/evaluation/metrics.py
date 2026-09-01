"""
评估模块

对比系统输出（星图识别 + 目标跟踪）与 ground truth，计算逐帧及总体准确率。

ground truth 来源：
  stars/        —— 恒星真值 (Frame, Star_ID, x, y, Magnitude)
  ground_truth/ —— 目标真值 (Frame, Target_ID, True_Position_X, True_Position_Y, ...)
"""

# Domain implementation: evaluation metrics.
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from configuration.values import dist_thresh, N_pixel_norm


# ══════════════════════════════════════════════════════════════════════
# Ground Truth 加载
# ══════════════════════════════════════════════════════════════════════

def _read_csv(path):
    """读取 CSV 文件，返回 dict 列表（列名已转小写）。"""
    rows = []
    with open(path, 'r') as f:
        header = f.readline().strip()
        cols = [c.strip().lower() for c in header.split(',')]
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [v.strip() for v in line.split(',')]
            if len(vals) < len(cols):
                continue
            rows.append(dict(zip(cols, vals)))
    return rows


def load_stars_gt(stars_path, frame_idx):
    """加载单帧恒星真值。

    支持两种格式：
      - 单文件: stars.csv (Frame, Star_ID, X, Y, Magnitude)
      - 逐帧目录: stars/stars_XXXX.csv

    参数 stars_path 可以是文件路径或目录路径。

    Star_ID 无效或为 0 时自动分配从 1 开始的序列号。

    返回 List[(star_id: int, x: float, y: float)]
    """
    rows = _load_gt_rows(stars_path, frame_idx, prefix="stars_")
    result = []
    auto_id = 0
    for r in rows:
        try:
            sid = int(float(r.get('star_id', 0)))
            x = float(r.get('x', 0))
            y = float(r.get('y', 0))
            auto_id += 1
            result.append((sid if sid != 0 else auto_id, x, y))
        except (ValueError, KeyError):
            continue
    return result


def load_targets_gt(ground_truth_path, frame_idx):
    """加载单帧目标真值。

    支持两种格式：
      - 单文件: ground_truth.csv (Frame, Target_ID, True_Position_X, True_Position_Y, ...)
      - 逐帧目录: ground_truth/ground_truth_XXXX.csv

    参数 ground_truth_path 可以是文件路径或目录路径。

    返回 List[(target_id: int, x: float, y: float)]
    """
    rows = _load_gt_rows(ground_truth_path, frame_idx, prefix="ground_truth_")
    result = []
    for r in rows:
        try:
            tid = int(float(r['target_id']))
            x = float(r.get('true_position_x', r.get('x', 0)))
            y = float(r.get('true_position_y', r.get('y', 0)))
            result.append((tid, x, y))
        except (ValueError, KeyError):
            continue
    return result


def _load_gt_rows(path, frame_idx, prefix=""):
    """通用的 GT 数据加载：支持单文件 CSV 或逐帧目录。"""
    # 格式 A：单文件 CSV（所有帧在一张表里）
    if os.path.isfile(path):
        rows = _read_csv(path)
        fid_str = str(frame_idx)
        return [r for r in rows if r.get('frame', '') == fid_str]

    # 格式 B：逐帧目录
    if os.path.isdir(path):
        fpath = os.path.join(path, "%s%04d.csv" % (prefix, frame_idx))
        if os.path.exists(fpath):
            return _read_csv(fpath)

    return []


# ══════════════════════════════════════════════════════════════════════
# 单帧评估
# ══════════════════════════════════════════════════════════════════════

def evaluate_star_id_frame(star_id_result, points, stars_gt, match_radius=5.0):
    """评估单帧星图识别准确率。

    匹配分两级：
    1. 位置匹配 —— 匈牙利算法全局最优，GT 与系统识别星一对一配对
    2. 星号校验 —— 验证配对星的星表编号是否一致

    参数
    ----
    star_id_result : dict  系统星图识别输出（含 'identified' 字段）
    points : List[(x, y, flux)]  星图识别模块的输入点集
    stars_gt : List[(star_id, x, y)]  该帧恒星真值
    match_radius : float  空间匹配半径（像素）

    返回
    ----
    dict  {tp, fp, fn, n_id_correct, n_id_wrong, n_gt_unpresentable,
           precision, recall, f1, position_error, n_gt, n_identified}
    """
    n_gt = len(stars_gt)
    identified_dict = star_id_result.get('identified', {}) if star_id_result else {}

    sys_stars = []
    for cand_idx, cat_id in identified_dict.items():
        idx = int(cand_idx)
        if idx < len(points):
            x, y, _ = points[idx]
            sys_stars.append((x, y, int(cat_id)))

    n_identified = len(sys_stars)

    base = {
        'tp': 0, 'fp': n_identified, 'fn': n_gt,
        'n_id_correct': 0, 'n_id_wrong': 0, 'n_gt_unpresentable': 0,
        'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
        'position_error': 0.0, 'n_gt': n_gt, 'n_identified': n_identified,
    }

    if n_gt == 0:
        return base

    # ── GT 星在输入点集中的可达性检查 ──
    pts_xy = np.array([[float(p[0]), float(p[1])] for p in points], dtype=np.float64)
    n_gt_unpresentable = 0
    if len(pts_xy) > 0:
        for _gt_id, gt_x, gt_y in stars_gt:
            if np.min(np.linalg.norm(pts_xy - np.array([[gt_x, gt_y]], dtype=np.float64),
                                     axis=1)) > match_radius:
                n_gt_unpresentable += 1
    else:
        n_gt_unpresentable = n_gt
    base['n_gt_unpresentable'] = n_gt_unpresentable

    if n_identified == 0:
        return base

    # ── 匈牙利全局最优位置匹配 ──
    n_g, n_s = n_gt, n_identified
    cost = np.full((n_g, n_s), 1e9, dtype=np.float64)
    for i, (_gt_id, gt_x, gt_y) in enumerate(stars_gt):
        for j, (sx, sy, _) in enumerate(sys_stars):
            d = np.sqrt((gt_x - sx) ** 2 + (gt_y - sy) ** 2)
            if d <= match_radius:
                cost[i, j] = d

    row_ind, col_ind = linear_sum_assignment(cost)
    matched_pairs = [(int(row_ind[k]), int(col_ind[k])) for k in range(len(row_ind))
                     if cost[row_ind[k], col_ind[k]] < 1e9]

    tp = len(matched_pairs)
    fn = n_gt - tp
    fp = n_identified - tp

    pos_errors = [float(cost[i, j]) for i, j in matched_pairs]
    mean_pos_err = float(np.mean(pos_errors)) if pos_errors else 0.0

    # ── 星号校验 ──
    n_id_correct = 0
    n_id_wrong = 0
    for gt_i, sys_j in matched_pairs:
        gt_star_id = int(stars_gt[gt_i][0])
        if gt_star_id == 0:
            continue
        sys_cat_id = int(sys_stars[sys_j][2])
        if sys_cat_id == gt_star_id:
            n_id_correct += 1
        else:
            n_id_wrong += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(n_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'n_id_correct': n_id_correct, 'n_id_wrong': n_id_wrong,
        'n_gt_unpresentable': n_gt_unpresentable,
        'precision': precision, 'recall': recall, 'f1': f1,
        'position_error': mean_pos_err,
        'n_gt': n_gt, 'n_identified': n_identified,
    }


def evaluate_star_removal_frame(candidates, target_candidates, stars_gt,
                                match_radius=5.0):
    """评估最终候选筛选对“已被检测到的恒星”的剔除效果。

    GT 恒星先与原始检测点做一对一位置匹配；只对确实形成了
    检测点的 GT 恒星计算剔除效果，避免把前端漏检错算成筛选成功。
    """
    n_gt = len(stars_gt)
    if not candidates or not stars_gt:
        return {
            'n_gt': n_gt, 'n_detected_stars': 0,
            'n_gt_undetected': n_gt, 'n_correctly_removed': 0,
            'n_leaked_stars': 0, 'removal_recall': 0.0,
            'leakage_rate': 0.0,
        }

    gt_xy = np.array([[float(s[1]), float(s[2])] for s in stars_gt],
                     dtype=np.float64)
    cand_xy = np.array([[float(c[0]), float(c[1])] for c in candidates],
                       dtype=np.float64)
    cost = np.linalg.norm(gt_xy[:, None, :] - cand_xy[None, :, :], axis=2)
    gated_cost = np.where(cost <= match_radius, cost, 1e9)
    rows, cols = linear_sum_assignment(gated_cost)
    detected_indices = [int(c) for r, c in zip(rows, cols)
                        if gated_cost[r, c] < 1e8]

    target_keys = {(round(float(c[0]), 3), round(float(c[1]), 3))
                   for c in target_candidates}
    leaked = 0
    for idx in detected_indices:
        cand = candidates[idx]
        key = (round(float(cand[0]), 3), round(float(cand[1]), 3))
        if key in target_keys:
            leaked += 1

    n_detected = len(detected_indices)
    removed = n_detected - leaked
    return {
        'n_gt': n_gt,
        'n_detected_stars': n_detected,
        'n_gt_undetected': n_gt - n_detected,
        'n_correctly_removed': removed,
        'n_leaked_stars': leaked,
        'removal_recall': removed / max(n_detected, 1),
        'leakage_rate': leaked / max(n_detected, 1),
    }


def evaluate_targets_frame(tracks, frame_idx, targets_gt, match_radius=10.0):
    """评估单帧目标跟踪准确率。

    参数
    ----
    tracks : List[dict]  当前所有轨迹（含 active/tentative/pruned）
    frame_idx : int  帧号
    targets_gt : List[(target_id, x, y)]  该帧目标真值
    match_radius : float  空间匹配半径（像素）

    返回
    ----
    dict  {tp, fp, fn, precision, recall, f1, position_error, n_gt, n_tracked}
    """
    n_gt = len(targets_gt)

    # 获取各轨迹在当前帧的位置: {track_id: (x, y)}
    track_positions = {}
    for t in tracks:
        sf = t.get('start_frame', 0)
        frames = t.get('frames')
        if (frames is not None and len(frames) == len(t['positions']) and
                all(f is not None for f in frames)):
            for idx, fid in enumerate(frames):
                if int(fid) == int(frame_idx):
                    track_positions[t['id']] = t['positions'][idx]
                    break
        else:
            idx = frame_idx - sf
            if 0 <= idx < len(t['positions']):
                track_positions[t['id']] = t['positions'][idx]

    n_tracked = len(track_positions)

    if n_gt == 0:
        return {'tp': 0, 'fp': n_tracked, 'fn': 0,
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'position_error': 0.0, 'n_gt': 0, 'n_tracked': n_tracked}

    if n_tracked == 0:
        return {'tp': 0, 'fp': 0, 'fn': n_gt,
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'position_error': 0.0, 'n_gt': n_gt, 'n_tracked': 0}

    # 匈牙利算法最优匹配
    gt_ids = [t[0] for t in targets_gt]
    gt_coords = {t[0]: (t[1], t[2]) for t in targets_gt}
    tk_ids = list(track_positions.keys())
    tk_coords = track_positions

    n_g, n_t = len(gt_ids), len(tk_ids)
    cost = np.zeros((n_g, n_t))
    for i, gid in enumerate(gt_ids):
        gx, gy = gt_coords[gid]
        for j, tid in enumerate(tk_ids):
            tx, ty = tk_coords[tid]
            d = np.sqrt((gx - tx) ** 2 + (gy - ty) ** 2)
            cost[i, j] = d if d <= match_radius else 1e9

    row_ind, col_ind = linear_sum_assignment(cost)
    matched_pairs = [(row_ind[i], col_ind[i]) for i in range(len(row_ind))
                     if cost[row_ind[i], col_ind[i]] < 1e8]

    tp = len(matched_pairs)
    fn = n_gt - tp
    fp = n_tracked - tp

    pos_errors = [cost[i, j] for i, j in matched_pairs]
    mean_pos_err = float(np.mean(pos_errors)) if pos_errors else 0.0

    precision = tp / max(tp + fp, 1)
    recall = tp / max(n_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision, 'recall': recall, 'f1': f1,
            'position_error': mean_pos_err,
            'n_gt': n_gt, 'n_tracked': n_tracked}


# ══════════════════════════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════════════════════════

def summarize(star_results, target_results):
    """汇总逐帧评估结果为总体指标。

    参数
    ----
    star_results : List[dict]   每帧恒星评估结果
    target_results : List[dict] 每帧目标评估结果

    返回
    ----
    dict  {star: {...}, target: {...}}
    """
    def _agg(results):
        if not results:
            return {}
        total_tp = sum(r['tp'] for r in results)
        total_fp = sum(r['fp'] for r in results)
        total_fn = sum(r['fn'] for r in results)
        total_gt = sum(r.get('n_gt', 0) for r in results)
        total_det = sum(r.get('n_identified', r.get('n_tracked', 0)) for r in results)
        total_id_correct = sum(r.get('n_id_correct', 0) for r in results)
        total_id_wrong = sum(r.get('n_id_wrong', 0) for r in results)
        total_unpresentable = sum(r.get('n_gt_unpresentable', 0) for r in results)
        errors = [r['position_error'] for r in results
                  if r['position_error'] > 0]
        precision = total_tp / max(total_tp + total_fp, 1)
        recall = total_tp / max(total_gt, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        return {
            'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
            'precision': precision, 'recall': recall, 'f1': f1,
            'position_error': float(np.mean(errors)) if errors else 0.0,
            'n_frames_evaluated': len(results),
            'n_gt_total': total_gt,
            'n_id_correct': total_id_correct,
            'n_id_wrong': total_id_wrong,
            'n_gt_unpresentable': total_unpresentable,
        }
    return {'star': _agg(star_results), 'target': _agg(target_results)}


# ══════════════════════════════════════════════════════════════════════
# 保留的旧接口（向后兼容）
# ══════════════════════════════════════════════════════════════════════

def bipartite_match(gt_pos, est_pos, dist_threshold=None):
    """二分图匹配，将 GT 位置与估计位置关联。

    返回 List[(gt_key, est_key)]
    """
    if dist_threshold is None:
        dist_threshold = dist_thresh

    if not gt_pos or not est_pos:
        return []

    gt_keys = list(gt_pos.keys())
    est_keys = list(est_pos.keys())

    n_gt = len(gt_keys)
    n_est = len(est_keys)

    cost = np.zeros((n_gt, n_est))
    for i, gk in enumerate(gt_keys):
        gx, gy = gt_pos[gk]
        for j, ek in enumerate(est_keys):
            ex, ey = est_pos[ek]
            d = np.sqrt((gx - ex) ** 2 + (gy - ey) ** 2)
            cost[i, j] = d if d <= dist_threshold else 1e9

    row_ind, col_ind = linear_sum_assignment(cost)

    matched = []
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] < 1e8:
            matched.append((gt_keys[i], est_keys[j]))

    return matched


def calc_mota_motp(track_results, ground_truth, dist_threshold=None, N_pixel_norm_val=None):
    """计算 MOTA 和 MOTP 指标。

    参数
    ----
    track_results : dict {frame_id: {track_id: (x, y)}}
    ground_truth  : dict {frame_id: {target_id: (x, y)}}

    返回 (MOTA, MOTP, stats)
    """
    if dist_threshold is None:
        dist_threshold = dist_thresh
    if N_pixel_norm_val is None:
        N_pixel_norm_val = N_pixel_norm

    total_gt = 0
    FP = 0
    FN = 0
    dist_sum = 0.0
    N_matched = 0

    for fid in ground_truth:
        gt_pos = ground_truth[fid]
        est_pos = track_results.get(fid, {})

        matched = bipartite_match(gt_pos, est_pos, dist_threshold)

        FN += len(gt_pos) - len(matched)
        FP += len(est_pos) - len(matched)
        total_gt += len(gt_pos)

        for gk, ek in matched:
            gx, gy = gt_pos[gk]
            ex, ey = est_pos[ek]
            dist_sum += np.sqrt((gx - ex) ** 2 + (gy - ey) ** 2)
            N_matched += 1

    MOTA = max(1.0 - (FN + FP) / max(total_gt, 1), 0.0)
    MOTP = max(1.0 - dist_sum / max(N_matched * N_pixel_norm_val, 1), 0.0)

    stats = {
        'total_gt': total_gt,
        'FP': FP,
        'FN': FN,
        'N_matched': N_matched,
        'dist_sum': dist_sum,
        'MOTA': MOTA,
        'MOTP': MOTP,
    }

    return MOTA, MOTP, stats
