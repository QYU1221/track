# Domain implementation: per-frame track lifecycle and association.
import numpy as np
from configuration.values import (eta, tentative_min,
                     mahalanobis_gate, R_diag,
                     prune_T_min, prune_T_max, prune_conf_weights, prune_hist_len,
                     prune_n_confirm, in_fov_timeout,
                     jpda_hit_threshold, jpda_new_track_threshold,
                     FRAME_W, FRAME_H,
                     use_imm, max_velocity,
                     candidate_trusted_radius,
                     enable_in_fov_reconnect, in_fov_reconnect_min_hits,
                     in_fov_reconnect_max_gap, in_fov_reconnect_min_conf,
                     in_fov_gate_growth, in_fov_gate_max_scale,
                     in_fov_reconnect_score_thresh,
                     fill_missing_with_prediction,
                     enable_fallback_association,
                     fallback_mahalanobis_gate,
                     fallback_min_cosine,
                     fallback_max_velocity_scale,
                     strict_one_update_per_frame,
                     max_record_jump_scale,
                     triangle_star_evidence_frames)
from .association_impl import sinkhorn_soft_assignment


_EPS = 1e-8


def _get_track_velocity(t):
    """获取轨迹速度 (vx, vy)，兼容 4D/5D 状态格式"""
    s = t['state']
    if len(s) == 4:
        return float(s[2]), float(s[3])
    elif len(s) >= 5:
        v, theta = float(s[3]), float(s[2])
        return float(v * np.cos(theta)), float(v * np.sin(theta))
    return 0.0, 0.0


def _track_speed_theta(t):
    """从位置历史推算速度和方向（比状态向量更稳定）"""
    p = t['positions']
    if len(p) >= 2:
        dx = p[-1][0] - p[-2][0]
        dy = p[-1][1] - p[-2][1]
        speed = float(np.sqrt(dx**2 + dy**2))
        theta = float(np.arctan2(dy, dx))
        return speed, theta
    else:
        vx, vy = _get_track_velocity(t)
        speed = float(np.sqrt(vx**2 + vy**2))
        theta = float(np.arctan2(vy, vx))
        return speed, theta


def _add_track_event(t, frame, event, x=None, y=None, reason='', score=None):
    """记录轨迹管理中的拒绝/接受事件，便于排查乱连来源。"""
    row = {
        'frame': frame,
        'type': event,
        'x': None if x is None else float(x),
        'y': None if y is None else float(y),
        'reason': reason,
    }
    if score is not None:
        row['score'] = float(score)
    t.setdefault('event_hist', []).append(row)


def _snapshot_filter(tracker):
    """保存滤波器关键状态；用于记录被拒绝时回滚一次更新。"""
    snap = {
        'X': tracker.X.copy(),
        'P': tracker.P.copy(),
    }
    if hasattr(tracker, 'cv') and hasattr(tracker, 'ctrv'):
        snap.update({
            'cv_X': tracker.cv.X.copy(),
            'cv_P': tracker.cv.P.copy(),
            'ctrv_X': tracker.ctrv.X.copy(),
            'ctrv_P': tracker.ctrv.P.copy(),
            '_X_cv': tracker._X_cv.copy(),
            '_P_cv': tracker._P_cv.copy(),
            '_X_ctrv': tracker._X_ctrv.copy(),
            '_P_ctrv': tracker._P_ctrv.copy(),
            'mu': tracker.mu.copy(),
            '_cbar': tracker._cbar.copy(),
        })
    return snap


def _restore_filter(tracker, snap):
    """恢复 _snapshot_filter 保存的滤波器状态。"""
    tracker.X = snap['X'].copy()
    tracker.P = snap['P'].copy()
    if 'cv_X' in snap:
        tracker.cv.X = snap['cv_X'].copy()
        tracker.cv.P = snap['cv_P'].copy()
        tracker.ctrv.X = snap['ctrv_X'].copy()
        tracker.ctrv.P = snap['ctrv_P'].copy()
        tracker._X_cv = snap['_X_cv'].copy()
        tracker._P_cv = snap['_P_cv'].copy()
        tracker._X_ctrv = snap['_X_ctrv'].copy()
        tracker._P_ctrv = snap['_P_ctrv'].copy()
        tracker.mu = snap['mu'].copy()
        tracker._cbar = snap['_cbar'].copy()


def _mahalanobis_d2(track, cx, cy, camera_info=None):
    """返回马氏距离平方 d2，所有门控阈值均按 d2 解释。"""
    pred_x, pred_y = track['state'][0], track['state'][1]
    if camera_info is not None:
        gate_scale = getattr(camera_info, 'mahalanobis_gate_scale', 1.0)
    else:
        gate_scale = 1.0
    S = track['P'][:2, :2] + np.diag(R_diag)
    z_diff = np.array([cx - pred_x, cy - pred_y], dtype=np.float64)
    try:
        d2 = z_diff @ np.linalg.inv(S) @ z_diff
        return float(max(d2 / max(gate_scale, _EPS), 0.0))
    except np.linalg.LinAlgError:
        diag = np.maximum(np.diag(S), np.asarray(R_diag, dtype=np.float64))
        diag = np.maximum(diag, _EPS)
        return float(np.sum((z_diff ** 2) / diag))


def _calc_confidence(t):
    """计算轨迹综合置信度 C ∈ [0, 1]

    四个维度加权融合：
      S_len    存活长度   —— 跟踪帧数越多越可信
      S_assoc  关联质量   —— 历史 Phi 均值，匹配越稳定越可信
      S_motion 运动一致性 —— 速度/方向越平稳越可信（真目标运动连续）
      S_photo  光度一致性 —— 能量越稳定越可信（同一目标亮度不突变）

    权重由 config.prune_conf_weights 控制，默认 (0.30, 0.30, 0.20, 0.20)。
    """
    w_len, w_assoc, w_motion, w_photo = prune_conf_weights

    # 维度1：存活长度
    n_hits = t.get('hit_count', len(t['positions']))
    S_len = min(n_hits / max(prune_n_confirm, 1), 1.0)

    # 维度2：关联质量（历史 Phi 均值，存入 assoc_hist）
    assoc_hist = t.get('assoc_hist', [])
    S_assoc = float(np.mean(assoc_hist)) if assoc_hist else 0.3

    # 维度3：运动一致性（速度标准差/均值 + 方向标准差/π）
    v_hist    = t.get('v_hist', [])
    theta_hist = t.get('theta_hist', [])
    if len(v_hist) >= 3:
        v_arr = np.array(v_hist, dtype=np.float64)
        t_arr = np.array(theta_hist, dtype=np.float64)
        sigma_v = v_arr.std()
        v_mean  = v_arr.mean() + 1e-8
        sigma_t = t_arr.std()
        S_motion = float(np.exp(-(sigma_v / v_mean) ** 2
                                - (sigma_t / np.pi) ** 2))
    else:
        S_motion = 0.5  # 历史不足，取中间值

    # 维度4：光度一致性（能量相对标准差）
    E_hist = [e for e in t.get('energy_hist', []) if e > 0]
    if len(E_hist) >= 3:
        E_arr  = np.array(E_hist, dtype=np.float64)
        S_photo = float(1.0 - E_arr.std() / (E_arr.mean() + 1e-8))
        S_photo = max(0.0, min(1.0, S_photo))
    else:
        S_photo = 0.5

    C = w_len * S_len + w_assoc * S_assoc + w_motion * S_motion + w_photo * S_photo
    return float(np.clip(C, 0.0, 1.0))


def _adaptive_prune_thresh(confidence):
    """根据置信度线性插值计算个性化剪枝容忍帧数

    confidence=0 → prune_T_min（默认3帧，快速清除噪声）
    confidence=1 → prune_T_max（默认25帧，充分保护高质量目标）
    """
    return int(round(prune_T_min + (prune_T_max - prune_T_min) * confidence))


def _append_hist(lst, val, maxlen):
    """向历史列表追加值并限制长度（原地操作）"""
    lst.append(val)
    if len(lst) > maxlen:
        lst.pop(0)


def _is_in_fov_xy(x, y):
    """判断像点是否位于当前图像视场内。"""
    return 0 <= float(x) < FRAME_W and 0 <= float(y) < FRAME_H




# ── 飞出视场后再入重连：几何与历史摘要工具 ─────────────────────
def _distance_to_border(x, y):
    """计算点到最近图像边界的距离。"""
    x, y = float(x), float(y)
    return float(min(x, FRAME_W - 1 - x, y, FRAME_H - 1 - y))


def _get_exit_edge(x, y):
    """根据点的位置判断最近边界：left/right/top/bottom。"""
    x, y = float(x), float(y)
    d = {
        'left': x,
        'right': FRAME_W - 1 - x,
        'top': y,
        'bottom': FRAME_H - 1 - y,
    }
    return min(d, key=d.get)


def _border_inner_normal(edge):
    """返回边界指向图像内部的单位法向量。"""
    if edge == 'left':
        return np.array([1.0, 0.0], dtype=np.float64)
    if edge == 'right':
        return np.array([-1.0, 0.0], dtype=np.float64)
    if edge == 'top':
        return np.array([0.0, 1.0], dtype=np.float64)
    if edge == 'bottom':
        return np.array([0.0, -1.0], dtype=np.float64)
    return np.zeros(2, dtype=np.float64)


def _opposite_edge(edge):
    return {'left': 'right', 'right': 'left', 'top': 'bottom', 'bottom': 'top'}.get(edge)


def _boundary_coordinate(point, edge):
    """把矩形边界点映射到顺时针周长坐标。"""
    x = float(np.clip(point[0], 0.0, FRAME_W - 1.0))
    y = float(np.clip(point[1], 0.0, FRAME_H - 1.0))
    width = float(FRAME_W - 1)
    height = float(FRAME_H - 1)
    if edge == 'top':
        return x
    if edge == 'right':
        return width + y
    if edge == 'bottom':
        return width + height + (width - x)
    if edge == 'left':
        return 2.0 * width + height + (height - y)
    return None


def _outside_boundary_distance(exit_point, exit_edge, entry_point, entry_edge):
    """估计目标在视场外从飞出点到再入点至少需要走过的距离。"""
    exit_s = _boundary_coordinate(exit_point, exit_edge)
    entry_s = _boundary_coordinate(entry_point, entry_edge)
    if exit_s is None or entry_s is None:
        return None
    perimeter = 2.0 * ((FRAME_W - 1.0) + (FRAME_H - 1.0))
    direct = abs(entry_s - exit_s)
    return float(min(direct, perimeter - direct))


def _safe_recent_values(values, k=5):
    vals = [float(v) for v in (values or [])[-k:]
            if v is not None and np.isfinite(float(v))]
    return vals


def _safe_recent_mean(values, k=5, default=0.0):
    vals = _safe_recent_values(values, k=k)
    return float(np.mean(vals)) if vals else float(default)


def _safe_recent_std(values, k=5, default=0.0):
    vals = _safe_recent_values(values, k=k)
    return float(np.std(vals)) if len(vals) >= 2 else float(default)


def _track_measured_ratio(track):
    """计算真实测量/恢复点占比，过滤主要由预测点组成的伪轨迹。"""
    sources = track.get('point_sources') or []
    if not sources:
        return 1.0
    good = sum(1 for s in sources if s in ('measured', 'recovered', 'reconnected'))
    return float(good / max(len(sources), 1))


def _new_track_measurement_quality(track):
    """返回再入短轨迹的真实观测质量，预测点不能充当确认次数。"""
    frames = track.get('frames') or []
    sources = track.get('point_sources') or []
    positions = track.get('positions') or []
    measured_sources = {'measured', 'recovered', 'reconnected'}

    measured_frames = []
    for index in range(min(len(frames), len(positions))):
        source = sources[index] if index < len(sources) else 'measured'
        frame_id = frames[index]
        if source in measured_sources and frame_id is not None:
            measured_frames.append(int(frame_id))
    measured_frames = sorted(set(measured_frames))

    max_gap = 0
    if len(measured_frames) >= 2:
        max_gap = max(b - a for a, b in zip(measured_frames[:-1], measured_frames[1:]))
    measured_ratio = float(len(measured_frames) / max(len(positions), 1))
    return {
        'measured_hits': len(measured_frames),
        'measured_ratio': measured_ratio,
        'max_observation_gap': max_gap,
        'latest_measured_frame': measured_frames[-1] if measured_frames else None,
    }


def _vector_from_recent_positions(track, k=3):
    """用最近位置估计速度向量。"""
    pts = track.get('positions') or []
    frames = track.get('frames') or []
    if len(pts) >= 2:
        i0 = max(0, len(pts) - k)
        p0 = np.array(pts[i0], dtype=np.float64)
        p1 = np.array(pts[-1], dtype=np.float64)
        if len(frames) == len(pts) and frames[i0] is not None and frames[-1] is not None:
            dt = max(int(frames[-1]) - int(frames[i0]), 1)
        else:
            dt = max(len(pts) - 1 - i0, 1)
        v = (p1 - p0) / float(dt)
        return float(v[0]), float(v[1])
    return _get_track_velocity(track)


def _track_entry_frame(track):
    frames = track.get('frames') or []
    return int(frames[0]) if frames and frames[0] is not None else int(track.get('start_frame', 0))


def _record_track_point(t, x, y, frame, source):
    """记录轨迹点，并保持 positions/frames/point_sources 一一对应。

    同一帧若已有 predicted 点，measured/recovered 可替换；若同一帧已有
    measured/recovered，则拒绝重复写入，避免轨迹时间轴错乱。
    """
    positions = t.setdefault('positions', [])
    n_old = len(positions)

    # 补齐旧轨迹的 frames。
    if 'frames' not in t or t.get('frames') is None:
        sf = int(t.get('start_frame', 0))
        t['frames'] = [sf + i for i in range(n_old)]
    elif len(t['frames']) < n_old:
        last = t['frames'][-1] if t['frames'] else None
        if last is None:
            last = int(t.get('start_frame', 0)) + len(t['frames']) - 1
        t['frames'].extend([last + i + 1 for i in range(n_old - len(t['frames']))])
    elif len(t['frames']) > n_old:
        del t['frames'][n_old:]

    # 补齐旧轨迹的 point_sources。
    if 'point_sources' not in t or t.get('point_sources') is None:
        t['point_sources'] = ['measured'] * n_old
    elif len(t['point_sources']) < n_old:
        t['point_sources'].extend(['measured'] * (n_old - len(t['point_sources'])))
    elif len(t['point_sources']) > n_old:
        del t['point_sources'][n_old:]

    if frame is None:
        if t['frames']:
            last = t['frames'][-1]
            frame = (int(t.get('start_frame', 0)) + len(t['frames'])
                     if last is None else int(last) + 1)
        else:
            frame = int(t.get('start_frame', 0))
    frame = int(frame) if frame is not None else None

    x = float(x)
    y = float(y)
    priority = {
        'predicted': 0,
        'extrapolated_exit': 0,
        'exit_predicted': 0,
        'measured': 1,
        'recovered': 2,
        'reconnected': 2,
    }

    def _jump_ok(ref_idx):
        if source not in ('measured', 'recovered', 'reconnected') or ref_idx < 0:
            return True
        ref_frame = t['frames'][ref_idx]
        if ref_frame is None or frame is None:
            return True
        frame_gap = frame - int(ref_frame)
        if frame_gap > 1:
            return True
        px, py = positions[ref_idx]
        dist = float(np.hypot(x - px, y - py))
        limit = max_velocity * max_record_jump_scale
        if dist > limit:
            _add_track_event(
                t, frame, 'reject_jump', x, y,
                'jump %.2f > %.2f from frame %s' % (dist, limit, ref_frame))
            return False
        return True

    same_frame = bool(positions and t['frames'] and t['frames'][-1] == frame)
    if same_frame and strict_one_update_per_frame:
        last_source = t['point_sources'][-1]
        old_pri = priority.get(last_source, 0)
        new_pri = priority.get(source, 0)
        if last_source in ('measured', 'recovered', 'reconnected') and source in ('measured', 'recovered', 'reconnected'):
            if new_pri <= old_pri:
                _add_track_event(
                    t, frame, 'reject_duplicate', x, y,
                    'same-frame %s already exists' % last_source)
                return False
        if new_pri <= old_pri:
            _add_track_event(
                t, frame, 'reject_duplicate', x, y,
                'same-frame %s not replaced by %s' % (last_source, source))
            return False
        if not _jump_ok(len(positions) - 2):
            return False
        positions[-1] = (x, y)
        t['point_sources'][-1] = source
        t['last_update_frame'] = frame
        if source in ('measured', 'recovered', 'reconnected'):
            t['gap_start_frame'] = None
        return True

    if not _jump_ok(len(positions) - 1):
        return False

    positions.append((x, y))
    t['frames'].append(frame)
    t['point_sources'].append(source)
    t['last_update_frame'] = frame
    if source in ('measured', 'recovered', 'reconnected'):
        t['gap_start_frame'] = None
    return True

def _can_coast_in_fov(t):
    """判断轨迹是否具备视场内短时滑行等待重连的基础条件。"""
    if not enable_in_fov_reconnect:
        return False
    if t.get('hit_count', 0) < in_fov_reconnect_min_hits:
        return False
    if t.get('confidence', 0.0) < in_fov_reconnect_min_conf:
        return False
    if t.get('miss_count', 0) >= in_fov_reconnect_max_gap:
        return False
    return _is_in_fov_xy(t['state'][0], t['state'][1])


def _score_in_fov_reconnect(t, cx, cy, cand_energy=None, cand_radius=None):
    """对视场内漏检后的候选重连进行软评分，返回 [0, 1]。

    位置项权重最高；miss_count 越大门控适度放宽，但 scale 不超过
    in_fov_gate_max_scale。该函数只提供第一阶段数据结构和评分能力，
    暂不接管 update_all 主流程。
    """
    if not _is_in_fov_xy(cx, cy):
        return 0.0

    miss_count = max(int(t.get('miss_count', 0)), 0)
    gate_scale = min(1.0 + in_fov_gate_growth * miss_count,
                     in_fov_gate_max_scale)
    gate_d2 = max(mahalanobis_gate * gate_scale, _EPS)
    d2 = _mahalanobis_d2(t, cx, cy)
    S_pos = float(np.exp(-0.5 * d2 / gate_d2))

    pred_x, pred_y = float(t['state'][0]), float(t['state'][1])
    d_eucl = float(np.sqrt((pred_x - cx) ** 2 + (pred_y - cy) ** 2))
    motion_gate = max(max_velocity * gate_scale, _EPS)
    S_motion = float(np.exp(-0.5 * (d_eucl / motion_gate) ** 2))

    S_photo = 0.5
    if cand_energy is not None and t.get('energy_hist'):
        old_energy = t['energy_hist'][-1]
        if old_energy > 0 and cand_energy > 0:
            delta = abs(np.log((float(cand_energy) + _EPS) /
                               (float(old_energy) + _EPS)))
            S_photo = float(np.exp(-0.5 * (delta / 0.75) ** 2))

    S_size = 0.5
    if cand_radius is not None and t.get('radius_hist'):
        old_radius = t['radius_hist'][-1]
        if old_radius > 0 and cand_radius > 0:
            delta = abs(np.log((float(cand_radius) + _EPS) /
                               (float(old_radius) + _EPS)))
            S_size = float(np.exp(-0.5 * (delta / 0.60) ** 2))

    score = 0.65 * S_pos + 0.20 * S_motion + 0.075 * S_photo + 0.075 * S_size
    return float(np.clip(score, 0.0, 1.0))


def _mark_track_missed(t, frame):
    """处理单帧未命中：视场内短时滑行，否则沿用原漏检/出界逻辑。"""
    pred_x, pred_y = float(t['state'][0]), float(t['state'][1])
    in_fov = _is_in_fov_xy(pred_x, pred_y)

    if in_fov and enable_in_fov_reconnect and _can_coast_in_fov(t):
        t['status'] = 'lost_in_fov'
        t['miss_count'] = t.get('miss_count', 0) + 1
        if t.get('gap_start_frame') is None:
            t['gap_start_frame'] = frame
        if fill_missing_with_prediction:
            _record_track_point(t, pred_x, pred_y, frame, 'predicted')
        return

    t['miss_count'] = t.get('miss_count', 0) + 1
    _check_edge_exit(t)


def _try_reconnect_lost_in_fov(tracks, candidate, frame, energy=None, radius=None):
    """尝试把未关联候选回接到 lost_in_fov 轨迹，成功时保持原轨迹 ID。"""
    cx, cy, _ = candidate
    best_track = None
    best_score = 0.0

    for t in tracks:
        if t.get('status') != 'lost_in_fov':
            continue
        if t.get('miss_count', 0) > in_fov_reconnect_max_gap:
            continue
        score = _score_in_fov_reconnect(t, cx, cy,
                                        cand_energy=energy,
                                        cand_radius=radius)
        if score > best_score:
            best_score = score
            best_track = t

    if best_track is None or best_score < in_fov_reconnect_score_thresh:
        return False

    ok = _record_track_point(best_track, cx, cy, frame, 'recovered')
    if not ok:
        return False

    best_track['ekf'].set_position(cx, cy)
    best_track['state'] = best_track['ekf'].X.copy()
    best_track['P'] = best_track['ekf'].P.copy()
    best_track['miss_count'] = 0
    best_track['edge_miss'] = 0
    best_track['gap_start_frame'] = None
    best_track['status'] = 'active'
    best_track['recovered_count'] = best_track.get('recovered_count', 0) + 1
    best_track['hit_count'] = best_track.get('hit_count', 0) + 1
    best_track.setdefault('energy_hist', []).append(float(energy) if energy is not None else 0.0)
    best_track.setdefault('radius_hist', []).append(float(radius) if radius is not None else 0.0)
    _append_hist(best_track.setdefault('assoc_hist', []),
                 float(best_score), prune_hist_len)
    speed, theta = _track_speed_theta(best_track)
    _append_hist(best_track.setdefault('v_hist', []), speed, prune_hist_len)
    _append_hist(best_track.setdefault('theta_hist', []), theta, prune_hist_len)
    _add_track_event(best_track, frame, 'lost_in_fov_recovered',
                     cx, cy, 'score %.3f' % best_score, best_score)
    return True


# 全局轨迹 ID 分配器
_track_id_counter = 0


def _next_id():
    global _track_id_counter
    _track_id_counter += 1
    return _track_id_counter


def create_track(x, y, vx=0.0, vy=0.0, frame=None):
    """创建新轨迹"""
    if use_imm:
        from .imm import IMMTracker
        tracker = IMMTracker(x, y, vx0=vx, vy0=vy)
    else:
        from .ctrv_ekf import EKFTracker
        v = np.sqrt(vx**2 + vy**2) if (vx != 0.0 or vy != 0.0) else 2.0
        theta = np.arctan2(vy, vx) if (vx != 0.0 or vy != 0.0) else 0.0
        tracker = EKFTracker(x, y, theta0=theta, v0=v, omega0=0.0)

    track = {
        'id': _next_id(),
        'state': tracker.X.copy(),
        'P': tracker.P.copy(),
        'ekf': tracker,
        'positions': [(x, y)],
        'frames': [frame],
        'point_sources': ['measured'],
        'gap_start_frame': None,
        'recovered_count': 0,
        'last_update_frame': frame,
        'start_frame': frame if frame is not None else 0,
        'energy_hist': [],
        'radius_hist': [],
        'miss_count': 0,
        'status': 'tentative',
        'tentative_count': 1,
        'hit_count': 1,
        'assoc_hist': [],
        'v_hist': [],
        'theta_hist': [],
        'event_hist': [],
        'confidence': 0.3,
        'star_evidence_count': 0,
        'triangle_star_evidence_count': 0,
        'nonstar_evidence_count': 0,
        'object_class': 'unknown',
        'edge_miss': 0,
    }
    return track


def _check_edge_exit(t):
    """检测轨迹预测位置是否已飞出画面"""
    px, py = t['state'][0], t['state'][1]
    outside = (px < 0 or px >= FRAME_W or py < 0 or py >= FRAME_H)
    if outside:
        t.setdefault('edge_miss', 0)
        t['edge_miss'] += 1


def _extrapolate_exit_point(t, frame=None):
    """外推轨迹最后位置到画面边界交点。

    长间隔飞出-再入重连中，exit_point 只是轨迹记忆库的摘要特征，
    不再直接写入 positions，避免 positions/frames/point_sources 长度不一致，
    也避免把预测边界点误当作真实观测点参与评估。
    """
    if not t.get('positions'):
        return None
    lx, ly = t['positions'][-1]
    vx, vy = _get_track_velocity(t)
    best_t = float('inf')
    best_x, best_y = lx, ly

    # 与左右边界求交
    for bx in (0.0, float(FRAME_W - 1)):
        if abs(vx) > 1e-8:
            tc = (bx - lx) / vx
            if 0 < tc < best_t:
                yc = ly + vy * tc
                if 0 <= yc < FRAME_H:
                    best_t, best_x, best_y = tc, bx, yc

    # 与上下边界求交
    for by in (0.0, float(FRAME_H - 1)):
        if abs(vy) > 1e-8:
            tc = (by - ly) / vy
            if 0 < tc < best_t:
                xc = lx + vx * tc
                if 0 <= xc < FRAME_W:
                    best_t, best_x, best_y = tc, xc, by

    if best_t < float('inf'):
        return (float(best_x), float(best_y))

    # 速度方向不可用时，用最后位置夹紧到边界作为保守摘要。
    return (float(np.clip(lx, 0, FRAME_W - 1)),
            float(np.clip(ly, 0, FRAME_H - 1)))


def update_all(tracks, candidates, Phi, frame=None, energy_func=None, radius_func=None,
               camera_info=None, energies=None, radii=None):
    """Sinkhorn 软分配 + JPDA-lite 概率关联

    流程：
      1. Sinkhorn-Knopp 将 Phi 转化为软分配概率矩阵 P
      2. 马氏距离 + 概率双重门控筛选有效候选对
      3. JPDA-lite 对各轨迹加权融合所有门内候选更新 EKF
      4. 低关联质量的候选创建新轨迹
      5. 置信度驱动自适应剪枝

    参数
    ----
    energies, radii : ndarray or None
        CSV 输入时预计算的候选点能量/半径，优先于 energy_func/radius_func。

    返回
    ----
    active_tracks : List[track_dict]
    pruned_tracks  : List[track_dict]
    """
    n_tracks = len(tracks)
    n_cands = len(candidates)

    # ── Step 1: Sinkhorn 软分配 ────────────────────────────
    if n_tracks > 0 and n_cands > 0:
        P = sinkhorn_soft_assignment(Phi)
    else:
        P = Phi  # 0 矩阵

    # ── Step 2 & 3: 门控 + JPDA-lite 加权 EKF 更新 ─────────
    track_assoc_mass = np.zeros(n_tracks)  # Σ_j P[i,j] per track
    cand_claimed_mass = np.zeros(n_cands)   # Σ_i P[i,j] per candidate

    if n_tracks > 0 and n_cands > 0:
        for i, t in enumerate(tracks):
            gated_meas = []  # [(cx, cy), ...]
            gated_w = []     # [weight, ...]
            gated_idx = []   # 原 candidates 下标

            for j in range(n_cands):
                cx, cy, _ = candidates[j]
                d2 = _mahalanobis_d2(t, cx, cy, camera_info=camera_info)
                # 自适应门控：Sinkhorn 高置信度时放宽马氏距离限制
                effective_gate_d2 = mahalanobis_gate * min(1.0 + 0.5 * P[i, j], 1.5)
                # 速度硬上限：帧间位移超过 max_velocity 视为物理不可能
                d_eucl = np.sqrt((t['state'][0] - cx) ** 2 + (t['state'][1] - cy) ** 2)
                if d2 < effective_gate_d2 and d_eucl <= max_velocity and P[i, j] > eta:
                    gated_meas.append((cx, cy))
                    gated_w.append(P[i, j])
                    gated_idx.append(j)
                    cand_claimed_mass[j] += P[i, j]

            track_assoc_mass[i] = sum(gated_w)

            if track_assoc_mass[i] >= jpda_hit_threshold and gated_meas:
                # JPDA-lite 加权概率融合更新
                old_status = t.get('status')
                filter_snap = _snapshot_filter(t['ekf'])
                old_fields = {
                    'status': t.get('status'),
                    'miss_count': t.get('miss_count', 0),
                    'edge_miss': t.get('edge_miss', 0),
                    'gap_start_frame': t.get('gap_start_frame'),
                    'tentative_count': t.get('tentative_count', 0),
                    'hit_count': t.get('hit_count', 0),
                }
                t['ekf'].update_jpda(gated_meas, gated_w)
                t['state'] = t['ekf'].X.copy()
                t['P'] = t['ekf'].P.copy()
                t['miss_count'] = 0
                if t['status'] == 'tentative':
                    t['tentative_count'] += 1
                    if t['tentative_count'] >= tentative_min:
                        t['status'] = 'active'
                if old_status in ('lost_in_fov', 'recovered'):
                    t['status'] = 'active'

                # 用最高权重的候选记录位置历史
                best_j = int(np.argmax(gated_w))
                best_cand_j = gated_idx[best_j]
                bx, by = gated_meas[best_j]
                source = 'recovered' if old_status == 'lost_in_fov' else 'measured'
                ok = _record_track_point(t, bx, by, frame, source)
                if not ok:
                    _restore_filter(t['ekf'], filter_snap)
                    t['state'] = t['ekf'].X.copy()
                    t['P'] = t['ekf'].P.copy()
                    t['status'] = old_fields['status']
                    t['miss_count'] = old_fields['miss_count']
                    t['edge_miss'] = old_fields['edge_miss']
                    t['gap_start_frame'] = old_fields['gap_start_frame']
                    t['tentative_count'] = old_fields['tentative_count']
                    t['hit_count'] = old_fields['hit_count']
                    _mark_track_missed(t, frame)
                    continue
                if source == 'recovered':
                    t['recovered_count'] = t.get('recovered_count', 0) + 1

                if energies is not None and radii is not None and best_cand_j < len(energies):
                    E = float(energies[best_cand_j])
                    R = float(radii[best_cand_j])
                else:
                    E = energy_func(frame, bx, by)
                    R = radius_func(frame, bx, by)
                t['energy_hist'].append(E)
                t['radius_hist'].append(R)

                # 置信度历史（用总关联质量而非单点 Phi）
                t['edge_miss'] = 0
                t['gap_start_frame'] = None
                t['hit_count'] = t.get('hit_count', 0) + 1
                _append_hist(t.setdefault('assoc_hist', []),
                             float(track_assoc_mass[i]), prune_hist_len)
                speed, theta = _track_speed_theta(t)
                _append_hist(t.setdefault('v_hist', []), speed, prune_hist_len)
                _append_hist(t.setdefault('theta_hist', []), theta, prune_hist_len)
            else:
                _mark_track_missed(t, frame)

    elif n_tracks > 0 and n_cands == 0:
        for t in tracks:
            _mark_track_missed(t, frame)

    elif n_tracks == 0 and n_cands > 0:
        for cx, cy, _ in candidates:
            tracks.append(create_track(cx, cy, frame=frame))

    # ── Step 4: 未关联候选 → 新轨迹或兜底关联 ─────────────
    if n_tracks > 0 and n_cands > 0:
        for j in range(n_cands):
            if cand_claimed_mass[j] < jpda_new_track_threshold:
                cx, cy, _ = candidates[j]
                cand_energy = float(energies[j]) if energies is not None and j < len(energies) else None
                cand_radius = float(radii[j]) if radii is not None and j < len(radii) else None

                if _try_reconnect_lost_in_fov(tracks, candidates[j], frame,
                                              energy=cand_energy,
                                              radius=cand_radius):
                    continue

                # 兜底：检查是否在已有轨迹的自适应门控内
                best_track = None
                best_dist = float('inf')
                fallback_reject_track = None
                fallback_reject_reason = ''
                if enable_fallback_association:
                    for i, t in enumerate(tracks):
                        if t.get('last_update_frame') == frame:
                            continue
                        if t.get('status') not in ('active', 'tentative'):
                            continue
                        d2 = _mahalanobis_d2(t, cx, cy, camera_info=camera_info)
                        d_eucl = np.sqrt((t['state'][0] - cx) ** 2 +
                                          (t['state'][1] - cy) ** 2)
                        reason = ''
                        if d2 >= fallback_mahalanobis_gate:
                            reason = 'd2 %.3f >= %.3f' % (d2, fallback_mahalanobis_gate)
                        elif d_eucl > max_velocity * fallback_max_velocity_scale:
                            reason = 'eucl %.3f > %.3f' % (
                                d_eucl, max_velocity * fallback_max_velocity_scale)
                        else:
                            vx, vy = _get_track_velocity(t)
                            v_norm = float(np.hypot(vx, vy))
                            if v_norm > 0.5:
                                disp_x = float(cx - t['state'][0])
                                disp_y = float(cy - t['state'][1])
                                disp_norm = float(np.hypot(disp_x, disp_y))
                                cos_sim = ((disp_x * vx + disp_y * vy) /
                                           max(disp_norm * v_norm, _EPS))
                                if cos_sim < fallback_min_cosine:
                                    reason = 'cos %.3f < %.3f' % (
                                        cos_sim, fallback_min_cosine)
                        if reason:
                            if d2 < best_dist:
                                fallback_reject_track = t
                                fallback_reject_reason = reason
                            continue
                        if d2 < best_dist:
                            best_dist = d2
                            best_track = t
                    if best_track is None and fallback_reject_track is not None:
                        _add_track_event(fallback_reject_track, frame,
                                         'reject_fallback_gate',
                                         cx, cy, fallback_reject_reason)

                if best_track is not None:
                    ok = _record_track_point(best_track, cx, cy, frame, 'measured')
                    if not ok:
                        continue
                    best_track['ekf'].update_jpda([(cx, cy)], [1.0])
                    best_track['state'] = best_track['ekf'].X.copy()
                    best_track['P'] = best_track['ekf'].P.copy()
                    best_track['miss_count'] = 0
                    best_track['edge_miss'] = 0
                    best_track['gap_start_frame'] = None
                    best_track['hit_count'] = best_track.get('hit_count', 0) + 1
                    if best_track['status'] == 'tentative':
                        best_track['tentative_count'] += 1
                        if best_track['tentative_count'] >= tentative_min:
                            best_track['status'] = 'active'
                    if energies is not None and radii is not None and j < len(energies):
                        E = float(energies[j])
                        R = float(radii[j])
                    else:
                        E = energy_func(frame, cx, cy)
                        R = radius_func(frame, cx, cy)
                    best_track['energy_hist'].append(E)
                    best_track['radius_hist'].append(R)
                    _append_hist(best_track.setdefault('assoc_hist', []),
                                 1.0, prune_hist_len)
                    speed, theta = _track_speed_theta(best_track)
                    _append_hist(best_track.setdefault('v_hist', []), speed, prune_hist_len)
                    _append_hist(best_track.setdefault('theta_hist', []), theta, prune_hist_len)
                    _add_track_event(best_track, frame, 'fallback_accepted',
                                     cx, cy, 'd2 %.3f' % best_dist, best_dist)
                else:
                    # 空间去重：若候选靠近已成功命中的轨迹，视为差分残差不创建新轨迹
                    near_hit_track = False
                    for i, t in enumerate(tracks):
                        # 安全访问 track_assoc_mass，tracks 可能在后续逻辑被扩展
                        assoc_mass = track_assoc_mass[i] if i < track_assoc_mass.shape[0] else 0.0
                        if assoc_mass >= jpda_hit_threshold:
                            d = np.sqrt((t['state'][0] - cx) ** 2 + (t['state'][1] - cy) ** 2)
                            if d <= candidate_trusted_radius:
                                near_hit_track = True
                                break
                    if near_hit_track:
                        continue
                    # 从最高 Sinkhorn 概率的轨迹继承速度，避免新轨迹重新收敛
                    best_sink_val = 0.0
                    best_vx, best_vy = 0.0, 0.0
                    for i, t in enumerate(tracks):
                        p_val = P[i, j] if i < P.shape[0] and j < P.shape[1] else 0.0
                        if p_val > best_sink_val:
                            best_sink_val = p_val
                            best_vx, best_vy = _get_track_velocity(t)
                    if best_sink_val > 0.3:
                        tracks.append(create_track(cx, cy, vx=best_vx, vy=best_vy, frame=frame))
                    else:
                        tracks.append(create_track(cx, cy, frame=frame))

    # ── Step 5: 置信度驱动自适应剪枝 + 飞出视场检测 ─────
    pruned = []
    active = []
    EXIT_EDGE_THRESH = 3
    for t in tracks:
        if t.get('edge_miss', 0) >= EXIT_EDGE_THRESH:
            t['status'] = 'exited'
            t['exit_frame'] = frame
            t['exit_point'] = _extrapolate_exit_point(t, frame=frame)
            if t.get('exit_point') is not None:
                t['exit_edge'] = _get_exit_edge(t['exit_point'][0], t['exit_point'][1])
                _add_track_event(t, frame, 'exited',
                                 t['exit_point'][0], t['exit_point'][1],
                                 'edge exit')
            pruned.append(t)
            continue
        px, py = t['state'][0], t['state'][1]
        in_fov = _is_in_fov_xy(px, py)
        if t.get('status') == 'lost_in_fov':
            if t.get('miss_count', 0) > in_fov_reconnect_max_gap:
                t['status'] = 'pruned'
                pruned.append(t)
            elif in_fov:
                active.append(t)
            else:
                conf = _calc_confidence(t)
                t['confidence'] = conf
                thresh = _adaptive_prune_thresh(conf)
                if t['miss_count'] >= thresh:
                    t['status'] = 'pruned'
                    pruned.append(t)
                else:
                    active.append(t)
            continue
        if in_fov and t['miss_count'] >= in_fov_timeout:
            t['status'] = 'pruned'
            pruned.append(t)
            continue
        conf = _calc_confidence(t)
        t['confidence'] = conf
        thresh = _adaptive_prune_thresh(conf)
        if t['miss_count'] >= thresh:
            t['status'] = 'pruned'
            pruned.append(t)
        else:
            active.append(t)

    return active, pruned


def get_active_tracks(tracks):
    """获取活跃轨迹"""
    return [t for t in tracks if t['status'] == 'active']
