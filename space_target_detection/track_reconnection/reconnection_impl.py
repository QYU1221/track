"""Long-gap exit-memory and re-entry reconnection implementation."""

import numpy as np

from configuration.values import (
    FRAME_W, FRAME_H, max_velocity, prune_hist_len, prune_n_confirm,
    triangle_star_evidence_frames,
    enable_exit_reconnect, exit_reconnect_max_gap, exit_reconnect_memory_ttl,
    exit_reconnect_min_old_hits, exit_reconnect_min_old_conf,
    exit_reconnect_border_margin,
    exit_reconnect_confirm_hits, exit_reconnect_score_margin,
    exit_reconnect_confirm_high, exit_reconnect_defer_low,
    exit_reconnect_defer_frames, exit_reconnect_photo_sigma_base,
    exit_reconnect_size_sigma_base, exit_reconnect_photo_sigma_gap_gain,
    exit_reconnect_size_sigma_gap_gain, exit_reconnect_maneuver_gain,
    exit_reconnect_min_measured_ratio, exit_reconnect_reentry_score_min,
    exit_reconnect_min_new_measured_ratio, exit_reconnect_max_observation_gap,
    exit_reconnect_max_required_speed, exit_reconnect_speed_tolerance_scale,
    exit_reconnect_unknown_min_hits, exit_reconnect_unknown_score_threshold,
    exit_reconnect_pending_max_stale_frames,
)
from target_tracking.trajectory_manager_impl import (
    _EPS, _border_inner_normal, _distance_to_border, _extrapolate_exit_point,
    _get_exit_edge, _get_track_velocity, _new_track_measurement_quality,
    _opposite_edge, _outside_boundary_distance, _safe_recent_mean,
    _safe_recent_std, _safe_recent_values, _track_entry_frame,
    _track_measured_ratio, _track_speed_theta, _vector_from_recent_positions,
)

# The minimum observations needed by a new re-entry track is the same as the
# configured confirmation-hit requirement.
exit_reconnect_min_new_hits = exit_reconnect_confirm_hits


def _track_maneuver_score(track):
    """根据速度/方向历史估计机动程度。值越大，旧速度方向越不可信。"""
    v_hist = _safe_recent_values(track.get('v_hist', []), k=prune_hist_len)
    theta_hist = _safe_recent_values(track.get('theta_hist', []), k=prune_hist_len)
    if len(v_hist) < 3 and len(theta_hist) < 3:
        return 0.3
    score = 0.0
    if len(v_hist) >= 3:
        v = np.array(v_hist, dtype=np.float64)
        score += float(np.std(v) / max(np.mean(np.abs(v)), _EPS))
    if len(theta_hist) >= 3:
        th = np.unwrap(np.array(theta_hist, dtype=np.float64))
        score += float(np.std(th) / np.pi)
    return float(np.clip(score, 0.0, 5.0))


def build_exit_memory_item(track, frame):
    """将高质量 exited 轨迹转换为长期重连记忆项。

    注意：exit_point 是辅助摘要，不写入 positions。
    """
    if not enable_exit_reconnect:
        return None
    # 已被候选筛选闭环确认成恒星的轨迹不得进入飞出记忆库。
    # 这既避免恒星成为旧轨迹模板，也与再入侧的恒星短轨迹否决保持对称。
    if track.get('object_class') == 'star':
        return None
    if track.get('status') != 'exited':
        return None
    if track.get('hit_count', 0) < exit_reconnect_min_old_hits:
        return None
    if track.get('confidence', 0.0) < exit_reconnect_min_old_conf:
        return None
    if len(track.get('positions', [])) < 5:
        return None
    if _track_measured_ratio(track) < exit_reconnect_min_measured_ratio:
        return None

    exit_point = track.get('exit_point')
    if exit_point is None:
        exit_point = _extrapolate_exit_point(track, frame=frame)
    if exit_point is None:
        return None
    exit_edge = track.get('exit_edge') or _get_exit_edge(exit_point[0], exit_point[1])
    vx, vy = _get_track_velocity(track)
    speed, theta = _track_speed_theta(track)

    memory_item = {
        'old_id': track.get('id'),
        'exit_frame': int(track.get('exit_frame', frame if frame is not None else track.get('last_update_frame', 0))),
        'exit_point': (float(exit_point[0]), float(exit_point[1])),
        'exit_edge': exit_edge,
        'exit_velocity': (float(vx), float(vy)),
        'speed_mean': _safe_recent_mean(track.get('v_hist', []), k=5, default=speed),
        'theta_mean': _safe_recent_mean(track.get('theta_hist', []), k=5, default=theta),
        'energy_mean': _safe_recent_mean(track.get('energy_hist', []), k=5, default=0.0),
        'energy_std': _safe_recent_std(track.get('energy_hist', []), k=5, default=0.0),
        'radius_mean': _safe_recent_mean(track.get('radius_hist', []), k=5, default=0.0),
        'radius_std': _safe_recent_std(track.get('radius_hist', []), k=5, default=0.0),
        'confidence': float(track.get('confidence', 0.0)),
        'hit_count': int(track.get('hit_count', len(track.get('positions', [])))),
        'track_ref': track,
        'used': False,
        'created_frame': int(frame if frame is not None else track.get('last_update_frame', 0)),
        'maneuver_score': _track_maneuver_score(track),
    }
    track['exit_memory_built'] = True
    return memory_item


def prune_exit_memory(memory, frame):
    """清理已使用或过期的飞出轨迹记忆。"""
    if not enable_exit_reconnect:
        return []
    kept = []
    for item in memory or []:
        if item.get('used'):
            continue
        if frame is not None:
            if int(frame) - int(item.get('created_frame', item.get('exit_frame', frame))) > exit_reconnect_memory_ttl:
                continue
            if int(frame) - int(item.get('exit_frame', frame)) > exit_reconnect_max_gap:
                continue
        kept.append(item)
    return kept


def _entry_info(track):
    pts = track.get('positions') or []
    if not pts:
        return None
    x0, y0 = pts[0]
    edge = _get_exit_edge(x0, y0)
    dist = _distance_to_border(x0, y0)
    vx, vy = _vector_from_recent_positions(track, k=min(4, len(pts)))
    v = np.array([vx, vy], dtype=np.float64)
    n_in = _border_inner_normal(edge)
    norm = float(np.linalg.norm(v))
    if norm < _EPS:
        s_reentry = 0.0
    else:
        s_reentry = float(max(0.0, np.dot(v / norm, n_in)))
    return {
        'entry_frame': _track_entry_frame(track),
        'entry_point': (float(x0), float(y0)),
        'entry_edge': edge,
        'border_dist': dist,
        'velocity': (float(vx), float(vy)),
        's_reentry': s_reentry,
    }


def find_reentry_candidates(tracks, frame):
    """筛选边界再入短轨迹。单帧点不参与重连，必须先形成短轨迹。"""
    if not enable_exit_reconnect:
        return []
    cands = []
    for t in tracks:
        # 候选筛选模块会通过多帧星表/姿态证据把误建轨恒星标为 star。
        # 此类短轨迹即使位于边界、运动方向也像“再入”，仍禁止参与重连。
        if t.get('object_class') == 'star':
            continue
        if t.get('triangle_star_evidence_count', 0) >= triangle_star_evidence_frames:
            continue
        if t.get('status') not in ('tentative', 'active'):
            continue
        if t.get('reconnected_count', 0) > 0:
            continue
        if t.get('recovered_count', 0) > 0 or t.get('gap_start_frame') is not None:
            continue
        quality = _new_track_measurement_quality(t)
        min_hits = exit_reconnect_min_new_hits
        if t.get('object_class', 'unknown') == 'unknown':
            min_hits = max(min_hits, exit_reconnect_unknown_min_hits)
        if quality['measured_hits'] < min_hits:
            continue
        if quality['measured_ratio'] < exit_reconnect_min_new_measured_ratio:
            continue
        if quality['max_observation_gap'] > exit_reconnect_max_observation_gap:
            continue
        entry_frame = _track_entry_frame(t)
        if frame is not None and int(frame) - int(entry_frame) > max(exit_reconnect_confirm_hits + 4, 8):
            continue
        info = _entry_info(t)
        if info is None:
            continue
        if info['border_dist'] > exit_reconnect_border_margin:
            continue
        if info['s_reentry'] < exit_reconnect_reentry_score_min:
            continue
        t['entry_edge'] = info['entry_edge']
        t['entry_frame'] = info['entry_frame']
        t['reentry_score'] = info['s_reentry']
        t['_reentry_measurement_quality'] = quality
        cands.append((t, info))
    return cands


def _ratio_similarity(a, b, sigma=0.75):
    if a <= 0 or b <= 0:
        return 0.5
    delta = abs(np.log((float(a) + _EPS) / (float(b) + _EPS)))
    return float(np.exp(-0.5 * (delta / max(sigma, _EPS)) ** 2))


def score_exit_reentry_pair(memory_item, new_track, frame):
    """计算飞出记忆与边界再入短轨迹的自适应软重连分数。"""
    if memory_item.get('used'):
        return None
    # 防御性检查：即使调用方绕过 find_reentry_candidates，恒星短轨迹也不能评分。
    if new_track.get('object_class') == 'star':
        return None
    old_track = memory_item.get('track_ref')
    if old_track is not None and old_track.get('object_class') == 'star':
        return None
    info = _entry_info(new_track)
    if info is None:
        return None
    delta_t = int(info['entry_frame']) - int(memory_item.get('exit_frame', info['entry_frame']))
    if delta_t <= 0 or delta_t > exit_reconnect_max_gap:
        return None

    outside_distance = _outside_boundary_distance(
        memory_item.get('exit_point', (0.0, 0.0)),
        memory_item.get('exit_edge'),
        info['entry_point'],
        info['entry_edge'],
    )
    if outside_distance is None:
        return None

    old_v_for_gate = np.array(memory_item.get('exit_velocity', (0.0, 0.0)), dtype=np.float64)
    new_v_for_gate = np.array(info['velocity'], dtype=np.float64)
    history_speed = max(float(np.linalg.norm(old_v_for_gate)),
                        float(np.linalg.norm(new_v_for_gate)))
    # 历史速度决定自适应容差，同时始终受绝对速度上限约束。
    # max_velocity 的四分之一仅用于低速/短历史轨迹，避免速度接近零时门限退化。
    adaptive_speed_limit = max(
        history_speed * float(exit_reconnect_speed_tolerance_scale),
        float(max_velocity) * 0.25,
    )
    speed_limit = min(float(exit_reconnect_max_required_speed), adaptive_speed_limit)
    required_speed = float(outside_distance / max(delta_t, 1))
    if required_speed > speed_limit:
        return None

    exit_edge = memory_item.get('exit_edge')
    entry_edge = info['entry_edge']
    if entry_edge == _opposite_edge(exit_edge):
        S_boundary = 1.0
    elif entry_edge == exit_edge:
        S_boundary = 0.8
    elif exit_edge in ('left', 'right') and entry_edge in ('top', 'bottom'):
        S_boundary = 0.5
    elif exit_edge in ('top', 'bottom') and entry_edge in ('left', 'right'):
        S_boundary = 0.5
    else:
        S_boundary = 0.25

    S_reentry = float(info['s_reentry'])

    old_v = np.array(memory_item.get('exit_velocity', (0.0, 0.0)), dtype=np.float64)
    new_v = np.array(info['velocity'], dtype=np.float64)
    old_speed = float(np.linalg.norm(old_v))
    new_speed = float(np.linalg.norm(new_v))
    S_speed = _ratio_similarity(old_speed, new_speed, sigma=0.8)
    if old_speed > _EPS and new_speed > _EPS:
        cos_dir = float(np.dot(old_v, new_v) / max(old_speed * new_speed, _EPS))
        S_dir = (np.clip(cos_dir, -1.0, 1.0) + 1.0) / 2.0
    else:
        S_dir = 0.5
    S_motion = float(0.65 * S_speed + 0.35 * S_dir)

    maneuver_score = max(float(memory_item.get('maneuver_score', 0.3)),
                         _track_maneuver_score(new_track))
    q_motion = float(np.exp(-maneuver_score))

    old_E = float(memory_item.get('energy_mean', 0.0))
    new_E = _safe_recent_mean(new_track.get('energy_hist', []), k=5, default=0.0)
    old_R = float(memory_item.get('radius_mean', 0.0))
    new_R = _safe_recent_mean(new_track.get('radius_hist', []), k=5, default=0.0)

    sigma_E = (exit_reconnect_photo_sigma_base +
               exit_reconnect_photo_sigma_gap_gain * delta_t +
               exit_reconnect_maneuver_gain * maneuver_score)
    sigma_R = (exit_reconnect_size_sigma_base +
               exit_reconnect_size_sigma_gap_gain * delta_t +
               exit_reconnect_maneuver_gain * maneuver_score)

    S_photo = _ratio_similarity(old_E, new_E, sigma=sigma_E)
    S_size = _ratio_similarity(old_R, new_R, sigma=sigma_R)

    # 光度/尺度自身波动大时降权，避免距离变化或相位变化导致误否决。
    E_cv = abs(float(memory_item.get('energy_std', 0.0))) / max(abs(old_E), _EPS) if old_E > 0 else 1.0
    R_cv = abs(float(memory_item.get('radius_std', 0.0))) / max(abs(old_R), _EPS) if old_R > 0 else 1.0
    q_photo = float(np.clip(np.exp(-E_cv), 0.25, 1.0))
    q_size = float(np.clip(np.exp(-R_cv), 0.25, 1.0))

    tau = max(exit_reconnect_max_gap / 2.0, 1.0)
    S_time = float(np.exp(-delta_t / tau))
    S_quality = float(np.clip(0.7 * memory_item.get('confidence', 0.0) +
                              0.3 * min(memory_item.get('hit_count', 0) / max(prune_n_confirm, 1), 1.0),
                              0.0, 1.0))

    score = (0.25 * S_boundary +
             0.25 * S_reentry +
             0.15 * q_motion * S_motion +
             0.12 * q_photo * S_photo +
             0.08 * q_size * S_size +
             0.10 * S_time +
             0.05 * S_quality)

    return {
        'score': float(np.clip(score, 0.0, 1.0)),
        'old_id': memory_item.get('old_id'),
        'temp_id': new_track.get('id'),
        'exit_frame': memory_item.get('exit_frame'),
        'entry_frame': info['entry_frame'],
        'exit_edge': exit_edge,
        'entry_edge': entry_edge,
        'gap': delta_t,
        'outside_path_distance': outside_distance,
        'required_speed': required_speed,
        'speed_limit': speed_limit,
        'new_track_quality': new_track.get('_reentry_measurement_quality',
                                           _new_track_measurement_quality(new_track)),
        'components': {
            'S_boundary': S_boundary, 'S_reentry': S_reentry,
            'S_motion': S_motion, 'q_motion': q_motion,
            'S_photo': S_photo, 'q_photo': q_photo,
            'S_size': S_size, 'q_size': q_size,
            'S_time': S_time, 'S_quality': S_quality,
            'maneuver_score': maneuver_score,
        }
    }


def merge_reconnected_tracks(old_track, new_track, memory_item, score_info, frame, mode='immediate'):
    """合并再入短轨迹，恢复旧 ID。中间空窗不补预测点。"""
    old_id = old_track.get('id')
    temp_id = new_track.get('id')

    # 合并位置序列，允许 frames 不连续。
    old_frames = old_track.setdefault('frames', [])
    old_pos = old_track.setdefault('positions', [])
    old_sources = old_track.setdefault('point_sources', ['measured'] * len(old_pos))
    new_frames = new_track.get('frames') or []
    new_pos = new_track.get('positions') or []
    new_sources = new_track.get('point_sources') or ['measured'] * len(new_pos)

    existing_frames = set(int(f) for f in old_frames if f is not None)
    for i, pos in enumerate(new_pos):
        fid = int(new_frames[i]) if i < len(new_frames) and new_frames[i] is not None else None
        if fid is not None and fid in existing_frames:
            continue
        old_pos.append((float(pos[0]), float(pos[1])))
        old_frames.append(fid)
        src = new_sources[i] if i < len(new_sources) else 'measured'
        old_sources.append('reconnected' if src in ('measured', 'recovered') else src)
        if fid is not None:
            existing_frames.add(fid)

    for key in ('energy_hist', 'radius_hist', 'assoc_hist', 'v_hist', 'theta_hist'):
        old_track.setdefault(key, [])
        old_track[key].extend(list(new_track.get(key, [])))
        if key in ('assoc_hist', 'v_hist', 'theta_hist') and len(old_track[key]) > prune_hist_len:
            old_track[key] = old_track[key][-prune_hist_len:]

    old_track['id'] = old_id
    old_track['status'] = 'active'
    old_track['miss_count'] = 0
    old_track['edge_miss'] = 0
    old_track['gap_start_frame'] = None
    old_track['last_update_frame'] = frame
    old_track['hit_count'] = max(old_track.get('hit_count', 0), len(old_pos))
    old_track['reconnected_count'] = old_track.get('reconnected_count', 0) + 1
    old_track['exit_memory_built'] = False  # 允许该目标后续再次飞出时重新入库
    old_track['state'] = new_track['state'].copy()
    old_track['P'] = new_track['P'].copy()
    old_track['ekf'] = new_track['ekf']

    memory_item['used'] = True
    event = {
        'old_id': old_id,
        'temp_id': temp_id,
        'exit_frame': score_info.get('exit_frame'),
        'entry_frame': score_info.get('entry_frame'),
        'reconnect_frame': int(frame) if frame is not None else None,
        'score': float(score_info.get('score', 0.0)),
        'mode': mode,
        'exit_edge': score_info.get('exit_edge'),
        'entry_edge': score_info.get('entry_edge'),
        'gap': score_info.get('gap'),
        'outside_path_distance': score_info.get('outside_path_distance'),
        'required_speed': score_info.get('required_speed'),
        'speed_limit': score_info.get('speed_limit'),
        'new_measured_hits': score_info.get('new_track_quality', {}).get('measured_hits'),
        'new_measured_ratio': score_info.get('new_track_quality', {}).get('measured_ratio'),
        'new_max_observation_gap': score_info.get('new_track_quality', {}).get('max_observation_gap'),
    }
    old_track.setdefault('event_hist', []).append({
        'frame': frame, 'type': 'exit_reconnected',
        'x': old_pos[-1][0] if old_pos else None,
        'y': old_pos[-1][1] if old_pos else None,
        'reason': 'temp_id=%s score=%.3f mode=%s' % (temp_id, event['score'], mode),
        'score': event['score'],
    })
    return event


def reconnect_exited_tracks(tracks, trajectory_memory, frame, pending_reconnect=None):
    """尝试将边界再入短轨迹与 exited 记忆库重连。"""
    if pending_reconnect is None:
        pending_reconnect = {}
    if not enable_exit_reconnect or not trajectory_memory:
        return tracks, trajectory_memory, pending_reconnect, []

    trajectory_memory = prune_exit_memory(trajectory_memory, frame)
    # 某条短轨迹可能先以 unknown 身份进入延迟确认，随后被多帧证据识别为恒星。
    # 立即删除它的 pending 记录，避免旧分数继续累积并触发错误合并。
    star_track_ids = {
        t.get('id') for t in tracks if t.get('object_class') == 'star'
    }
    for track_id in star_track_ids:
        pending_reconnect.pop(track_id, None)
    if frame is not None:
        pending_reconnect = {
            track_id: rec for track_id, rec in pending_reconnect.items()
            if int(frame) - int(rec.get('last_evidence_frame', frame))
            <= exit_reconnect_pending_max_stale_frames
        }
    candidates = find_reentry_candidates(tracks, frame)
    events = []
    remove_ids = set()
    append_tracks = []

    for new_track, _info in candidates:
        scored = []
        for item in trajectory_memory:
            if item.get('used'):
                continue
            score_info = score_exit_reentry_pair(item, new_track, frame)
            if score_info is not None:
                scored.append((score_info['score'], item, score_info))
        if not scored:
            continue
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_item, best_info = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        margin_ok = (best_score - second_score) >= exit_reconnect_score_margin
        if not margin_ok:
            continue

        if (new_track.get('object_class', 'unknown') == 'unknown' and
                best_score < exit_reconnect_unknown_score_threshold):
            continue

        mode = None
        if best_score >= exit_reconnect_confirm_high:
            mode = 'immediate'
        elif best_score >= exit_reconnect_defer_low:
            key = new_track.get('id')
            rec = pending_reconnect.get(key)
            quality = best_info.get('new_track_quality') or _new_track_measurement_quality(new_track)
            measured_hits = int(quality.get('measured_hits', 0))
            latest_measured_frame = quality.get('latest_measured_frame')
            if rec and rec.get('old_id') == best_item.get('old_id'):
                rec['score'] = best_score
                rec['score_info'] = best_info
                has_new_measurement = (
                    measured_hits > int(rec.get('last_measured_hits', 0)) and
                    latest_measured_frame is not None and
                    latest_measured_frame > rec.get('last_measured_frame', -1)
                )
                if has_new_measurement:
                    rec['count'] += 1
                    rec['last_measured_hits'] = measured_hits
                    rec['last_measured_frame'] = latest_measured_frame
                    rec['last_evidence_frame'] = int(frame)
            else:
                rec = {'old_id': best_item.get('old_id'), 'count': 1,
                       'score': best_score, 'score_info': best_info,
                       'last_measured_hits': measured_hits,
                       'last_measured_frame': latest_measured_frame,
                       'last_evidence_frame': int(frame) if frame is not None else latest_measured_frame}
            pending_reconnect[key] = rec
            if rec['count'] >= exit_reconnect_defer_frames:
                mode = 'deferred'
        if mode is None:
            continue

        old_track = best_item.get('track_ref')
        if old_track is None:
            continue
        event = merge_reconnected_tracks(old_track, new_track, best_item, best_info, frame, mode=mode)
        events.append(event)
        remove_ids.add(new_track.get('id'))
        append_tracks.append(old_track)
        pending_reconnect.pop(new_track.get('id'), None)

    if remove_ids:
        tracks = [t for t in tracks if t.get('id') not in remove_ids]
        existing = {t.get('id') for t in tracks}
        for t in append_tracks:
            if t.get('id') not in existing:
                tracks.append(t)
                existing.add(t.get('id'))
    trajectory_memory = prune_exit_memory(trajectory_memory, frame)
    return tracks, trajectory_memory, pending_reconnect, events
