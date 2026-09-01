"""Result artifacts and terminal summaries for a completed sequence."""

import csv
import os

import numpy as np

from configuration.values import FRAME_H, FRAME_W, dist_thresh, prune_len
from configuration.camera import external_quaternion_to_wxyz
from visualization import draw_tracks_on_frame, save_image


def save_frame_statistics(run_dir, rows):
    fields = ['frame', 'n_stars', 'n_tracked_targets_pre', 'n_displacement_targets_pre', 'n_stars_predicted', 'n_triangles', 'n_triangle_background_triangles', 'n_matched_pairs', 'triangle_available', 'triangle_reason', 'triangle_model', 'triangle_point_matches', 'triangle_inliers', 'triangle_rejected', 'triangle_conflicts', 'triangle_protected', 'triangle_median_residual', 'catalog_available', 'catalog_target_available', 'catalog_matches', 'catalog_match_ratio', 'catalog_median_residual', 'n_candidates', 'n_targets', 'n_active_tracks', 'n_star_class_tracks', 'n_target_class_tracks', 'n_unknown_class_tracks', 'n_exit_memory', 'n_reconnect_events', 'match_ratio', 'inlier_ratio', 'edge_preserve', 'n_identified_stars', 'star_id_selected_method', 'star_id_primary_method', 'star_id_fallback_used', 'star_id_switch_reason', 'star_id_primary_time_ms', 'star_id_fallback_time_ms', 'star_id_quality_score', 'star_id_quality_matches', 'star_id_quality_median_residual', 'star_id_quality_max_residual', 'star_id_quality_coverage', 'star_id_consecutive_failures', 'attitude_ok', 'q_w', 'q_x', 'q_y', 'q_z', 'time_ms']
    with open(os.path.join(run_dir, 'per_frame_stats.csv'), 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def quat_to_euler(w, x, y, z):
    roll = np.degrees(np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
    pitch = np.degrees(np.arcsin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))
    yaw = np.degrees(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    return roll, pitch, yaw


def save_attitude(run_dir, rows, input_dir=None):
    path = os.path.join(run_dir, 'attitude.csv')
    valid_rows = [row for row in rows if row.get('attitude_ok')]
    q_gt_all = None
    if input_dir:
        candidate = os.path.join(os.path.dirname(os.path.normpath(input_dir)), 'truth', 'quaternion_history.npy')
        if os.path.exists(candidate):
            q_gt_all = np.load(candidate)
    with open(path, 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        header = ['frame', 'q_w', 'q_x', 'q_y', 'q_z', 'roll_deg', 'pitch_deg', 'yaw_deg', 'n_identified']
        if q_gt_all is not None:
            header.append('error_vs_gt_deg')
        writer.writerow(header)
        for row in rows:
            if not row.get('attitude_ok'):
                continue
            qw, qx, qy, qz = row['q_w'], row['q_x'], row['q_y'], row['q_z']
            if qw < 0.0:
                qw, qx, qy, qz = -qw, -qx, -qy, -qz
            roll, pitch, yaw = quat_to_euler(qw, qx, qy, qz)
            result = [row['frame'], '%.8f' % qw, '%.8f' % qx, '%.8f' % qy, '%.8f' % qz, '%.4f' % roll, '%.4f' % pitch, '%.4f' % yaw, row['n_identified_stars']]
            if q_gt_all is not None and row['frame'] < len(q_gt_all):
                q_gt = external_quaternion_to_wxyz(q_gt_all[row['frame']])
                dot = min(abs(np.dot(np.array([qw, qx, qy, qz]), q_gt)), 1.0)
                result.append('%.6f' % np.degrees(2.0 * np.arccos(dot)))
            writer.writerow(result)
    if valid_rows:
        print('Attitude saved: %s  (%d frames)' % (path, len(set(row['frame'] for row in valid_rows))))


def save_tracks(run_dir, tracks):
    with open(os.path.join(run_dir, 'trajectories.csv'), 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['track_id', 'frame', 'x', 'y', 'status', 'source'])
        for track in tracks:
            frames, positions = track.get('frames'), track['positions']
            use_frames = frames is not None and len(frames) == len(positions) and all(frame is not None for frame in frames)
            for index, (x, y) in enumerate(positions):
                frame = int(frames[index]) if use_frames else track.get('start_frame', 0) + index + 1
                source = track.get('point_sources', [])[index] if index < len(track.get('point_sources', [])) else 'measured'
                writer.writerow([track['id'], frame, '%.2f' % x, '%.2f' % y, track['status'], source])


def save_track_events(run_dir, tracks):
    rows = [{'track_id': track.get('id'), 'frame': event.get('frame'), 'event': event.get('type'), 'x': event.get('x'), 'y': event.get('y'), 'reason': event.get('reason', ''), 'score': event.get('score', '')} for track in tracks for event in track.get('event_hist', [])]
    with open(os.path.join(run_dir, 'track_events.csv'), 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['track_id', 'frame', 'event', 'x', 'y', 'reason', 'score'])
        writer.writeheader()
        writer.writerows(rows)


def save_track_overlay(run_dir, tracks):
    tracks = [track for track in tracks if len(track['positions']) >= prune_len]
    canvas = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    path = os.path.join(run_dir, 'tracks_overlay.png')
    save_image(draw_tracks_on_frame(canvas, tracks, show_ids=True) if tracks else canvas, path)
    print('Tracks overlay saved: ' + path)


def save_evaluation(run_dir, star_rows, target_rows, star_removal_rows=None):
    for filename, fields, rows in [
        ('eval_stars.csv', ['frame', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'position_error', 'n_gt', 'n_identified', 'n_id_correct', 'n_id_wrong', 'n_gt_unpresentable'], star_rows),
        ('eval_targets.csv', ['frame', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'position_error', 'n_gt', 'n_tracked'], target_rows),
        ('eval_star_removal.csv', ['frame', 'n_gt', 'n_detected_stars', 'n_gt_undetected', 'n_correctly_removed', 'n_leaked_stars', 'removal_recall', 'leakage_rate'], star_removal_rows),
    ]:
        if rows:
            with open(os.path.join(run_dir, filename), 'w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(rows)


def print_star_removal_summary(rows):
    if not rows:
        return
    total = lambda field: sum(int(row[field]) for row in rows)
    n_gt, detected, undetected, removed, leaked = (total(field) for field in ('n_gt', 'n_detected_stars', 'n_gt_undetected', 'n_correctly_removed', 'n_leaked_stars'))
    print('\n  [Star Removal]\n    GT stars total       : %d\n    Detected GT stars    : %d\n    Frontend undetected  : %d\n    Correctly removed    : %d\n    Leaked to targets    : %d\n    Removal recall       : %.4f\n    Leakage rate         : %.4f' % (n_gt, detected, undetected, removed, leaked, removed / max(detected, 1), leaked / max(detected, 1)))


def print_evaluation_summary(summary):
    print('\n' + '=' * 60 + '\n  Evaluation Summary\n' + '=' * 60)
    for title, data, extra in [('Star Identification', summary.get('star', {}), True), ('Target Tracking', summary.get('target', {}), False)]:
        if not data:
            continue
        print('  [%s]\n    Frames evaluated : %d\n    GT %s total%s: %d\n    TP / FP / FN      : %d / %d / %d\n    Precision         : %.4f\n    Recall            : %.4f\n    F1                : %.4f\n    Position Error    : %.2f px' % (title, data.get('n_frames_evaluated', 0), 'stars' if extra else 'targets', '    ' if extra else ' ', data.get('n_gt_total', 0), data['tp'], data['fp'], data['fn'], data['precision'], data['recall'], data['f1'], data['position_error']))
        if extra:
            ok, bad, n_gt = data.get('n_id_correct', 0), data.get('n_id_wrong', 0), data.get('n_gt_total', 0)
            print('    GT not in input   : %d  (pre-separation loss)\n    ID correct / wrong: %d / %d  (rate=%.1f%% of GT, pos-match=%.1f%%)' % (data.get('n_gt_unpresentable', 0), ok, bad, 100.0 * ok / max(n_gt, 1), 100.0 * ok / max(ok + bad, 1)))
    print('=' * 60)


def print_mota_motp(input_dir, tracks, num_frames):
    from evaluation import calc_mota_motp
    truth_dir = os.path.join(os.path.dirname(os.path.normpath(input_dir)), 'truth')
    if not os.path.isdir(truth_dir): return
    trajectories = [np.load(os.path.join(truth_dir, name)) for name in sorted(os.listdir(truth_dir)) if name.startswith('target_') and name.endswith('_trajectory.npy')]
    if not trajectories: return
    gt = {fid: {idx: tuple(path[fid]) for idx, path in enumerate(trajectories) if fid < len(path) and 0 <= path[fid][0] < FRAME_W and 0 <= path[fid][1] < FRAME_H} for fid in range(num_frames)}
    est = {}
    for track in tracks:
        frames, positions = track.get('frames'), track['positions']
        use_frames = frames is not None and len(frames) == len(positions) and all(frame is not None for frame in frames)
        for index, point in enumerate(positions): est.setdefault(int(frames[index]) if use_frames else track.get('start_frame', 0) + index, {})[track['id']] = point
    mota, motp, stats = calc_mota_motp(est, gt, dist_threshold=dist_thresh)
    if stats['total_gt']:
        print('MOTA: %.2f%%  MOTP: %.2f%%  (GT=%d  FN=%d  FP=%d  err=%.2fpx)' % (mota * 100, motp * 100, stats['total_gt'], stats['FN'], stats['FP'], stats['dist_sum'] / max(stats['N_matched'], 1)))


def print_attitude_evaluation(input_dir, rows):
    path = os.path.join(os.path.dirname(os.path.normpath(input_dir)), 'truth', 'quaternion_history.npy')
    if not os.path.exists(path): return
    truth, errors = np.load(path), []
    for row in rows:
        if row['attitude_ok'] and row['frame'] < len(truth):
            q_gt = external_quaternion_to_wxyz(truth[row['frame']])
            errors.append(np.degrees(2.0 * np.arccos(min(abs(np.dot(np.array([row['q_w'], row['q_x'], row['q_y'], row['q_z']]), q_gt)), 1.0))))
    if errors:
        print('Attitude Error vs GT: mean=%.4f°  median=%.4f°  max=%.4f°  (N=%d/%d frames)' % (np.mean(errors), np.median(errors), np.max(errors), len(errors), len(rows)))
