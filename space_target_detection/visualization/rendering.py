"""可视化辅助函数（仅保留 CSV 输入模式下使用的函数）"""

# Domain implementation: visualization.
import numpy as np
import cv2
import os
from configuration.values import max_velocity


COLOR_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (0, 128, 255), (128, 0, 255), (255, 128, 128), (128, 255, 128),
]


def _ensure_bgr(frame):
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame.copy()


def draw_tracks_on_frame(frame, tracks, show_ids=True):
    """在画布上绘制所有轨迹路径，不同 ID 使用不同颜色。"""
    out = _ensure_bgr(frame)
    for t in tracks:
        if len(t['positions']) < 2:
            continue
        is_exited = t.get('status') == 'exited'
        color = COLOR_PALETTE[t['id'] % len(COLOR_PALETTE)]
        pts = np.array(t['positions'], dtype=np.int32)
        frames = t.get('frames')
        sources = t.get('point_sources', [])
        use_frames = (frames is not None and len(frames) == len(t['positions']) and
                      all(f is not None for f in frames))
        for i in range(1, len(pts)):
            if use_frames:
                frame_diff = int(frames[i]) - int(frames[i - 1])
                if frame_diff <= 0:
                    continue
                source = sources[i] if i < len(sources) else 'measured'
                dist = float(np.hypot(pts[i][0] - pts[i - 1][0],
                                      pts[i][1] - pts[i - 1][1]))
                limit = max_velocity * max(1, frame_diff) * 2.0
                if source not in ('extrapolated_exit', 'recovered') and dist > limit:
                    print("Warning: skip large track segment ID=%s frame=%s->%s dist=%.1f" %
                          (t.get('id'), frames[i - 1], frames[i], dist))
                    continue
            cv2.line(out, tuple(pts[i - 1]), tuple(pts[i]), color, 2)
        cx, cy = pts[-1]
        if is_exited:
            cv2.circle(out, (cx, cy), 7, color, 2)
            if len(pts) >= 2:
                px, py = pts[-2]
                dx, dy = cx - px, cy - py
                d_norm = np.sqrt(dx**2 + dy**2) + 1e-8
                dx, dy = dx / d_norm * 12, dy / d_norm * 12
                cv2.arrowedLine(out, (cx, cy), (int(cx+dx), int(cy+dy)), color, 2, tipLength=0.3)
        else:
            cv2.circle(out, (cx, cy), 6, color, -1)
        if show_ids:
            tag = f"ID:{t['id']}"
            if is_exited:
                tag += " X"
            cv2.putText(out, tag, (cx + 8, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return out


def draw_star_recognition_result(frame_w, frame_h, measurements, stars_gt,
                                 star_points, star_id_result):
    """在全部当帧 measurement 底图上绘制恒星真值与识别结果。"""
    out = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    measurement_color = (120, 120, 120)
    gt_color = (0, 255, 0)
    recognized_color = (0, 0, 255)

    for measurement in measurements:
        x = int(round(float(measurement[0])))
        y = int(round(float(measurement[1])))
        cv2.circle(out, (x, y), 2, measurement_color, -1, cv2.LINE_AA)

    cv2.putText(out, "All measurements", (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, measurement_color, 1,
                cv2.LINE_AA)
    cv2.putText(out, "GT: green (id, x, y)", (12, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, gt_color, 1, cv2.LINE_AA)
    cv2.putText(out, "Sys: blue (id, x, y)", (12, 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, recognized_color, 1,
                cv2.LINE_AA)

    if not stars_gt:
        cv2.putText(out, "GT unavailable", (12, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, gt_color, 1,
                    cv2.LINE_AA)

    for star_id, x, y in stars_gt:
        px, py = int(round(float(x))), int(round(float(y)))
        cv2.circle(out, (px, py), 6, gt_color, 1, cv2.LINE_AA)
        cv2.drawMarker(out, (px, py), gt_color, cv2.MARKER_CROSS,
                       8, 1, cv2.LINE_AA)
        cv2.putText(out, "#%d (%.0f,%.0f)" % (int(star_id), x, y),
                    (px + 7, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, gt_color, 1,
                    cv2.LINE_AA)

    identified = (star_id_result or {}).get('identified', {})
    for candidate_idx, sys_cat_id in identified.items():
        try:
            idx = int(candidate_idx)
            cat_id = int(sys_cat_id)
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(star_points):
            continue
        x, y = float(star_points[idx][0]), float(star_points[idx][1])
        px, py = int(round(x)), int(round(y))
        cv2.circle(out, (px, py), 3, recognized_color, -1, cv2.LINE_AA)
        cv2.circle(out, (px, py), 8, recognized_color, 1, cv2.LINE_AA)
        cv2.putText(out, "#%d (%.0f,%.0f)" % (cat_id, x, y),
                    (px + 7, py + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, recognized_color, 1,
                    cv2.LINE_AA)
    return out


def draw_measurements_and_tracks(frame_w, frame_h, measurements, tracks):
    """在全部当帧 measurement 上叠加截至当前帧的彩色轨迹。"""
    canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    measurement_color = (180, 180, 180)
    for measurement in measurements:
        x, y = int(round(float(measurement[0]))), int(round(float(measurement[1])))
        cv2.circle(canvas, (x, y), 2, measurement_color, -1, cv2.LINE_AA)

    cv2.putText(canvas, "All measurements", (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, measurement_color, 1,
                cv2.LINE_AA)
    cv2.putText(canvas, "Colored lines: tracks", (12, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)
    return draw_tracks_on_frame(canvas, tracks, show_ids=True)


def save_image(img, path):
    ext = os.path.splitext(path)[1]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ret, buf = cv2.imencode(ext, img)
    if ret:
        buf.tofile(path)
