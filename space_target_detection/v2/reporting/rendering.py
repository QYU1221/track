"""像面与惯性画布的轨迹可视化辅助函数"""

# Domain implementation: visualization.
import os

import cv2
import numpy as np
from ..numerical_policy import DISPLAY_VECTOR_NORM_FLOOR


COLOR_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (0, 128, 255), (128, 0, 255), (255, 128, 128), (128, 255, 128),
]


def draw_track_path(canvas, track, points, frames=None, sources=None,
                    color=None, show_ids=True):
    """在画布上绘制一条轨迹路径（轨迹绘制的唯一实现）。

    ``points`` 是画布坐标序列，元素可为 None（跳过该点）；
    ``frames``/``sources`` 与 ``points`` 对齐时只连接相邻帧。
    绘制使用画布坐标，不以原图速度阈值过滤缩放后的线段。
    像面叠加与惯性切平面叠加都调用本函数。
    """
    if len(points) < 2:
        return
    points = list(points)
    if color is None:
        color = COLOR_PALETTE[track.id % len(COLOR_PALETTE)]
    use_frames = (frames is not None and len(frames) == len(points) and
                  all(f is not None for f in frames))

    for index in range(1, len(points)):
        prev_pt, curr_pt = points[index - 1], points[index]
        if prev_pt is None or curr_pt is None:
            continue
        if use_frames:
            frame_diff = int(frames[index]) - int(frames[index - 1])
            if frame_diff != 1:
                continue

        source = sources[index] if sources and index < len(sources) else 'measured'
        previous_source = (sources[index - 1] if sources and index - 1 < len(sources)
                           else 'measured')
        if 'predicted' in (source, previous_source):
            start, end = np.asarray(prev_pt), np.asarray(curr_pt)
            for fraction in (0.0, 0.4, 0.8):
                a = tuple((start + fraction * (end-start)).astype(int))
                b = tuple((start + min(fraction+0.2, 1.0) * (end-start)).astype(int))
                cv2.line(canvas, a, b, tuple(int(c * .45) for c in color), 1)
        else:
            cv2.line(canvas, tuple(prev_pt), tuple(curr_pt), color, 2)

    # An absent current position must not relabel an old point as current.
    if points[-1] is None:
        return
    last_index = len(points) - 1
    cx, cy = points[last_index]
    predicted = bool(sources and last_index < len(sources) and sources[last_index] == 'predicted')
    if predicted:
        color = tuple(int(c * .45) for c in color)
    is_exited = getattr(track, 'status', None) in ('RETIRED','TERMINATED','NO_RETAINED_SUPPORT')
    if is_exited:
        cv2.circle(canvas, (cx, cy), 7, color, 2)
        prev_index = last_index - 1
        while prev_index >= 0 and points[prev_index] is None:
            prev_index -= 1
        if prev_index >= 0:
            px, py = points[prev_index]
            dx, dy = cx - px, cy - py
            d_norm = np.sqrt(dx ** 2 + dy ** 2) + DISPLAY_VECTOR_NORM_FLOOR
            dx, dy = dx / d_norm * 12, dy / d_norm * 12
            cv2.arrowedLine(canvas, (cx, cy), (int(cx + dx), int(cy + dy)),
                            color, 2, tipLength=0.3)
    else:
        cv2.circle(canvas, (cx, cy), 6, color, 1 if predicted else -1)
    if show_ids:
        tag = "ID:%s" % track.id
        if predicted:
            tag += " P"
        if is_exited:
            tag += " X"
        cv2.putText(canvas, tag, (cx + 8, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def save_image(img, path):
    ext = os.path.splitext(path)[1]
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    ret, buf = cv2.imencode(ext, img)
    if not ret:
        raise OSError('Image encoding failed: %s' % path)
    buf.tofile(path)
