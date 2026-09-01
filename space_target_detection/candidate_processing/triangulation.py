# Shared triangulation implementation used by candidate and star processing.
import numpy as np
from scipy.spatial import Delaunay


def bsp_delaunay(points):
    """
    Delaunay 三角剖分

    使用 scipy.spatial.Delaunay（C 实现，等价于论文 BSP 树改进的效果）
    对结果构建三角形特征元数据，供后续融合特征计算与匹配使用。

    参数
    ----
    points : List[(x, y, ...)]
        输入点列表

    返回
    ----
    triangles : List[dict]
        三角形列表，每个 dict 包含 vertices, circum_center, circum_radius,
        aspect_ratio, perimeter, area, fusion_feat
    """
    if len(points) < 3:
        return []

    pts = np.array([(p[0], p[1]) for p in points], dtype=np.float64)

    tri = Delaunay(pts)

    triangles = []
    for simplex in tri.simplices:
        v0, v1, v2 = simplex
        p0 = pts[v0]
        p1 = pts[v1]
        p2 = pts[v2]

        # 边长
        d01 = np.linalg.norm(p0 - p1)
        d12 = np.linalg.norm(p1 - p2)
        d20 = np.linalg.norm(p2 - p0)
        sides = sorted([d01, d12, d20])

        perimeter = d01 + d12 + d20

        # 海伦公式求面积
        s = perimeter / 2.0
        area_val = max(np.sqrt(max(s * (s - d01) * (s - d12) * (s - d20), 0.0)), 1e-12)

        a, b, c = sides
        r1 = b / max(a, 1e-12)         # 边长比 b/a（≥1）
        r2 = c / max(a, 1e-12)         # 边长比 c/a（≥1）
        r1_inv = a / max(b, 1e-12)     # 边长比 a/b（∈(0,1]），V2 使用的 r₁
        aspect_ratio = r2              # alias of r2

        # 最小内角（最短边 a 对角），余弦定理
        cos_alpha = (b * b + c * c - a * a) / max(2.0 * b * c, 1e-12)
        cos_alpha = max(-1.0, min(1.0, cos_alpha))
        alpha_min = np.arccos(cos_alpha)

        # 外接圆圆心和半径
        cx, cy, r = _circumcircle(p0, p1, p2)

        triangles.append({
            'vertices': [tuple(p0.tolist()), tuple(p1.tolist()), tuple(p2.tolist())],
            'circum_center': (cx, cy),
            'circum_radius': r,
            'aspect_ratio': aspect_ratio,
            'perimeter': perimeter,
            'area': area_val,
            'r1': r1,
            'r2': r2,
            'r1_inv': r1_inv,
            'alpha_min': alpha_min,
            'fusion_feat': None,
        })

    return triangles


def _circumcircle(p0, p1, p2):
    """计算三角形的外接圆圆心和半径"""
    a = p0
    b = p1
    c = p2
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-12:
        # 共线，返回退化值
        center = np.mean([a, b, c], axis=0)
        return center[0], center[1], 1e9

    ax, ay = a
    bx, by = b
    cx, cy = c
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy

    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    r = np.sqrt((ax - ux) ** 2 + (ay - uy) ** 2)

    return ux, uy, r
