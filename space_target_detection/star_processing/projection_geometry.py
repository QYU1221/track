"""恒星投影几何工具。

提供图像像素、相机坐标系单位视线和导航星表投影之间的转换。
姿态解算统一由 :mod:`star_processing.attitude_solvers` 负责。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from configuration.values import camera_projection_model


def _projection_model(profile=None):
    model = camera_projection_model if profile is None else profile
    if model not in ("standard", "k4"):
        raise ValueError("Unsupported camera projection model: %s" % model)
    return model


def image_point_to_body_vector(
    x: float,
    y: float,
    frame_w: int = 1024,
    frame_h: int = 1024,
    focal_length: Optional[float] = None,
    principal_point: Optional[Tuple[float, float]] = None,
    camera_profile: Optional[str] = None,
) -> np.ndarray:
    """将图像像素点转换为单位机体坐标向量。

    这个函数使用简单的针孔相机模型，将图像平面坐标映射到相机光轴方向上的单位矢量。
    """
    if principal_point is None:
        principal_point = (frame_w / 2.0, frame_h / 2.0)
    cx, cy = principal_point
    if focal_length is None:
        focal_length = max(frame_w, frame_h)
    if _projection_model(camera_profile) == "k4":
        vx = (y - cy) / focal_length
        vy = -(x - cx) / focal_length
    else:
        vx = (x - cx) / focal_length
        vy = -(y - cy) / focal_length
    vec = np.array([vx, vy, 1.0], dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError("Invalid image point for body vector conversion")
    return vec / norm


def body_vector_to_image_point(
    body_vector: np.ndarray,
    frame_w: int = 1024,
    frame_h: int = 1024,
    focal_length: Optional[float] = None,
    principal_point: Optional[Tuple[float, float]] = None,
    camera_profile: Optional[str] = None,
) -> np.ndarray:
    """将机体坐标系单位向量投影回图像像素坐标。"""
    vec = np.asarray(body_vector, dtype=np.float64)
    if vec.shape != (3,):
        raise ValueError("body_vector must be a 3-element vector")
    if principal_point is None:
        principal_point = (frame_w / 2.0, frame_h / 2.0)
    cx, cy = principal_point
    if focal_length is None:
        focal_length = max(frame_w, frame_h)
    if abs(vec[2]) < 1e-12:
        raise ValueError("Cannot project body vector with near-zero z component")
    if _projection_model(camera_profile) == "k4":
        x = cx - (vec[1] / vec[2]) * focal_length
        y = cy + (vec[0] / vec[2]) * focal_length
    else:
        x = cx + (vec[0] / vec[2]) * focal_length
        y = cy - (vec[1] / vec[2]) * focal_length
    return np.array([x, y], dtype=np.float64)


def project_catalog_stars(
    catalog_stars,
    rotation_matrix: np.ndarray,
    frame_w: int = 1024,
    frame_h: int = 1024,
    focal_length: Optional[float] = None,
    principal_point: Optional[Tuple[float, float]] = None,
    camera_profile: Optional[str] = None,
):
    """根据姿态将星表惯性向量投影到当前观测图像坐标系。"""
    projected = []
    for star in catalog_stars:
        if star.inertial_vector is None:
            continue
        body_vec = rotation_matrix.T @ star.inertial_vector
        if body_vec[2] <= 0:
            continue
        try:
            proj = body_vector_to_image_point(
                body_vec, frame_w=frame_w, frame_h=frame_h,
                focal_length=focal_length, principal_point=principal_point,
                camera_profile=camera_profile)
        except ValueError:
            continue
        projected.append({
            'catalog_id': star.id,
            'projected_x': float(proj[0]),
            'projected_y': float(proj[1]),
            'body_vector': body_vec,
            'inertial_vector': star.inertial_vector,
        })
    return projected


def project_inertial_vectors(
    inertial_vectors: np.ndarray,
    rotation_matrix: np.ndarray,
    frame_w: int,
    frame_h: int,
    focal_length: float,
    principal_point: Tuple[float, float],
    camera_profile: Optional[str] = None,
):
    """批量投影惯性方向，返回像素坐标及对应输入索引。"""
    vectors = np.asarray(inertial_vectors, dtype=np.float64).reshape(-1, 3)
    body_vectors = vectors @ np.asarray(rotation_matrix, dtype=np.float64)
    front = body_vectors[:, 2] > 1e-12
    input_indices = np.where(front)[0]
    body_visible = body_vectors[front]
    if len(body_visible) == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=int), body_visible
    cx, cy = principal_point
    if _projection_model(camera_profile) == "k4":
        pixels = np.column_stack([
            cx - focal_length * body_visible[:, 1] / body_visible[:, 2],
            cy + focal_length * body_visible[:, 0] / body_visible[:, 2]])
    else:
        pixels = np.column_stack([
            cx + focal_length * body_visible[:, 0] / body_visible[:, 2],
            cy - focal_length * body_visible[:, 1] / body_visible[:, 2]])
    in_frame = (
        (pixels[:, 0] >= 0.0) & (pixels[:, 0] < frame_w) &
        (pixels[:, 1] >= 0.0) & (pixels[:, 1] < frame_h))
    return pixels[in_frame], input_indices[in_frame], body_visible[in_frame]


def project_catalog_vectors(
    inertial_vectors,
    catalog_ids,
    rotation_matrix,
    frame_w,
    frame_h,
    focal_length,
    principal_point,
    magnitudes=None,
    camera_profile=None,
):
    """基于公共 CatalogData 的批量星表投影，不构造临时 CatalogStar 对象。

    返回 List[dict]，每个元素包含:
        catalog_id, projected_x, projected_y, body_vector, inertial_vector,
        以及可选的 magnitude。
    保持 body_vector = R.T @ inertial_vector 的旋转方向。
    """
    vectors = np.asarray(inertial_vectors, dtype=np.float64).reshape(-1, 3)
    ids = np.asarray(catalog_ids, dtype=np.int64).ravel()
    R = np.asarray(rotation_matrix, dtype=np.float64)
    cx, cy = principal_point

    body_vecs = vectors @ R  # (N, 3)  body = R.T @ inertial → @R for row-wise
    front = body_vecs[:, 2] > 1e-12
    if not np.any(front):
        return []

    body_visible = body_vecs[front]
    ids_visible = ids[front]
    if _projection_model(camera_profile) == "k4":
        px = cx - focal_length * body_visible[:, 1] / body_visible[:, 2]
        py = cy + focal_length * body_visible[:, 0] / body_visible[:, 2]
    else:
        px = cx + focal_length * body_visible[:, 0] / body_visible[:, 2]
        py = cy - focal_length * body_visible[:, 1] / body_visible[:, 2]
    in_frame = (px >= 0) & (px < frame_w) & (py >= 0) & (py < frame_h)

    if magnitudes is not None:
        mags = np.asarray(magnitudes, dtype=np.float64).ravel()
        mags_visible = mags[front] if len(mags) == len(vectors) else None
    else:
        mags_visible = None

    result = []
    for k in np.where(in_frame)[0]:
        entry = {
            'catalog_id': int(ids_visible[k]),
            'projected_x': float(px[k]),
            'projected_y': float(py[k]),
            'body_vector': body_visible[k].copy(),
            'inertial_vector': vectors[int(np.where(front)[0][k])].copy(),
        }
        if mags_visible is not None:
            entry['magnitude'] = float(mags_visible[k])
        result.append(entry)
    return result

