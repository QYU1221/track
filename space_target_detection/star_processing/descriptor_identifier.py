"""星图识别模块

基于全天星表惯性单位向量和检测点相机单位视线构建球面局部几何描述子，
采用 Top-K 候选匹配 + 旋转 RANSAC 完成未知姿态下的全天星图识别。

模块独立，可放在轨迹关联之前使用。

用法示例：
    catalog = StarCatalog.from_records([
        {'id': 'S1', 'x': 10.3, 'y': 20.5, 'mag': 5.1},
        {'id': 'S2', 'x': 50.2, 'y': 12.8, 'mag': 4.8},
        ...
    ])
    recognizer = StarMapRecognizer(catalog)
    result = recognizer.recognize_frame(candidates, meta=meta)

结果包含：
    recognized_stars  已识别恒星列表
    unmatched_stars   未识别星点列表
    transform         估计的 body→inertial 旋转矩阵
    global_confidence 识别置信度
"""
from __future__ import annotations

import math
import heapq
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

import numpy as np

from .attitude_solvers import solve_star_attitude

try:
    from scipy.spatial import cKDTree as KDTree
    from scipy.optimize import linear_sum_assignment
except ImportError:
    KDTree = None
    linear_sum_assignment = None


DEFAULT_K_NEIGHBORS = 7
DEFAULT_TOP_K = 8
DEFAULT_RANSAC_ITERS = 50
DEFAULT_RESIDUAL_THRESHOLD = 5.0
DEFAULT_MIN_CONFIDENCE = 0.35


class FallbackKDTree:
    def __init__(self, data: np.ndarray):
        self.data = np.asarray(data, dtype=np.float64)

    def query(self, points, k=1):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if self.data.size == 0:
            shape = (points.shape[0], k)
            return np.full(shape, np.inf), np.full(shape, -1, dtype=int)

        diff = points[:, None, :] - self.data[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)
        idx = np.argpartition(d2, min(k, d2.shape[1]) - 1, axis=1)[:, :k]
        idx = np.array([row[np.argsort(d2[i, row])] for i, row in enumerate(idx)], dtype=int)
        dist = np.sqrt(np.take_along_axis(d2, idx, axis=1))
        if dist.shape[1] == 1:
            return dist[:, 0], idx[:, 0]
        return dist, idx


def build_kdtree(points: np.ndarray):
    if KDTree is not None:
        return KDTree(points)
    return FallbackKDTree(points)


def _one_to_one_assignment(cost: np.ndarray):
    """返回矩形代价矩阵的全局一对一分配；无 SciPy 时退化为贪心。"""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    if linear_sum_assignment is not None:
        return linear_sum_assignment(cost)
    rows, cols = [], []
    used_rows, used_cols = set(), set()
    for _, row, col in sorted(
            (float(cost[r, c]), r, c)
            for r in range(cost.shape[0]) for c in range(cost.shape[1])):
        if row in used_rows or col in used_cols:
            continue
        used_rows.add(row)
        used_cols.add(col)
        rows.append(row)
        cols.append(col)
        if len(rows) >= min(cost.shape):
            break
    return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)


def _rank_correlation(first, second):
    """不依赖 scipy.stats 的局部亮度排名相关系数。"""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if len(first) < 3 or len(first) != len(second):
        return 0.5
    rank_a = np.argsort(np.argsort(first)).astype(np.float64)
    rank_b = np.argsort(np.argsort(second)).astype(np.float64)
    if np.std(rank_a) < 1e-12 or np.std(rank_b) < 1e-12:
        return 0.5
    corr = float(np.corrcoef(rank_a, rank_b)[0, 1])
    return float(np.clip((corr + 1.0) * 0.5, 0.0, 1.0))


def _build_spherical_descriptor(vectors, center_idx, neighbor_indices,
                                k_neighbors):
    """构建与坐标系旋转无关的球面局部几何描述子。"""
    pair_count = k_neighbors * (k_neighbors - 1) // 2
    desired_length = k_neighbors + pair_count
    neighbors = np.asarray(neighbor_indices, dtype=int)[:k_neighbors]
    if len(neighbors) == 0:
        return np.zeros(desired_length, dtype=np.float64)

    center = vectors[int(center_idx)]
    neighbor_vectors = vectors[neighbors]
    radial = np.arccos(np.clip(neighbor_vectors @ center, -1.0, 1.0))
    radial = np.sort(radial)

    pairwise = []
    for i in range(len(neighbor_vectors)):
        for j in range(i + 1, len(neighbor_vectors)):
            pairwise.append(float(np.arccos(np.clip(
                np.dot(neighbor_vectors[i], neighbor_vectors[j]), -1.0, 1.0))))
    pairwise = np.sort(np.asarray(pairwise, dtype=np.float64))

    descriptor = np.concatenate([radial, pairwise])
    if descriptor.size < desired_length:
        descriptor = np.pad(descriptor, (0, desired_length - descriptor.size),
                            constant_values=np.pi)
    return descriptor[:desired_length]


@dataclass
class CatalogStar:
    id: Any
    x: float
    y: float
    mag: float
    descriptor: Optional[np.ndarray] = None
    inertial_vector: Optional[np.ndarray] = None


@dataclass
class CandidateStar:
    x: float
    y: float
    flux: float
    energy: Optional[float] = None
    area: Optional[float] = None
    snr: Optional[float] = None
    confidence: Optional[float] = None
    descriptor: Optional[np.ndarray] = None
    body_vector: Optional[np.ndarray] = None
    radial_signature: Optional[np.ndarray] = None
    neighbor_indices: Optional[np.ndarray] = None


@dataclass
class RecognitionResult:
    recognized_stars: List[Dict[str, Any]]
    unmatched_stars: List[Dict[str, Any]]
    transform: Optional[np.ndarray]
    global_confidence: float


def _ra_dec_to_unit_vector(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    return np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ], dtype=np.float64)


def _infer_inertial_vector(record: Dict[str, Any]) -> Optional[np.ndarray]:
    if 'ix' in record and 'iy' in record and 'iz' in record:
        return np.array([float(record['ix']), float(record['iy']), float(record['iz'])], dtype=np.float64)
    if 'ux' in record and 'uy' in record and 'uz' in record:
        return np.array([float(record['ux']), float(record['uy']), float(record['uz'])], dtype=np.float64)
    if 'ra' in record and 'dec' in record:
        return _ra_dec_to_unit_vector(float(record['ra']), float(record['dec']))
    return None


class StarCatalog:
    def __init__(self, stars: List[CatalogStar], k_neighbors: int = DEFAULT_K_NEIGHBORS):
        self.stars = stars
        self.k_neighbors = k_neighbors
        self._build_index()

    @classmethod
    def from_catalog_data(cls, catalog_data, k_neighbors: int = DEFAULT_K_NEIGHBORS):
        """由公共 CatalogData 直接构造描述子星表，不重复读取文件。"""
        stars = []
        for i in range(len(catalog_data.catalog_ids)):
            stars.append(CatalogStar(
                id=int(catalog_data.catalog_ids[i]),
                x=0.0,
                y=0.0,
                mag=float(catalog_data.magnitudes[i]),
                inertial_vector=catalog_data.unit_vectors[i].copy(),
            ))
        return cls(stars, k_neighbors=k_neighbors)

    @classmethod
    def from_records(cls, records: List[Dict[str, Any]], k_neighbors: int = DEFAULT_K_NEIGHBORS):
        stars = []
        for idx, r in enumerate(records):
            stars.append(CatalogStar(
                id=r.get('id', idx),
                x=float(r['x']),
                y=float(r['y']),
                mag=float(r.get('mag', 0.0)),
                inertial_vector=_infer_inertial_vector(r),
            ))
        return cls(stars, k_neighbors=k_neighbors)

    def _build_index(self):
        vectors = np.asarray([
            s.inertial_vector for s in self.stars
        ], dtype=np.float64).reshape(-1, 3)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        valid = np.all(np.isfinite(vectors), axis=1) & (norms[:, 0] > 1e-12)
        if not np.all(valid):
            raise ValueError("全天描述子星表中的每颗星都必须具有有效惯性单位向量")
        self._vectors = vectors / norms
        self.index = build_kdtree(self._vectors)
        self._compute_descriptors()

    def _compute_descriptors(self):
        vectors = self._vectors
        self._descriptors = []
        self._radial_signatures = []
        self._neighbor_indices = []

        if len(vectors) == 0:
            self._descriptor_matrix = np.zeros((0, 0), dtype=np.float64)
            self._radial_matrix = np.zeros((0, 0), dtype=np.float64)
            return

        k_query = min(self.k_neighbors + 1, len(vectors))
        _, idx = self.index.query(vectors, k=k_query)
        if idx.ndim == 1:
            idx = idx[:, None]

        for row_idx, neighbors in enumerate(idx):
            neighbors = neighbors[neighbors != row_idx]
            neighbors = neighbors[: self.k_neighbors]
            radial = np.arccos(np.clip(
                vectors[neighbors] @ vectors[row_idx], -1.0, 1.0))
            order = np.argsort(radial)
            radial = radial[order]
            neighbors = neighbors[order]
            self._radial_signatures.append(radial)
            self._neighbor_indices.append(np.asarray(neighbors, dtype=int))
            desc = _build_spherical_descriptor(
                vectors, row_idx, neighbors, self.k_neighbors)
            self._descriptors.append(desc)
            self.stars[row_idx].descriptor = desc
        self._descriptor_matrix = np.asarray(self._descriptors, dtype=np.float64)
        self._radial_matrix = np.asarray(
            self._radial_signatures, dtype=np.float64)

    def query_partial_candidates(self, observed_radial, observed_neighbor_vectors,
                                 observed_neighbor_fluxes, top_k, min_support,
                                 trimmed_matches, tolerance, pair_tolerance):
        """径向一对一分配后，以邻星两两角距离评价完整局部几何。"""
        if len(self._radial_signatures) == 0:
            return []
        observed = np.asarray(observed_radial, dtype=np.float64)
        if observed.size == 0:
            return []
        obs_vectors = np.asarray(observed_neighbor_vectors, dtype=np.float64)
        obs_fluxes = np.asarray(observed_neighbor_fluxes, dtype=np.float64)
        results = []
        for catalog_idx, catalog_radial in enumerate(self._radial_signatures):
            cost = np.abs(observed[:, None] - catalog_radial[None, :])
            obs_rows, cat_cols = _one_to_one_assignment(cost)
            errors = cost[obs_rows, cat_cols]
            valid = errors <= tolerance
            obs_rows, cat_cols, errors = (
                obs_rows[valid], cat_cols[valid], errors[valid])
            support = len(errors)
            if support < min_support:
                continue

            radial_kept = np.sort(errors)[:trimmed_matches]
            radial_median = float(np.median(radial_kept))
            support_score = support / max(1, len(observed))
            radial_score = support_score * math.exp(
                -radial_median / max(tolerance, 1e-12))

            cat_neighbor_ids = self._neighbor_indices[catalog_idx][cat_cols]
            cat_vectors = self._vectors[cat_neighbor_ids]
            matched_obs_vectors = obs_vectors[obs_rows]
            pair_errors = []
            for first in range(support):
                for second in range(first + 1, support):
                    obs_angle = np.arccos(np.clip(np.dot(
                        matched_obs_vectors[first], matched_obs_vectors[second]),
                        -1.0, 1.0))
                    cat_angle = np.arccos(np.clip(np.dot(
                        cat_vectors[first], cat_vectors[second]), -1.0, 1.0))
                    pair_errors.append(abs(float(obs_angle - cat_angle)))
            pair_errors = np.asarray(pair_errors, dtype=np.float64)
            if pair_errors.size:
                pair_support_ratio = float(np.mean(pair_errors <= pair_tolerance))
                pair_median = float(np.median(pair_errors))
                pair_score = pair_support_ratio * math.exp(
                    -pair_median / max(pair_tolerance, 1e-12))
            else:
                pair_support_ratio, pair_median, pair_score = 0.0, float('inf'), 0.0

            cat_mags = np.asarray([
                self.stars[index].mag for index in cat_neighbor_ids
            ], dtype=np.float64)
            magnitude_score = _rank_correlation(
                obs_fluxes[obs_rows], -cat_mags)
            geometry_score = 0.45 * radial_score + 0.55 * pair_score
            total_score = (
                0.68 * geometry_score +
                0.20 * support_score +
                0.12 * magnitude_score)
            results.append({
                'catalog_index': int(catalog_idx),
                'descriptor_distance': radial_median,
                'support': int(support),
                'support_score': float(support_score),
                'radial_score': float(radial_score),
                'pair_score': float(pair_score),
                'pair_support_ratio': float(pair_support_ratio),
                'pair_median': float(pair_median),
                'magnitude_score': float(magnitude_score),
                'geometry_score': float(geometry_score),
                'score': float(total_score),
            })
        results.sort(key=lambda item: (
            -item['score'], -item['support'], item['descriptor_distance']))
        return results[:top_k]


class StarMapRecognizer:
    def __init__(
        self,
        catalog: StarCatalog,
        k_neighbors: Optional[int] = None,
        top_k: int = DEFAULT_TOP_K,
        ransac_iters: int = DEFAULT_RANSAC_ITERS,
        residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        focal_length_px: Optional[float] = None,
        principal_point: Optional[Tuple[float, float]] = None,
        min_support: int = 4,
        trimmed_matches: int = 6,
        radial_tolerance: float = 0.0015,
        pair_tolerance: float = 0.003,
        max_observations: Optional[int] = 50,
        max_top_k: int = 30,
        candidate_score_margin: float = 0.15,
        min_pair_support_ratio: float = 0.60,
        triple_min_separation: float = 0.002,
        triple_min_shape: float = 0.05,
        sampling_grid: Tuple[int, int] = (4, 4),
        max_per_cell: int = 4,
        attitude_method: str = 'svd',
        debug: bool = False,
    ):
        self.catalog = catalog
        self.k_neighbors = k_neighbors if k_neighbors is not None else catalog.k_neighbors
        self.top_k = top_k
        self.ransac_iters = ransac_iters
        self.residual_threshold = residual_threshold
        self.min_confidence = min_confidence
        self.focal_length_px = float(focal_length_px) if focal_length_px else None
        self.principal_point = principal_point
        self.min_support = int(min_support)
        self.trimmed_matches = int(trimmed_matches)
        self.radial_tolerance = float(radial_tolerance)
        self.pair_tolerance = float(pair_tolerance)
        self.max_observations = (
            None if max_observations is None else int(max_observations))
        self.max_top_k = max(int(max_top_k), int(top_k))
        self.candidate_score_margin = float(candidate_score_margin)
        self.min_pair_support_ratio = float(min_pair_support_ratio)
        self.triple_min_separation = float(triple_min_separation)
        self.triple_min_shape = float(triple_min_shape)
        self.sampling_grid = (max(1, int(sampling_grid[0])),
                              max(1, int(sampling_grid[1])))
        self.max_per_cell = max(1, int(max_per_cell))
        self.attitude_method = str(attitude_method)
        self.debug = bool(debug)
        if self.focal_length_px is None or self.principal_point is None:
            raise ValueError("全天球面描述子需要 focal_length_px 和 principal_point")

        if self.catalog.k_neighbors < self.k_neighbors:
            raise ValueError("星表邻域池不能小于观测描述子邻域")

    def _build_candidate_records(
        self,
        candidates: List[Tuple[float, float, float]],
        meta: Optional[Dict[Tuple[float, float], Dict[str, Any]]] = None,
    ) -> List[CandidateStar]:
        result = []
        for cx, cy, flux in candidates:
            key = (round(cx, 2), round(cy, 2))
            extra = meta.get(key, {}) if meta is not None else {}
            result.append(CandidateStar(
                x=float(cx),
                y=float(cy),
                flux=float(flux),
                energy=float(extra.get('energy', flux)) if extra.get('energy', None) is not None else float(flux),
                area=float(extra.get('area', 0.0)) if extra.get('area', None) is not None else None,
                snr=float(extra.get('snr', 0.0)) if extra.get('snr', None) is not None else None,
                confidence=float(extra.get('confidence', 0.0)) if extra.get('confidence', None) is not None else None,
            ))
        return result

    def _compute_candidate_descriptors(self, candidates: List[CandidateStar]) -> None:
        if not candidates:
            return

        cx, cy = self.principal_point
        vectors = np.asarray([
            [(c.y - cy) / self.focal_length_px,
             -(c.x - cx) / self.focal_length_px,
             1.0]
            for c in candidates
        ], dtype=np.float64)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        for candidate, vector in zip(candidates, vectors):
            candidate.body_vector = vector

        tree = build_kdtree(vectors)
        k_query = min(self.k_neighbors + 1, len(vectors))
        _, idx = tree.query(vectors, k=k_query)
        if idx.ndim == 1:
            idx = idx[:, None]

        for row_idx, neighbors in enumerate(idx):
            neighbors = neighbors[neighbors != row_idx]
            neighbors = neighbors[: self.k_neighbors]
            radial = np.arccos(np.clip(
                vectors[neighbors] @ vectors[row_idx], -1.0, 1.0))
            order = np.argsort(radial)
            neighbors = neighbors[order]
            candidates[row_idx].radial_signature = radial[order]
            candidates[row_idx].neighbor_indices = np.asarray(neighbors, dtype=int)
            desc = _build_spherical_descriptor(
                vectors, row_idx, neighbors, self.k_neighbors)
            candidates[row_idx].descriptor = desc

    def _make_topk_matches(self, candidates: List[CandidateStar]) -> List[List[Dict[str, Any]]]:
        matches = []
        for cand in candidates:
            if cand.descriptor is None or cand.neighbor_indices is None:
                matches.append([])
                continue
            neighbor_vectors = np.asarray([
                candidates[index].body_vector for index in cand.neighbor_indices
            ], dtype=np.float64)
            neighbor_fluxes = np.asarray([
                candidates[index].flux for index in cand.neighbor_indices
            ], dtype=np.float64)
            ranked = self.catalog.query_partial_candidates(
                cand.radial_signature, neighbor_vectors, neighbor_fluxes,
                self.max_top_k, self.min_support, self.trimmed_matches,
                self.radial_tolerance, self.pair_tolerance)
            cand_matches = []
            frame_w = max(2.0 * self.principal_point[0], 1.0)
            frame_h = max(2.0 * self.principal_point[1], 1.0)
            edge_distance = min(cand.x, frame_w - cand.x, cand.y, frame_h - cand.y)
            center_reliability = float(np.clip(
                edge_distance / max(0.1 * min(frame_w, frame_h), 1.0), 0.0, 1.0))
            snr_score = float(np.clip((cand.snr or 0.0) / 20.0, 0.0, 1.0))
            detection_score = max(snr_score, float(np.clip(
                cand.confidence or 0.0, 0.0, 1.0)))
            for item in ranked:
                if item['pair_support_ratio'] < self.min_pair_support_ratio:
                    continue
                score = (0.82 * item['score'] +
                         0.10 * detection_score +
                         0.08 * center_reliability)
                cand_matches.append({
                    **item,
                    'catalog_star': self.catalog.stars[item['catalog_index']],
                    'score': float(score),
                    'detection_score': detection_score,
                    'center_reliability': center_reliability,
                })
            cand_matches.sort(key=lambda item: -item['score'])
            if cand_matches:
                cutoff = cand_matches[0]['score'] - self.candidate_score_margin
                cand_matches = [item for item in cand_matches if item['score'] >= cutoff]
            cand_matches = cand_matches[:self.max_top_k]
            matches.append(cand_matches)
        return matches

    def _select_spatially_distributed(self, candidates: List[CandidateStar]):
        """优先保留高质量星，同时避免最亮星全部集中在局部区域。"""
        if self.max_observations is None or len(candidates) <= self.max_observations:
            return candidates
        frame_w = max(2.0 * self.principal_point[0], 1.0)
        frame_h = max(2.0 * self.principal_point[1], 1.0)
        grid_x, grid_y = self.sampling_grid
        ordered = sorted(candidates, key=lambda candidate: (
            candidate.energy if candidate.energy is not None else candidate.flux),
            reverse=True)
        counts = {}
        selected, deferred = [], []
        for candidate in ordered:
            cell = (
                min(grid_x - 1, max(0, int(candidate.x / frame_w * grid_x))),
                min(grid_y - 1, max(0, int(candidate.y / frame_h * grid_y))),
            )
            if counts.get(cell, 0) < self.max_per_cell:
                selected.append(candidate)
                counts[cell] = counts.get(cell, 0) + 1
            else:
                deferred.append(candidate)
            if len(selected) >= self.max_observations:
                return selected
        selected.extend(deferred[:self.max_observations - len(selected)])
        return selected

    def _search_consistent_matches(
        self,
        candidates: List[CandidateStar],
        candidate_matches: List[List[Dict[str, Any]]],
    ) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
        all_pairs = []
        for i, cand_match_list in enumerate(candidate_matches):
            for match in cand_match_list:
                all_pairs.append((i, match['catalog_index'], match['score']))

        if len(all_pairs) < 3:
            return [], None

        best_inliers = []
        best_transform = None
        best_total_score = 0.0
        body_vectors = np.asarray([c.body_vector for c in candidates], dtype=np.float64)
        catalog_vectors = self.catalog._vectors
        candidate_positions = np.asarray(
            [[candidate.x, candidate.y] for candidate in candidates],
            dtype=np.float64)
        descriptor_distance_lookup = {
            (candidate_idx, match['catalog_index']): match['descriptor_distance']
            for candidate_idx, matches in enumerate(candidate_matches)
            for match in matches
        }

        def _evaluate_transform(rotation: np.ndarray):
            # body→inertial 的逆投影：行向量形式下 camera = inertial @ R。
            camera_vectors = catalog_vectors @ rotation
            visible = camera_vectors[:, 2] > 1e-12
            visible_indices = np.where(visible)[0]
            camera_visible = camera_vectors[visible]
            from configuration.values import camera_projection_model
            if camera_projection_model == 'k4':
                px = (self.principal_point[0] - self.focal_length_px *
                      camera_visible[:, 1] / camera_visible[:, 2])
                py = (self.principal_point[1] + self.focal_length_px *
                      camera_visible[:, 0] / camera_visible[:, 2])
            else:
                px = (self.principal_point[0] + self.focal_length_px *
                      camera_visible[:, 0] / camera_visible[:, 2])
                py = (self.principal_point[1] - self.focal_length_px *
                      camera_visible[:, 1] / camera_visible[:, 2])
            in_frame = ((px >= 0.0) & (px < 2.0 * self.principal_point[0]) &
                        (py >= 0.0) & (py < 2.0 * self.principal_point[1]))
            visible_indices = visible_indices[in_frame]
            px, py = px[in_frame], py[in_frame]
            projected = np.column_stack([px, py])
            if len(projected) == 0:
                return [], 0.0

            distances = np.linalg.norm(
                projected[:, None, :] - candidate_positions[None, :, :], axis=2)
            assignments = []
            total_score = 0.0
            gated_cost = distances.copy()
            gated_cost[gated_cost > self.residual_threshold] = 1e9
            cat_rows, cand_cols = _one_to_one_assignment(gated_cost)
            for visible_row, cand_idx in zip(cat_rows, cand_cols):
                residual = float(distances[visible_row, cand_idx])
                if residual > self.residual_threshold:
                    continue
                cat_idx = int(visible_indices[visible_row])
                score = float(np.exp(
                    -residual / max(self.residual_threshold, 1e-6)))
                assignments.append({
                    'candidate_index': cand_idx,
                    'catalog_index': cat_idx,
                    'catalog_star': self.catalog.stars[cat_idx],
                    'descriptor_distance': descriptor_distance_lookup.get(
                        (cand_idx, cat_idx), float('inf')),
                    'residual': float(residual),
                    'score': score,
                })
                total_score += score
            return assignments, total_score

        sample_size = 3
        # 建立候选对应一致性图：两组对应只有在观测夹角与星表夹角一致时连边。
        adjacency = [set() for _ in all_pairs]
        edge_scores = {}
        for first in range(len(all_pairs)):
            cand_a, cat_a, _ = all_pairs[first]
            for second in range(first + 1, len(all_pairs)):
                cand_b, cat_b, _ = all_pairs[second]
                if cand_a == cand_b or cat_a == cat_b:
                    continue
                body_angle = np.arccos(np.clip(np.dot(
                    body_vectors[cand_a], body_vectors[cand_b]), -1.0, 1.0))
                catalog_angle = np.arccos(np.clip(np.dot(
                    catalog_vectors[cat_a], catalog_vectors[cat_b]), -1.0, 1.0))
                angle_error = abs(float(body_angle - catalog_angle))
                if angle_error <= self.pair_tolerance:
                    adjacency[first].add(second)
                    adjacency[second].add(first)
                    edge_scores[(first, second)] = math.exp(
                        -angle_error / max(self.pair_tolerance, 1e-12))

        # 保留综合质量最高的三元组，而不是固定枚举顺序中的前 N 个。
        triple_heap = []
        triple_serial = 0
        for first in range(len(all_pairs)):
            for second in sorted(idx for idx in adjacency[first] if idx > first):
                common = adjacency[first].intersection(adjacency[second])
                for third in sorted(idx for idx in common if idx > second):
                    triple = (all_pairs[first], all_pairs[second], all_pairs[third])
                    cand_ids = [item[0] for item in triple]
                    cat_ids = [item[1] for item in triple]
                    if len(set(cand_ids)) == 3 and len(set(cat_ids)) == 3:
                        angular_sides = np.asarray([
                            np.arccos(np.clip(np.dot(
                                body_vectors[cand_ids[a]], body_vectors[cand_ids[b]]),
                                -1.0, 1.0))
                            for a, b in ((0, 1), (0, 2), (1, 2))
                        ], dtype=np.float64)
                        min_separation = float(np.min(angular_sides))
                        semiperimeter = float(np.sum(angular_sides) * 0.5)
                        area = math.sqrt(max(
                            semiperimeter * np.prod(
                                np.maximum(semiperimeter - angular_sides, 0.0)),
                            0.0))
                        shape_quality = float(
                            4.0 * math.sqrt(3.0) * area /
                            max(float(np.sum(angular_sides ** 2)), 1e-12))
                        if (min_separation < self.triple_min_separation or
                                shape_quality < self.triple_min_shape):
                            continue
                        node_score = float(np.mean([item[2] for item in triple]))
                        edge_score = float(np.mean([
                            edge_scores[(first, second)],
                            edge_scores[(first, third)],
                            edge_scores[(second, third)],
                        ]))
                        spread_score = float(np.clip(
                            min_separation /
                            max(5.0 * self.triple_min_separation, 1e-12),
                            0.0, 1.0))
                        triple_score = (
                            0.45 * node_score + 0.35 * edge_score +
                            0.12 * spread_score + 0.08 * shape_quality)
                        heap_item = (triple_score, triple_serial, triple)
                        triple_serial += 1
                        if len(triple_heap) < self.ransac_iters:
                            heapq.heappush(triple_heap, heap_item)
                        elif triple_score > triple_heap[0][0]:
                            heapq.heapreplace(triple_heap, heap_item)
        compatible_triples = [
            item[2] for item in sorted(triple_heap, reverse=True)
        ]
        sample_iterator = compatible_triples

        for sample in sample_iterator:
            cand_ids = [p[0] for p in sample]
            cat_ids = [p[1] for p in sample]
            if len(set(cand_ids)) < len(cand_ids) or len(set(cat_ids)) < len(cat_ids):
                continue

            # 旋转保持任意两条视线的夹角。先用三对角距离一致性排除明显
            # 错误假设，再执行 SVD 和全候选评分。
            angular_gate = self.pair_tolerance
            angle_consistent = True
            for a in range(sample_size):
                for b in range(a + 1, sample_size):
                    body_angle = np.arccos(np.clip(np.dot(
                        body_vectors[cand_ids[a]], body_vectors[cand_ids[b]]),
                        -1.0, 1.0))
                    catalog_angle = np.arccos(np.clip(np.dot(
                        catalog_vectors[cat_ids[a]], catalog_vectors[cat_ids[b]]),
                        -1.0, 1.0))
                    if abs(body_angle - catalog_angle) > angular_gate:
                        angle_consistent = False
                        break
                if not angle_consistent:
                    break
            if not angle_consistent:
                continue

            try:
                _, rotation = solve_star_attitude(
                    body_vectors[cand_ids], catalog_vectors[cat_ids],
                    method=self.attitude_method)
            except Exception:
                continue

            inliers, total_score = _evaluate_transform(rotation)
            if len(inliers) > len(best_inliers) or (
                len(inliers) == len(best_inliers) and total_score > best_total_score
            ):
                best_inliers = inliers
                best_transform = rotation
                best_total_score = total_score

        if not best_inliers:
            if self.debug:
                print('[descriptor-allsky] assignments=%d graph_edges=%d '
                      'triples=%d best_inliers=0' % (
                          len(all_pairs),
                          sum(len(items) for items in adjacency) // 2,
                          len(compatible_triples)))
            return [], None

        if self.debug:
            print('[descriptor-allsky] assignments=%d graph_edges=%d '
                  'triples=%d best_inliers=%d' % (
                      len(all_pairs),
                      sum(len(items) for items in adjacency) // 2,
                      len(compatible_triples), len(best_inliers)))

        return best_inliers, best_transform

    def recognize_frame(
        self,
        candidates: List[Tuple[float, float, float]],
        meta: Optional[Dict[Tuple[float, float], Dict[str, Any]]] = None,
    ) -> RecognitionResult:
        candidate_list = self._build_candidate_records(candidates, meta)
        candidate_list = self._select_spatially_distributed(candidate_list)
        self._compute_candidate_descriptors(candidate_list)
        candidate_matches = self._make_topk_matches(candidate_list)
        inliers, transform = self._search_consistent_matches(candidate_list, candidate_matches)

        recognized = []
        matched_candidate_indices = set()
        for item in inliers:
            idx = item['candidate_index']
            cand = candidate_list[idx]
            confidence = float(np.clip(item['score'], 0.0, 1.0))
            if confidence < self.min_confidence:
                continue
            matched_candidate_indices.add(idx)
            recognized_item = {
                'x': cand.x,
                'y': cand.y,
                'flux': cand.flux,
                'catalog_id': item['catalog_star'].id,
                'catalog_x': item['catalog_star'].x,
                'catalog_y': item['catalog_star'].y,
                'magnitude': item['catalog_star'].mag,
                'descriptor_distance': item['descriptor_distance'],
                'residual': item['residual'],
                'confidence': confidence,
                'body_vector': cand.body_vector,
            }
            if item['catalog_star'].inertial_vector is not None:
                recognized_item['catalog_inertial_vector'] = item['catalog_star'].inertial_vector
                recognized_item['inertial_vector'] = item['catalog_star'].inertial_vector
            recognized.append(recognized_item)

        unmatched = []
        for idx, cand in enumerate(candidate_list):
            if idx in matched_candidate_indices:
                continue
            unmatched.append({
                'x': cand.x,
                'y': cand.y,
                'flux': cand.flux,
                'energy': cand.energy,
                'area': cand.area,
                'snr': cand.snr,
                'confidence': cand.confidence,
            })

        global_confidence = float(
            len(recognized) / max(1, len(candidate_list)) if candidate_list else 0.0
        )

        return RecognitionResult(
            recognized_stars=recognized,
            unmatched_stars=unmatched,
            transform=transform,
            global_confidence=global_confidence,
        )

    @staticmethod
    def _match_candidate_indices(
        candidates: List[Tuple[float, float, float]],
        background: List[Dict[str, Any]],
    ) -> set:
        matched_indices = set()
        candidate_index = {
            (round(cx, 3), round(cy, 3)): idx
            for idx, (cx, cy, _) in enumerate(candidates)
        }
        for entry in background:
            mx = entry.get('matched_x')
            my = entry.get('matched_y')
            if mx is None or my is None:
                continue
            key = (round(mx, 3), round(my, 3))
            idx = candidate_index.get(key)
            if idx is not None:
                matched_indices.add(idx)
        return matched_indices

    @staticmethod
    def extract_unknown_targets(
        candidates: List[Tuple[float, float, float]],
        background: List[Dict[str, Any]],
        match_radius: float = 6.0,
        missed_radius: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[float, float, float]]]:
        """根据背景模型剔除星表匹配点，仅保留未知目标候选。"""
        if missed_radius is None:
            missed_radius = match_radius * 2.0

        matched_candidate_indices = StarMapRecognizer._match_candidate_indices(candidates, background)
        projected_points = np.array(
            [[entry['projected_x'], entry['projected_y']] for entry in background],
            dtype=np.float64,
        ) if background else np.zeros((0, 2), dtype=np.float64)
        tree = build_kdtree(projected_points) if len(projected_points) > 0 else None

        unknown_points = []
        for idx, (cx, cy, flux) in enumerate(candidates):
            if idx in matched_candidate_indices:
                continue

            source = 'unknown_target'
            matched_catalog_id = None
            residual = None
            if tree is not None and len(projected_points) > 0:
                dist, proj_idx = tree.query(np.array([cx, cy], dtype=np.float64), k=1)
                if np.isscalar(dist):
                    dist = float(dist)
                    proj_idx = int(proj_idx)
                else:
                    dist = float(dist[0])
                    proj_idx = int(proj_idx[0])
                matched_catalog_id = background[proj_idx].get('catalog_id')
                residual = dist
                if dist <= match_radius:
                    source = 'missed_star_candidate'
                elif dist <= missed_radius:
                    source = 'noise_candidate'
                else:
                    source = 'unknown_target'

            unknown_points.append({
                'x': float(cx),
                'y': float(cy),
                'flux': float(flux),
                'source': source,
                'matched_catalog_id': matched_catalog_id,
                'residual': residual,
            })

        unknown_candidates = [(p['x'], p['y'], p['flux']) for p in unknown_points
                              if p['source'] == 'unknown_target']
        return unknown_points, unknown_candidates

    @staticmethod
    def build_background_model(
        projected_catalog: List[Dict[str, Any]],
        candidates: List[Tuple[float, float, float]],
        recognized_stars: List[Dict[str, Any]],
        match_radius: float = 6.0,
    ) -> List[Dict[str, Any]]:
        """基于姿态反馈，将投影星表与当前检测候选再次匹配，形成完整背景模型。"""
        if not projected_catalog:
            return []

        cand_points = np.array([[cx, cy] for cx, cy, _ in candidates], dtype=np.float64)
        if cand_points.size == 0:
            return [
                {
                    'catalog_id': p['catalog_id'],
                    'projected_x': p['projected_x'],
                    'projected_y': p['projected_y'],
                    'matched_x': None,
                    'matched_y': None,
                    'confidence': 0.1,
                    'residual': None,
                    'status': 'predicted',
                }
                for p in projected_catalog
            ]

        tree = build_kdtree(cand_points)
        recognized_index = {
            (round(r['x'], 2), round(r['y'], 2)): r
            for r in recognized_stars
        }
        recognized_by_catalog = {
            r['catalog_id']: r
            for r in recognized_stars
            if r.get('catalog_id') is not None
        }
        used_candidate = set()
        matched = []
        for proj in projected_catalog:
            catalog_id = proj['catalog_id']
            recognized_entry = recognized_by_catalog.get(catalog_id)
            if recognized_entry is not None:
                # Preserve Stage1 recognition when catalog ID is already known.
                cand_x, cand_y = float(recognized_entry['x']), float(recognized_entry['y'])
                dist = float(np.hypot(proj['projected_x'] - cand_x, proj['projected_y'] - cand_y))
                confidence = float(np.clip(
                    float(recognized_entry.get('confidence', 0.0)) + 0.2,
                    0.0, 1.0))
                status = 'recognized'
                matched.append({
                    'catalog_id': catalog_id,
                    'projected_x': proj['projected_x'],
                    'projected_y': proj['projected_y'],
                    'matched_x': cand_x,
                    'matched_y': cand_y,
                    'confidence': confidence,
                    'residual': dist,
                    'status': status,
                })
                # Mark candidate as used if it exists in the current frame.
                key = (round(cand_x, 2), round(cand_y, 2))
                matched_idx = None
                for idx, (cx, cy, _) in enumerate(candidates):
                    if (round(cx, 2), round(cy, 2)) == key:
                        matched_idx = idx
                        break
                if matched_idx is not None:
                    used_candidate.add(matched_idx)
                continue

            projected_pt = np.array([proj['projected_x'], proj['projected_y']], dtype=np.float64)
            dist, idx = tree.query(projected_pt, k=1)
            if np.isscalar(dist):
                idx = int(idx)
                dist = float(dist)
            else:
                idx = int(idx[0])
                dist = float(dist[0])

            if dist <= match_radius and idx not in used_candidate:
                cand_x, cand_y, _ = candidates[idx]
                used_candidate.add(idx)
                recognized_entry = recognized_index.get((round(cand_x, 2), round(cand_y, 2)))
                if recognized_entry is not None:
                    base_conf = float(recognized_entry.get('confidence', 0.0))
                else:
                    base_conf = 0.0
                confidence = float(np.clip(
                    base_conf + np.exp(-dist / max(1.0, match_radius)) * 0.5,
                    0.0, 1.0))
                if recognized_entry is not None:
                    if recognized_entry.get('catalog_id') == proj['catalog_id']:
                        status = 'recognized'
                    else:
                        status = 'corrected'
                else:
                    status = 'recovered'
                matched.append({
                    'catalog_id': proj['catalog_id'],
                    'projected_x': proj['projected_x'],
                    'projected_y': proj['projected_y'],
                    'matched_x': float(cand_x),
                    'matched_y': float(cand_y),
                    'confidence': confidence,
                    'residual': dist,
                    'status': status,
                })
            else:
                matched.append({
                    'catalog_id': proj['catalog_id'],
                    'projected_x': proj['projected_x'],
                    'projected_y': proj['projected_y'],
                    'matched_x': None,
                    'matched_y': None,
                    'confidence': 0.1,
                    'residual': None,
                    'status': 'predicted',
                })

        return matched
