"""Delaunay catalog star identification.

星图识别模块

基于 Delaunay 三角剖分和 RANSAC 验证的星图识别算法。

流程：
1. 选取一个 Delaunay 三角形，在星表中搜索三边角距匹配的三角形
2. 对每个候选匹配三角形，用 SVD 求解试探姿态
3. 将星表亮星批量投影到像面，统计与观测星的重合数
4. 选取得分最高的姿态，用所有匹配星对重新求解最终姿态
"""

import hashlib
import os

import numpy as np
from collections import defaultdict

from .attitude_solvers import solve_star_attitude
from utils.point_matching import match_projected_points
from .projection_geometry import (
    image_point_to_body_vector, project_inertial_vectors)


class DelaunayCatalogIndex:
    """Delaunay 星图识别专用的星表索引。

    基于 CatalogData 构建星对角距数据库和二分查询索引。
    内部使用筛选后的数组行号 catalog_row，对外输出原始星表编号 catalog_id。
    """

    def __init__(self, catalog_data, cache_enabled=True, cache_rebuild=False,
                 cache_dir=None, catalog_path=None):
        self._data = catalog_data
        self._n = len(catalog_data.catalog_ids)
        self.catalog_ids = catalog_data.catalog_ids
        self.unit_vectors = catalog_data.unit_vectors
        self.magnitudes = catalog_data.magnitudes

        self.pair_angles = None
        self.pair_ids = None
        self._pair_cache_enabled = bool(cache_enabled)
        self._pair_cache_rebuild = bool(cache_rebuild)
        self._pair_cache_dir = cache_dir
        self._catalog_path = os.path.abspath(catalog_path) if catalog_path else None

        self._bright_order = np.argsort(self.magnitudes)

        if not self._load_pair_cache():
            self._build_pair_database()
            self._save_pair_cache()

    # ── 编号转换 ──────────────────────────────────────────────

    def get_catalog_id(self, catalog_row):
        """内部行号 → 原始星表编号。"""
        return int(self.catalog_ids[catalog_row])

    def get_star_vector(self, catalog_row):
        """内部行号 → ICRS 单位方向向量（副本）。"""
        return self.unit_vectors[catalog_row].copy()

    def get_star_magnitude(self, catalog_row):
        """内部行号 → 星等。"""
        return float(self.magnitudes[catalog_row])

    def __len__(self):
        return self._n

    # ── 亮星查询 ──────────────────────────────────────────────

    def get_bright_vectors(self, n_bright=2000):
        """返回按亮度排序的前 n_bright 颗星的向量及行号。"""
        n = min(n_bright, self._n)
        rows = self._bright_order[:n]
        return self.unit_vectors[rows], rows

    # ── 星对数据库构建 ────────────────────────────────────────

    def _build_pair_database(self):
        n = self._n
        n_pairs = n * (n - 1) // 2
        pair_angles = np.zeros(n_pairs, dtype=np.float64)
        pair_ids = np.zeros((n_pairs, 2), dtype=np.int32)
        k = 0
        for i in range(n):
            count = n - i - 1
            if count <= 0:
                continue
            end = k + count
            dots = self.unit_vectors[i + 1:] @ self.unit_vectors[i]
            pair_angles[k:end] = np.arccos(np.clip(dots, -1.0, 1.0))
            pair_ids[k:end, 0] = i
            pair_ids[k:end, 1] = np.arange(i + 1, n, dtype=np.int32)
            k = end
        order = np.argsort(pair_angles)
        self.pair_angles = pair_angles[order]
        self.pair_ids = pair_ids[order]

    # ── 缓存 ──────────────────────────────────────────────────

    def _pair_cache_path(self):
        if not self._pair_cache_enabled or self._catalog_path is None:
            return None
        stat = os.stat(self._catalog_path)
        signature = '%s|%d|%d|%d|v3' % (
            self._catalog_path, stat.st_size, int(stat.st_mtime), self._n)
        digest = hashlib.sha1(signature.encode('utf-8')).hexdigest()[:16]
        cache_dir = self._pair_cache_dir or os.path.join(
            os.path.dirname(self._catalog_path), '.star_cache')
        return os.path.join(cache_dir, 'delaunay_pairs_%s.npz' % digest)

    def _load_pair_cache(self):
        path = self._pair_cache_path()
        if path is None or self._pair_cache_rebuild or not os.path.exists(path):
            return False
        try:
            with np.load(path, allow_pickle=False) as data:
                pair_angles = data['pair_angles']
                pair_ids = data['pair_ids']
            expected = self._n * (self._n - 1) // 2
            if len(pair_angles) != expected or pair_ids.shape != (expected, 2):
                return False
            self.pair_angles = pair_angles
            self.pair_ids = pair_ids
            return True
        except Exception:
            return False

    def _save_pair_cache(self):
        path = self._pair_cache_path()
        if path is None:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temp_path = path + '.tmp.npz'
        np.savez(temp_path, pair_angles=self.pair_angles, pair_ids=self.pair_ids)
        os.replace(temp_path, path)

    # ── 查询 ──────────────────────────────────────────────────

    def _ensure_pair_index(self):
        if self.pair_angles is None:
            raise RuntimeError(
                "Delaunay pair index not built; ensure cache_enabled=True")

    def query_pairs(self, angle, tolerance=0.005):
        """查询与给定角距匹配的星表星对，返回 (catalog_row1, catalog_row2) 列表。"""
        self._ensure_pair_index()
        left = int(np.searchsorted(self.pair_angles, angle - tolerance, side='left'))
        right = int(np.searchsorted(self.pair_angles, angle + tolerance, side='right'))
        return [(int(i), int(j)) for i, j in self.pair_ids[left:right]]

    def query_triangles(self, d1, d2, d3, tolerance=0.005, max_results=300):
        """给定排序三边角距 (d1≤d2≤d3)，查找匹配的星表三角。

        返回 List[(catalog_row1, catalog_row2, catalog_row3)]。
        """
        pairs_d1 = self.query_pairs(d1, tolerance)
        if not pairs_d1:
            return []

        results = []
        angles_from_p = {}
        d2_lo, d2_hi = d2 - tolerance, d2 + tolerance
        d3_lo, d3_hi = d3 - tolerance, d3 + tolerance

        for p, q in pairs_d1:
            vq = self.unit_vectors[q]
            angles_p = angles_from_p.get(p)
            if angles_p is None:
                angles_p = np.arccos(np.clip(
                    self.unit_vectors @ self.unit_vectors[p], -1.0, 1.0))
                angles_from_p[p] = angles_p
            candidate_ids = np.where(
                (angles_p >= d2_lo) & (angles_p <= d2_hi))[0]
            for r in candidate_ids:
                r = int(r)
                if r == q:
                    continue
                cos_qr = np.clip(np.dot(vq, self.unit_vectors[r]), -1.0, 1.0)
                d_qr = np.arccos(cos_qr)
                if d3_lo <= d_qr <= d3_hi:
                    results.append((p, q, r))
                    if max_results is not None and len(results) >= max_results:
                        return results

        return results


class StarIdentifier:
    """星图识别器"""

    def __init__(self, catalog, K, angle_tolerance=0.005, vote_threshold=1,
                 max_triangle_candidates=300, match_pixel_threshold=5.0,
                 frame_width=1024, frame_height=1024,
                 attitude_method='svd', min_hypotheses=20):
        self.catalog = catalog
        self.K = K
        self.K_inv = np.linalg.inv(K)
        self.f_px = K[0, 0]
        self.cx = K[0, 2]
        self.cy = K[1, 2]
        self.angle_tolerance = angle_tolerance
        self.vote_threshold = vote_threshold
        self.max_triangle_candidates = max_triangle_candidates
        self.match_pixel_threshold = match_pixel_threshold
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.attitude_method = str(attitude_method)
        self.min_hypotheses = int(min_hypotheses)

    def pixel_to_unit_vector(self, x, y):
        """像素坐标 → 相机坐标系单位方向向量 (y-up)"""
        return image_point_to_body_vector(
            x, y, frame_w=self.frame_width, frame_h=self.frame_height,
            focal_length=self.f_px, principal_point=(self.cx, self.cy))

    def identify(self, points, triangles=None):
        """
        星图识别主入口

        参数
        ----
        points : List[(x, y, flux)]
        triangles : List[dict] or None

        返回
        ----
        result : dict
        """
        n_obs = len(points)
        if n_obs < 3 or triangles is None or len(triangles) < 1:
            return self._identify_fallback(points)

        self._point_coords = np.array([(x, y) for x, y, _ in points], dtype=np.float64)
        self._point_fluxes = np.array([p[2] for p in points], dtype=np.float64)

        # 计算观测星在相机系下的单位向量
        obs_vectors = np.zeros((n_obs, 3), dtype=np.float64)
        for i in range(n_obs):
            obs_vectors[i] = self.pixel_to_unit_vector(
                self._point_coords[i, 0], self._point_coords[i, 1])

        # 坐标 → 索引映射
        coord_to_idx = {}
        for i in range(n_obs):
            x, y = self._point_coords[i]
            coord_to_idx[(round(x, 1), round(y, 1))] = i

        # 提取三角形观测星索引
        tri_indices = []
        for tri in triangles:
            verts = tri['vertices']
            idx = []
            for vx, vy in verts:
                key = (round(vx, 1), round(vy, 1))
                obs_idx = coord_to_idx.get(key)
                if obs_idx is None:
                    break
                idx.append(obs_idx)
            if len(idx) == 3 and len(set(idx)) == 3:
                tri_indices.append(tuple(idx))

        if not tri_indices:
            return self._identify_fallback(points)

        # 预计算亮星的惯性系单位向量 (按星等排序，用于批量投影)
        bright_limit = min(2000, len(self.catalog))
        bright_vecs, bright_idx = self.catalog.get_bright_vectors(bright_limit)

        best_match_count = 0
        best_ident = {}
        best_residuals = {}
        best_score = -float('inf')
        hypotheses_tested = 0

        tri_indices.sort(
            key=lambda ids: self._triangle_quality(ids, obs_vectors, points),
            reverse=True)

        # 尝试多个三角形，找到第一个给出足够匹配数的三角形即停止
        for tri_idx, (i, j, k) in enumerate(tri_indices):
            if (hypotheses_tested >= self.min_hypotheses and
                    best_match_count >= 6 and best_residuals and
                    np.median(list(best_residuals.values())) <= 2.0 and
                    max(best_residuals.values()) <= 4.0):
                break

            vi, vj, vk = obs_vectors[i], obs_vectors[j], obs_vectors[k]
            sides = sorted([
                (self._angle_between(vi, vj), i, j),
                (self._angle_between(vj, vk), j, k),
                (self._angle_between(vk, vi), k, i),
            ], key=lambda x: x[0])

            d1, a1, b1 = sides[0]
            d2, a2, b2 = sides[1]
            d3, a3, b3 = sides[2]

            pivot = None
            if a1 == a2 or a1 == b2:
                pivot = a1
            elif b1 == a2 or b1 == b2:
                pivot = b1
            if pivot is None:
                continue

            other_d1 = b1 if a1 == pivot else a1
            other_d2 = b2 if a2 == pivot else a2

            tri_matches = self.catalog.query_triangles(
                d1, d2, d3, self.angle_tolerance,
                max_results=max(4 * self.max_triangle_candidates,
                                self.max_triangle_candidates))
            tri_matches = self._rank_catalog_triangles(
                tri_matches, d1, d2, d3)[:self.max_triangle_candidates]

            for cat_p, cat_q, cat_r, _ in tri_matches:
                trial_map = {pivot: cat_p, other_d1: cat_q, other_d2: cat_r}

                try:
                    bv = np.array([obs_vectors[oi] for oi in trial_map])
                    rv = np.array([self.catalog.get_star_vector(ci)
                                   for ci in trial_map.values()])
                    _, R_trial = solve_star_attitude(
                        bv, rv, method=self.attitude_method)
                except Exception:
                    continue

                # 批量投影并匹配
                matched, residuals = self._project_and_match_batch(
                    R_trial, obs_vectors, n_obs, bright_vecs, bright_idx)
                hypotheses_tested += 1
                score = self._hypothesis_score(matched, residuals)

                if score > best_score:
                    best_score = score
                    best_match_count = len(matched)
                    best_ident = matched
                    best_residuals = residuals

        # 迭代精匹配：用高质量子集重解姿态，逐步改善匹配
        if best_match_count >= 4:
            best_ident, best_residuals, best_match_count = self._iterative_refine(
                best_ident, best_residuals, obs_vectors, n_obs,
                bright_vecs, bright_idx)

        if best_match_count >= 3:
            n_votes = {oi: best_match_count for oi in best_ident}
            return self._build_result(best_ident, n_votes, obs_vectors)

        return self._identify_fallback(points)

    def _iterative_refine(self, initial_ident, initial_residuals,
                          obs_vectors, n_obs, bright_vecs, bright_idx):
        """迭代精匹配：用上一轮高质量匹配子集重解姿态，逐步改善匹配质量。

        第一轮用全部匹配点重解姿态，后续轮次只保留残差最小的点作为
        锚点求解姿态，再用改善后的姿态重新投影匹配。最多 3 轮。
        """
        current_ident = dict(initial_ident)
        current_residuals = dict(initial_residuals)
        best_ident = dict(initial_ident)
        best_residuals = dict(initial_residuals)

        for iteration in range(3):
            # 选择锚点：第二轮起只用残差较小的子集
            if iteration == 0:
                anchors = current_ident
            else:
                sorted_pairs = sorted(current_residuals.items(), key=lambda x: x[1])
                keep = max(3, int(len(sorted_pairs) * 0.75))
                anchor_ids = {int(oi) for oi, _ in sorted_pairs[:keep]}
                anchors = {oi: ci for oi, ci in current_ident.items()
                           if oi in anchor_ids}

            if len(anchors) < 3:
                break

            try:
                bv = np.array([obs_vectors[oi] for oi in anchors])
                rv = np.array([self.catalog.get_star_vector(ci)
                               for ci in anchors.values()])
                _, R_new = solve_star_attitude(bv, rv, method=self.attitude_method)
                new_matched, new_residuals = self._project_and_match_batch(
                    R_new, obs_vectors, n_obs, bright_vecs, bright_idx)
            except Exception:
                break

            if len(new_matched) > len(best_ident) or (
                    len(new_matched) == len(best_ident) and
                    new_residuals and best_residuals and
                    np.median(list(new_residuals.values())) <
                    np.median(list(best_residuals.values()))):
                best_ident = new_matched
                best_residuals = new_residuals

            # 收敛判断：匹配数和残差不再改善
            if (len(new_matched) <= len(current_ident) and
                    new_residuals and current_residuals and
                    np.median(list(new_residuals.values())) >=
                    np.median(list(current_residuals.values())) * 0.95):
                current_ident = dict(best_ident)
                current_residuals = dict(best_residuals)
                break

            current_ident = dict(best_ident)
            current_residuals = dict(best_residuals)

        return best_ident, best_residuals, len(best_ident)

    def _project_and_match_batch(self, R, obs_vectors, n_obs, bright_vecs, bright_idx):
        """
        批量投影星表星到像面并匹配观测星

        参数
        ----
        R : ndarray (3, 3)
            姿态矩阵 (body → inertial)
        bright_vecs : ndarray (M, 3)
            亮星星表向量 (inertial)，按星等排序
        bright_idx : ndarray (M,)
            亮星在完整星表中的索引

        返回
        ----
        matched : Dict[int, int]
            {obs_idx: catalog_id}
        """
        projected, visible_rows, _ = project_inertial_vectors(
            bright_vecs, R, self.frame_width, self.frame_height,
            self.f_px, (self.cx, self.cy))
        if len(projected) == 0:
            return {}, {}
        assignments = match_projected_points(
            projected, self._point_coords, self.match_pixel_threshold)
        matched, residuals = {}, {}
        for projected_row, obs_idx, residual in assignments:
            bright_row = int(visible_rows[projected_row])
            matched[int(obs_idx)] = int(bright_idx[bright_row])
            residuals[int(obs_idx)] = float(residual)
        return matched, residuals

    def _brightness_consistency(self, matched):
        """匹配星对的通量排序与星等排序一致性比率。

        对每对已匹配点，比较观测通量差与星表星等差的方向是否一致。
        返回 0.0 ~ 1.0，越高表示越一致。
        """
        obs_indices = list(matched.keys())
        if len(obs_indices) < 2:
            return 1.0
        n_pairs = 0
        n_consistent = 0
        for a in range(len(obs_indices)):
            for b in range(a + 1, len(obs_indices)):
                oi, oj = obs_indices[a], obs_indices[b]
                fi = float(self._point_fluxes[oi])
                fj = float(self._point_fluxes[oj])
                if fi <= 0 or fj <= 0:
                    continue
                mag_i = self.catalog.get_star_magnitude(matched[oi])
                mag_j = self.catalog.get_star_magnitude(matched[oj])
                n_pairs += 1
                if (fi > fj) == (mag_i < mag_j):
                    n_consistent += 1
        if n_pairs == 0:
            return 1.0
        return float(n_consistent) / float(n_pairs)

    def _hypothesis_score(self, matched, residuals):
        """融合匹配数、残差、空间覆盖率和亮度一致性评价姿态假设。"""
        count = len(matched)
        if count == 0 or not residuals:
            return -float('inf')
        values = np.asarray(list(residuals.values()), dtype=np.float64)
        indices = list(matched.keys())
        coords = self._point_coords[indices]
        width = float(np.ptp(coords[:, 0])) / max(self.frame_width, 1)
        height = float(np.ptp(coords[:, 1])) / max(self.frame_height, 1)
        coverage = float(np.sqrt(max(width * height, 0.0)))
        match_score = min(count / 8.0, 1.0)
        residual_score = float(np.exp(
            -np.median(values) / max(self.match_pixel_threshold, 1e-6)))
        max_penalty = float(np.exp(
            -np.max(values) / max(self.match_pixel_threshold, 1e-6)))
        brightness = self._brightness_consistency(matched)
        return (0.47 * match_score + 0.20 * residual_score +
                0.08 * max_penalty + 0.10 * min(coverage, 1.0) +
                0.15 * brightness)

    def _triangle_quality(self, ids, obs_vectors, points):
        """优先选择展开充分、非退化且顶点检测质量较高的观测三角形。"""
        i, j, k = ids
        angular = np.asarray([
            self._angle_between(obs_vectors[i], obs_vectors[j]),
            self._angle_between(obs_vectors[j], obs_vectors[k]),
            self._angle_between(obs_vectors[k], obs_vectors[i]),
        ], dtype=np.float64)
        semiperimeter = float(np.sum(angular) * 0.5)
        area = np.sqrt(max(
            semiperimeter * np.prod(np.maximum(semiperimeter - angular, 0.0)),
            0.0))
        shape = float(4.0 * np.sqrt(3.0) * area /
                      max(float(np.sum(angular ** 2)), 1e-12))
        spread = float(np.clip(np.min(angular) / 0.01, 0.0, 1.0))
        fluxes = np.asarray([points[index][2] for index in ids], dtype=np.float64)
        flux_quality = float(np.mean(fluxes / max(float(np.max(fluxes)), 1e-12)))
        return 0.50 * shape + 0.35 * spread + 0.15 * flux_quality

    def _rank_catalog_triangles(self, matches, d1, d2, d3):
        """按三条球面边的联合误差排序星表三角候选。"""
        ranked = []
        target = np.asarray([d1, d2, d3], dtype=np.float64)
        for p, q, r in matches:
            actual = np.sort(np.asarray([
                self._angle_between(
                    self.catalog.get_star_vector(p), self.catalog.get_star_vector(q)),
                self._angle_between(
                    self.catalog.get_star_vector(p), self.catalog.get_star_vector(r)),
                self._angle_between(
                    self.catalog.get_star_vector(q), self.catalog.get_star_vector(r)),
            ], dtype=np.float64))
            errors = np.abs(actual - target)
            ranked.append((int(p), int(q), int(r), float(
                np.median(errors) + 0.5 * np.max(errors))))
        ranked.sort(key=lambda item: item[3])
        return ranked

    def _identify_fallback(self, points):
        """退化情况：星对投票"""
        n_obs = len(points)
        if n_obs < 2:
            return self._empty_result()

        obs_vectors = np.zeros((n_obs, 3), dtype=np.float64)
        for i in range(n_obs):
            x, y, _ = points[i]
            obs_vectors[i] = self.pixel_to_unit_vector(x, y)

        votes = defaultdict(lambda: defaultdict(int))
        for i in range(n_obs):
            for j in range(i + 1, n_obs):
                angle = self._angle_between(obs_vectors[i], obs_vectors[j])
                matches = self.catalog.query_pairs(angle, self.angle_tolerance)
                for cat_id1, cat_id2 in matches:
                    votes[i][cat_id1] += 1
                    votes[j][cat_id2] += 1
                    votes[i][cat_id2] += 1
                    votes[j][cat_id1] += 1

        identified = {}
        n_votes = {}
        for obs_idx in range(n_obs):
            if obs_idx in votes and votes[obs_idx]:
                best_cat = max(votes[obs_idx], key=votes[obs_idx].get)
                if votes[obs_idx][best_cat] >= self.vote_threshold:
                    identified[obs_idx] = best_cat
                    n_votes[obs_idx] = votes[obs_idx][best_cat]

        identified = self._resolve_conflicts(identified, n_votes)
        return self._build_result(identified, n_votes, obs_vectors)

    def _build_result(self, identified, n_votes, obs_vectors):
        body_vecs = []
        ref_vecs = []
        mags = []
        ident_final = {}
        for obs_idx, cat_row in identified.items():
            body_vecs.append(obs_vectors[obs_idx])
            ref_vecs.append(self.catalog.get_star_vector(cat_row))
            mags.append(self.catalog.get_star_magnitude(cat_row))
            ident_final[obs_idx] = self.catalog.get_catalog_id(cat_row)

        return {
            'identified': ident_final,
            'body_vectors': np.array(body_vecs, dtype=np.float64) if body_vecs else np.zeros((0, 3)),
            'ref_vectors': np.array(ref_vecs, dtype=np.float64) if ref_vecs else np.zeros((0, 3)),
            'magnitudes': np.array(mags, dtype=np.float64) if mags else np.zeros(0),
            'n_votes': n_votes,
            'n_identified': len(ident_final),
        }

    def _angle_between(self, v1, v2):
        return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

    def _resolve_conflicts(self, identified, n_votes):
        cat_to_obs = defaultdict(list)
        for obs_idx, cat_id in identified.items():
            cat_to_obs[cat_id].append(obs_idx)
        resolved = {}
        for cat_id, obs_list in cat_to_obs.items():
            if len(obs_list) == 1:
                resolved[obs_list[0]] = cat_id
            else:
                resolved[max(obs_list, key=lambda o: n_votes.get(o, 0))] = cat_id
        return resolved

    def _empty_result(self):
        return {
            'identified': {},
            'body_vectors': np.zeros((0, 3)),
            'ref_vectors': np.zeros((0, 3)),
            'magnitudes': np.zeros(0),
            'n_votes': {},
            'n_identified': 0,
        }
