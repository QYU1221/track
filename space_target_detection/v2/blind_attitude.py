"""Independent, bounded triangle-search attitude diagnostic; never feeds tracking.

No frame attitude, LOS metadata, visible-star shortlist or previous-frame solution
is accepted by the solver. Four one-to-one matches are required, so a seed
triangle alone cannot establish success. Thresholds are diagnostic engineering
settings, not calibrated confidence probabilities.
"""
from itertools import combinations
import json
import time
from math import isfinite

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from openpyxl import Workbook

from .geometry_errors import GeometryError

BLIND_CATALOG_MAG_LIMIT = 7.5


def _fit(body, inertial):
    u, _, vt = np.linalg.svd(inertial.T @ body)
    return u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt


class CatalogPairIndex:
    """Only pairs that can share a field of view; no N-by-N arrays.

    Build in bounded batches. A preflight pair count bounds allocation, and
    the compact array sorts in place instead of allocating a global argsort.
    Float32 angle keys are padded at query time, then checked in float64.
    """
    def __init__(self, directions, tree, max_angle, max_pairs=20000000):
        self.directions = directions
        radius = 2 * np.sin(min(np.pi, max_angle) / 2)
        count = (int(tree.count_neighbors(tree, radius)) - len(directions)) // 2
        self.count = count
        self.available = count <= max_pairs
        self.entries = np.empty(0, dtype=[('angle', '<f4'), ('i', '<u4'), ('j', '<u4')])
        if not self.available:
            return
        self.entries = np.empty(count, dtype=self.entries.dtype)
        offset = 0
        for start in range(0, len(directions), 128):
            neighbors = tree.query_ball_point(directions[start:start + 128], radius)
            for local, candidates in enumerate(neighbors):
                i = start + local
                js = np.asarray(candidates, dtype=np.int64)
                js = js[js > i]
                end = offset + len(js)
                self.entries['i'][offset:end] = i
                self.entries['j'][offset:end] = js
                self.entries['angle'][offset:end] = np.arccos(np.clip(directions[js] @ directions[i], -1, 1))
                offset = end
        if offset != count:
            raise ValueError('catalog spatial pair count changed while building index')
        self.entries.sort(order='angle', kind='quicksort')

    def query(self, angle, tolerance):
        padding = 4 * np.finfo(np.float32).eps
        lo = np.searchsorted(self.entries['angle'], angle - tolerance - padding, side='left')
        hi = np.searchsorted(self.entries['angle'], angle + tolerance + padding, side='right')
        # Yield small exact-filtered batches, rather than copying an annulus.
        for start in range(lo, hi, 4096):
            batch = self.entries[start:min(hi, start + 4096)]
            i, j = batch['i'], batch['j']
            exact = np.arccos(np.clip(np.einsum('ij,ij->i', self.directions[i], self.directions[j]), -1, 1))
            for entry in batch[np.abs(exact - angle) <= tolerance]:
                yield int(entry['i']), int(entry['j'])


class BlindAttitudeSolver:
    def __init__(self, camera, catalog, *, tolerance_px=8., max_candidates=10000,
                 max_pair_trials=200000, max_seconds=5., max_index_pairs=20000000):
        if not isfinite(tolerance_px) or tolerance_px <= 0:
            raise ValueError('blind attitude pixel tolerance must be finite and positive')
        if not isfinite(max_seconds) or max_seconds < 0:
            raise ValueError('blind attitude time budget must be finite and nonnegative')
        for budget in (max_candidates, max_pair_trials, max_index_pairs):
            if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
                raise ValueError('blind attitude count budgets must be nonnegative integers')
        self.camera = camera
        self.catalog = tuple(catalog)
        self.tolerance = tolerance_px
        self.max_candidates = max_candidates
        self.max_pair_trials = max_pair_trials
        self.max_seconds = max_seconds
        self.max_index_pairs = max_index_pairs
        self.directions = np.asarray([s.direction_j2000 for s in catalog]).reshape(-1, 3)
        self.tree = cKDTree(self.directions)
        self.index = None  # Built once, lazily; frames with <4 points need no pair index.
        radius = camera.max_normalized_radius
        if camera.distortion_model == 'none':
            c = camera.camera
            corners = [(x, y) for x in (0, c.width) for y in (0, c.height)]
            radius = min(radius, max(np.linalg.norm(np.linalg.solve(
                camera.K[:2, :2], np.asarray(xy) - camera.K[:2, 2])) for xy in corners))
        self.cap_angle = np.arctan(radius)
        self.angular_tolerance = 2 * self.tolerance / min(camera.camera.fx_px, camera.camera.fy_px)

    def _ensure_index(self):
        if self.index is None:
            self.index = CatalogPairIndex(self.directions, self.tree,
                2 * self.cap_angle + self.angular_tolerance, self.max_index_pairs)

    def _annulus(self, i, angle):
        radius = 2 * np.sin(min(np.pi, angle + self.angular_tolerance) / 2)
        neighbors = np.asarray(self.tree.query_ball_point(self.directions[i], radius), dtype=int)
        distances = np.arccos(np.clip(self.directions[neighbors] @ self.directions[i], -1, 1))
        return neighbors[np.abs(distances - angle) <= self.angular_tolerance]

    def _matches(self, rotation, pixels):
        # The cap uses only the candidate rotation, never the input attitude.
        radius = 2 * np.sin(self.cap_angle / 2) + 1e-12
        candidates = np.asarray(self.tree.query_ball_point(rotation[:, 2], radius), dtype=int)
        projected, indices = [], []
        if self.camera.distortion_model == 'none' and len(candidates):
            body = self.directions[candidates] @ rotation
            good = body[:, 2] > 0
            body, candidates = body[good], candidates[good]
            image = (body[:, :2] / body[:, 2, None]) @ self.camera.image_to_camera_rotation[:2, :2]
            pixels_all = image @ self.camera.K[:2, :2].T + self.camera.K[:2, 2]
            c = self.camera.camera
            keep = ((pixels_all[:, 0] >= 0) & (pixels_all[:, 0] < c.width) &
                    (pixels_all[:, 1] >= 0) & (pixels_all[:, 1] < c.height) &
                    (np.linalg.norm(image, axis=1) <= self.camera.max_normalized_radius))
            projected, indices = pixels_all[keep], candidates[keep]
            candidates = ()
        for i in candidates:
            direction = self.directions[i]
            try:
                pixel = self.camera.camera_to_pixel(rotation.T @ direction)
            except GeometryError:
                continue
            if self.camera.camera.contains(pixel):
                projected.append(pixel)
                indices.append(i)
        if len(projected) == 0:
            return (), (), np.array([])
        distance = np.linalg.norm(pixels[:, None, :] - np.asarray(projected)[None, :, :], axis=2)
        # Dummy columns permit unmatched detections; maximize inlier count first.
        penalty = (len(pixels) + 1) * self.tolerance
        cost = np.where(distance <= self.tolerance, distance, 2 * penalty)
        cost = np.column_stack((cost, np.full((len(pixels), len(pixels)), penalty)))
        a, b = linear_sum_assignment(cost)
        keep = b < len(indices)
        a, b = a[keep], b[keep]
        keep = distance[a, b] <= self.tolerance
        a, b = a[keep], b[keep]
        return a, np.asarray(indices)[b], distance[a, b]

    def solve(self, detections):
        row = dict(Status='FAILED', Reason='', Matched_Stars=0, q1=None, q2=None,
                   q3=None, q4_scalar=None, Reprojection_RMS_px=None,
                   Reprojection_Max_px=None, Matches='[]', Candidates_Tested=0,
                   Tolerance_px=self.tolerance, Input_Attitude_Used=False,
                   Quaternion_Convention='xyzw_camera_to_j2000',
                   Camera_Axis_Rotation_deg=self.camera.camera_axis_rotation_deg,
                   Pair_Trials=0, Index_Pair_Count=0, Index_Memory_MB=0.,
                   Search_Complete=False, Solve_Time_s=0.,
                   Candidate_Limit=self.max_candidates, Pair_Trial_Limit=self.max_pair_trials,
                   Time_Limit_s=self.max_seconds, Index_Pair_Limit=self.max_index_pairs,
                   Best_Candidate_Stars=0, Best_Candidate_RMS_px=None)
        if len(self.catalog) < 4:
            row['Reason'] = 'CATALOG_REQUIRES_AT_LEAST_4_STARS'
            return row
        valid, rays = [], []
        for d in detections:
            if not d.valid or not self.camera.camera.contains((d.x_px, d.y_px)):
                continue
            try:
                ray = self.camera.pixel_to_camera((d.x_px, d.y_px))['camera_los']
            except GeometryError:
                continue
            valid.append(d)
            rays.append(ray)
        if len(valid) < 4:
            row['Reason'] = 'INSUFFICIENT_VALID_POINTS_REQUIRES_4'
            return row
        self._ensure_index()
        row.update(Index_Pair_Count=self.index.count,
                   Index_Memory_MB=self.index.entries.nbytes / 1024**2)
        if not self.index.available:
            row['Reason'] = 'PAIR_INDEX_MEMORY_LIMIT_REACHED'
            return row
        started = time.perf_counter()
        rays = np.asarray(rays)
        pixels = np.asarray([(d.x_px, d.y_px) for d in valid])
        # Select seeds without consulting attitude or prior classifications.
        seeds = sorted(range(len(valid)), key=lambda i: (
            -dict(valid[i].features).get('Flux', 0.), valid[i].detection_id))[:15]
        observed = np.arccos(np.clip(rays @ rays.T, -1, 1))
        angular_tolerance = self.angular_tolerance
        solutions = {}
        exhausted = False
        for a, b, c in combinations(seeds, 3):
            if min(observed[a, b], observed[a, c], observed[b, c]) < 2 * angular_tolerance:
                continue
            annuli = {}
            for pair in self.index.query(observed[a, b], angular_tolerance):
                for i, j in (pair, pair[::-1]):
                    row['Pair_Trials'] += 1
                    if row['Pair_Trials'] > self.max_pair_trials:
                        exhausted = True
                        row['Reason'] = 'PAIR_SEARCH_LIMIT_REACHED'
                        break
                    if (row['Pair_Trials'] - 1) % 32 == 0 and time.perf_counter() - started >= self.max_seconds:
                        exhausted = True
                        row['Reason'] = 'TIME_LIMIT_REACHED'
                        break
                    if i not in annuli:
                        annuli[i] = self._annulus(i, observed[a, c])
                    nearby = annuli[i]
                    angles_j = np.arccos(np.clip(self.directions[nearby] @ self.directions[j], -1, 1))
                    candidates = nearby[np.abs(angles_j - observed[b, c]) <= angular_tolerance]
                    for k in candidates:
                        if k in (i, j):
                            continue
                        if time.perf_counter() - started >= self.max_seconds:
                            exhausted = True
                            row['Reason'] = 'TIME_LIMIT_REACHED'
                            break
                        if row['Candidates_Tested'] >= self.max_candidates:
                            exhausted = True
                            row['Reason'] = 'SEARCH_LIMIT_REACHED'
                            break
                        row['Candidates_Tested'] += 1
                        rotation = _fit(rays[[a, b, c]], self.directions[[i, j, k]])
                        aa, bb, residual = self._matches(rotation, pixels)
                        if len(aa) < 4:
                            continue
                        for _ in range(3):
                            rotation = _fit(rays[aa], self.directions[bb])
                            aa, bb, residual = self._matches(rotation, pixels)
                            if len(aa) < 4:
                                break
                        if len(aa) >= 4:
                            key = tuple(zip(map(int, aa), map(int, bb)))
                            rms = float(np.sqrt(np.mean(residual**2)))
                            if key not in solutions or rms < solutions[key][0]:
                                solutions[key] = (rms, rotation, residual)
                    if exhausted:
                        break
                if exhausted:
                    break
            if exhausted:
                break
        row['Solve_Time_s'] = time.perf_counter() - started
        if solutions:
            provisional = min(solutions, key=lambda key: (-len(key), solutions[key][0]))
            row.update(Best_Candidate_Stars=len(provisional),
                       Best_Candidate_RMS_px=solutions[provisional][0])
        if exhausted:
            return row
        row['Search_Complete'] = True
        if not solutions:
            row['Reason'] = 'NO_CONSISTENT_FOUR_STAR_MATCH'
            return row
        ranked = sorted(solutions, key=lambda key: (-len(key), solutions[key][0]))
        best = ranked[0]
        rms, rotation, residual = solutions[best]
        # Comparable, materially different rotations are not silently accepted.
        for key in ranked[1:]:
            other_rms, other_rotation, _ = solutions[key]
            separation = Rotation.from_matrix(rotation.T @ other_rotation).magnitude()
            if len(key) >= len(best) - 1 and other_rms <= rms + 1. and separation > angular_tolerance:
                row['Reason'] = 'AMBIGUOUS_STAR_PATTERN'
                return row
        q = Rotation.from_matrix(rotation).as_quat()
        if q[3] < 0:
            q = -q
        row.update(Status='SUCCESS', Reason='INDEPENDENT_CATALOG_PATTERN_MATCH',
                   Matched_Stars=len(best), q1=float(q[0]), q2=float(q[1]),
                   q3=float(q[2]), q4_scalar=float(q[3]), Reprojection_RMS_px=rms,
                   Reprojection_Max_px=float(max(residual)),
                   Matches=json.dumps([dict(Detection_ID=valid[a].detection_id,
                       KnownStar_ID=self.catalog[b].known_star_id,
                       Residual_px=float(e)) for (a, b), e in zip(best, residual)]))
        return row


def save_attitude_table(path, rows):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = 'Blind_Attitude'
    if rows:
        fields = list(rows[0])
        sheet.append(fields)
        for row in rows:
            sheet.append([row[k] for k in fields])
        sheet.freeze_panes = 'C2'
        sheet.auto_filter.ref = sheet.dimensions
        from openpyxl.utils import get_column_letter
        for i, field in enumerate(fields, 1):
            sheet.column_dimensions[get_column_letter(i)].width = min(48, max(15, len(field) + 2))
    workbook.save(path)
    workbook.close()
