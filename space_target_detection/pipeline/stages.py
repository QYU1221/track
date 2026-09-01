"""Thin, behavior-preserving wrappers around the existing pipeline algorithms."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from candidate_processing.triangulation import bsp_delaunay
from target_tracking.association_impl import build_cost_matrix
from target_tracking.trajectory_manager_impl import (
    update_all,
)


class TrackPredictionStage:
    """Predict legacy track filters before candidate pre-separation."""

    @staticmethod
    def run(tracks, frame_idx, last_valid_frame_idx, max_gap_frames, enabled=True):
        if enabled and tracks and last_valid_frame_idx is not None:
            predict_steps = min(frame_idx - last_valid_frame_idx, max_gap_frames)
            for _step in range(predict_steps):
                for track in tracks:
                    track['ekf'].predict()
                    track['state'] = track['ekf'].X.copy()
                    track['P'] = track['ekf'].P.copy()
        return tracks


@dataclass
class CandidatePreprocessingResult:
    star_points: Any
    tracked_targets: Any
    displacement_targets: Any
    star_triangles: Any
    background_triangles: Any
    triangle_background: Any
    next_background_triangles: Any


class CandidatePreprocessingStage:
    """Prepare star-recognition input and adjacent-frame triangle evidence."""

    @staticmethod
    def run(candidates, frame_idx, tracks, separator, previous_triangles,
            trajectory_association_enabled, triangle_background_enabled,
            target_exclusion_radius, triangle_debug=False):
        if trajectory_association_enabled:
            star_points, tracked_targets, displacement_targets = \
                separator.pre_separate(candidates, frame_idx, tracks)
        else:
            star_points = list(candidates)
            tracked_targets = []
            displacement_targets = []

        return CandidatePreprocessingStage.run_geometry(
            candidates=candidates,
            frame_idx=frame_idx,
            tracks=tracks,
            star_points=star_points,
            tracked_targets=tracked_targets,
            displacement_targets=displacement_targets,
            previous_triangles=previous_triangles,
            triangle_background_enabled=triangle_background_enabled,
            target_exclusion_radius=target_exclusion_radius,
            triangle_debug=triangle_debug,
        )

    @staticmethod
    def run_geometry(candidates, frame_idx, tracks, star_points,
                     tracked_targets, displacement_targets,
                     previous_triangles, triangle_background_enabled,
                     target_exclusion_radius, triangle_debug=False):
        """Continue after the legacy ``pre_separate`` call.

        Keeping this boundary separate is important: ``pre_separate`` mutates
        temporal separator state and must execute exactly once per frame.
        """
        star_triangles = bsp_delaunay(star_points) if len(star_points) >= 3 else []

        confirmed_positions = np.asarray([
            track['state'][:2] for track in tracks
            if track.get('object_class') == 'target'
        ], dtype=np.float64).reshape(-1, 2)
        triangle_seed_points = []
        for candidate in candidates:
            point = np.asarray(candidate[:2], dtype=np.float64)
            if (confirmed_positions.size > 0 and
                    np.min(np.linalg.norm(confirmed_positions - point, axis=1)) <=
                    target_exclusion_radius):
                continue
            triangle_seed_points.append(candidate)
        background_triangles = (bsp_delaunay(triangle_seed_points)
                                if len(triangle_seed_points) >= 3 else [])

        triangle_background = None
        if (triangle_background_enabled and previous_triangles and
                background_triangles):
            try:
                from candidate_processing.triangle_background import match_background_points
                triangle_background = match_background_points(
                    previous_triangles, background_triangles, tracks=tracks)
            except Exception as exc:
                triangle_background = {
                    'available': False,
                    'matched_points': [],
                    'residual_by_point': {},
                    'reason': 'exception:%s' % exc,
                }
                if triangle_debug:
                    print('[triangle-background] frame=%d failed: %s' % (
                        frame_idx, exc))

        return CandidatePreprocessingResult(
            star_points=star_points,
            tracked_targets=tracked_targets,
            displacement_targets=displacement_targets,
            star_triangles=star_triangles,
            background_triangles=background_triangles,
            triangle_background=triangle_background,
            next_background_triangles=(background_triangles
                                       if background_triangles else None),
        )


@dataclass
class FrameTrackingResult:
    tracks: Any
    pruned_tracks: Any
    cost_matrix: Any
    candidate_energies: Any
    candidate_radii: Any


class FrameTrackingStage:
    """Build association costs and execute the existing per-frame update."""

    @staticmethod
    def run(tracks, target_candidates, loader, frame_idx, enabled=True):
        if tracks and target_candidates:
            energies, radii = loader.get_energies_and_radii(target_candidates)
            cost_matrix = build_cost_matrix(
                tracks, target_candidates, energies, radii)
        else:
            cost_matrix = np.zeros((len(tracks), len(target_candidates)))

        # Deliberately preserve the legacy second metadata lookup.
        if tracks and target_candidates:
            candidate_energies, candidate_radii = \
                loader.get_energies_and_radii(target_candidates)
        else:
            candidate_energies, candidate_radii = np.array([]), np.array([])

        if enabled:
            tracks, pruned = update_all(
                tracks, target_candidates, cost_matrix,
                frame=frame_idx, energy_func=None, radius_func=None,
                camera_info=None, energies=candidate_energies,
                radii=candidate_radii)
        else:
            pruned = []

        return FrameTrackingResult(
            tracks=tracks,
            pruned_tracks=pruned,
            cost_matrix=cost_matrix,
            candidate_energies=candidate_energies,
            candidate_radii=candidate_radii,
        )

