"""Named facade for the two established candidate-processing phases."""

from pipeline import CandidateClassificationResult, CandidatePreprocessingStage


class CandidateProcessingStage:
    def __init__(self, separator):
        self.separator = separator

    def prepare(self, candidates, frame_idx, tracks, previous_triangles,
                association_enabled, triangle_enabled,
                target_exclusion_radius, triangle_debug=False):
        if association_enabled:
            star_points, tracked_targets, displacement_targets = \
                self.separator.pre_separate(candidates, frame_idx, tracks)
        else:
            star_points = list(candidates)
            tracked_targets = []
            displacement_targets = []
        return CandidatePreprocessingStage.run_geometry(
            candidates, frame_idx, tracks, star_points, tracked_targets,
            displacement_targets, previous_triangles, triangle_enabled,
            target_exclusion_radius, triangle_debug)

    def prepare_geometry(self, candidates, frame_idx, tracks, star_points,
                         tracked_targets, displacement_targets,
                         previous_triangles, triangle_background_enabled,
                         target_exclusion_radius, triangle_debug=False):
        """Continue from an already executed legacy pre-separation step."""
        return CandidatePreprocessingStage.run_geometry(
            candidates, frame_idx, tracks, star_points, tracked_targets,
            displacement_targets, previous_triangles,
            triangle_background_enabled,
            target_exclusion_radius, triangle_debug)

    def classify(self, candidates, tracks, star_result, preprocessing,
                 association_enabled):
        if association_enabled:
            target_candidates, background, tracked_targets, removed_count = \
                self.separator.refine(
                    candidates=candidates,
                    tracks=tracks,
                    attitude_R=star_result.attitude_matrix,
                    attitude_ok=star_result.attitude_valid,
                    projected_catalog=star_result.projected_catalog,
                    recognized_stars=star_result.recognized_stars,
                    tracked_targets=preprocessing.tracked_targets,
                    displacement_targets=preprocessing.displacement_targets,
                    triangle_background=preprocessing.triangle_background,
                )
        else:
            target_candidates = []
            background = []
            tracked_targets = []
            removed_count = 0
        return CandidateClassificationResult(
            target_candidates=target_candidates,
            background_candidates=background,
            tracked_targets=tracked_targets,
            motion_removed_count=removed_count,
            catalog_diagnostics=self.separator.catalog_diagnostics,
        )
