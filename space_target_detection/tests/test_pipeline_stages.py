"""Behavior checks for the orchestration wrappers.

The tests use small fakes so they verify call order and state propagation without
requiring a complete dataset run.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.stages import (
    CandidatePreprocessingStage,
    FrameTrackingStage,
    TrackPredictionStage,
)
from pipeline import FullPipeline, load_stage_snapshot, save_stage_snapshot
from candidate_processing import CandidateProcessingStage
from star_processing import StarProcessingStage


class _FakeFilter:
    def __init__(self):
        self.calls = 0
        self.X = np.array([1.0, 2.0])
        self.P = np.eye(2)

    def predict(self):
        self.calls += 1


class _FakeLoader:
    def __init__(self):
        self.calls = 0

    def get_energies_and_radii(self, candidates):
        self.calls += 1
        return np.array([3.0]), np.array([4.0])


class PipelineStageTests(unittest.TestCase):
    def test_snapshot_round_trip_preserves_numpy_and_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'stage.pkl')
            payload = {'matrix': np.eye(2), 'tracks': [{'id': 7}]}
            save_stage_snapshot(path, 'tracking', payload)
            loaded = load_stage_snapshot(path, expected_stage='tracking')

        np.testing.assert_array_equal(loaded.payload['matrix'], np.eye(2))
        self.assertEqual(loaded.payload['tracks'][0]['id'], 7)

    def test_full_pipeline_forwards_arguments_without_transforming(self):
        calls = []

        def processor(path, **kwargs):
            calls.append((path, kwargs))
            return 'result'

        result = FullPipeline(processor).run(
            'measurements', output_dir='output', end_frame=3,
            verbose=False, dataset_name='sample')

        self.assertEqual(result, 'result')
        self.assertEqual(calls[0][0], 'measurements')
        self.assertEqual(calls[0][1]['end_frame'], 3)

    def test_candidate_classification_disabled_is_empty(self):
        class Separator:
            catalog_diagnostics = {'available': False}

        result = CandidateProcessingStage(Separator()).classify(
            candidates=[(1, 2)], tracks=[], star_result=None,
            preprocessing=None, association_enabled=False)

        self.assertEqual(result.target_candidates, [])
        self.assertEqual(result.background_candidates, [])

    def test_star_stage_disabled_returns_named_empty_result(self):
        stage = StarProcessingStage(
            adaptive_identifier=None,
            fixed_identifier=None,
            catalog_projector=None,
            recognized_star_resolver=None,
            runtime_state={'consecutive_failures': 2},
        )

        result = stage.process(
            star_points=[], triangles=[], meta={}, frame_idx=0,
            enabled=False, method='adaptive')

        self.assertFalse(result.attitude_valid)
        self.assertIsNone(result.identification)
        self.assertEqual(result.diagnostics['consecutive_failures'], 2)

    def test_star_stage_fixed_method_preserves_selected_result(self):
        selected = {
            'method': 'descriptor',
            'star_id_result': {'n_identified': 4},
            'attitude_q': np.array([1.0, 0.0, 0.0, 0.0]),
            'attitude_R': np.eye(3),
            'attitude_ok': True,
            'time_ms': 5.0,
            'quality': {
                'score': 0.9, 'matches': 4, 'median_residual': 0.2,
                'max_residual': 0.5, 'coverage': 0.8,
            },
            'recognized_stars': ['recognized'],
        }
        stage = StarProcessingStage(
            adaptive_identifier=None,
            fixed_identifier=lambda *args: selected,
            catalog_projector=lambda *args: ['projected'],
            recognized_star_resolver=None,
            runtime_state={},
        )

        result = stage.process(
            star_points=[(0, 0), (1, 1), (2, 2)], triangles=[],
            meta={}, frame_idx=0, enabled=True, method='descriptor')

        self.assertIs(result.identification, selected['star_id_result'])
        self.assertIs(result.attitude_matrix, selected['attitude_R'])
        self.assertEqual(result.projected_catalog, ['projected'])
        self.assertEqual(result.recognized_stars, ['recognized'])

    @patch('pipeline.stages.bsp_delaunay')
    def test_candidate_geometry_keeps_two_distinct_triangle_inputs(self, delaunay):
        star_points = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
        candidates = star_points + [(100.0, 100.0)]
        tracks = [{
            'object_class': 'target',
            'state': np.array([100.0, 100.0]),
        }]
        delaunay.side_effect = ['star-triangles', 'background-triangles']

        result = CandidatePreprocessingStage.run_geometry(
            candidates=candidates,
            frame_idx=1,
            tracks=tracks,
            star_points=star_points,
            tracked_targets=[],
            displacement_targets=[],
            previous_triangles=None,
            triangle_background_enabled=True,
            target_exclusion_radius=12.0,
        )

        self.assertEqual(delaunay.call_args_list[0].args[0], star_points)
        self.assertEqual(delaunay.call_args_list[1].args[0], star_points)
        self.assertEqual(result.star_triangles, 'star-triangles')
        self.assertEqual(result.background_triangles, 'background-triangles')
        self.assertEqual(result.next_background_triangles, 'background-triangles')

    def test_prediction_preserves_legacy_step_count_and_writeback(self):
        tracker = _FakeFilter()
        track = {'ekf': tracker}

        TrackPredictionStage.run(
            [track], frame_idx=8, last_valid_frame_idx=5,
            max_gap_frames=2, enabled=True)

        self.assertEqual(tracker.calls, 2)
        np.testing.assert_array_equal(track['state'], tracker.X)
        np.testing.assert_array_equal(track['P'], tracker.P)

    @patch('pipeline.stages.update_all')
    @patch('pipeline.stages.build_cost_matrix')
    def test_tracking_preserves_two_metadata_reads(self, build_cost, update_all):
        loader = _FakeLoader()
        tracks = [{'id': 1}]
        candidates = [(10.0, 20.0)]
        expected_cost = np.array([[0.25]])
        build_cost.return_value = expected_cost
        update_all.return_value = (tracks, [])

        result = FrameTrackingStage.run(
            tracks, candidates, loader, frame_idx=7, enabled=True)

        self.assertEqual(loader.calls, 2)
        self.assertIs(result.tracks, tracks)
        self.assertIs(result.cost_matrix, expected_cost)
        update_all.assert_called_once()


if __name__ == '__main__':
    unittest.main()
