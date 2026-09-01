"""Replay one pipeline stage from an exact trusted local snapshot."""

import argparse
import os
import sys


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from candidate_processing.stage import CandidateProcessingStage
from data_io import DetectionLoader
from pipeline import load_stage_snapshot, save_stage_snapshot
from star_processing import StarProcessingStage
from target_tracking.stage import FrameTrackingStage, TrackPredictionStage
from track_reconnection import TrackReconnectionStage


def _run_star(payload):
    from pipeline import legacy_sequence as legacy
    stage = StarProcessingStage(
        adaptive_identifier=legacy._adaptive_star_identification,
        fixed_identifier=legacy._run_star_id_method,
        catalog_projector=legacy._get_projected_catalog,
        recognized_star_resolver=legacy._get_recognized_stars_for_background,
        runtime_state=payload.pop('runtime_state', legacy._star_id_runtime),
    )
    return stage.process(**payload)


def _run_candidate_prepare(payload):
    separator = payload.pop('separator')
    return CandidateProcessingStage(separator).prepare(**payload)


def _run_candidate_classify(payload):
    separator = payload.pop('separator')
    return CandidateProcessingStage(separator).classify(**payload)


def _run_tracking(payload):
    measurements_dir = payload.pop('measurements_dir')
    return FrameTrackingStage.run(
        loader=DetectionLoader(measurements_dir), **payload)


STAGE_RUNNERS = {
    'star': _run_star,
    'candidate_prepare': _run_candidate_prepare,
    'candidate_classify': _run_candidate_classify,
    'prediction': lambda payload: TrackPredictionStage.run(**payload),
    'tracking': _run_tracking,
    'reconnection': lambda payload: TrackReconnectionStage.run(**payload),
}


def main(fixed_stage=None):
    parser = argparse.ArgumentParser(
        description='Replay one module from a trusted local stage snapshot')
    if fixed_stage is None:
        parser.add_argument('stage', choices=sorted(STAGE_RUNNERS))
    parser.add_argument('--input', '-i', required=True,
                        help='Trusted input .pkl stage snapshot')
    parser.add_argument('--output', '-o', required=True,
                        help='Output .pkl stage snapshot')
    args = parser.parse_args()
    if fixed_stage is not None:
        args.stage = fixed_stage

    snapshot = load_stage_snapshot(args.input, expected_stage=args.stage)
    if not isinstance(snapshot.payload, dict):
        raise TypeError('Stage input payload must be a dictionary')
    result = STAGE_RUNNERS[args.stage](dict(snapshot.payload))
    save_stage_snapshot(args.output, args.stage + '_result', result)
    print('Saved %s result: %s' % (args.stage, args.output))


if __name__ == '__main__':
    main()
