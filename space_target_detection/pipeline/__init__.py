"""High-level orchestration helpers for the detection and tracking pipeline.

The package deliberately contains no algorithm implementations.  It gives names
to the existing processing stages while preserving the legacy data flow.
"""

from .runtime import (CandidateClassificationResult, FrameRuntime,
                      SequenceRuntime, StarProcessingResult)
from .full_pipeline import FullPipeline
from .snapshots import (StageSnapshot, load_stage_snapshot,
                        save_stage_snapshot)
from .stages import (
    CandidatePreprocessingResult,
    CandidatePreprocessingStage,
    FrameTrackingResult,
    FrameTrackingStage,
    TrackPredictionStage,
)
from track_reconnection.stage import TrackReconnectionResult, TrackReconnectionStage

__all__ = [
    "FrameRuntime",
    "SequenceRuntime",
    "StarProcessingResult",
    "CandidateClassificationResult",
    "FullPipeline",
    "StageSnapshot",
    "save_stage_snapshot",
    "load_stage_snapshot",
    "CandidatePreprocessingResult",
    "CandidatePreprocessingStage",
    "FrameTrackingResult",
    "FrameTrackingStage",
    "TrackPredictionStage",
    "TrackReconnectionResult",
    "TrackReconnectionStage",
]
