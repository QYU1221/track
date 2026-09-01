"""Named containers for sequence-level and frame-level pipeline state.

These containers intentionally keep the original lists, dictionaries and NumPy
arrays unchanged.  They document ownership; they do not transform algorithm data.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SequenceRuntime:
    """Mutable state that must survive from one frame to the next."""

    separator: Any
    tracks: List[Dict[str, Any]] = field(default_factory=list)
    archived_tracks: List[Dict[str, Any]] = field(default_factory=list)
    trajectory_memory: List[Dict[str, Any]] = field(default_factory=list)
    pending_reconnect: Dict[Any, Any] = field(default_factory=dict)
    reconnect_events: List[Dict[str, Any]] = field(default_factory=list)
    previous_background_triangles: Optional[Any] = None


@dataclass
class FrameRuntime:
    """Values produced while one frame moves through the pipeline."""

    frame_idx: int
    candidates: Any
    meta: Any
    star_points: Any = None
    tracked_targets: Any = None
    displacement_targets: Any = None
    star_triangles: Any = None
    background_triangles: Any = None
    triangle_background: Any = None
    star_id_result: Any = None
    attitude_q: Any = None
    attitude_R: Any = None
    attitude_ok: bool = False
    projected_catalog: Any = None
    recognized_stars: Any = None
    target_candidates: Any = None
    background_candidates: Any = None
    cost_matrix: Any = None
    pruned_tracks: Any = None
    reconnect_events: Any = None


@dataclass
class StarProcessingResult:
    """Outputs of recognition, attitude solving and catalog projection."""

    identification: Any
    attitude_quaternion: Any
    attitude_matrix: Any
    attitude_valid: bool
    selected_method: Any
    displayed_method: Any
    diagnostics: Any
    projected_catalog: Any
    recognized_stars: Any


@dataclass
class CandidateClassificationResult:
    """Outputs of the existing candidate-refinement step."""

    target_candidates: Any
    background_candidates: Any
    tracked_targets: Any
    motion_removed_count: int
    catalog_diagnostics: Any
