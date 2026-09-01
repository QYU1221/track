"""Candidate separation and triangle-background configuration namespace."""

from .values import *  # noqa: F401,F403

CANDIDATE_PROCESSING_KEYS = tuple(name for name in globals() if (
    name.startswith('candidate_') or name.startswith('motion_separation_') or
    name.startswith('triangle_') or name.startswith('pre_separation_') or
    name.startswith('track_star_') or name.startswith('track_nonstar_') or
    name.startswith('background_subtraction_') or name == 'max_gap_frames'
))
