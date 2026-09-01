"""Candidate separation implementations and temporal evidence."""

from .separator_impl import CandidateSeparator
from .temporal_filter import CandidateTemporalFilter
from .triangle_background import match_background_points

__all__ = [
    "CandidateSeparator",
    "CandidateTemporalFilter",
    "match_background_points",
]
