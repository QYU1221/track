"""Candidate preparation and star/target classification."""

__all__ = ["CandidateProcessingStage"]


def __getattr__(name):
    if name == "CandidateProcessingStage":
        from .stage import CandidateProcessingStage
        return CandidateProcessingStage
    raise AttributeError(name)
