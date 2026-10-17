"""Consecutive low-existence retirement for candidate and confirmed tracks."""
from dataclasses import dataclass, replace
from ..exact.contracts import probability


def consecutive_low_run(previous, first, existence, frame, previous_frame, policy):
    if first is None:
        return 0  # No retained real support has established a candidate clock.
    # Count evaluated consecutive frames; omitted frames cannot establish a run.
    run=previous.low_probability_run if previous_frame is not None and frame.frame==previous_frame+1 else 0
    return run+1 if existence<policy.candidate_low_probability else 0


@dataclass(frozen=True)
class RetiredCandidate:
    """Archive record for either maturity; legacy name keeps checkpoint imports stable."""
    label: str
    frame: int
    timestamp_s: float
    existence_probability: float
    management: object
    conditional_branches: tuple
    state_version: int
    reason: str = 'CONSECUTIVE_LOW_EXISTENCE'
    approximation: str = 'MARGINAL_PROJECTION; FUTURE_SOURCE_REMOVED; NOT_PHYSICAL_DEATH'

    def __post_init__(self):
        probability(self.existence_probability,'retired candidate existence')
        if self.label!=self.management.label:
            raise ValueError('retired track and management labels must match')
        if self.state_version<1 or self.frame<0:
            raise ValueError('invalid retirement version/frame')


def retire_candidates(histories, management, frame, policy, version):
    """Archive low-existence tracks of either maturity without reweighting histories."""
    eligible=tuple(r for r in management if policy.enable_candidate_pruning
        and r.first_real_support_frame is not None
        and r.last_existence_probability is not None
        and r.last_existence_probability<policy.candidate_low_probability
        and r.low_probability_run>=policy.candidate_low_frames)
    if not eligible:
        return histories,management,()
    labels={r.label for r in eligible}
    retired=tuple(RetiredCandidate(r.label,frame.frame,frame.timestamp_s,r.last_existence_probability,r,
        tuple((h.hypothesis_id,h.weight,t) for h in histories for t in h.tracks if t.label==r.label),
        version) for r in eligible)
    # Assignments, claims, support provenance and environment counts describe
    # consumed observations, including those owned by the retired labels.
    feedback=tuple(replace(h,tracks=tuple(t for t in h.tracks if t.label not in labels)) for h in histories)
    return feedback,tuple(r for r in management if r.label not in labels),retired
