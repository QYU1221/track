"""Retirement of tracks that leave the field of view, decay, or miss their confirmation deadline."""
from dataclasses import dataclass, replace
from ..exact.contracts import probability


def low_existence_start(previous, first, existence, frame, previous_frame, policy):
    """Time at which the current uninterrupted low-existence period began, or None."""
    if first is None or existence>=policy.candidate_low_probability:
        return None  # No retained real support has established a candidate clock, or existence recovered.
    # Omitted frames cannot establish a consecutive low-existence period.
    consecutive=previous_frame is not None and frame.frame==previous_frame+1
    if consecutive and previous.low_since_time is not None:
        return previous.low_since_time
    return frame.timestamp_s


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
    reason: str
    approximation: str = 'MARGINAL_PROJECTION; FUTURE_SOURCE_REMOVED; NOT_PHYSICAL_DEATH'

    def __post_init__(self):
        probability(self.existence_probability,'retired candidate existence')
        if self.label!=self.management.label:
            raise ValueError('retired track and management labels must match')
        if self.state_version<1 or self.frame<0:
            raise ValueError('invalid retirement version/frame')


def retirement_reason(record, frame, policy):
    """Why a track leaves joint inference, or None while it is still maintained."""
    if not policy.enable_candidate_pruning or record.first_real_support_frame is None:
        return None
    # Targets are assumed not to return after leaving the field of view.
    if record.outside_since_time is not None and frame.timestamp_s-record.outside_since_time>=policy.outside_fov_stop_s:
        return 'LEFT_FIELD_OF_VIEW'
    if (record.last_existence_probability is not None and record.low_since_time is not None
            and frame.timestamp_s-record.low_since_time>=policy.candidate_low_s):
        return 'CONSECUTIVE_LOW_EXISTENCE'
    if record.overdue_unconfirmed:
        return 'CANDIDATE_DEADLINE'
    return None


def retire_candidates(histories, management, frame, policy, version):
    """Archive tracks that must no longer be maintained, without reweighting histories."""
    reasons={r.label:retirement_reason(r,frame,policy) for r in management}
    eligible=tuple(r for r in management if reasons[r.label] is not None)
    if not eligible:
        return histories,management,()
    labels={r.label for r in eligible}
    retired=tuple(RetiredCandidate(r.label,frame.frame,frame.timestamp_s,r.last_existence_probability,r,
        tuple((h.hypothesis_id,h.weight,t) for h in histories for t in h.tracks if t.label==r.label),
        version,reasons[r.label]) for r in eligible)
    # Assignments, claims, support provenance and environment counts describe
    # consumed observations, including those owned by the retired labels.
    feedback=tuple(replace(h,tracks=tuple(t for t in h.tracks if t.label not in labels)) for h in histories)
    return feedback,tuple(r for r in management if r.label not in labels),retired
