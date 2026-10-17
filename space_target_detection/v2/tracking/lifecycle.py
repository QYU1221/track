"""Lifecycle policy. Joint histories own all active probability states.

Thresholds do not manufacture observation evidence. Track retirement is
an explicit approximate projection in candidates.py.
"""
from dataclasses import dataclass, replace
from enum import Enum
import math
from ..contracts import finite
from ..exact.contracts import InferenceError, probability


class TrackLifecycle(str, Enum):
    ACTIVE = 'ACTIVE'
    LOST = 'LOST'
    DORMANT = 'DORMANT'
    TERMINATED = 'TERMINATED'


class TrackMaturity(str, Enum):
    PROVISIONAL = 'PROVISIONAL'
    CANDIDATE = 'CANDIDATE'
    CONFIRMED = 'CONFIRMED'
    REJECTED = 'REJECTED'


def canonical_id(label):
    return 'C:%d:%s' % (len(label), label)


@dataclass(frozen=True)
class LifecycleConfig:
    dormant_after_s: float
    reliable_association_probability: float
    outside_visibility_probability: float
    candidate_existence_probability: float
    confirmed_existence_probability: float
    rejection_existence_probability: float
    enable_dormancy: bool
    confirmation_min_frames: int
    confirmation_window_frames: int
    candidate_deadline_frames: int
    candidate_low_probability: float
    candidate_low_frames: int
    enable_candidate_pruning: bool
    max_dormant_tracks_per_group: int
    policy_version: str
    decision_calibration: str

    def __post_init__(self):
        for name in ('reliable_association_probability','outside_visibility_probability',
                     'candidate_existence_probability','confirmed_existence_probability','rejection_existence_probability'):
            probability(getattr(self, name), name)
        finite(self.dormant_after_s, 'dormancy seconds')
        if self.dormant_after_s <= 0: raise ValueError('positive dormancy seconds required')
        if not 0 <= self.rejection_existence_probability < self.candidate_existence_probability <= self.confirmed_existence_probability <= 1:
            raise ValueError('ordered maturity policy required')
        if not isinstance(self.enable_dormancy,bool): raise ValueError('boolean dormancy switch required')
        probability(self.candidate_low_probability,'candidate low probability')
        if not 0 < self.candidate_low_probability < 1:
            raise ValueError('candidate low probability must lie strictly between zero and one')
        for name in ('confirmation_min_frames','confirmation_window_frames',
                     'candidate_deadline_frames','candidate_low_frames'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError('positive frame count required: '+name)
        if self.confirmation_window_frames<self.confirmation_min_frames:
            raise ValueError('confirmation window must include the required support frames')
        if not isinstance(self.enable_candidate_pruning,bool):
            raise ValueError('boolean candidate pruning switch required')
        if isinstance(self.max_dormant_tracks_per_group,bool) or not isinstance(self.max_dormant_tracks_per_group,int) or self.max_dormant_tracks_per_group<1:
            raise ValueError('positive dormant-track budget required')
        if not self.policy_version or self.decision_calibration != 'UNCALIBRATED_DECISION_COST':
            raise ValueError('explicit policy version and current uncalibrated cost status required')


@dataclass(frozen=True)
class ManagementState:
    label: str
    lifecycle: str = TrackLifecycle.ACTIVE.value
    maturity: str = TrackMaturity.PROVISIONAL.value
    first_state_time: float = 0.
    last_reliable_observation_time: object = None
    last_effective_observation_time: object = None
    visibility_probability: object = None
    association_probability: float = 0.
    ever_confirmed: bool = False
    epoch: int = 0
    version: int = 0
    reason: str = 'INITIAL_PRIOR'
    first_real_support_frame: object = None
    last_evaluated_frame: object = None
    last_existence_probability: object = None
    low_probability_run: int = 0
    confirmation_probability: float = 0.
    overdue_unconfirmed: bool = False

    def __post_init__(self):
        TrackLifecycle(self.lifecycle); TrackMaturity(self.maturity)
        finite(self.first_state_time,'management timestamp')
        for v in (self.last_reliable_observation_time,self.last_effective_observation_time):
            if v is not None: finite(v,'observation time')
        if self.visibility_probability is not None: probability(self.visibility_probability,'visibility')
        probability(self.association_probability,'association')
        probability(self.confirmation_probability,'confirmation support')
        if self.last_existence_probability is not None:
            probability(self.last_existence_probability,'last existence')
        for value in (self.first_real_support_frame,self.last_evaluated_frame):
            if value is not None and (isinstance(value,bool) or not isinstance(value,int) or value<0):
                raise ValueError('invalid management frame')
        if isinstance(self.low_probability_run,bool) or not isinstance(self.low_probability_run,int) or self.low_probability_run<0:
            raise ValueError('invalid consecutive low-probability count')
        if not self.label or self.epoch < 0 or self.version < 0: raise ValueError('invalid management identity/version')


@dataclass(frozen=True)
class TrackEvent:
    frame: int
    timestamp_s: float
    label: str
    kind: str
    before: str
    after: str
    reason: str
    state_version: int
    probability_changed_by_management: bool = False


class TrackMaturityManager:
    @staticmethod
    def decide(previous, existence, measured_support, config, confirmation_support=None):
        if existence <= config.rejection_existence_probability: return 'REJECTED'
        # Callers must provide the multi-frame event posterior. The old
        # ever-measured marginal is not a substitute for that event.
        if confirmation_support is not None and confirmation_support >= config.confirmed_existence_probability:
            return 'CONFIRMED'
        if previous.ever_confirmed: return 'CONFIRMED'  # historic maturity, not existence
        return 'CANDIDATE' if measured_support >= config.candidate_existence_probability else 'PROVISIONAL'


class TrackLifecycleManager:
    @staticmethod
    def update(previous, marginals, diagnostics, predicted, frame, config, version,
               previous_frame=None):
        records = {r.label:r for r in previous}; result=[]; events=[]
        for row in marginals:
            label=row['Track_Label']; old=records.get(label,ManagementState(label,first_state_time=frame.timestamp_s))
            assoc=min(1.,math.fsum(p for _,p in row['Association_Marginals']))
            components=[]
            for h in predicted:
                t=next((t for t in h.tracks if t.label==label),None)
                diag=next((d for d in diagnostics if d.get('parent_id')==h.hypothesis_id and d['source_id']=='T:'+label),None)
                if t and diag and t.existence>0:
                    components.append((h.weight*t.existence,diag.get('visibility_mass_cubature',0.),diag['pd_bar']))
            den=math.fsum(w for w,_,_ in components)
            visibility=math.fsum(w*v for w,v,_ in components)/den if den>0 else (0. if frame.inertial_observation_available else None)
            if visibility is not None: visibility=min(1.,max(0.,visibility))  # roundoff of normalized positive mixture
            effective=den>0 and any(w>0 and pd>0 for w,_,pd in components)
            reliable=assoc>=config.reliable_association_probability and assoc>0
            last=frame.timestamp_s if reliable else old.last_reliable_observation_time
            age=frame.timestamp_s-(last if last is not None else old.first_state_time)
            # Roundoff in the normalized mixture can exceed one by a few ulps.
            # This bounds only the management view, never the owning posterior.
            r=min(1.,max(0.,row['Existence_Probability']))
            # A zero posterior is the only automatic physical termination.
            if r==0:
                lifecycle,reason='TERMINATED','ZERO_PHYSICAL_POSTERIOR; software termination'
            elif reliable:
                lifecycle,reason='ACTIVE','JOINT_ASSOCIATION_POSTERIOR_SUPPORT'
            elif config.enable_dormancy and (age>=config.dormant_after_s or visibility is not None and visibility<=config.outside_visibility_probability):
                lifecycle,reason='DORMANT',('OUTSIDE_SAMPLED_FOV_SUPPORT' if visibility is not None and visibility<=config.outside_visibility_probability else 'OBSERVATION_GAP_MANAGEMENT_POLICY')
            else:
                lifecycle,reason='LOST',('OBSERVATION_UNAVAILABLE' if not frame.inertial_observation_available else 'INSUFFICIENT_ASSOCIATION_SUPPORT')
            confirmation=row.get('Confirmation_Support_Probability',0.)
            maturity=TrackMaturityManager.decide(old,r,row['Probability_Exists_And_Ever_Measured'],config,confirmation)
            first=old.first_real_support_frame
            if first is None: first=row.get('First_Real_Support_Frame')
            from .candidates import consecutive_low_run
            low=consecutive_low_run(old,first,r,frame,previous_frame,config)
            epoch=old.epoch+int(lifecycle=='DORMANT' and old.lifecycle!='DORMANT')
            record=replace(old,lifecycle=lifecycle,maturity=maturity,last_reliable_observation_time=last,
                last_effective_observation_time=frame.timestamp_s if effective else old.last_effective_observation_time,
                visibility_probability=visibility,association_probability=assoc,ever_confirmed=old.ever_confirmed or maturity=='CONFIRMED',
                epoch=epoch,version=version,reason=reason,first_real_support_frame=first,
                last_evaluated_frame=frame.frame,last_existence_probability=r,low_probability_run=low,
                confirmation_probability=confirmation,overdue_unconfirmed=bool(
                    not (old.ever_confirmed or maturity=='CONFIRMED') and first is not None
                    and frame.frame-first>=config.candidate_deadline_frames))
            result.append(record)
            for kind,before,after in (('LIFECYCLE',old.lifecycle,lifecycle),('MATURITY',old.maturity,maturity)):
                if before!=after: events.append(TrackEvent(frame.frame,frame.timestamp_s,label,kind,before,after,reason,version))
        if sum(r.lifecycle=='DORMANT' for r in result)>config.max_dormant_tracks_per_group:
            raise InferenceError('MANAGEMENT_BUDGET_EXCEEDED','max_dormant_tracks exceeded; management transaction not committed')
        return tuple(result),tuple(events)
