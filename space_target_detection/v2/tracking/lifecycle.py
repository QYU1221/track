"""Lifecycle policy. Joint histories own all active probability states.

Thresholds do not manufacture observation evidence. Track retirement is
an explicit approximate projection in candidates.py.
"""
from dataclasses import dataclass, replace
from enum import Enum
import math
from ..contracts import finite
from ..exact.contracts import probability


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
    outside_fov_stop_s: float
    candidate_existence_probability: float
    confirmed_existence_probability: float
    rejection_existence_probability: float
    enable_dormancy: bool
    confirmation_min_frames: int
    confirmation_window_frames: int
    candidate_deadline_s: float
    candidate_low_probability: float
    candidate_low_s: float
    enable_candidate_pruning: bool
    policy_version: str
    decision_calibration: str

    def __post_init__(self):
        for name in ('reliable_association_probability','outside_visibility_probability',
                     'candidate_existence_probability','confirmed_existence_probability','rejection_existence_probability'):
            probability(getattr(self, name), name)
        finite(self.dormant_after_s, 'dormancy seconds')
        if self.dormant_after_s <= 0: raise ValueError('positive dormancy seconds required')
        finite(self.outside_fov_stop_s, 'field-of-view exit seconds')
        if self.outside_fov_stop_s <= 0: raise ValueError('positive field-of-view exit seconds required')
        if not 0 <= self.rejection_existence_probability < self.candidate_existence_probability <= self.confirmed_existence_probability <= 1:
            raise ValueError('ordered maturity policy required')
        if not isinstance(self.enable_dormancy,bool): raise ValueError('boolean dormancy switch required')
        probability(self.candidate_low_probability,'candidate low probability')
        if not 0 < self.candidate_low_probability < 1:
            raise ValueError('candidate low probability must lie strictly between zero and one')
        for name in ('candidate_deadline_s','candidate_low_s'):
            finite(getattr(self,name),name)
            if getattr(self,name)<=0: raise ValueError('positive seconds required: '+name)
        for name in ('confirmation_min_frames','confirmation_window_frames'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError('positive frame count required: '+name)
        if self.confirmation_window_frames<self.confirmation_min_frames:
            raise ValueError('confirmation window must include the required support frames')
        if not isinstance(self.enable_candidate_pruning,bool):
            raise ValueError('boolean candidate pruning switch required')
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
    first_real_support_time: object = None
    last_evaluated_frame: object = None
    last_existence_probability: object = None
    low_since_time: object = None
    confirmation_probability: float = 0.
    overdue_unconfirmed: bool = False
    outside_since_time: object = None

    def __post_init__(self):
        TrackLifecycle(self.lifecycle); TrackMaturity(self.maturity)
        finite(self.first_state_time,'management timestamp')
        for v in (self.last_reliable_observation_time,self.last_effective_observation_time,self.outside_since_time,
                  self.first_real_support_time,self.low_since_time):
            if v is not None: finite(v,'observation time')
        if self.visibility_probability is not None: probability(self.visibility_probability,'visibility')
        probability(self.association_probability,'association')
        probability(self.confirmation_probability,'confirmation support')
        if self.last_existence_probability is not None:
            probability(self.last_existence_probability,'last existence')
        for value in (self.first_real_support_frame,self.last_evaluated_frame):
            if value is not None and (isinstance(value,bool) or not isinstance(value,int) or value<0):
                raise ValueError('invalid management frame')
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


MATURITY_REASONS = {
    'REJECTED': 'EXISTENCE_BELOW_REJECTION',
    'CONFIRMED': 'CONFIRMATION_SUPPORT',
    'CANDIDATE': 'MEASURED_SUPPORT_ABOVE_CANDIDATE',
    'PROVISIONAL': 'MEASURED_SUPPORT_BELOW_CANDIDATE',
}


class TrackMaturityManager:
    @staticmethod
    def decide(previous, existence, measured_support, config, confirmation_support=None):
        # Confirmation is accumulated evidence; later misses lower existence
        # but never revoke it.
        if previous.ever_confirmed: return 'CONFIRMED'
        if existence <= config.rejection_existence_probability: return 'REJECTED'
        # Callers must provide the multi-frame event posterior. The old
        # ever-measured marginal is not a substitute for that event.
        if confirmation_support is not None and confirmation_support >= config.confirmed_existence_probability:
            return 'CONFIRMED'
        return 'CANDIDATE' if measured_support >= config.candidate_existence_probability else 'PROVISIONAL'


class TrackLifecycleManager:
    @staticmethod
    def update(previous, marginals, diagnostics, predicted, frame, config, version,
               previous_frame=None):
        records = {r.label:r for r in previous}; result=[]; events=[]
        # Preserve the first-match semantics of the former linear scans.
        # Diagnostics are shared by all marginal labels in this group.
        diagnostic_index={}
        for diagnostic in diagnostics:
            diagnostic_index.setdefault((diagnostic.get('parent_id'),diagnostic['source_id']),diagnostic)
        predicted_index=[]
        for history in predicted:
            tracks={}
            for track in history.tracks:tracks.setdefault(track.label,track)
            predicted_index.append((history,tracks))
        for row in marginals:
            label=row['Track_Label']; old=records.get(label,ManagementState(label,first_state_time=frame.timestamp_s))
            assoc=min(1.,math.fsum(p for _,p in row['Association_Marginals']))
            components=[]
            for h,tracks in predicted_index:
                t=tracks.get(label)
                diag=diagnostic_index.get((h.hypothesis_id,'T:'+label))
                if t and diag and t.existence>0:
                    components.append((h.weight*t.existence,diag.get('visibility_mass_cubature'),diag['pd_bar']))
            den=math.fsum(w for w,_,_ in components)
            known_visibility=tuple((w,v) for w,v,_ in components if v is not None)
            visibility_den=math.fsum(w for w,_ in known_visibility)
            visibility=math.fsum(w*v for w,v in known_visibility)/visibility_den if visibility_den>0 else None
            if visibility is not None: visibility=min(1.,max(0.,visibility))  # roundoff of normalized positive mixture
            effective=den>0 and any(w>0 and pd>0 for w,_,pd in components)
            reliable=assoc>=config.reliable_association_probability and assoc>0
            last=frame.timestamp_s if reliable else old.last_reliable_observation_time
            age=frame.timestamp_s-(last if last is not None else old.first_state_time)
            # Roundoff in the normalized mixture can exceed one by a few ulps.
            # This bounds only the management view, never the owning posterior.
            r=min(1.,max(0.,row['Existence_Probability']))
            # A zero posterior is the only automatic physical termination.
            if r==0 and row.get("Log_Existence_Probability",-math.inf)==-math.inf:
                lifecycle,reason='TERMINATED','ZERO_PHYSICAL_POSTERIOR; software termination'
            elif reliable:
                lifecycle,reason='ACTIVE','JOINT_ASSOCIATION_POSTERIOR_SUPPORT'
            elif config.enable_dormancy and (age>=config.dormant_after_s or visibility is not None and visibility<=config.outside_visibility_probability):
                lifecycle,reason='DORMANT',('OUTSIDE_SAMPLED_FOV_SUPPORT' if visibility is not None and visibility<=config.outside_visibility_probability else 'OBSERVATION_GAP_MANAGEMENT_POLICY')
            else:
                lifecycle,reason='LOST',('OBSERVATION_UNAVAILABLE' if not frame.inertial_observation_available else 'INSUFFICIENT_ASSOCIATION_SUPPORT')
            outside=not reliable and visibility is not None and visibility<=config.outside_visibility_probability
            outside_since=(frame.timestamp_s if old.outside_since_time is None else old.outside_since_time) if outside else None
            confirmation=row.get('Confirmation_Support_Probability',0.)
            maturity=TrackMaturityManager.decide(old,r,row['Probability_Exists_And_Ever_Measured'],config,confirmation)
            first=old.first_real_support_frame
            if first is None: first=row.get('First_Real_Support_Frame')
            first_time=old.first_real_support_time
            if first_time is None and first is not None: first_time=frame.timestamp_s
            from .candidates import low_existence_start
            low_since=low_existence_start(old,first,r,frame,previous_frame,config)
            epoch=old.epoch+int(lifecycle=='DORMANT' and old.lifecycle!='DORMANT')
            record=replace(old,lifecycle=lifecycle,maturity=maturity,last_reliable_observation_time=last,
                last_effective_observation_time=frame.timestamp_s if effective else old.last_effective_observation_time,
                visibility_probability=visibility,association_probability=assoc,ever_confirmed=old.ever_confirmed or maturity=='CONFIRMED',
                epoch=epoch,version=version,reason=reason,first_real_support_frame=first,first_real_support_time=first_time,
                last_evaluated_frame=frame.frame,last_existence_probability=r,low_since_time=low_since,
                confirmation_probability=confirmation,overdue_unconfirmed=bool(
                    not (old.ever_confirmed or maturity=='CONFIRMED') and first_time is not None
                    and frame.timestamp_s-first_time>=config.candidate_deadline_s),
                outside_since_time=outside_since)
            result.append(record)
            for kind,before,after,why in (('LIFECYCLE',old.lifecycle,lifecycle,reason),
                                          ('MATURITY',old.maturity,maturity,MATURITY_REASONS[maturity])):
                if before!=after: events.append(TrackEvent(frame.frame,frame.timestamp_s,label,kind,before,after,why,version))
        return tuple(result),tuple(events)
