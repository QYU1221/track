"""Immutable online joint-history state and fixed-plane CV prediction."""
from dataclasses import dataclass,replace
from typing import Optional
import numpy as np
from ..contracts import finite
from ..gnomonic import GnomonicPlane
from ..exact.contracts import GaussianState,probability,identifier,InferenceError


def cv_matrices(dt: float,q_a: float) -> tuple:
    finite((dt,q_a),'seconds and acceleration spectral density')
    if dt<0 or q_a<0:
        raise ValueError('dt and q_a must be nonnegative')
    F = np.eye(4)
    F[:2,2:] = dt*np.eye(2)
    with np.errstate(over='raise',invalid='raise'):
        Q = q_a*np.kron(np.array([[dt**3/3,dt**2/2],[dt**2/2,dt]]),np.eye(2))
    finite(Q,'process covariance')
    return F,Q


def predict_gaussian(state: GaussianState,dt: float,q_a: float) -> GaussianState:
    F,Q = cv_matrices(dt,q_a)
    P = F @ np.asarray(state.covariance) @ F.T+Q
    return GaussianState(tuple(F @ state.mean),tuple(map(tuple,(P+P.T)/2)))


def mixture_moments(weights: tuple,states: tuple) -> Optional[GaussianState]:
    w = finite(weights,'mixture weights')
    if len(w)!=len(states) or np.any(w<0):
        raise ValueError('invalid mixture')
    if not len(w) or sum(w)==0:
        return None
    w = w/sum(w)
    means = np.asarray([s.mean for s in states])
    m = w @ means
    P = sum(a*(np.asarray(s.covariance)+np.outer(mu-m,mu-m)) for a,s,mu in zip(w,states,means))
    return type(states[0])(tuple(m),tuple(map(tuple,(P+P.T)/2)))


@dataclass(frozen=True)
class Track:
    label: str
    existence: float
    state: Optional[GaussianState]
    plane: GnomonicPlane
    support_keys: tuple = ()

    def __post_init__(self):
        identifier(self.label,'track label')
        probability(self.existence,'existence')
        if (self.state is None)!=(self.existence==0):
            raise ValueError('state density is defined iff existence is positive')
        object.__setattr__(self,'support_keys',tuple(tuple(k) for k in self.support_keys))
        if len(set(self.support_keys))!=len(self.support_keys):
            raise ValueError('duplicate measurement in conditional history')


@dataclass(frozen=True)
class JointHistory:
    hypothesis_id: str
    weight: float
    tracks: tuple
    target_assignments: tuple = ()
    star_assignments: tuple = ()
    clutter_keys: tuple = ()
    backgrounds: tuple = ()
    background_assignments: tuple = ()

    def __post_init__(self):
        probability(self.weight,'joint weight')
        object.__setattr__(self,'tracks',tuple(self.tracks))
        if len({t.label for t in self.tracks})!=len(self.tracks):
            raise ValueError('duplicate track labels')
        object.__setattr__(self,'backgrounds',tuple(self.backgrounds))
        if len({b.background_id for b in self.backgrounds})!=len(self.backgrounds):
            raise ValueError('duplicate Background_ID')
        support=[k for t in self.tracks for k in t.support_keys]+[k for b in self.backgrounds for k in b.support_keys]
        if len(set(support))!=len(support):
            raise InferenceError('DUPLICATE_OBSERVATION_OWNER','target/background histories share a detection')


@dataclass(frozen=True)
class TrackingConfig:
    survival_probability: float
    q_a: float
    p_detection: float
    measurement_covariance_px2: tuple
    parameter_provenance: str
    max_dt_s: float
    survival_reference_dt_s: float
    miss_update_mode: str = 'retain'

    def __post_init__(self):
        from ..contracts import checked_covariance
        probability(self.survival_probability,'constant per-transition survival')
        probability(self.p_detection,'detection probability')
        finite((self.q_a,self.max_dt_s),'dynamic parameters')
        if self.q_a<0 or self.max_dt_s<=0:
            raise ValueError('q_a>=0 and max_dt_s>0 required')
        finite(self.survival_reference_dt_s,'survival reference seconds')
        if self.survival_reference_dt_s<=0: raise ValueError('positive survival reference seconds required')
        object.__setattr__(self,'measurement_covariance_px2',tuple(map(tuple,checked_covariance(self.measurement_covariance_px2))))
        identifier(self.parameter_provenance,'parameter provenance')
        if self.miss_update_mode not in ('retain','moment_matched'):
            raise ValueError('miss_update_mode must be retain or moment_matched')


def predict_track(track: Track,dt: float,config: TrackingConfig) -> Track:
    ps = config.survival_probability**(dt/config.survival_reference_dt_s) if dt>0 else 1.
    r = track.existence*ps if dt>0 else track.existence
    if dt>0 and track.existence>0 and config.survival_probability>0 and r==0:
        raise InferenceError('PROBABILITY_UNDERFLOW','positive survival/existence is not representable; no physical-death substitution')
    return replace(track,existence=r,state=predict_gaussian(track.state,dt,config.q_a) if r>0 else None)
