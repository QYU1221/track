"""Marginalize static/CV classes inside one Bernoulli and observation owner.

Class is a persistent latent property, not a Markov switch. Finite log odds
retain both explanations even when their displayed probability rounds to 0/1.
Association beams rank objects; they cannot independently prune a class.
"""
from dataclasses import dataclass,replace
import math
import numpy as np
from ..exact.contracts import SourceFactors,InferenceError
from ..efficient.graph import ProbabilisticGate
from .model import predict_gaussian
from .online_factors import affine_factors


@dataclass(frozen=True)
class ClassificationConfig:
    enabled: bool

    def __post_init__(self):
        if not isinstance(self.enabled,bool):raise ValueError('boolean classification switch required')


def log_class_weights(log_odds):
    return -float(np.logaddexp(0.,-log_odds)),-float(np.logaddexp(0.,log_odds))


def mixture_gate_mask(log_weights,geometries,pixels,probability):
    """Exclude at most 1-probability of the Gaussian-mixture pixel mass."""
    log_tail_budget=math.log1p(-probability)-math.log(len(log_weights))
    keep=np.zeros(len(pixels),dtype=bool)
    for log_weight,geometry in zip(log_weights,geometries):
        if log_weight<=log_tail_budget:continue
        limit=-2*(log_tail_budget-log_weight)
        keep|=ProbabilisticGate.distances(pixels-np.asarray(geometry[0]),geometry[1])<=limit
    return keep


def predict_object(track,dt,tracking,background):
    """Survival conditions both existence and class, with no new evidence."""
    lm,ls=log_class_weights(track.motion_log_odds)
    pm=tracking.survival_probability**(dt/tracking.survival_reference_dt_s) if dt else 1.
    ps=background.persistence_probability if dt else 1.
    # Both classes must stay representable; zero-survival models require a
    # separate model definition instead of silently removing the alternative.
    if pm<=0 or ps<=0:raise ValueError('class mixture requires positive class survival')
    lm+=math.log(pm);ls+=math.log(ps)
    survival=float(np.exp(np.logaddexp(lm,ls)))
    existence=track.existence*survival
    if track.existence>0 and existence==0:
        raise InferenceError('PROBABILITY_UNDERFLOW','positive mixture existence is not representable')
    return replace(track,existence=existence,
        state=predict_gaussian(track.state,dt,tracking.q_a),motion_log_odds=lm-ls)


def object_factors(track,frame,camera,window,detections,tracking,background,gate,rotation=None,cache=None):
    """One mixture likelihood; condition class and both states exactly once.

    Gate a probability-budgeted union of model ellipses. Evaluate both Gaussian
    likelihoods on admitted edges; per-class gates would permanently kill the
    alternative. Miss factors keep the original finite-window PD integrals.
    """
    source='T:'+track.label  # persistent object ID; class is carried separately
    if cache is None:cache={}
    def model_factors(kind,state,config):
        # Frame-local only: geometry and conditional hit states are independent
        # of Bernoulli existence and class odds. Histories often share them.
        key=(kind,state,track.plane,config.p_detection,config.measurement_covariance_px2)
        if key not in cache:
            cache[key]=affine_factors(source,kind,state,track.existence,track.plane,
                frame,camera,window,detections,config.p_detection,config.measurement_covariance_px2,None,rotation=rotation)
        return cache[key]
    fm,sm,gm=model_factors('T',track.state,tracking)
    fs,ss,gs=model_factors('S_U',track.static_state,background)
    if tracking.p_detection<=0 or background.p_detection<=0:
        raise ValueError('class mixture requires positive class detection probabilities')
    lm,ls=log_class_weights(track.motion_log_odds)
    logs=tuple(float(np.logaddexp(lm+a,ls+b)) for a,b in zip(fm.log_eta,fs.log_eta))
    pd=math.fsum((math.exp(lm)*fm.pd_bar,math.exp(ls)*fs.pd_bar))
    pd=min(1.,max(0.,pd))
    miss_m=math.log1p(-fm.pd_bar) if fm.pd_bar<1 else -math.inf
    miss_s=math.log1p(-fs.pd_bar) if fs.pd_bar<1 else -math.inf
    conditional={}
    if math.isfinite(miss_m) and math.isfinite(miss_s):
        conditional[None]=(track.state,track.static_state,track.motion_log_odds+miss_m-miss_s)
    else:
        # An exactly impossible class miss cannot be represented by finite
        # log odds. Reject the transaction explicitly rather than floor odds.
        raise ValueError('class mixture requires nonzero model miss probability')
    keep=np.ones(len(detections),dtype=bool)
    if gate is not None and detections:
        pixels=np.asarray([(d.x_px,d.y_px) for d in detections])
        # Each of two classes gets half the excluded mass alpha. A class with
        # mass <= alpha/2 needs no ellipse; otherwise w*tail = alpha/2.
        # Hence mixture mass outside the union is <= alpha. Keeping an ellipse
        # for an astronomically unlikely, ever-widening CV component would
        # connect unrelated groups and exhaust all association/birth slots.
        keep=mixture_gate_mask((lm,ls),(gm,gs),pixels,gate)
    for d,a,b,allowed in zip(detections,fm.log_eta,fs.log_eta,keep):
        if allowed and math.isfinite(a) and math.isfinite(b):
            conditional[d.key]=(sm[d.key],ss[d.key],track.motion_log_odds+a-b)
    gated=tuple(value if allowed else -math.inf for value,allowed in zip(logs,keep))
    # Geometry is intentionally absent: the union gate was already applied;
    # a second single-Gaussian gate would remove valid mixture edges.
    return SourceFactors(source,'T',track.existence,pd,gated),conditional,None


def display_moments(components):
    """Read-only mixture moments, embedding static velocities as exactly zero."""
    parts=[]
    for weight,track in components:
        c=track.motion_probability
        parts.append((weight*c,np.asarray(track.state.mean),np.asarray(track.state.covariance)))
        if track.static_state is not None:
            mean=np.r_[track.static_state.mean,0.,0.]
            covariance=np.zeros((4,4));covariance[:2,:2]=track.static_state.covariance
            parts.append((weight*(1-c),mean,covariance))
    den=math.fsum(w for w,_,_ in parts)
    if den<=0:return None,None
    mean=sum(w*m for w,m,_ in parts)/den
    covariance=sum(w*(p+np.outer(m-mean,m-mean)) for w,m,p in parts)/den
    return tuple(mean),tuple(map(tuple,covariance))
