"""Persistent 2-D inertial background Bernoullis; never target CV tracks.

Births are independent physical-object Bernoullis, not alternative types of
one opportunity. Coverage/identity assumptions are explicit prior metadata.
"""
from dataclasses import dataclass,replace
from typing import Optional
from ..exact.existence import checked_log_existence
import numpy as np
from ..contracts import finite,checked_covariance
from ..gnomonic import GnomonicPlane
from ..exact.contracts import probability,identifier


@dataclass(frozen=True)
class GaussianDirection:
    mean: tuple
    covariance: tuple

    def __post_init__(self):
        mean = finite(self.mean,'2D gnomonic direction')
        if mean.shape!=(2,):
            raise ValueError('unknown star requires 2D direction, not CV')
        object.__setattr__(self,'mean',tuple(mean))
        object.__setattr__(self,'covariance',tuple(map(tuple,checked_covariance(self.covariance))))


@dataclass(frozen=True)
class Background:
    background_id: str
    existence: float
    state: Optional[GaussianDirection]
    plane: GnomonicPlane
    support_keys: tuple = ()
    visible_miss_count: int = 0
    log_existence: Optional[float] = None

    def __post_init__(self):
        identifier(self.background_id,'Background_ID')
        probability(self.existence,'background existence')
        object.__setattr__(self,'log_existence',checked_log_existence(self.existence,self.log_existence))
        if (self.state is None)!=(self.log_existence == -np.inf):
            raise ValueError('direction density defined iff positive existence')
        if self.state is not None and not isinstance(self.state,GaussianDirection):
            raise TypeError('background state must be GaussianDirection')
        object.__setattr__(self,'support_keys',tuple(tuple(k) for k in self.support_keys))
        if len(set(self.support_keys))!=len(self.support_keys):
            raise ValueError('duplicate background measurement')
        if isinstance(self.visible_miss_count,bool) or not isinstance(self.visible_miss_count,int) or self.visible_miss_count<0:
            raise ValueError('background visible miss count must be a nonnegative integer')


@dataclass(frozen=True)
class BackgroundConfig:
    persistence_probability: float
    p_detection: float
    measurement_covariance_px2: tuple
    parameter_provenance: str
    enable_view_dormancy: bool
    dormancy_margin_deg: float
    dormancy_sigma: float
    enable_retirement: bool
    retirement_existence_probability: float
    retirement_visible_misses: int
    retirement_min_pd: float
    retirement_mature_mass_fraction: float

    def __post_init__(self):
        probability(self.persistence_probability,'background persistence')
        probability(self.p_detection,'unknown-star detection')
        identifier(self.parameter_provenance,'background parameter provenance')
        if not isinstance(self.enable_view_dormancy,bool) or not isinstance(self.enable_retirement,bool):
            raise ValueError('background lifecycle switches must be boolean')
        finite((self.dormancy_margin_deg,self.dormancy_sigma),'background dormancy policy')
        if not 0<=self.dormancy_margin_deg<90 or self.dormancy_sigma<3:
            raise ValueError('background dormancy requires margin in [0,90) and sigma >= 3')
        probability(self.retirement_existence_probability,'background retirement probability')
        probability(self.retirement_min_pd,'background retirement minimum detection probability')
        probability(self.retirement_mature_mass_fraction,'background retirement mature mass fraction')
        if self.retirement_mature_mass_fraction<=0:
            raise ValueError('background retirement mature mass fraction must be positive')
        if self.retirement_min_pd<=0:
            raise ValueError('background retirement minimum detection probability must be positive')
        if isinstance(self.retirement_visible_misses,bool) or not isinstance(self.retirement_visible_misses,int) or self.retirement_visible_misses<1:
            raise ValueError('background retirement requires positive integer visible misses')
        object.__setattr__(self,'measurement_covariance_px2',tuple(map(tuple,checked_covariance(self.measurement_covariance_px2))))


def predict_background(background: Background,dt: float,config: BackgroundConfig) -> Background:
    finite(dt,'background time interval')
    if dt<0: raise ValueError('negative background dt')
    log_r = background.log_existence + (checked_log_existence(config.persistence_probability) if dt>0 else 0.)
    r = float(np.exp(log_r))
    # Q_S=0 is the only implemented physical direction model. No fake CV.
    if log_r==background.log_existence:return background
    return replace(background,existence=r,log_existence=log_r,state=background.state if log_r > -np.inf else None)


def project_background(histories: tuple,frame,source_diagnostics: tuple,*,detail='full') -> tuple:
    from .model import mixture_moments
    import math
    rows = []
    indexed=tuple((h,{b.background_id:b for b in h.backgrounds},dict(h.background_assignments)) for h in histories)
    diagnostics={}
    for d in source_diagnostics:diagnostics.setdefault(d.get('source_id'),[]).append(d)
    for identity in sorted({b.background_id for h in histories for b in h.backgrounds}):
        parts,associations,last = [],{},[]
        miss = 0.
        template = None
        for h,objects,assignments in indexed:
            b = objects.get(identity)
            key = assignments.get('S_U:'+identity)
            if key is None: miss += h.weight
            else: associations[key] = associations.get(key,0.)+h.weight
            if b is None: continue
            template = b
            if b.log_existence > -math.inf and h.weight>0:
                parts.append((h.hypothesis_id,math.log(h.weight)+b.log_existence,b.state))
            if b.support_keys: last.append((h.hypothesis_id,b.support_keys[-1],h.weight*b.existence))
        maximum=max((p[1] for p in parts),default=-math.inf)
        scaled=tuple(math.exp(p[1]-maximum) for p in parts)
        total=math.fsum(scaled)
        log_r=maximum+math.log(total) if total else -math.inf
        r=math.exp(log_r)
        normalized=tuple(w/total for w in scaled)
        moments = mixture_moments(normalized,tuple(p[2] for p in parts))
        direction,covariance = None,None
        if moments is not None:
            direction,J = template.plane.inverse(moments.mean)
            covariance = J @ np.asarray(moments.covariance) @ J.T
        visibility = tuple({'parent_id':d['parent_id'],'mass':d.get('visibility_mass_cubature')} for d in diagnostics.get('S_U:'+identity,()))
        rows.append({'Background_ID':identity,'Frame':frame.frame,'Timestamp':frame.timestamp_s,
          'Existence_Probability':r,'Log_Existence_Probability':log_r,'J2000_Direction':tuple(direction) if direction is not None else None,
          'Direction_Covariance':tuple(map(tuple,covariance)) if covariance is not None else None,
          'Direction_Covariance_Approximation':'Jacobian at mixture mean; rank-two LOS covariance',
          'Tangent_Reference':template.plane,'State_Mean':moments.mean if moments else None,
          'State_Covariance':moments.covariance if moments else None,
          'Associated_Detection_Marginals':tuple(sorted(associations.items())),
          'Miss_Association_Probability':miss,'Last_Valid_Observation':tuple(last),
          'Visibility':visibility,'Model_Status':'STATIC_INERTIAL_Q_ZERO; CONTINUOUS_APPROXIMATE',
          'Background_Hypothesis_Status':'Measurement_Supported' if last else 'Latent_Bernoulli'})
        if detail=='full':rows[-1]['State_Mixture']=tuple({'hypothesis_id':i,'weight':w,'state':s} for (i,_,s),w in zip(parts,normalized))
    return tuple(rows)
