"""Immutable predicted priors, affine observation models and factor snapshots.

All observation densities use pixel Lebesgue area. CV state coordinates are
gnomonic ratios / ratio per second; state projection is locally linearized.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
from ..numerical_policy import PROBABILITY_EPS_MULTIPLIER
import numpy as np
from ..contracts import checked_covariance,finite
from ..observation_geometry import frozen_array


class InferenceError(ValueError):
    def __init__(self,code: str,message: str):
        super().__init__(code+': '+message)
        self.code = code


def probability(value: float,name: str) -> float:
    finite(value,name)
    if not 0 <= value <= 1:
        raise ValueError(name+' must be in [0,1]')
    return float(value)


def identifier(value: str,name: str) -> None:
    if not isinstance(value,str) or not value.strip():
        raise ValueError(name+' must be nonempty')


@dataclass(frozen=True)
class GaussianState:
    mean: tuple
    covariance: tuple

    def __post_init__(self):
        mean = finite(self.mean,'state mean')
        if mean.shape != (4,):
            raise ValueError('gnomonic CV state must be [xi,eta,v_xi,v_eta]')
        P = checked_covariance(self.covariance,4)
        object.__setattr__(self,'mean',frozen_array(mean))
        object.__setattr__(self,'covariance',frozen_array(P))


@dataclass(frozen=True)
class StarSpec:
    known_star_id: str
    p_detection: float
    measurement_covariance_px2: tuple

    def __post_init__(self):
        identifier(self.known_star_id,'KnownStar_ID')
        probability(self.p_detection,'star p_detection')
        object.__setattr__(self,'measurement_covariance_px2',frozen_array(checked_covariance(self.measurement_covariance_px2)))


@dataclass(frozen=True)
class ObservationWindow:
    bounds_px: tuple  # xmin,xmax,ymin,ymax, half open

    def __post_init__(self):
        b = finite(self.bounds_px,'observation window')
        if b.shape != (4,) or b[1]<=b[0] or b[3]<=b[2]:
            raise ValueError('window must have positive width/height')
        object.__setattr__(self,'bounds_px',frozen_array(b))

    @property
    def area(self) -> float:
        x0,x1,y0,y1 = self.bounds_px
        return float(finite((x1-x0)*(y1-y0),'window area'))

    def contains(self,pixel: tuple) -> bool:
        x0,x1,y0,y1 = self.bounds_px
        return bool(x0<=pixel[0]<x1 and y0<=pixel[1]<y1)


@dataclass(frozen=True)
class ClutterModel:
    expected_count: float
    window: ObservationWindow
    parameter_provenance: str

    def __post_init__(self):
        finite(self.expected_count,'Lambda_C')
        if self.expected_count < 0:
            raise ValueError('expected clutter count must be nonnegative')
        identifier(self.parameter_provenance,'clutter parameter provenance')

@dataclass(frozen=True)
class AffinePixelModel:
    mean_px: tuple
    noise_px2: tuple
    p_detection_pre_window: float
    window: ObservationWindow
    integration_error: float = 0.0

    def __post_init__(self):
        mean = finite(self.mean_px,'predicted pixel mean')
        if mean.shape!=(2,):
            raise ValueError('pixel model requires a two-dimensional mean')
        object.__setattr__(self,'mean_px',frozen_array(mean))
        object.__setattr__(self,'noise_px2',frozen_array(checked_covariance(self.noise_px2)))
        probability(self.p_detection_pre_window,'pre-window PD')


@dataclass(frozen=True)
class SourceFactors:
    source_id: str  # T:<Track_Hypothesis_ID> or S_K:<KnownStar_ID>
    kind: str
    existence_prior: float
    pd_bar: float
    log_eta: tuple  # log integral(PD*g*p); one per canonical detection key
    pixel_model: Optional[AffinePixelModel] = None
    log_miss_given_exists: Optional[float] = None

    def __post_init__(self):
        identifier(self.source_id,'source ID')
        if self.kind not in ('T','S_K','S_U'):
            raise ValueError('source kind must be T, S_K or S_U')
        probability(self.existence_prior,'existence prior')
        probability(self.pd_bar,'PD_bar')
        if self.kind=='S_K' and self.existence_prior!=1:
            raise ValueError('known stars have certain existence')
        values = np.asarray(self.log_eta,dtype=float)
        if values.ndim!=1 or np.any(np.isnan(values)) or np.any(np.isposinf(values)):
            raise ValueError('log_eta must be finite or -inf for zero density')
        object.__setattr__(self,'log_eta',tuple(float(v) for v in values))
        if self.log_miss_given_exists is not None and (np.isnan(self.log_miss_given_exists) or self.log_miss_given_exists>0):
            raise ValueError('log miss probability must be <=0, including -inf')
        if self.log_miss_given_exists is not None:
            allowance = max(PROBABILITY_EPS_MULTIPLIER*np.finfo(float).eps,self.pixel_model.integration_error if self.pixel_model else 0.)
            if abs(np.exp(self.log_miss_given_exists)-(1-self.pd_bar))>allowance:
                raise ValueError('miss log probability and PD_bar disagree beyond quadrature/roundoff')


@dataclass(frozen=True)
class FactorTable:
    parent_id: str
    parent_weight: float
    detection_keys: tuple
    sources: Tuple[SourceFactors,...]
    log_kappa: tuple

    def __post_init__(self):
        identifier(self.parent_id,'parent ID')
        probability(self.parent_weight,'parent weight')
        keys = tuple(tuple(k) for k in self.detection_keys)
        if len(set(keys))!=len(keys):
            raise InferenceError('DUPLICATE_DETECTION_ID','keys must be unique')
        if any(len(k)!=2 or not isinstance(k[0],int) or not isinstance(k[1],str) or not k[1] for k in keys):
            raise ValueError('Detection_Key must be (Frame:int,Detection_ID:str)')
        object.__setattr__(self,'detection_keys',keys)
        object.__setattr__(self,'sources',tuple(self.sources))
        logs = np.asarray(self.log_kappa,dtype=float)
        if logs.shape!=(len(keys),) or np.any(np.isnan(logs)) or np.any(np.isposinf(logs)):
            raise ValueError('log_kappa dimensions/values invalid')
        object.__setattr__(self,'log_kappa',tuple(float(v) for v in logs))
        if len({s.source_id for s in self.sources})!=len(self.sources) or any(len(s.log_eta)!=len(keys) for s in self.sources):
            raise ValueError('duplicate source ID or factor dimensions mismatch')


@dataclass(frozen=True)
class Association:
    hypothesis_id: str
    assignments: tuple  # (source_id, detection index or None)
    clutter_indices: tuple
