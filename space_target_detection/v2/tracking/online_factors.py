"""Affine, finite-image likelihoods for the production joint filter.

The observation model is z=mu+H(x-m)+epsilon, accepted inside W. Its
effective detection probability integrates the predictive Gaussian over W.
The target miss state is either retained or moment-projected according to
configuration. Background miss states retain the previous approximation.
Neither mode claims a Gaussian-exact finite-window posterior.
"""
import math
from functools import lru_cache
import numpy as np
from ..exact.contracts import SourceFactors
from ..exact.sources import project_direction_pixel
from ..exact.numerics import gaussian_window_mass, gaussian_hit_updates
from ..efficient.graph import ProbabilisticGate
from ..geometry_errors import GeometryError

_INVISIBLE_CODES = ('camera_singularity', 'distortion_domain', 'tangent_domain')


def _invisible_factor(source_id, kind, state, existence, detections):
    """Source outside the declared geometry domains: effective PD=0.

    Same declared degradation as the exact backend's out-of-domain sources;
    the miss factor is 1 so existence only decays by survival. Gate geometry
    is None so callers skip distance computation entirely.
    """
    return SourceFactors(source_id, kind, existence, 0.0,
                         tuple(-math.inf for _ in detections)), {None: state}, None


@lru_cache(maxsize=4096)
def _inertial_geometry(mean,plane):
    """Bounded cache of geometry from immutable validated state coordinates."""
    u,J=plane.inverse(mean)
    u.setflags(write=False);J.setflags(write=False)
    return u,J


def affine_factors(source_id,kind,state,existence,plane,frame,camera,window,detections,pd,R,gate,*,rotation=None,miss_update_mode='retain'):
    if state is None:
        if existence!=0:raise ValueError('positive existence requires a conditional state')
        return _invisible_factor(source_id,kind,state,existence,detections)
    try:
        mu,H,R,S,mass=_affine_projection(state,plane,frame,camera,window,R,rotation)
    except GeometryError as error:
        if error.code not in _INVISIBLE_CODES:
            raise
        return _invisible_factor(source_id,kind,state,existence,detections)
    residuals=np.asarray([(d.x_px-mu[0],d.y_px-mu[1]) for d in detections]).reshape(-1,2)
    distances=ProbabilisticGate.distances(residuals,S) if detections else ()
    logdet=np.linalg.slogdet(S)[1]
    return _affine_result(source_id,kind,state,existence,detections,pd,gate,
        mu,H,R,S,mass,residuals,distances,logdet,window,miss_update_mode)


def _affine_projection(state,plane,frame,camera,window,R,rotation):
    mean=state.mean[:2]
    mu,Hpos=project_direction_pixel(mean,plane,frame,camera,rotation=rotation,
        inertial_geometry=_inertial_geometry(mean,plane))
    H=np.column_stack((Hpos,np.zeros((2,len(state.mean)-2))))
    R=np.asarray(R); P=np.asarray(state.covariance)
    S=H @ P @ H.T+R
    mass,_=gaussian_window_mass(mu,S,window)
    return mu,H,R,S,mass


def _affine_result(source_id,kind,state,existence,detections,pd,gate,
                   mu,H,R,S,mass,residuals,distances,logdet,window=None,miss_update_mode='retain'):
    pd_bar=pd*mass
    threshold=ProbabilisticGate.threshold(gate) if gate is not None else math.inf
    logs=[];states={None:state}
    if kind=='T' and miss_update_mode=='moment_matched':
        from .missed_update import miss_conditioned_gaussian
        states[None]=miss_conditioned_gaussian(state,H,R,mu,S,window,pd,mass)
    for d,residual,distance in zip(detections,residuals,distances):
        value=math.log(pd)-math.log(2*math.pi)-.5*(logdet+distance) if pd>0 and distance<=threshold else -math.inf
        logs.append(value)
    hits=[(d.key,residual) for d,residual,value in zip(detections,residuals,logs) if math.isfinite(value)]
    if hits:
        updated=gaussian_hit_updates(state,H,R,[residual for _,residual in hits])
        states.update((key,value) for (key,_),value in zip(hits,updated))
    factor=SourceFactors(source_id,kind,existence,pd_bar,tuple(logs))
    return factor,states,(tuple(mu),tuple(map(tuple,S)))


def affine_factors_many(requests,frame,camera,window,detections,gate,*,rotation=None,check=None,miss_update_mode='retain'):
    """Batch independent gates/log determinants, retaining scalar window integrals.

    A request is (source ID, kind, state, existence, plane, PD, R). Chunking
    bounds temporary arrays; probability sums and conditional updates retain
    the scalar order, and neither priors nor association groups are merged.
    """
    if len(requests)<16:
        return tuple(affine_factors(source,kind,state,existence,plane,frame,camera,window,detections,pd,R,gate,
            rotation=rotation,miss_update_mode=miss_update_mode) for source,kind,state,existence,plane,pd,R in requests)
    output=[]
    pixels=np.asarray([(d.x_px,d.y_px) for d in detections]).reshape(-1,2)
    for start in range(0,len(requests),128):
        if check is not None:check()
        chunk=requests[start:start+128]
        projected=[]
        for request in chunk:
            _,_,state,existence,plane,_,R=request
            if state is None:
                if existence!=0:raise ValueError('positive existence requires a conditional state')
                projected.append(None)
                continue
            try:
                projected.append(_affine_projection(state,plane,frame,camera,window,R,rotation))
            except GeometryError as error:
                if error.code not in _INVISIBLE_CODES:
                    raise
                projected.append(None)
        positions=[position for position,value in enumerate(projected) if value is not None]
        means=np.asarray([projected[position][0] for position in positions])
        covariances=np.asarray([projected[position][3] for position in positions])
        residuals=pixels[None,:,:]-means[:,None,:] if positions else np.zeros((0,0,2))
        distances=ProbabilisticGate.distances_many(residuals,covariances) if detections and positions else [()]*len(positions)
        logdets=np.linalg.slogdet(covariances)[1] if positions else np.asarray([])
        computed={position:(residuals[row],distances[row],logdets[row])
                  for row,position in enumerate(positions)}
        for position,request in enumerate(chunk):
            source,kind,state,existence,_,pd,_=request
            projection=projected[position]
            if projection is None:
                output.append(_invisible_factor(source,kind,state,existence,detections))
                continue
            mu,H,R,S,mass=projection
            residual,distance,logdet=computed[position]
            output.append(_affine_result(source,kind,state,existence,detections,pd,gate,
                mu,H,R,S,mass,residual,distance,logdet,window,miss_update_mode))
    return tuple(output)
