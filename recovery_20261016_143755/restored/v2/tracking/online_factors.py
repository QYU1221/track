"""Affine, finite-image likelihoods for the production joint filter.

The observation model is z=mu+H(x-m)+epsilon, accepted inside W. Its
effective detection probability integrates the predictive Gaussian over W.
The miss-conditioned state is retained as a Gaussian approximation (not
claimed to be an exact or moment-matched finite-window posterior).
"""
import math
from functools import lru_cache
import numpy as np
from ..exact.contracts import SourceFactors
from ..exact.sources import project_direction_pixel
from ..exact.numerics import gaussian_window_mass, gaussian_hit_updates
from ..efficient.graph import ProbabilisticGate


@lru_cache(maxsize=4096)
def _inertial_geometry(mean,plane):
    """Bounded cache of geometry from immutable validated state coordinates."""
    u,J=plane.inverse(mean)
    u.setflags(write=False);J.setflags(write=False)
    return u,J


def affine_factors(source_id,kind,state,existence,plane,frame,camera,window,detections,pd,R,gate,*,rotation=None):
    mu,H,R,S,mass=_affine_projection(state,plane,frame,camera,window,R,rotation)
    residuals=np.asarray([(d.x_px-mu[0],d.y_px-mu[1]) for d in detections]).reshape(-1,2)
    distances=ProbabilisticGate.distances(residuals,S) if detections else ()
    logdet=np.linalg.slogdet(S)[1]
    return _affine_result(source_id,kind,state,existence,detections,pd,gate,
        mu,H,R,S,mass,residuals,distances,logdet)


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
                   mu,H,R,S,mass,residuals,distances,logdet):
    pd_bar=pd*mass
    threshold=ProbabilisticGate.threshold(gate) if gate is not None else math.inf
    logs=[];states={None:state}
    for d,residual,distance in zip(detections,residuals,distances):
        value=math.log(pd)-math.log(2*math.pi)-.5*(logdet+distance) if pd>0 and distance<=threshold else -math.inf
        logs.append(value)
    hits=[(d.key,residual) for d,residual,value in zip(detections,residuals,logs) if math.isfinite(value)]
    if hits:
        updated=gaussian_hit_updates(state,H,R,[residual for _,residual in hits])
        states.update((key,value) for (key,_),value in zip(hits,updated))
    factor=SourceFactors(source_id,kind,existence,pd_bar,tuple(logs))
    return factor,states,(tuple(mu),tuple(map(tuple,S)))


def affine_factors_many(requests,frame,camera,window,detections,gate,*,rotation=None,check=None):
    """Batch independent gates/log determinants, retaining scalar window integrals.

    A request is (source ID, kind, state, existence, plane, PD, R). Chunking
    bounds temporary arrays; probability sums and conditional updates retain
    the scalar order, and neither priors nor association groups are merged.
    """
    if len(requests)<16:
        return tuple(affine_factors(source,kind,state,existence,plane,frame,camera,window,detections,pd,R,gate,
            rotation=rotation) for source,kind,state,existence,plane,pd,R in requests)
    output=[]
    pixels=np.asarray([(d.x_px,d.y_px) for d in detections]).reshape(-1,2)
    for start in range(0,len(requests),128):
        if check is not None:check()
        chunk=requests[start:start+128]
        projected=[_affine_projection(state,plane,frame,camera,window,R,rotation)
                   for _,_,state,_,plane,_,R in chunk]
        means=np.asarray([v[0] for v in projected])
        covariances=np.asarray([v[3] for v in projected])
        residuals=pixels[None,:,:]-means[:,None,:]
        distances=ProbabilisticGate.distances_many(residuals,covariances) if detections else [()]*len(chunk)
        logdets=np.linalg.slogdet(covariances)[1]
        for request,projection,residual,distance,logdet in zip(chunk,projected,residuals,distances,logdets):
            source,kind,state,existence,_,pd,_=request
            mu,H,R,S,mass=projection
            output.append(_affine_result(source,kind,state,existence,detections,pd,gate,
                mu,H,R,S,mass,residual,distance,logdet))
    return tuple(output)
