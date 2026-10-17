"""Known-star pixel densities and finite-window detection factors."""
from dataclasses import replace
from functools import lru_cache
import numpy as np
from scipy.special import ndtr,log_ndtr
from ..contracts import checked_covariance
from ..numerical_policy import WINDOW_INTERIOR_SIGMA
from .contracts import SourceFactors
from .numerics import _gaussian_logpdf_prepared,log_probability
from .numerics import _gaussian_window_mass_prepared,_log_nondetection_prepared


@lru_cache(maxsize=256)
def _star_covariance(noise):
    """Only immutable, constructor-validated pixel-model covariances enter."""
    S=checked_covariance(noise);L=np.linalg.cholesky(S)
    S.setflags(write=False);L.setflags(write=False)
    return S,L


def _normal_interval_many(lower,upper):
    """Elementwise standard Gaussian interval mass.

    Identical per-element arithmetic to numerics.normal_interval, including
    the positive-tail log-space branch and its exact-zero underflow guard.
    Callers guarantee lower<upper elementwise.
    """
    a=log_ndtr(-lower)
    with np.errstate(over='ignore',invalid='ignore'):
        tail=np.exp(a)*(-np.expm1(log_ndtr(-upper)-a))
    direct=ndtr(upper)-ndtr(lower)
    values=np.where(lower>0,tail,direct)
    return np.where(np.isneginf(a),0.,values)


def _diagonal_star_mass_log_miss(means,S,window,pds):
    """Batch (mass,error,log_miss) for stars sharing one diagonal covariance.

    Replicates the diagonal branches of _gaussian_window_mass_prepared and
    _gaussian_window_log_outside_prepared elementwise. Correlated covariances
    must keep the scalar quadrature path.
    """
    x0,x1,y0,y1 = window.bounds_px
    sx,sy = np.sqrt(np.diag(S))
    m0,m1 = means[:,0],means[:,1]
    l0=(x0-m0)/sx;u0=(x1-m0)/sx;l1=(y0-m1)/sy;u1=(y1-m1)/sy
    nx=_normal_interval_many(l0,u0);ny=_normal_interval_many(l1,u1)
    interior=(np.minimum(np.minimum(u0,-l0),np.minimum(u1,-l1))>=WINDOW_INTERIOR_SIGMA)
    mass=np.where(interior,1.,nx*ny)
    log_y=np.logaddexp(log_ndtr(l1),log_ndtr(-u1))
    log_x=np.logaddexp(log_ndtr(l0),log_ndtr(-u0))
    with np.errstate(divide='ignore'):
        log_y_inside=np.where(ny>0,np.log(ny),-np.inf)
    log_y_inside=np.where(log_y<0,log_y_inside,-np.inf)
    value=np.logaddexp(log_y,log_x+log_y_inside)
    log_outside=np.minimum(0.,value)
    log_miss=np.logaddexp(np.log(1-np.asarray(pds,dtype=float)),
                          np.log(np.asarray(pds,dtype=float))+log_outside)
    return mass,np.zeros(len(means)),log_miss


def build_known_star_factors(sources: tuple,detections: tuple) -> tuple:
    """PD_bar=PD*P(z in W); eta(z)=PD*N(z;h(mean),HPH'+R)*1_W.

    Catalog measurement R is fixed before association. Raw detector Sigma
    remains diagnostic metadata; no calibrated R is invented.
    """
    detections = tuple(sorted(detections,key=lambda d:d.key))
    factors = []
    density_cache = {};inside_cache={};groups={};densities_by_index={}
    pixels=np.asarray([(d.x_px,d.y_px) for d in detections]).reshape(-1,2)
    for index,source in enumerate(sources):
        groups.setdefault(source.model.noise_px2,[]).append(index)
    for key,indices in groups.items():
        density_cache[key]=_star_covariance(key)
        _,L=density_cache[key]
        if len(detections)==1:
            # LAPACK may use a different arithmetic kernel for one RHS.
            # Retain that path to preserve the scalar result bit for bit.
            for index in indices:
                densities_by_index[index]=_gaussian_logpdf_prepared(pixels-np.asarray(sources[index].model.mean_px),L)
            continue
        for start in range(0,len(indices),128):
            chunk=indices[start:start+128]
            means=np.asarray([sources[index].model.mean_px for index in chunk])
            residuals=pixels[None,:,:]-means[:,None,:]
            densities=_gaussian_logpdf_prepared(residuals.reshape(-1,2),L).reshape(len(chunk),len(detections))
            densities_by_index.update((index,values) for index,values in zip(chunk,densities))
    # Window mass and miss factors: batch the diagonal-covariance stars with
    # elementwise-identical arithmetic; correlated covariances keep the
    # scalar quadrature path so the probability model stays untouched.
    mass_cache={};miss_cache={}
    for key,indices in groups.items():
        S,_=density_cache[key]
        windows={sources[index].model.window for index in indices}
        if S[0,1]==0 and len(windows)==1:
            means=np.asarray([sources[index].model.mean_px for index in indices])
            pds=[sources[index].model.p_detection_pre_window for index in indices]
            masses,errors,misses=_diagonal_star_mass_log_miss(means,S,next(iter(windows)),pds)
            for index,m,e,l in zip(indices,masses,errors,misses):
                mass_cache[index]=(float(m),float(e));miss_cache[index]=float(l)
        else:
            for index in indices:
                model=sources[index].model
                mass_cache[index]=_gaussian_window_mass_prepared(np.asarray(model.mean_px),S,model.window)
                miss_cache[index]=_log_nondetection_prepared(model.p_detection_pre_window,
                                                             np.asarray(model.mean_px),S,model.window)
    for index,source in enumerate(sources):
        model = source.model
        mass,error = mass_cache[index]
        log_miss = miss_cache[index]
        if model.integration_error!=error:
            model = replace(model,integration_error=error)
        densities=densities_by_index[index]
        log_pd=log_probability(model.p_detection_pre_window)
        if model.window not in inside_cache:
            inside_cache[model.window]=tuple(model.window.contains((d.x_px,d.y_px)) for d in detections)
        logs = tuple(log_pd+value if inside else -np.inf for inside,value in zip(inside_cache[model.window],densities))
        factors.append(SourceFactors(source.source_id,'S_K',1.,
                                     model.p_detection_pre_window*mass,logs,model,log_miss))
    return tuple(factors)
