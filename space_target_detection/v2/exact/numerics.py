"""Gaussian log densities, deterministic rectangle mass and Joseph update."""
import warnings
from ..numerical_policy import BASIS_EPS_MULTIPLIER
import numpy as np
from scipy.special import ndtr,log_ndtr
from ..contracts import checked_covariance,finite
from ..numerical_policy import (QUADRATURE_ABS_TOL, QUADRATURE_REL_TOL, QUADRATURE_ERROR_LIMIT,
    QUADRATURE_SUBDIVISIONS, NORMAL_INTEGRATION_CUTS, WINDOW_INTERIOR_SIGMA, PROBABILITY_EPS_MULTIPLIER)
from .contracts import GaussianState,ObservationWindow,InferenceError


def log_probability(value: float) -> float:
    return -np.inf if value==0 else float(np.log(value))


def _gaussian_logpdf_prepared(residuals,cholesky):
    """Reuse a validated covariance factor; still validate every residual batch."""
    values=finite(residuals,'residuals')
    if values.ndim!=2 or values.shape[1]!=len(cholesky):
        raise ValueError('residual matrix dimensions must match covariance')
    with np.errstate(over='ignore'):
        whitened=np.linalg.solve(cholesky,values.T)
        squares=np.sum(whitened*whitened,axis=0)
    return -.5*(values.shape[1]*np.log(2*np.pi)+2*np.log(np.diag(cholesky)).sum()+squares)


def normal_interval(lower: float,upper: float) -> float:
    """Stable standard Gaussian interval mass, including positive tails."""
    if lower>=upper:
        return 0.
    if lower>0:
        a = float(log_ndtr(-lower))
        if np.isneginf(a):
            return 0.
        return float(np.exp(a)*(-np.expm1(log_ndtr(-upper)-a)))
    return float(ndtr(upper)-ndtr(lower))


def gaussian_window_mass(mean: np.ndarray,covariance: np.ndarray,window: ObservationWindow) -> tuple:
    """Deterministic 1D conditional integral; absolute 1e-12 / relative 1e-10.

    Integration error is returned and recorded. No random multivariate CDF.
    The standard-normal integration interval is split around 0 to avoid missing
    a narrow density when pixel windows span many standard deviations.
    """
    mean = finite(mean,'pixel mean')
    C = checked_covariance(covariance)
    if mean.shape!=(2,):
        raise ValueError('pixel mean must have dimension 2')
    return _gaussian_window_mass_prepared(mean,C,window)


def _gaussian_window_mass_prepared(mean,C,window):
    """Internal path for an already validated 2-D mean and covariance."""
    x0,x1,y0,y1 = window.bounds_px
    sx,sy = np.sqrt(np.diag(C))
    # Union bound: outside mass <= 4*Phi(-9) < 5e-19, below one ulp at 1.
    if min((mean[0]-x0)/sx,(x1-mean[0])/sx,(mean[1]-y0)/sy,(y1-mean[1])/sy)>=WINDOW_INTERIOR_SIGMA:
        return 1.,0.
    lower,upper = (x0-mean[0])/sx,(x1-mean[0])/sx
    if C[0,1]==0:
        return normal_interval(lower,upper)*normal_interval((y0-mean[1])/sy,(y1-mean[1])/sy),0.
    rho = C[0,1]/(sx*sy)
    sigma = np.linalg.cholesky(C)[1,1]
    if not np.isfinite(sigma) or sigma<=0:
        raise InferenceError('DEGENERATE_COVARIANCE','conditional normal variance is not positive')
    def integrand(t):
        conditional_mean = mean[1]+rho*sy*t
        return np.exp(-t*t/2)/np.sqrt(2*np.pi)*normal_interval((y0-conditional_mean)/sigma,(y1-conditional_mean)/sigma)
    cuts = [lower,*[v for v in NORMAL_INTEGRATION_CUTS if lower<v<upper],upper]
    value,error = 0.,0.
    from scipy.integrate import quad,IntegrationWarning
    with warnings.catch_warnings():
        warnings.simplefilter('error',IntegrationWarning)
        try:
            for a,b in zip(cuts,cuts[1:]):
                part,err = quad(integrand,a,b,epsabs=QUADRATURE_ABS_TOL,epsrel=QUADRATURE_REL_TOL,limit=QUADRATURE_SUBDIVISIONS)
                value,error = value+part,error+err
        except IntegrationWarning as exc:
            raise InferenceError('INTEGRATION_FAILED','Gaussian window integral did not converge') from exc
    tolerance=PROBABILITY_EPS_MULTIPLIER*np.finfo(float).eps
    if not np.isfinite(value) or not -tolerance<=value<=1+tolerance or error>QUADRATURE_ERROR_LIMIT:
        raise InferenceError('INTEGRATION_FAILED','invalid window mass or excessive quadrature error')
    return float(min(1.,max(0.,value))),float(error)


def _gaussian_window_log_outside_prepared(mean,C,window):
    """Same complement arithmetic, reusing the caller's validated inputs."""
    sx,sy = np.sqrt(np.diag(C))
    x0,x1,y0,y1 = window.bounds_px
    log_y = float(np.logaddexp(log_ndtr((y0-mean[1])/sy),log_ndtr((mean[1]-y1)/sy)))
    if C[0,1]==0:
        log_x = float(np.logaddexp(log_ndtr((x0-mean[0])/sx),log_ndtr((mean[0]-x1)/sx)))
        log_y_inside = log_probability(normal_interval((y0-mean[1])/sy,(y1-mean[1])/sy)) if log_y<0 else -np.inf
        value = float(np.logaddexp(log_y,log_x+log_y_inside))
        return min(0.,value)
    rho_sy = C[0,1]/sx
    sigma = np.linalg.cholesky(C)[1,1]
    from scipy.integrate import quad,IntegrationWarning
    def tail(threshold,sign):
        def conditional(t):
            ymean = mean[1]+sign*rho_sy*t
            return normal_interval((y0-ymean)/sigma,(y1-ymean)/sigma)
        if threshold>=0:
            scale = max(1.,threshold)
            def integrand(s):
                return np.exp(-threshold*s/scale-.5*(s/scale)**2)*conditional(threshold+s/scale)
            value,_ = quad(integrand,0,np.inf,epsabs=QUADRATURE_ABS_TOL,epsrel=QUADRATURE_REL_TOL,limit=QUADRATURE_SUBDIVISIONS)
            return -np.inf if value==0 else float(-threshold**2/2-np.log(2*np.pi)/2-np.log(scale)+np.log(value))
        def integrand(t):
            return np.exp(-t*t/2)/np.sqrt(2*np.pi)*conditional(t)
        a,_ = quad(integrand,threshold,0,epsabs=QUADRATURE_ABS_TOL,epsrel=QUADRATURE_REL_TOL,limit=QUADRATURE_SUBDIVISIONS)
        b,_ = quad(integrand,0,np.inf,epsabs=QUADRATURE_ABS_TOL,epsrel=QUADRATURE_REL_TOL,limit=QUADRATURE_SUBDIVISIONS)
        return -np.inf if a+b==0 else float(np.log(a+b))
    with warnings.catch_warnings():
        warnings.simplefilter('error',IntegrationWarning)
        try:
            logs = (log_y,tail((mean[0]-x0)/sx,-1),tail((x1-mean[0])/sx,1))
        except IntegrationWarning as error:
            raise InferenceError('INTEGRATION_FAILED','Gaussian complement quadrature failed') from error
    from scipy.special import logsumexp
    value = float(logsumexp(logs))
    if np.isnan(value) or value>BASIS_EPS_MULTIPLIER*np.finfo(float).eps:
        raise InferenceError('INTEGRATION_FAILED','invalid Gaussian complement')
    return min(0.,value)  # At most floating-point probability-domain rounding.


def _log_nondetection_prepared(pd,mean,C,window):
    if pd==0:return 0.
    return float(np.logaddexp(log_probability(1-pd),log_probability(pd)+_gaussian_window_log_outside_prepared(mean,C,window)))


def gaussian_hit_updates(state: GaussianState,H: np.ndarray,R: np.ndarray,residuals) -> tuple:
    """Alternative hits share a gain/covariance, never each other's evidence."""
    residuals=finite(residuals,'innovations')
    if residuals.ndim!=2 or residuals.shape[1]!=2:
        raise ValueError('two-dimensional innovations required')
    P = np.asarray(state.covariance)
    H = finite(H,'observation matrix')
    R = checked_covariance(R)
    S = checked_covariance(H @ P @ H.T+R)
    K = np.linalg.solve(S,H @ P).T
    A = np.eye(len(state.mean))-K @ H
    covariance=A @ P @ A.T+K @ R @ K.T
    # Joseph's expression is symmetric analytically. Remove multiplication
    # roundoff here, after validating inputs; do not alter eigenvalues or
    # relax the public covariance validation contract.
    covariance=tuple(map(tuple,(covariance+covariance.T)/2))
    mean=np.asarray(state.mean)
    return tuple(type(state)(tuple(mean+K @ residual),covariance) for residual in residuals)
