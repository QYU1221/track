"""Moment projection of the affine finite-window nondetection posterior.

Conditional on existence, p(x|miss) is proportional to
p(x) [1-PD P(z in W|x)]. This projection does not change the miss evidence
or existence update. It is not a Gaussian-exact posterior or a gate update.
"""
import math
import numpy as np
from scipy.integrate import quad_vec
from ..exact.numerics import normal_interval
from ..exact.contracts import InferenceError
from ..numerical_policy import NORMAL_INTEGRATION_CUTS


def miss_conditioned_gaussian(state,H,R,mu,S,window,pd,mass):
    if pd==0 or mass==0 or mass==1:
        return state
    P=np.asarray(state.covariance);S=np.asarray(S);mu=np.asarray(mu)
    sx,sy=np.sqrt(np.diag(S));rho=S[0,1]/(sx*sy)
    sigma=np.linalg.cholesky(S)[1,1]/sy
    x0,x1,y0,y1=window.bounds_px
    lo,hi=(x0-mu[0])/sx,(x1-mu[0])/sx
    ya,yb=(y0-mu[1])/sy,(y1-mu[1])/sy
    def phi(x):return math.exp(-x*x/2)/math.sqrt(2*math.pi)
    def integrand(t):
        center=rho*t;a=(ya-center)/sigma;b=(yb-center)/sigma
        pa,pb=phi(a),phi(b);m=normal_interval(a,b)
        first=center*m+sigma*(pa-pb)
        second=(center*center+sigma*sigma)*m+2*center*sigma*(pa-pb)+sigma*sigma*(a*pa-b*pb)
        return phi(t)*np.array((m,t*m,first,t*t*m,t*first,second))
    cuts=[lo,*[v for v in NORMAL_INTEGRATION_CUTS if lo<v<hi],hi]
    moments=np.zeros(6);error=0.
    for a,b in zip(cuts,cuts[1:]):
        value,err=quad_vec(integrand,a,b,epsabs=1e-11,epsrel=1e-9)
        moments+=value;error+=err
    if not np.isfinite(moments).all() or error>1e-8 or abs(moments[0]-mass)>1e-9:
        raise InferenceError('MISS_MOMENT_INTEGRATION_FAILED','finite-window moments inconsistent with detection mass')
    first=np.array((sx*moments[1],sy*moments[2]))
    second=np.array(((sx*sx*moments[3],sx*sy*moments[4]),(sx*sy*moments[4],sy*sy*moments[5])))
    K=np.linalg.solve(S,H@P).T
    A=np.eye(len(state.mean))-K@H
    conditional=A@P@A.T+K@R@K.T
    denominator=1-pd*mass
    shift=-pd*(K@first)/denominator
    covariance=(P-pd*(conditional*mass+K@second@K.T))/denominator-np.outer(shift,shift)
    covariance=(covariance+covariance.T)/2
    return type(state)(tuple(np.asarray(state.mean)+shift),tuple(map(tuple,covariance)))
