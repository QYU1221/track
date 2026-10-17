"""Two PPP populations: undetected CV targets and static uncatalogued stars.

Poisson thinning supplies first-detection intensities. A detection is used
once: intensity integration and its conditional state are the same Bayes
update, never a measurement-built prior followed by another hit update.
Gaussian intensity merging and miss-state retention are explicit continuous
approximations. Rates are declared engineering parameters, not calibration.
"""
from dataclasses import dataclass,replace
import math
import numpy as np
from ..gnomonic import GnomonicPlane
from ..geometry import quaternion_matrix
from ..exact.contracts import GaussianState
from .background import GaussianDirection
from .model import predict_gaussian,mixture_moments
from .online_factors import affine_factors


@dataclass(frozen=True)
class Population:
    kind: str
    expected_count: float
    state: object
    plane: GnomonicPlane


def spatial_prior(camera,frame,max_tangent_angle_rad):
    rotation=quaternion_matrix(frame.quaternion_xyzw)
    plane=GnomonicPlane('population-frame-'+str(frame.frame),tuple(rotation[:,2]),max_tangent_angle_rad,
        tuple(rotation[:,0]),tuple(rotation[:,1]))
    c=camera.camera
    # Independent of all detection coordinates. Match a uniform image's
    # first two moments in the boresight gnomonic plane (pinhole prior).
    mean=(-(c.width/2-c.cx_px)/c.fx_px,-(c.height/2-c.cy_px)/c.fy_px)
    variance=(c.width**2/(12*c.fx_px**2),c.height**2/(12*c.fy_px**2))
    return plane,mean,variance


def rebase_population(population,plane):
    """Transport moments to the current camera plane without replaying data.

    Gnomonic coordinates are related by a projective homography. Its exact
    position Jacobian and velocity-position derivative transport the full CV
    covariance, including position/velocity cross terms.
    """
    old=population.plane;state=population.state
    basis=np.column_stack((old.e1,old.e2,old.center))
    H=np.asarray((plane.e1,plane.e2,plane.center))@basis
    q=H@np.r_[state.mean[:2],1.];position=q[:2]/q[2]
    # Preserve the declared geometry domain, including front hemisphere.
    direction,_=old.inverse(state.mean[:2]);plane.project(direction)
    J=(H[:2,:2]-np.outer(position,H[2,:2]))/q[2]
    if population.kind=='T':
        velocity=np.asarray(state.mean[2:]);lower=np.empty((2,2))
        for i in range(2):
            hessian=-(np.outer(J[i],H[2,:2])+np.outer(H[2,:2],J[i]))/q[2]
            lower[i]=hessian@velocity
        transform=np.block([[J,np.zeros((2,2))],[lower,J]])
        mean=np.r_[position,J@velocity]
    else:transform=J;mean=position
    covariance=transform@np.asarray(state.covariance)@transform.T
    return replace(population,plane=plane,state=type(state)(tuple(mean),tuple(map(tuple,(covariance+covariance.T)/2))))


def prepare(populations,frame,camera,tracking,background,options,dt,max_tangent_angle_rad,*,rebase=False):
    if not frame.inertial_observation_available:
        return tuple(replace(p,expected_count=p.expected_count*(tracking.survival_probability**(dt/tracking.survival_reference_dt_s) if p.kind=='T' else background.persistence_probability),
            state=predict_gaussian(p.state,dt,tracking.q_a) if p.kind=='T' else p.state) for p in populations)
    newplane,center,variance=spatial_prior(camera,frame,max_tangent_angle_rad)
    if rebase:
        populations=tuple(rebase_population(p,newplane) for p in populations)
    output=[]
    for kind in ('T','S_U'):
        previous=next((p for p in populations if p.kind==kind),None)
        plane=previous.plane if previous else newplane
        if previous:
            # Transform the current footprint prior, not any measured point.
            direction,J=newplane.inverse(center)
            mean,A=plane.project(direction)
            cov=A @ J @ np.diag(variance) @ J.T @ A.T
        else:mean=np.asarray(center);cov=np.diag(variance)
        if kind=='T':
            P=np.zeros((4,4));P[:2,:2]=cov;P[2:,2:]=np.eye(2)*options['velocity_sigma_rad_s']**2
            injected=GaussianState(tuple(mean)+(0.,0.),tuple(map(tuple,P)))
            survival=tracking.survival_probability**(dt/tracking.survival_reference_dt_s)
        else:
            injected=GaussianDirection(tuple(mean),tuple(map(tuple,cov)))
            survival=background.persistence_probability
        prefix='target' if kind=='T' else 'unknown_star'
        added=options[prefix+'_birth_rate_per_second']*dt if previous else options[prefix+'_initial_expected_count']
        old_count=previous.expected_count*survival if previous else 0.
        old_state=predict_gaussian(previous.state,dt,tracking.q_a) if previous and kind=='T' else previous.state if previous else None
        count=old_count+added
        state=mixture_moments((old_count,added),(old_state or injected,injected)) if count>0 else injected
        output.append(Population(kind,count,state,plane))
    return tuple(output)


def observe(populations,frame,camera,clutter,detections,tracking,background,*,rotation=None):
    intensities={d.key:{} for d in detections};conditionals={};remaining=[];detected_count=0.
    for p in populations:
        config=tracking if p.kind=='T' else background
        factor,states,_=affine_factors('PPP:'+p.kind,p.kind,p.state,1.,p.plane,frame,camera,
            clutter.window,detections,config.p_detection,config.measurement_covariance_px2,None,rotation=rotation)
        detected_count+=p.expected_count*factor.pd_bar
        remaining.append(replace(p,expected_count=p.expected_count*(1-factor.pd_bar)))
        for d,log_eta in zip(detections,factor.log_eta):
            intensity=p.expected_count*math.exp(log_eta)
            intensities[d.key][p.kind]=intensity
            if intensity>0:conditionals[(d.key,p.kind)]=(states[d.key],p.plane)
    return tuple(remaining),intensities,conditionals,detected_count
