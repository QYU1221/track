"""Two PPP populations: undetected CV targets and static uncatalogued stars.

Nonlinear quadrature supplies first-detection intensities and conditional
states with the same weights, using each detection once. Velocity retains
the prior conditional uncertainty. Gaussian intensity merging, affine
finite-window thinning and miss-state retention remain explicit continuous
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
from .birth_update import nonlinear_birth_updates


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
    transform=camera.image_to_camera_rotation[:2,:2] @ np.linalg.inv(camera.K[:2,:2])
    mean=transform @ (np.array([c.width/2,c.height/2])-camera.K[:2,2])
    covariance=transform @ np.diag([c.width**2/12,c.height**2/12]) @ transform.T
    return plane,mean,covariance


def prepare(populations,frame,camera,tracking,background,options,dt,max_tangent_angle_rad):
    if not frame.inertial_observation_available:
        return tuple(replace(p,expected_count=p.expected_count*(tracking.survival_probability**(dt/tracking.survival_reference_dt_s) if p.kind=='T' else background.persistence_probability),
            state=predict_gaussian(p.state,dt,tracking.q_a) if p.kind=='T' else p.state) for p in populations)
    newplane,center,footprint_covariance=spatial_prior(camera,frame,max_tangent_angle_rad)
    output=[]
    for kind in ('T','S_U'):
        previous=next((p for p in populations if p.kind==kind),None)
        plane=previous.plane if previous else newplane
        if previous:
            # Transform the current footprint prior, not any measured point.
            direction,J=newplane.inverse(center)
            mean,A=plane.project(direction)
            cov=A @ J @ footprint_covariance @ J.T @ A.T
        else:mean=np.asarray(center);cov=footprint_covariance
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
        # Retain the declared affine finite-window thinning approximation.
        # First-detection evidence and states instead share a nonlinear integral.
        factor,_,_=affine_factors('PPP:'+p.kind,p.kind,p.state,1.,p.plane,frame,camera,
            clutter.window,(),config.p_detection,config.measurement_covariance_px2,None,rotation=rotation)
        detected_count+=p.expected_count*factor.pd_bar
        remaining.append(replace(p,expected_count=p.expected_count*(1-factor.pd_bar)))
        updates=(nonlinear_birth_updates(p.state,p.plane,frame,camera,
            config.measurement_covariance_px2,detections,rotation=rotation)
            if p.expected_count>0 and config.p_detection>0 else ((-math.inf,None),)*len(detections))
        for d,(log_evidence,state) in zip(detections,updates):
            intensity=p.expected_count*config.p_detection*math.exp(log_evidence)
            intensities[d.key][p.kind]=intensity
            if intensity>0:conditionals[(d.key,p.kind)]=(state,p.plane)
    return tuple(remaining),intensities,conditionals,detected_count
