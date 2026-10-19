"""Bind the online profile to public camera attitudes and the star catalog."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .configuration import DEFAULT_PATH


def default_profile_path():
    return DEFAULT_PATH


def bind_profile(packet,raw,config_path,catalog=None):
    from ..v2.geometry import quaternion_matrix
    from ..v2.exact.io import public_navstar_catalog
    from ..v2.camera_model import CameraModel
    from ..v2.geometry_errors import GeometryError
    from ..v2.numerical_policy import PROBABILITY_EPS_MULTIPLIER
    if raw.get('tracker_engine')!='factorized_joint_ppp' or raw['bootstrap']['schema']!='population-bootstrap-2':
        raise ValueError('only the online population bootstrap is supported')
    result=deepcopy(raw);options=result.pop('bootstrap');result['bootstrap_definition']=options
    camera=packet.camera;bounds=[0,camera.width,0,camera.height]
    result['clutter']['bounds_px']=bounds
    result['initial_timestamp_s']=packet.frames[0].timestamp_s if packet.frames else 0.
    result['bound_input']=dict(camera=asdict(camera),time_scale=packet.time_scale)
    path=Path(raw['catalog_path'])
    if not path.is_absolute():path=Path(config_path).parent/path
    result['catalog_path']=str(path.absolute())
    if catalog is None:
        catalog=public_navstar_catalog(path,raw.get('catalog_mag_limit'))
    directions=np.asarray([star.direction_j2000 for star in catalog]).reshape(-1,3)
    visible=np.zeros(len(catalog),dtype=bool)
    model=CameraModel(camera,**raw['calibration'])
    # A spherical cap only shortlists rays. The actual visibility decision
    # uses the same calibrated projection as the online star likelihoods.
    radius=model.max_normalized_radius
    if model.distortion_model=='none':
        radius=min(radius,max(np.linalg.norm(np.linalg.solve(model.K[:2,:2],np.asarray((x,y))-model.K[:2,2]))
            for x in (0,camera.width) for y in (0,camera.height)))
    cap_cosine=1/np.sqrt(1+radius*radius)
    allowance=PROBABILITY_EPS_MULTIPLIER*np.finfo(float).eps
    valid_frames=tuple(frame for frame in packet.frames if frame.inertial_observation_available)
    for frame in valid_frames:
        rotation=quaternion_matrix(frame.quaternion_xyzw)
        indices=np.flatnonzero(~visible & (directions@rotation[:,2]>=cap_cosine-allowance))
        for index in indices:
            try:
                pixel=model.camera_to_pixel(rotation.T@directions[index])
            except GeometryError as error:
                if error.code not in ('camera_singularity','distortion_domain'):
                    raise
                continue
            visible[index]=camera.contains(pixel)
    identifiers=[star.known_star_id for star,selected in zip(catalog,visible) if selected]
    result['stars']=[dict(known_star_id=identifier,p_detection=options['star_detection_probability'],
        measurement_covariance_px2=result['tracking']['measurement_covariance_px2']) for identifier in identifiers]
    result['profile_binding']=dict(profile_id=raw.get('profile_id'),public_frames=len(packet.frames),
        valid_attitude_frames=len(valid_frames),catalog_selected_by_magnitude=len(catalog),catalog_visible_union=len(identifiers),
        scope=options['birth_scope'],birth_representation='persistent undetected PPP populations; first detections compete every valid frame')
    return result,catalog
