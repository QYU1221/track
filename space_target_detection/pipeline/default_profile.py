"""Bind the online profile to public camera attitudes and the star catalog."""
from copy import deepcopy
from pathlib import Path
import numpy as np


def default_profile_path():
    return Path(__file__).with_name('profiles')/'default.json'


def bind_profile(packet,raw,config_path,catalog=None):
    from ..v2.geometry import quaternion_matrix
    from ..v2.exact.io import public_navstar_catalog
    if raw.get('tracker_engine')!='factorized_joint_ppp' or raw['bootstrap']['schema']!='population-bootstrap-2':
        raise ValueError('only the online population bootstrap is supported')
    result=deepcopy(raw);options=result.pop('bootstrap');result['bootstrap_definition']=options
    camera=packet.camera;bounds=[0,camera.width,0,camera.height]
    result['clutter']['bounds_px']=bounds
    result['initial_timestamp_s']=packet.frames[0].timestamp_s if packet.frames else 0.
    path=Path(raw['catalog_path'])
    if not path.is_absolute():path=Path(config_path).parent/path
    result['catalog_path']=str(path.absolute())
    if catalog is None:
        catalog=public_navstar_catalog(path,raw.get('catalog_mag_limit'))
    directions=np.asarray([star.direction_j2000 for star in catalog])
    visible=np.zeros(len(catalog),dtype=bool)
    valid_frames=tuple(frame for frame in packet.frames if frame.inertial_observation_available)
    for frame in valid_frames:
        body=directions@quaternion_matrix(frame.quaternion_xyzw);front=body[:,2]>0;indices=np.flatnonzero(front)
        xy=-body[front,:2]/body[front,2:3]
        x=xy[:,0]*camera.fx_px+camera.cx_px;y=xy[:,1]*camera.fy_px+camera.cy_px
        visible[indices]|=(x>=0)&(x<camera.width)&(y>=0)&(y<camera.height)
    identifiers=[star.known_star_id for star,selected in zip(catalog,visible) if selected]
    result['stars']=[dict(known_star_id=identifier,p_detection=options['star_detection_probability'],
        measurement_covariance_px2=result['tracking']['measurement_covariance_px2']) for identifier in identifiers]
    result['profile_binding']=dict(profile_id=raw.get('profile_id'),public_frames=len(packet.frames),
        valid_attitude_frames=len(valid_frames),catalog_selected_by_magnitude=len(catalog),catalog_visible_union=len(identifiers),
        scope=options['birth_scope'],birth_representation='persistent undetected PPP populations; first detections compete every valid frame')
    return result,catalog
