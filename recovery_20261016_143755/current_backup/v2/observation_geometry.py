"""Causal observation geometry; no classification, tracks or hidden truth."""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from .numerical_policy import UNIT_VECTOR_EPS_MULTIPLIER, PROBABILITY_EPS_MULTIPLIER
from .contracts import Detection, FrameInfo, checked_covariance, finite
from .camera_model import CameraModel
from .attitude import camera_to_inertial
from .gnomonic import GnomonicPlane
from .geometry_errors import GeometryError
from .measurement_covariance import propagate_los_covariance,propagate_tangent_covariance,checked_los_covariance

def frozen_array(value):
    if value is None:
        return None
    a = np.asarray(value)
    return tuple(tuple(float(v) for v in row) for row in a) if a.ndim == 2 else tuple(float(v) for v in a)


@dataclass(frozen=True)
class GeometryObservation:
    detection_key: Tuple[int,str]
    timestamp_s: float
    pixel_position: tuple
    pixel_covariance: Optional[tuple]
    camera_los: Optional[tuple]
    j2000_los: Optional[tuple]
    los_covariance: Optional[tuple]
    tangent_position: Optional[tuple]
    tangent_covariance: Optional[tuple]
    tangent_reference_id: str
    attitude_valid: bool
    geometry_valid: bool
    covariance_valid: bool
    covariance_source: str
    calibration_version: str
    calibration_status: str
    quality_flags: tuple
    normalized_image: Optional[tuple] = None
    jacobian_los_pixel: Optional[tuple] = None
    jacobian_tangent_pixel: Optional[tuple] = None
    tangent_coordinate_type: str = 'gnomonic_ratio'

    def __post_init__(self):
        object.__setattr__(self,'detection_key',tuple(self.detection_key))
        object.__setattr__(self,'quality_flags',tuple(self.quality_flags))
        for name,shape in (('pixel_position',(2,)),('pixel_covariance',(2,2)),
                           ('camera_los',(3,)),('j2000_los',(3,)),('los_covariance',(3,3)),
                           ('tangent_position',(2,)),('tangent_covariance',(2,2)),
                           ('normalized_image',(2,)),('jacobian_los_pixel',(3,2)),('jacobian_tangent_pixel',(2,2))):
            value = getattr(self,name)
            if value is not None:
                a = finite(value,name)
                if a.shape != shape:
                    raise ValueError('invalid geometry observation dimensions: '+name)
                object.__setattr__(self,name,frozen_array(a))
        finite(self.timestamp_s,'geometry timestamp')
        for direction in (self.camera_los,self.j2000_los):
            if (direction is not None and not np.isclose(np.dot(direction,direction),1.,rtol=0,
                    atol=UNIT_VECTOR_EPS_MULTIPLIER*np.finfo(float).eps)):
                raise ValueError('geometry LOS must be a unit direction')
        if not self.attitude_valid and self.j2000_los is not None:
            raise ValueError('invalid attitude cannot expose trusted J2000 LOS')
        if self.los_covariance is not None:
            C = checked_los_covariance(self.los_covariance)
            if (self.j2000_los is None or np.linalg.norm(C @ self.j2000_los) >
                    PROBABILITY_EPS_MULTIPLIER*np.finfo(float).eps*np.max(np.abs(C))):
                raise ValueError('LOS covariance must be tangent to the observed unit direction')
        if self.geometry_valid and any(v is None for v in (self.camera_los,self.j2000_los,self.tangent_position)):
            raise ValueError('valid geometry requires complete directional observations')
        if self.covariance_valid and (self.los_covariance is None or self.tangent_covariance is None):
            raise ValueError('valid covariance requires LOS and tangent covariance')
        for matrix in (self.pixel_covariance,self.tangent_covariance):
            if matrix is not None:
                checked_covariance(matrix)
        if self.tangent_coordinate_type != 'gnomonic_ratio':
            raise ValueError('geometry1.0 accepts only gnomonic ratio observations')


def observe_detection(detection: Detection, frame: FrameInfo, model: CameraModel,
                      plane: GnomonicPlane) -> GeometryObservation:
    """Keep every raw record; flag expected invalid geometry explicitly.

    Covariance_Valid means numerical validity, not scientific calibration.
    Missing Sigma retains valid directional geometry but no fabricated R.
    """
    if detection.frame != frame.frame:
        raise ValueError('detection key and frame metadata disagree')
    flags = []
    if model.calibration_status not in ('calibrated','synthetic_test_value'):
        flags.append('camera_calibration_unverified')
    if detection.covariance_status != 'simulation_known':
        flags.append(detection.covariance_status)
    if not detection.valid:
        flags.append(detection.invalid_reason)
    if not frame.input_available:
        flags.append('input_missing')
    if not frame.attitude_valid:
        flags.append('attitude_invalid')
    base = dict(detection_key=detection.key,timestamp_s=frame.timestamp_s,
                pixel_position=(detection.x_px,detection.y_px),pixel_covariance=detection.covariance_px2,
                camera_los=None,j2000_los=None,los_covariance=None,tangent_position=None,
                tangent_covariance=None,tangent_reference_id=plane.reference_id,
                attitude_valid=frame.attitude_valid,geometry_valid=False,covariance_valid=False,
                covariance_source=detection.covariance_status,calibration_version=model.calibration_version,
                calibration_status=model.calibration_status,quality_flags=())
    if detection.valid and frame.input_available:
        try:
            ray = model.pixel_to_camera(base['pixel_position'])
            base.update(camera_los=frozen_array(ray['camera_los']),normalized_image=frozen_array(ray['normalized_image']))
            if frame.attitude_valid:
                u,J = camera_to_inertial(ray['camera_los'],ray['jacobian_camera_pixel'],frame.quaternion_xyzw)
                base.update(j2000_los=frozen_array(u),jacobian_los_pixel=frozen_array(J))
                if detection.covariance_px2 is not None:
                    base['los_covariance'] = frozen_array(propagate_los_covariance(detection.covariance_px2,J))
                position,G = plane.project(u)
                base.update(tangent_position=frozen_array(position),jacobian_tangent_pixel=frozen_array(G @ J),geometry_valid=True)
                if detection.covariance_px2 is not None:
                    covariance = propagate_tangent_covariance(base['los_covariance'],G)
                    base.update(tangent_covariance=frozen_array(covariance),covariance_valid=True)
        except GeometryError as error:
            flags.append(error.code)
        # Unexpected programming exceptions are not suppressed as invalid data.
    base['quality_flags'] = tuple(flags)
    return GeometryObservation(**base)
