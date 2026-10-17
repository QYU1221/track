"""Immutable public observations; no hidden object IDs or truth fields.

Positions use zero-origin pixels; timestamps are seconds in an explicit scale.
Rotations map camera vectors to J2000, with external quaternions (x,y,z,w).
"""
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Sequence
from .numerical_policy import SYMMETRY_EPS_MULTIPLIER, UNIT_VECTOR_EPS_MULTIPLIER
import numpy as np

Vector3 = Tuple[float, float, float]
Matrix2 = Tuple[Tuple[float, float], Tuple[float, float]]


def finite(value: Any, name: str) -> np.ndarray:
    # Numeric ndarray dtypes cannot hide booleans or complex elements. Keep
    # element-wise checking for containers, where e.g. [1., True] must fail.
    numeric_array=isinstance(value,np.ndarray) and value.dtype.kind in 'iuf'
    numeric_scalar=isinstance(value,(int,float,np.integer,np.floating)) and not isinstance(value,(bool,np.bool_))
    if not (numeric_array or numeric_scalar):
        raw = np.asarray(value, dtype=object)
        if any(isinstance(item, (bool, np.bool_, complex, np.complexfloating)) for item in raw.flat):
            raise ValueError('%s must be real numeric values, not boolean or complex' % name)
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError('%s must be finite' % name)
    return result


def checked_covariance(value: Sequence[Sequence[float]], dimension: int = 2) -> np.ndarray:
    """Require finite symmetric positive-definite covariance; never repair it.

    Symmetry tolerance is 32 floating-point epsilons relative to matrix scale.
    LOS covariance is rank two and is validated separately as semidefinite.
    """
    matrix = finite(value, 'covariance')
    if matrix.shape != (dimension, dimension):
        raise ValueError('covariance has incorrect dimensions')
    tolerance = SYMMETRY_EPS_MULTIPLIER * np.finfo(float).eps * np.max(np.abs(matrix))
    if np.max(np.abs(matrix - matrix.T)) > tolerance:
        raise ValueError('covariance must be symmetric')
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError('covariance must be positive definite') from error
    return matrix


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
               for v in (self.width, self.height)):
            raise ValueError('camera image dimensions must be positive integers')
        finite((self.fx_px, self.fy_px, self.cx_px, self.cy_px), 'camera')
        if self.fx_px <= 0 or self.fy_px <= 0:
            raise ValueError('camera focal lengths must be positive')
        if not (0 <= self.cx_px < self.width and 0 <= self.cy_px < self.height):
            raise ValueError('camera principal point must be inside the image')

    def contains(self, xy: Sequence[float]) -> bool:
        return bool(0 <= xy[0] < self.width and 0 <= xy[1] < self.height)


@dataclass(frozen=True)
class FrameInfo:
    frame: int
    timestamp_s: float
    input_available: bool
    attitude_valid: bool
    quaternion_xyzw: Optional[Tuple[float, float, float, float]]

    def __post_init__(self):
        if self.quaternion_xyzw is not None:
            object.__setattr__(self, 'quaternion_xyzw', tuple(self.quaternion_xyzw))
        if isinstance(self.frame, bool) or not isinstance(self.frame, int) or not 0 <= self.frame <= 2147483647:
            raise ValueError('Frame must be a non-negative integer')
        finite(self.timestamp_s, 'timestamp_s')
        if not isinstance(self.input_available, bool) or not isinstance(self.attitude_valid, bool):
            raise ValueError('frame validity flags must be booleans')
        if self.attitude_valid:
            q = finite(self.quaternion_xyzw, 'quaternion')
            if q.shape != (4,) or not np.isclose(q @ q, 1.0, rtol=0, atol=UNIT_VECTOR_EPS_MULTIPLIER*np.finfo(float).eps):
                raise ValueError('valid attitude requires a normalized xyzw quaternion')
        elif self.quaternion_xyzw is not None:
            raise ValueError('invalid attitude must not expose a trusted quaternion')

    @property
    def inertial_observation_available(self) -> bool:
        return self.input_available and self.attitude_valid


@dataclass(frozen=True)
class Detection:
    frame: int
    detection_id: str
    x_px: float
    y_px: float
    covariance_px2: Optional[Matrix2]
    covariance_status: str
    features: Tuple[Tuple[str, float], ...] = ()
    valid: bool = True
    invalid_reason: str = ''

    def __post_init__(self):
        object.__setattr__(self, 'features', tuple(tuple(item) for item in self.features))
        if self.covariance_px2 is not None:
            object.__setattr__(self, 'covariance_px2', tuple(tuple(row) for row in self.covariance_px2))
        if isinstance(self.frame, bool) or not isinstance(self.frame, int) or not 0 <= self.frame <= 2147483647:
            raise ValueError('detection Frame must be a non-negative integer')
        if not isinstance(self.detection_id, str) or not self.detection_id.strip():
            raise ValueError('Detection_ID must be non-empty')
        finite((self.x_px, self.y_px), 'pixel position')
        if self.covariance_px2 is not None:
            checked_covariance(self.covariance_px2)
        if self.covariance_status not in ('simulation_known', 'reported_pending_calibration', 'missing_pending_calibration'):
            raise ValueError('unknown covariance status')
        if (self.covariance_px2 is None) != (self.covariance_status == 'missing_pending_calibration'):
            raise ValueError('covariance and calibration status disagree')
        allowed = {'Flux', 'Peak_Intensity', 'Area', 'SNR', 'Confidence',
                   'Background_Mean', 'Background_Std', 'PSF_Width_X', 'PSF_Width_Y', 'Fit_Residual'}
        if len({k for k, _ in self.features}) != len(self.features):
            raise ValueError('duplicate feature')
        for name, value in self.features:
            if name not in allowed:
                raise ValueError('unsupported or hidden feature: ' + name)
            finite(value, name)
            if name == 'Confidence' and not 0 <= value <= 1:
                raise ValueError('Confidence is a front-end score in [0,1], not a posterior')
            if name != 'Background_Mean' and value < 0:
                raise ValueError(name + ' must be non-negative')
        if not self.valid and not self.invalid_reason:
            raise ValueError('invalid detections require a reason')

    @property
    def key(self) -> Tuple[int, str]:
        return self.frame, self.detection_id


@dataclass(frozen=True)
class KnownStar:
    known_star_id: str
    direction_j2000: Vector3

    def __post_init__(self):
        object.__setattr__(self, 'direction_j2000', tuple(self.direction_j2000))
        vector = finite(self.direction_j2000, 'known star direction')
        if not isinstance(self.known_star_id, str) or not self.known_star_id or vector.shape != (3,) or not np.isclose(vector @ vector, 1.0, rtol=0, atol=UNIT_VECTOR_EPS_MULTIPLIER*np.finfo(float).eps):
            raise ValueError('known star requires a unique ID and unit direction')


@dataclass(frozen=True)
class ObservationPacket:
    """Only public detector observations, frame metadata and known catalog.

    There are no Track_IDs at stage 0. Object_ID and background IDs belong to
    hidden truth or later inference. Tuple fields avoid shared mutable state.
    """
    camera: Camera
    time_scale: str
    frames: Tuple[FrameInfo, ...]
    detections: Tuple[Detection, ...]
    catalog: Tuple[KnownStar, ...] = ()
    origin: str = 'real_detection_table'

    def __post_init__(self):
        for name in ('frames', 'detections', 'catalog'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not isinstance(self.time_scale, str) or not self.time_scale.strip():
            raise ValueError('time scale must be explicitly declared')
        frame_map = {f.frame: f for f in self.frames}
        if len(frame_map) != len(self.frames):
            raise ValueError('duplicate Frame')
        if any(b.frame <= a.frame or b.timestamp_s <= a.timestamp_s
               for a, b in zip(self.frames, self.frames[1:])):
            raise ValueError('frames and timestamps must be strictly increasing')
        if len({d.key for d in self.detections}) != len(self.detections):
            raise ValueError('duplicate (Frame, Detection_ID)')
        for detection in self.detections:
            if detection.frame not in frame_map:
                raise ValueError('detection refers to an absent frame')
            if not frame_map[detection.frame].input_available:
                raise ValueError('missing input frames cannot contain detections')
            if detection.valid and not self.camera.contains((detection.x_px, detection.y_px)):
                raise ValueError('valid detection is outside the camera image')
        if len({s.known_star_id for s in self.catalog}) != len(self.catalog):
            raise ValueError('duplicate KnownStar_ID')
