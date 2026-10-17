"""Explicit adapters; never infer quaternion order or rotation direction."""
from typing import Iterable, Tuple
import numpy as np
from .numerical_policy import UNIT_VECTOR_EPS_MULTIPLIER
from .geometry import normalize_quaternion, quaternion_matrix, checked_rotation
from .contracts import finite


def adapt_quaternion(values: Iterable[float], *, order: str,
                     direction: str) -> Tuple[float, ...]:
    """Return canonical xyzw camera-to-J2000 attitude.

    order: xyzw or wxyz; direction: camera_to_j2000 or j2000_to_camera.
    Input is normalized. The inverse of a unit quaternion is its conjugate.
    Neither convention has a default in this external adapter.
    """
    q = normalize_quaternion(values)
    if order == 'wxyz':
        q = (q[1],q[2],q[3],q[0])
    elif order != 'xyzw':
        raise ValueError('declare quaternion order xyzw or wxyz')
    if direction == 'j2000_to_camera':
        q = (-q[0],-q[1],-q[2],q[3])
    elif direction != 'camera_to_j2000':
        raise ValueError('declare attitude rotation direction')
    return q


def camera_to_inertial(camera_los: np.ndarray, jacobian_camera_pixel: np.ndarray,
                       quaternion_xyzw: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    """D-12 A: deterministic R_IC rotates LOS and its measurement Jacobian."""
    b = finite(camera_los,'camera unit LOS')
    J = finite(jacobian_camera_pixel,'camera Jacobian')
    if (b.shape != (3,) or J.shape != (3,2) or
            not np.isclose(b @ b,1.,rtol=0,atol=UNIT_VECTOR_EPS_MULTIPLIER*np.finfo(float).eps)):
        raise ValueError('camera LOS must be unit (3,) and Jacobian must be (3,2)')
    R = checked_rotation(quaternion_matrix(quaternion_xyzw))
    return R @ b, R @ J
