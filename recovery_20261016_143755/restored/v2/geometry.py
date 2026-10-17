"""Direction normalization and camera-to-J2000 rotation primitives."""
from typing import Iterable, Tuple
from .numerical_policy import BASIS_EPS_MULTIPLIER
import numpy as np
from .contracts import finite


def unit(vector: Iterable[float]) -> np.ndarray:
    result = finite(vector, 'direction')
    if result.shape != (3,) or np.max(np.abs(result)) == 0:
        raise ValueError('direction must be a non-zero 3-vector')
    result = result / np.max(np.abs(result))
    return result / np.linalg.norm(result)


def normalize_quaternion(value: Iterable[float]) -> Tuple[float, ...]:
    q = finite(value, 'quaternion')
    if q.shape != (4,) or np.max(np.abs(q)) == 0:
        raise ValueError('quaternion must be a non-zero 4-vector')
    q = q / np.max(np.abs(q))
    return tuple(q / np.linalg.norm(q))


def quaternion_matrix(q_xyzw: Iterable[float]) -> np.ndarray:
    """Return camera-to-J2000 rotation from an explicitly xyzw quaternion."""
    x, y, z, w = normalize_quaternion(q_xyzw)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def checked_rotation(rotation: np.ndarray) -> np.ndarray:
    R = finite(rotation, 'rotation')
    tolerance = BASIS_EPS_MULTIPLIER*np.finfo(float).eps
    if R.shape != (3, 3) or not np.allclose(R.T @ R, np.eye(3), atol=tolerance, rtol=0) or abs(np.linalg.det(R)-1) > tolerance:
        raise ValueError('rotation must be a proper orthogonal 3x3 matrix')
    return R
