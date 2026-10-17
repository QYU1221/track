"""Independent pixel measurement covariance propagation (D-12 A)."""
import numpy as np
from .numerical_policy import BASIS_EPS_MULTIPLIER
from .contracts import checked_covariance, finite


def checked_los_covariance(value: np.ndarray) -> np.ndarray:
    """Finite symmetric PSD covariance with numerical rank 2; no repairs."""
    C_los = finite(value,'LOS covariance')
    if C_los.shape != (3,3):
        raise ValueError('LOS covariance must have shape (3,3)')
    scale = np.max(np.abs(C_los))
    tolerance = BASIS_EPS_MULTIPLIER*np.finfo(float).eps*scale
    if scale == 0 or np.max(np.abs(C_los-C_los.T)) > tolerance:
        raise ValueError('LOS covariance must be nonzero and symmetric')
    eigenvalues = np.linalg.eigvalsh(C_los)
    if eigenvalues[0] < -tolerance or eigenvalues[1] <= tolerance or abs(eigenvalues[0]) > tolerance:
        raise ValueError('LOS covariance must be PSD with numerical rank 2')
    return C_los


def propagate_los_covariance(covariance_px2: np.ndarray,
                            jacobian_inertial_pixel: np.ndarray) -> np.ndarray:
    """Validate and propagate independent pixel error to rank-2 LOS space."""
    R = checked_covariance(covariance_px2)
    J = finite(jacobian_inertial_pixel,'LOS Jacobian')
    if J.shape != (3,2):
        raise ValueError('LOS Jacobian dimensions must be (3,2)')
    return checked_los_covariance(J @ R @ J.T)


def propagate_tangent_covariance(covariance_los: np.ndarray,
                                 jacobian_tangent_inertial: np.ndarray) -> np.ndarray:
    """Transport an already computed LOS covariance, without recounting error."""
    C = checked_los_covariance(covariance_los)
    G = finite(jacobian_tangent_inertial,'tangent Jacobian')
    if G.shape != (2,3):
        raise ValueError('tangent Jacobian dimensions must be (2,3)')
    return checked_covariance(G @ C @ G.T)


