"""Pinhole intrinsics and five-coefficient Brown distortion, geometry1.0.

Normalized image axes follow positive pixel axes. The camera_-X_-Y ray is
rotated about +Z by camera_axis_rotation_deg into the attitude's camera frame.
Distortion is applied in normalized image coordinates before K. Its inverse
Jacobian is computed by the implicit function theorem, not ignored.
"""
from dataclasses import dataclass
import math
from typing import Iterable, Tuple
from .numerical_policy import BASIS_EPS_MULTIPLIER, UNIT_VECTOR_EPS_MULTIPLIER
import numpy as np
from .contracts import Camera, finite
from .geometry import unit
from .geometry_errors import GeometryError


@dataclass(frozen=True)
class CameraModel:
    camera: Camera
    skew_px: float
    distortion_model: str
    distortion_coefficients: tuple  # k1,k2,p1,p2,k3
    max_normalized_radius: float
    inverse_tolerance_px: float
    max_iterations: int
    calibration_version: str
    calibration_status: str
    camera_axis_rotation_deg: float

    def __post_init__(self):
        object.__setattr__(self, 'distortion_coefficients', tuple(self.distortion_coefficients))
        values = finite(self.distortion_coefficients, 'distortion coefficients')
        if values.shape != (5,) or self.distortion_model not in ('none', 'brown5'):
            raise ValueError('distortion supports only none or brown5 (k1,k2,p1,p2,k3)')
        if self.distortion_model == 'none' and np.any(values != 0):
            raise ValueError('none distortion requires zero coefficients')
        finite((self.skew_px, self.max_normalized_radius, self.inverse_tolerance_px), 'camera model')
        if self.max_normalized_radius <= 0 or self.inverse_tolerance_px <= 0:
            raise ValueError('camera radius and inverse pixel tolerance must be positive')
        if isinstance(self.max_iterations, bool) or not isinstance(self.max_iterations, int) or self.max_iterations <= 0:
            raise ValueError('max_iterations must be a positive integer')
        if not self.calibration_version or self.calibration_status not in ('unverified','reported','synthetic_test_value','calibrated'):
            raise ValueError('declare calibration version and status')
        if (isinstance(self.camera_axis_rotation_deg, bool)
                or not isinstance(self.camera_axis_rotation_deg, (int, float))
                or not math.isfinite(self.camera_axis_rotation_deg)):
            raise ValueError('camera_axis_rotation_deg must be a finite angle in degrees')

    @property
    def image_to_camera_rotation(self) -> np.ndarray:
        """Map (normalized pixel x, pixel y, 1) into the quaternion frame.

        Positive angles use a right-handed rotation about the optical +Z axis.
        Zero preserves (-x,-y,1); -90 degrees gives (-y,x,1).
        """
        angle = self.camera_axis_rotation_deg % 360.0
        if angle % 90.0 == 0:
            cosine, sine = ((1., 0.), (0., 1.), (-1., 0.), (0., -1.))[int(angle / 90)]
        else:
            cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        return np.array([[-cosine, sine, 0.], [-sine, -cosine, 0.], [0., 0., 1.]])

    @property
    def K(self) -> np.ndarray:
        c = self.camera
        return np.array([[c.fx_px,self.skew_px,c.cx_px], [0.,c.fy_px,c.cy_px], [0.,0.,1.]])

    def distort(self, normalized: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Return distorted image coordinates (2,) and their Jacobian (2,2).

        Coordinates and coefficients are dimensionless. Radius is a declared
        calibration domain, not a hidden coordinate clamp. A non-positive or
        numerically singular local Jacobian is rejected (folded calibration).
        This local check does not prove global injectivity for arbitrary D.
        """
        p = finite(normalized, 'normalized image')
        if p.shape != (2,) or np.linalg.norm(p) > self.max_normalized_radius:
            raise GeometryError('distortion_domain', 'normalized image outside declared distortion domain')
        if self.distortion_model=='none':
            # The validated zero-distortion model is exactly the identity.
            return p.copy(),np.eye(2)
        x, y = p
        k1,k2,p1,p2,k3 = self.distortion_coefficients
        r2 = x*x+y*y
        with np.errstate(over='raise', invalid='raise'):
            try:
                radial = 1+k1*r2+k2*r2*r2+k3*r2*r2*r2
                dr = 2*(k1+2*k2*r2+3*k3*r2*r2)*p
                out = np.array([x*radial+2*p1*x*y+p2*(r2+2*x*x),
                                y*radial+p1*(r2+2*y*y)+2*p2*x*y])
                J = np.array([[radial+x*dr[0]+2*p1*y+6*p2*x, x*dr[1]+2*p1*x+2*p2*y],
                              [y*dr[0]+2*p1*x+2*p2*y, radial+y*dr[1]+6*p1*y+2*p2*x]])
            except FloatingPointError as error:
                raise GeometryError('distortion_numeric', 'distortion exceeds floating-point range') from error
        finite(out, 'distorted image')
        finite(J, 'distortion Jacobian')
        singular = np.linalg.svd(J, compute_uv=False)
        if np.linalg.det(J) <= 0 or singular[-1] <= BASIS_EPS_MULTIPLIER*np.finfo(float).eps*singular[0]:
            raise GeometryError('distortion_noninvertible', 'folded or singular local distortion calibration')
        return out, J

    def undistort_pixel(self, pixel: Iterable[float]) -> Tuple[np.ndarray, np.ndarray, int]:
        """Correct pixel to normalized image; Jacobian units px^-1.

        Solve D(p)=K^-1 z with damped Newton, then J=J_D^-1 J_K.
        Residual tolerance is measured in pixels. Non-convergence fails.
        """
        xy = finite(pixel, 'pixels')
        if xy.shape != (2,):
            raise ValueError('pixel position must have dimension 2')
        A = self.K[:2,:2]
        try:
            target = np.linalg.solve(A, xy-self.K[:2,2])
            # Distorted coordinates may lie beyond the undistorted calibration
            # radius even when their inverse is valid. Start at the optical
            # center in that case; Newton's iterates remain in the declared
            # domain. This is a solver initialization, not coordinate clipping.
            estimate = (np.zeros(2) if self.distortion_model == 'brown5' and
                        np.linalg.norm(target) > self.max_normalized_radius else target.copy())
            for iteration in range(self.max_iterations):
                forward, J = self.distort(estimate)
                residual = forward-target
                error_px = np.linalg.norm(A @ residual)
                if error_px <= self.inverse_tolerance_px:
                    return estimate, np.linalg.solve(J, np.linalg.solve(A, np.eye(2))), iteration
                step = np.linalg.solve(J, residual)
                accepted = False
                for power in range(20):
                    candidate = estimate-step*(0.5**power)
                    try:
                        mapped, _ = self.distort(candidate)
                    except GeometryError:
                        continue
                    if np.linalg.norm(A @ (mapped-target)) < error_px:
                        estimate, accepted = candidate, True
                        break
                if not accepted:
                    break
        except np.linalg.LinAlgError as error:
            raise GeometryError('distortion_noninvertible', 'cannot solve calibration Jacobian') from error
        raise GeometryError('distortion_nonconvergence', 'inverse distortion did not meet declared pixel tolerance')

    def pixel_to_camera(self, pixel: Iterable[float]) -> dict:
        """Normalized image (2,), camera LOS (3,), dLOS/dpixel (3,2)."""
        p, J_image, iterations = self.undistort_pixel(pixel)
        rotation = self.image_to_camera_rotation
        a = rotation @ np.array([p[0],p[1],1.])
        norm = np.linalg.norm(a)
        b = a/norm
        J_normalize = (np.eye(3)-np.outer(b,b))/norm
        J = J_normalize @ rotation[:,:2] @ J_image
        return {'normalized_image':p, 'corrected_pixel':self.K[:2,:2] @ p+self.K[:2,2],
                'camera_los':b, 'jacobian_camera_pixel':J, 'inverse_iterations':iterations}

    def camera_to_pixel(self, direction: Iterable[float]) -> np.ndarray:
        """Project a front-facing camera ray to a distorted pixel; no bounds clamp."""
        b = unit(direction)
        if b[2] <= UNIT_VECTOR_EPS_MULTIPLIER*np.finfo(float).eps:
            raise GeometryError('camera_singularity', 'camera ray is rear-facing or at projection singularity')
        normalized = self.image_to_camera_rotation[:2,:2].T @ (b[:2]/b[2])
        image, _ = self.distort(normalized)
        return finite(self.K[:2,:2] @ image+self.K[:2,2], 'projected pixel')
