"""Camera parameters grouped without changing their source values."""

from dataclasses import dataclass

from . import values as legacy


@dataclass(frozen=True)
class CameraConfig:
    profile: str
    frame_width: int
    frame_height: int
    focal_length_m: float
    pixel_size_m: float
    principal_x: float
    principal_y: float
    optical_axis: str
    projection_model: str
    quaternion_order: str
    quaternion_inverse: bool


def current_camera_config():
    return CameraConfig(
        profile=legacy.camera_profile,
        frame_width=legacy.FRAME_W,
        frame_height=legacy.FRAME_H,
        focal_length_m=legacy.camera_focal_length_m,
        pixel_size_m=legacy.camera_pixel_size_m,
        principal_x=legacy.camera_cx,
        principal_y=legacy.camera_cy,
        optical_axis=legacy.camera_optical_axis,
        projection_model=legacy.camera_projection_model,
        quaternion_order=legacy.camera_quaternion_order,
        quaternion_inverse=legacy.camera_quaternion_inverse,
    )


def external_quaternion_to_wxyz(quaternion, config=None):
    """Convert a profile-specific external quaternion to internal wxyz form."""
    import numpy as np

    camera = config or current_camera_config()
    raw = np.asarray(quaternion, dtype=np.float64).reshape(4)
    if camera.quaternion_order == "wxyz":
        result = raw.copy()
    elif camera.quaternion_order == "xyzw":
        result = raw[[3, 0, 1, 2]]
    else:
        raise ValueError("Unsupported quaternion order: %s" % camera.quaternion_order)
    if camera.quaternion_inverse:
        result[1:] *= -1.0
    norm = np.linalg.norm(result)
    if norm < 1e-12:
        raise ValueError("External quaternion has zero norm")
    return result / norm
