"""Domain-oriented views of the behavior-compatible legacy configuration."""

from .camera import CameraConfig, current_camera_config, external_quaternion_to_wxyz
from .features import FeatureFlags, current_feature_flags

__all__ = [
    "CameraConfig", "current_camera_config", "external_quaternion_to_wxyz",
    "FeatureFlags", "current_feature_flags",
]
