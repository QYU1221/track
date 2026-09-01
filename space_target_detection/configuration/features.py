"""Top-level feature switches grouped for experiment inspection."""

from dataclasses import dataclass

from . import values as legacy


@dataclass(frozen=True)
class FeatureFlags:
    star_identification: bool
    attitude: bool
    trajectory_association: bool
    background_subtraction: bool
    triangle_background: bool
    in_fov_reconnect: bool
    exit_reconnect: bool


def current_feature_flags():
    return FeatureFlags(
        star_identification=legacy.enable_star_id,
        attitude=legacy.enable_attitude,
        trajectory_association=legacy.enable_trajectory_association,
        background_subtraction=legacy.enable_background_subtraction,
        triangle_background=legacy.triangle_background_enabled,
        in_fov_reconnect=legacy.enable_in_fov_reconnect,
        exit_reconnect=legacy.enable_exit_reconnect,
    )
