"""Star recognition and attitude configuration namespace."""

from .values import *  # noqa: F401,F403 - compatibility view of established values

STAR_PROCESSING_KEYS = tuple(name for name in globals() if (
    name.startswith('star_id_') or name.startswith('attitude_') or
    name.startswith('star_catalog_') or name.startswith('camera_') or
    name in {'FRAME_W', 'FRAME_H'}
))
