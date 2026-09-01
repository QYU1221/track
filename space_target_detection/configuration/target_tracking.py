"""Filtering, association and lifecycle configuration namespace."""

from .values import *  # noqa: F401,F403

TARGET_TRACKING_KEYS = tuple(name for name in globals() if (
    name.startswith(('imm_', 'cv_', 'ctrv_', 'jpda_', 'sinkhorn_',
                     'prune_', 'fallback_', 'in_fov_', 'Q_', 'R_', 'P_')) or
    name in {'use_imm', 'eta', 'tentative_min', 'mahalanobis_gate',
             'max_velocity', 'strict_one_update_per_frame'}
))
