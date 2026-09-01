"""Public attitude solvers and star-projection geometry."""

from .attitude_solvers import (
    solve_attitude,
    solve_davenport,
    solve_quest,
    solve_star_attitude,
    solve_svd,
)
from .projection_geometry import (
    body_vector_to_image_point,
    image_point_to_body_vector,
    project_catalog_stars,
    project_catalog_vectors,
    project_inertial_vectors,
)
from .projection_tracking import track_stars_by_attitude_projection

__all__ = [
    "solve_star_attitude",
    "solve_attitude",
    "solve_svd",
    "solve_davenport",
    "solve_quest",
    "image_point_to_body_vector",
    "body_vector_to_image_point",
    "project_catalog_stars",
    "project_inertial_vectors",
    "project_catalog_vectors",
    "track_stars_by_attitude_projection",
]
