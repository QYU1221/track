"""Implementation tolerances, not detector/model tuning parameters.

Changing these requires numerical regression, not a runtime JSON override.
Multipliers below are applied to IEEE floating-point machine epsilon.
"""
POLICY_VERSION = 'numerical-policy-3'
BIRTH_QUADRATURE_ORDER = 7
SYMMETRY_EPS_MULTIPLIER = 32
UNIT_VECTOR_EPS_MULTIPLIER = 64
BASIS_EPS_MULTIPLIER = 128
PROBABILITY_EPS_MULTIPLIER = 256
POSTERIOR_NORMALIZATION_ATOL = 1e-9
CHECKPOINT_NORMALIZATION_ATOL = 1e-10
QUADRATURE_ABS_TOL = 1e-12
QUADRATURE_REL_TOL = 1e-10
QUADRATURE_ERROR_LIMIT = 1e-9
QUADRATURE_SUBDIVISIONS = 200
NORMAL_INTEGRATION_CUTS = (-12., -6., 0., 6., 12.)
WINDOW_INTERIOR_SIGMA = 9.
DISPLAY_CLASS_TIE_ATOL = 1e-13
DISPLAY_PROJECTION_MIN_Z = 1e-10
DISPLAY_SPAN_FLOOR = 1e-9
DISPLAY_DIRECTION_SINE_FLOOR = 1e-12
DISPLAY_VECTOR_NORM_FLOOR = 1e-8


def snapshot():
    return {name: value for name, value in globals().items() if name.isupper()}
