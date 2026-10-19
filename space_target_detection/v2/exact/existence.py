"""Authoritative log existence with a float projection for reporting."""
import math


def checked_log_existence(existence, log_existence=None):
    if log_existence is None:
        return math.log(existence) if existence else -math.inf
    value = float(log_existence)
    if math.isnan(value) or value > 0:
        raise ValueError('log existence must be <= 0')
    if not math.isclose(math.exp(value), existence, rel_tol=1e-12, abs_tol=0.):
        raise ValueError('existence and log existence disagree')
    return value


def log_absence(log_existence):
    if log_existence == 0:
        return -math.inf
    if log_existence < -math.log(2):
        return math.log1p(-math.exp(log_existence))
    return math.log(-math.expm1(log_existence))
