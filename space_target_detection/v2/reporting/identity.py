"""Target identity of a finished track, decided from its accumulated observations.

Identity is evidence, not current existence: later misses or a departure from
the field of view never revoke it. These checks never feed back into inference.
The local jump gate assumes independent Gaussian measurement errors and local
constant velocity. Other identity thresholds remain uncalibrated presentation
policy, including the allowed fraction of jumps.
"""
import math
import numpy as np
from ..contracts import finite
from ..geometry_errors import GeometryError
from ..gnomonic import GnomonicPlane
from ..measurement_covariance import propagate_tangent_covariance

MIN_OBSERVED_POINTS = 10
MIN_MEAN_TARGET_POSTERIOR = 0.85
MAX_JUMP_FRACTION = 0.1


def jump_threshold_squared(false_alarm_probability):
    """Chi-square(2) threshold: P(D^2 > threshold) = alpha = exp(-threshold/2)."""
    alpha = false_alarm_probability
    if (isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha) or not 0 < alpha < 1):
        raise ValueError('jump_false_alarm_probability must lie in (0,1)')
    return -2. * math.log(alpha)


def jump_diagnostics(observed, *, false_alarm_probability):
    """Check CV residuals against their propagated measurement covariance.

    For a=dt_next/dt_previous, e=z_next-(1+a)z_current+a*z_previous and
    S=R_next+(1+a)^2*R_current+a^2*R_previous. A jump has e.T S^-1 e
    above the two-dimensional chi-square threshold. No speed-based gate or
    truth input is used. Overlapping triples are correlated; their fraction
    is a quality rule, NOT a binomial significance test for the whole track.

    Unavailable geometry/covariance is reported explicitly, never as zero jumps.
    """
    threshold = jump_threshold_squared(false_alarm_probability)
    result = dict(status='INSUFFICIENT_OBSERVATIONS', check_count=0, jump_count=0,
                  jump_fraction=None, threshold_squared=threshold, checks=[],
                  covariance_sources=sorted({r.get('jump_covariance_source') or 'UNSPECIFIED'
                                             for r in observed}))
    if len(observed) < 3:
        return result
    if any(r.get('_direction') is None for r in observed):
        return dict(result, status='MISSING_DIRECTION')
    if any(r.get('_los_covariance') is None for r in observed):
        return dict(result, status='MISSING_COVARIANCE')
    try:
        times = finite([r['timestamp_s'] for r in observed], 'jump timestamps')
        if np.any(np.diff(times) <= 0):
            return dict(result, status='INVALID_TIMESTAMPS')
    except (KeyError, TypeError, ValueError):
        return dict(result, status='INVALID_TIMESTAMPS')
    try:
        directions = finite([r['_direction'] for r in observed], 'jump directions')
        if directions.shape != (len(observed), 3):
            return dict(result, status='INVALID_GEOMETRY_OR_COVARIANCE')
        # An orthonormal basis avoids the old polar singularity. Every observed
        # LOS and its covariance use this same fixed plane and projection.
        plane = GnomonicPlane('identity-quality', tuple(directions.mean(axis=0)), math.radians(89.))
        positions, covariances = [], []
        for row, direction in zip(observed, directions):
            position, jacobian = plane.project(direction)
            positions.append(position)
            covariances.append(propagate_tangent_covariance(row['_los_covariance'], jacobian))
        checks = []
        for i in range(1, len(observed) - 1):
            a = (times[i+1]-times[i]) / (times[i]-times[i-1])
            residual = (positions[i+1]-positions[i]) - a*(positions[i]-positions[i-1])
            covariance = covariances[i+1] + (1+a)**2*covariances[i] + a*a*covariances[i-1]
            whitened = np.linalg.solve(np.linalg.cholesky(covariance), residual)
            distance = float(whitened @ whitened)
            if not math.isfinite(distance):
                return dict(result, status='INVALID_GEOMETRY_OR_COVARIANCE')
            checks.append(dict(frame=observed[i+1].get('frame'),
                               residual_norm=float(np.linalg.norm(residual)),
                               mahalanobis_squared=distance, jump=distance > threshold))
    except (GeometryError, TypeError, ValueError, np.linalg.LinAlgError, FloatingPointError):
        return dict(result, status='INVALID_GEOMETRY_OR_COVARIANCE')
    count = sum(r['jump'] for r in checks)
    return dict(result, status='AVAILABLE', check_count=len(checks), jump_count=count,
                jump_fraction=count/len(checks), checks=checks)


def jump_fraction(observed, *, false_alarm_probability):
    """Return the noise-normalized jump fraction, or None when it is unavailable."""
    return jump_diagnostics(observed, false_alarm_probability=false_alarm_probability)['jump_fraction']


def target_identity_diagnostics(ever_confirmed, observed, *, false_alarm_probability):
    """Keep both the decision and the reason for final identity acceptance/rejection."""
    jumps = jump_diagnostics(observed, false_alarm_probability=false_alarm_probability)
    posteriors = [r['target_posterior'] for r in observed if r.get('target_posterior') is not None]
    if not ever_confirmed:
        reason = 'NEVER_CONFIRMED'
    elif len(observed) < MIN_OBSERVED_POINTS:
        reason = 'INSUFFICIENT_OBSERVATIONS'
    elif not posteriors or not np.isfinite(posteriors).all() or np.mean(posteriors) < MIN_MEAN_TARGET_POSTERIOR:
        reason = 'INSUFFICIENT_TARGET_POSTERIOR'
    elif jumps['status'] != 'AVAILABLE':
        reason = 'JUMP_CHECK_' + jumps['status']
    elif jumps['jump_fraction'] > MAX_JUMP_FRACTION:
        reason = 'EXCESS_NOISE_NORMALIZED_JUMPS'
    else:
        reason = 'CONFIRMED_TARGET_IDENTITY'
    return dict(confirmed=reason == 'CONFIRMED_TARGET_IDENTITY', reason=reason, jumps=jumps)


def is_target_identity(ever_confirmed, observed, *, false_alarm_probability):
    """True when a once-confirmed track has enough, consistent target-like observations."""
    return target_identity_diagnostics(ever_confirmed, observed,
        false_alarm_probability=false_alarm_probability)['confirmed']
