"""Target identity of a finished track, decided from its accumulated observations.

Identity is evidence, not current existence: later misses or a departure from
the field of view never revoke it. All thresholds are presentation policy and
UNCALIBRATED; they never feed back into inference.
"""
import numpy as np

MIN_OBSERVED_POINTS = 10
MIN_MEAN_TARGET_POSTERIOR = 0.85
# A step is a jump when it misses the constant-velocity extrapolation by more
# than this multiple of the extrapolated step length.
JUMP_STEP_RATIO = 1.0
MAX_JUMP_FRACTION = 0.1


def jump_fraction(observed):
    """Fraction of constant-velocity extrapolations that a track's observations break."""
    directions = np.asarray([r['_direction'] for r in observed if r.get('_direction') is not None], float)
    times = np.asarray([r['timestamp_s'] for r in observed if r.get('_direction') is not None], float)
    if len(directions) < 3:
        return 0.
    center = directions.mean(axis=0)
    center /= np.linalg.norm(center)
    east = np.cross([0., 0., 1.], center)
    east /= np.linalg.norm(east)
    north = np.cross(center, east)
    along = directions @ center
    plane = np.c_[directions @ east, directions @ north] / along[:, None]
    jumps = 0
    for i in range(1, len(plane) - 1):
        step = plane[i + 1] - plane[i]
        expected = (plane[i] - plane[i - 1]) * (times[i + 1] - times[i]) / (times[i] - times[i - 1])
        jumps += np.linalg.norm(step - expected) > JUMP_STEP_RATIO * np.linalg.norm(expected)
    return jumps / (len(plane) - 2)


def is_target_identity(ever_confirmed, observed):
    """True when a once-confirmed track has enough, consistent and target-like observations."""
    posteriors = [r['target_posterior'] for r in observed if r.get('target_posterior') is not None]
    return bool(ever_confirmed and len(observed) >= MIN_OBSERVED_POINTS and posteriors
                and float(np.mean(posteriors)) >= MIN_MEAN_TARGET_POSTERIOR
                and jump_fraction(observed) <= MAX_JUMP_FRACTION)
