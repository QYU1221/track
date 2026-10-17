"""Joint weights, bounded normalization and conditional existence states."""
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from ..numerical_policy import BASIS_EPS_MULTIPLIER
from .contracts import Association, FactorTable, InferenceError
from .numerics import log_probability


@dataclass(frozen=True)
class ConditionalExistence:
    reference: str
    existence: float


@dataclass(frozen=True)
class JointPosterior:
    hypothesis_id: str
    parent_hypothesis_id: str
    target_assignments: tuple
    star_assignments: tuple
    unknown_star_assignments: tuple
    clutter_detection_keys: tuple
    log_unnormalized_weight: float
    posterior_weight: float
    conditional_state_references: tuple


@dataclass(frozen=True)
class InferenceResult:
    hypotheses: tuple
    conditional_states: tuple
    log_predictive_evidence: float


def joint_log_weight(table: FactorTable, association: Association) -> float:
    """Finite-set density under one injective source assignment."""
    sources = {source.source_id: source for source in table.sources}
    terms = [log_probability(table.parent_weight)]
    for source_id, detection_index in association.assignments:
        source = sources[source_id]
        if detection_index is None:
            terms.append(source_log_miss_factor(source))
        else:
            terms.append(log_probability(source.existence_prior) + source.log_eta[detection_index])
    terms.extend(table.log_kappa[index] for index in association.clutter_indices)
    if any(np.isneginf(value) for value in terms):
        return -np.inf
    try:
        value = math.fsum(terms)
    except OverflowError as error:
        raise InferenceError('NUMERIC_RANGE_EXCEEDED', 'log weight exceeds numeric range') from error
    if not np.isfinite(value):
        raise InferenceError('NUMERIC_RANGE_EXCEEDED', 'nonfinite log weight')
    return value


def source_log_miss_given_exists(source) -> float:
    if source.log_miss_given_exists is not None:
        return source.log_miss_given_exists
    return float(np.log1p(-source.pd_bar)) if source.pd_bar < 1 else -np.inf


def source_log_miss_factor(source) -> float:
    return float(np.logaddexp(
        log_probability(1 - source.existence_prior),
        log_probability(source.existence_prior) + source_log_miss_given_exists(source)))


def normalize_log_weights(log_weights: tuple) -> tuple:
    values = np.asarray(log_weights, dtype=float)
    if not len(values) or np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise InferenceError('INVALID_LOG_WEIGHT', 'weights must be finite or -inf')
    if np.all(np.isneginf(values)):
        raise InferenceError('ZERO_EVIDENCE', 'no hypothesis has positive model density')
    maximum = float(np.max(values))
    shifted = values - maximum
    normalizer = math.log(math.fsum(math.exp(value) for value in shifted))
    posterior = np.exp(shifted - normalizer)
    evidence = maximum + normalizer
    if not np.isfinite(evidence):
        raise InferenceError('NUMERIC_RANGE_EXCEEDED', 'evidence exceeds numeric range')
    return tuple(float(value) for value in posterior), float(evidence)


def _conditional_existence(table: FactorTable, source, detection_index) -> ConditionalExistence:
    key = table.detection_keys[detection_index] if detection_index is not None else None
    reference = 'STATE:' + hashlib.sha256(json.dumps(
        (table.parent_id, source.source_id, key), separators=(',', ':')).encode()).hexdigest()
    if detection_index is not None:
        return ConditionalExistence(reference, 1.)
    log_denominator = source_log_miss_factor(source)
    if np.isneginf(log_denominator):
        raise InferenceError('IMPOSSIBLE_CONDITION', 'zero-weight miss has no conditional posterior')
    log_existence = (log_probability(source.existence_prior) +
                     source_log_miss_given_exists(source) - log_denominator)
    existence = float(np.exp(log_existence))
    if existence == 0 and np.isfinite(log_existence):
        raise InferenceError('PROBABILITY_UNDERFLOW',
                             'positive conditional existence cannot be stored as zero')
    return ConditionalExistence(reference, existence)


def infer_factors(tables: tuple, computation_policy) -> InferenceResult:
    """Normalize candidates supplied by the configured bounded graph policy."""
    if not tables:
        raise ValueError('at least one parent required')
    tables = tuple(sorted(tables, key=lambda table: table.parent_id))
    if (len({table.parent_id for table in tables}) != len(tables) or
            any(set(table.detection_keys) != set(tables[0].detection_keys) for table in tables)):
        raise ValueError('parents must be unique and share one detection set')
    if abs(math.fsum(table.parent_weight for table in tables) - 1) > BASIS_EPS_MULTIPLIER * np.finfo(float).eps:
        raise ValueError('parent prior weights must sum to 1')

    computation_policy.parent_count = len(tables)
    plans = []
    for table in tables:
        for association in computation_policy.proposals(table):
            computation_policy.controller.consume()
            plans.append((table, association))
    logs = tuple(joint_log_weight(table, association) for table, association in plans)
    keep = computation_policy.select(plans, logs)
    plans = [plans[index] for index in keep]
    logs = tuple(logs[index] for index in keep)
    weights, evidence = normalize_log_weights(logs)

    cache = {}
    rows = []
    for (table, association), log_weight, weight in zip(plans, logs, weights):
        lookup = {source.source_id: source for source in table.sources}
        targets, stars, unknown_stars, references = [], [], [], []
        for source_id, detection_index in association.assignments:
            source = lookup[source_id]
            key = table.detection_keys[detection_index] if detection_index is not None else None
            if source.kind in ('T', 'S_U'):
                (targets if source.kind == 'T' else unknown_stars).append((source_id, key))
                if np.isfinite(log_weight):
                    cache_key = (table.parent_id, source_id, key)
                    if cache_key not in cache:
                        cache[cache_key] = _conditional_existence(table, source, detection_index)
                    references.append((source_id, cache[cache_key].reference))
            else:
                stars.append((source_id, key))
        rows.append(JointPosterior(
            association.hypothesis_id, table.parent_id, tuple(targets), tuple(stars),
            tuple(unknown_stars),
            tuple(table.detection_keys[index] for index in association.clutter_indices),
            log_weight, weight, tuple(references)))
    return InferenceResult(tuple(rows),
                           tuple(sorted(cache.values(), key=lambda state: state.reference)), evidence)
