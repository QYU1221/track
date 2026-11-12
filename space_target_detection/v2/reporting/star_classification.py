"""Conservative display policy; thresholds are engineering choices, not calibration.

Unknown-star evidence is a joint event: the same background owns this detection
and has observations in at least four distinct frames within the last eight
frame numbers. Prediction-only rows never count as observations.
"""
STAR_CANDIDATE_SUPPORT = 0.70
STAR_CONFIRMED_SUPPORT = 0.95


def classify_star(point):
    """Return display class, reason, source identity and conservative support.

Retained mass guards against evaluated-branch truncation. It does not bound
the error from gating or heuristic search.
"""
    mass = point.get('Association_Retained_Mass')
    mass = 0.0 if mass is None else max(0.0, min(1.0, mass))
    probability = point['P_Known_Star'] + point['P_Unknown_Star']
    support = probability * mass
    if support < STAR_CANDIDATE_SUPPORT:
        return '未知', 'STAR_SUPPORT_BELOW_CANDIDATE_THRESHOLD', None, support
    evidence = dict(point.get('Unknown_Star_Confirmation_Support', ()))
    confirmed = []
    for source, posterior in point.get('Association_Marginals', ()):
        if source.startswith('S_K:'):
            strength = posterior * mass
        elif source.startswith('S_U:'):
            strength = min(posterior, evidence.get(source, 0.0)) * mass
        else:
            continue
        if strength >= STAR_CONFIRMED_SUPPORT:
            confirmed.append((source, strength))
    if support >= STAR_CONFIRMED_SUPPORT and len(confirmed) == 1:
        source, strength = confirmed[0]
        reason = ('CONFIRMED_CATALOG_STAR' if source.startswith('S_K:')
                  else 'CONFIRMED_STATIC_STAR_4_OF_8_FRAMES')
        return '恒星', reason, source, strength
    return '恒星候选', 'STAR_IDENTITY_OR_MULTIFRAME_EVIDENCE_INSUFFICIENT', None, support
