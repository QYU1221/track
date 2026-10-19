"""Bounded joint selection that reserves distinct existing-target associations.

Weights remain the original joint weights. Diversity selection is an explicit
truncation, not marginal rescaling or a claim of globally optimal top K.
"""
import math


def select_joint_histories(histories, limit, existing_targets, preserve=True):
    ranked = sorted(histories, key=lambda h: (-h.weight, h.hypothesis_id))
    if not preserve or not existing_targets:
        kept = ranked[:limit]
    else:
        groups = {}
        for history in ranked:
            assignments = dict(history.target_assignments)
            present = {track.label for track in history.tracks}
            signature = tuple((label, label in present, assignments.get('T:' + label))
                              for label in sorted(existing_targets))
            groups.setdefault(signature, []).append(history)
        ordered = sorted(groups.values(), key=lambda group: (
            -math.fsum(h.weight for h in group), group[0].hypothesis_id))
        # First reserve one representative for each association explanation.
        kept = [group[0] for group in ordered[:limit]]
        selected = {h.hypothesis_id for h in kept}
        for history in ranked:
            if len(kept) >= limit:
                break
            if history.hypothesis_id not in selected:
                kept.append(history)
        kept.sort(key=lambda h: (-h.weight, h.hypothesis_id))
    retained = math.fsum(h.weight for h in kept)
    discarded = math.fsum(h.weight for h in ranked) - retained
    return kept, max(0., discarded)
