"""View-limited background approximation, preserving whole joint groups.

No conditional history is split or replaced by marginal feedback. Outside
the padded field of view, observation likelihoods are approximated by one.
Static Q=0 states and persistence predictions remain in checkpoint ownership.
"""
from dataclasses import replace
import math
import numpy as np
from scipy.spatial import cKDTree


class BackgroundViewIndex:
    """Cached inertial directions, queried as a padded spherical-cap index."""
    def __init__(self):
        self.cache={}

    def active_groups(self,groups,rotation,camera,window,config):
        if not config.enable_view_dormancy:return set(range(len(groups)))
        # For Brown distortion use the entire declared calibration domain;
        # arbitrary distortion need not attain its largest ray at a corner.
        if camera.distortion_model=='none':
            x0,x1,y0,y1=window.bounds_px
            radius=max(np.linalg.norm(np.linalg.solve(camera.K[:2,:2],np.asarray((x,y))-camera.K[:2,2]))
                       for x in (x0,x1) for y in (y0,y1))
        else:radius=camera.max_normalized_radius
        angle=math.atan(radius)+math.radians(config.dormancy_margin_deg)
        # Measurement noise supplies an additional angular safety margin.
        angle+=config.dormancy_sigma*math.sqrt(np.linalg.eigvalsh(config.measurement_covariance_px2)[-1])*np.linalg.norm(np.linalg.inv(camera.K[:2,:2]),2)
        active={i for i,g in enumerate(groups) if any(h.tracks for h in g.histories)}
        entries=[];cache={}
        for i,g in enumerate(groups):
            if i in active:continue
            seen=set()
            for h in g.histories:
                for b in h.backgrounds:
                    if b.state is None:continue
                    key=(b.state,b.plane)
                    if key in seen:continue
                    seen.add(key)
                    value=self.cache.get(key)
                    if value is None:
                        direction,_=b.plane.inverse(b.state.mean)
                        # Normalization of (x,y,1) is globally 1-Lipschitz
                        # in tangent coordinates. Chord->angle bounds the
                        # entire sigma ellipse, without a local Jacobian.
                        chord=config.dormancy_sigma*math.sqrt(np.linalg.eigvalsh(b.state.covariance)[-1])
                        spread=2*math.asin(min(1.,chord/2))
                        value=(direction,spread)
                    cache[key]=value;entries.append((i,*value))
        self.cache=cache  # Bound memory to states still owned this frame.
        if not entries:return active
        directions=np.asarray([v[1] for v in entries])
        spans=np.asarray([angle+v[2] for v in entries])
        candidates=cKDTree(directions).query_ball_point(rotation[:,2],2*math.sin(min(math.pi,float(max(spans)))/2))
        for j in candidates:
            i,direction,_=entries[j]
            if direction@rotation[:,2]>=math.cos(min(math.pi,float(spans[j]))):active.add(i)
        return active


def retire_backgrounds(histories,frame,config):
    """Drop low marginal existence only when its mass has visible miss evidence.

    Births have conditional existence one. Their low existence is carried by
    joint-history weights, so branch-local existence thresholds cannot prune
    them. All branches are removed together with explicit discarded mass.
    """
    if not config.enable_retirement:return histories,()
    identities={b.background_id for h in histories for b in h.backgrounds}
    removed=set();events=[]
    for identity in sorted(identities):
        parts=[(h.weight,b) for h in histories for b in h.backgrounds if b.background_id==identity]
        mass=math.fsum(w*b.existence for w,b in parts)
        mature_mass=math.fsum(w*b.existence for w,b in parts if b.visible_miss_count>=config.retirement_visible_misses)
        logs=[(math.log(w)+b.log_existence,b) for w,b in parts if w>0 and b.log_existence > -math.inf]
        maximum=max((value for value,_ in logs),default=-math.inf)
        scaled=math.fsum(math.exp(value-maximum) for value,_ in logs)
        mature_scaled=math.fsum(math.exp(value-maximum) for value,b in logs if b.visible_miss_count>=config.retirement_visible_misses)
        if mass<config.retirement_existence_probability and mature_scaled>0 and mature_scaled>=config.retirement_mature_mass_fraction*scaled:
            removed.add(identity)
            events.append(dict(Background_ID=identity,frame=frame.frame,timestamp_s=frame.timestamp_s,
                reason='LOW_MARGINAL_EXISTENCE_AFTER_VISIBLE_MISSES',discarded_existence_mass=mass,
                mature_existence_mass=mature_mass))
    if not removed:return histories,()
    return tuple(replace(h,backgrounds=tuple(b for b in h.backgrounds if b.background_id not in removed),
        background_assignments=tuple((s,k) for s,k in h.background_assignments if s[4:] not in removed)) for h in histories),tuple(events)
