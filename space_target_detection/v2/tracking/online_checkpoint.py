"""Integrity-checked checkpoint for the product-of-joint-groups backend."""
from pathlib import Path
import hashlib
import json
from ..numerical_policy import CHECKPOINT_NORMALIZATION_ATOL
from ..exact.io import serializable
from ..exact.contracts import GaussianState
from ..gnomonic import GnomonicPlane
from .model import Track,JointHistory
from .background import Background,GaussianDirection
from .lifecycle import ManagementState
from .candidates import RetiredCandidate
from .undetected import Population
from .factorized import Group,MODEL_VERSION
from .input_evidence import validate_consumed_input

SCHEMA='factorized-joint-populations-4'


def digest(value):
    return hashlib.sha256(json.dumps(serializable(value),sort_keys=True,allow_nan=False).encode()).hexdigest()


def binding(tracker):
    from ..numerical_policy import snapshot
    r=tracker.reference
    return digest(dict(version=MODEL_VERSION,camera=r.camera,catalog=r.catalog,stars=r.stars,clutter=r.clutter,
        tracking=r.config,background=r.background_config,lifecycle=r.lifecycle_config,inference=r.inference_config,
        population=tracker.options,time_scale=r.time_scale,initial_time=r.initial_timestamp_s,
        max_tangent_angle_rad=tracker.max_tangent_angle_rad,numerical_policy=snapshot()))


def save(tracker,path):
    path=Path(path)
    if path.exists():raise FileExistsError('checkpoint is immutable; choose a new path')
    validate_consumed_input(tracker.consumed_frame_ids,tracker.input_evidence_sha256,
        tracker.reference.input_frame_digests,last_frame=tracker.frame,version=tracker.version)
    snapshot=dict(frame=tracker.frame,timestamp_s=tracker.timestamp_s,version=tracker.version,groups=tracker.groups,
        populations=tracker.populations,management=tracker.management,retired=tracker.retired,
        dormant_background_ids=tracker.dormant_background_ids,background_retired_count=tracker.background_retired_count,
        consumed_frame_ids=tracker.consumed_frame_ids,input_evidence_sha256=tracker.input_evidence_sha256,
        segment_events=tracker.segment_events)
    raw=dict(schema=SCHEMA,model_digest=binding(tracker),snapshot=serializable(snapshot),snapshot_sha256=digest(snapshot))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(raw,allow_nan=False),encoding='utf-8')


def restore(tracker,path):
    if tracker.frame is not None:raise ValueError('restore only into a fresh tracker')
    raw=json.loads(Path(path).read_text(encoding='utf-8'))
    if raw['schema']!=SCHEMA:
        raise ValueError('checkpoint schema mismatch; rerun the original input with this code version')
    if raw['model_digest']!=binding(tracker):raise ValueError('checkpoint model mismatch')
    data=raw['snapshot']
    if raw['snapshot_sha256']!=digest(data):raise ValueError('checkpoint integrity mismatch')
    validate_consumed_input(data['consumed_frame_ids'],data['input_evidence_sha256'],
        tracker.reference.input_frame_digests,last_frame=data['frame'],version=data['version'])
    def obj(v,star=False):
        v=dict(v);v['plane']=GnomonicPlane(**v['plane'])
        v['state']=(GaussianDirection if star else GaussianState)(**v['state']) if v['state'] else None
        return (Background if star else Track)(**v)
    groups=[];seen=set()
    for g in data['groups']:
        histories=[]
        for h in g['histories']:
            h=dict(h);h['tracks']=tuple(obj(t) for t in h['tracks']);h['backgrounds']=tuple(obj(t,True) for t in h['backgrounds'])
            h['target_assignments']=tuple((s,tuple(k) if k else None) for s,k in h['target_assignments'])
            h['star_assignments']=tuple((s,tuple(k) if k else None) for s,k in h['star_assignments'])
            h['background_assignments']=tuple((s,tuple(k) if k else None) for s,k in h['background_assignments'])
            h['clutter_keys']=tuple(map(tuple,h['clutter_keys']))
            history=JointHistory(**h)
            if any(k[0]>data['frame'] for t in history.tracks+history.backgrounds for k in t.support_keys):
                raise ValueError('checkpoint contains unconsumed future evidence')
            histories.append(history)
        if (not histories or
                abs(sum(h.weight for h in histories)-1)>CHECKPOINT_NORMALIZATION_ATOL):
            raise ValueError('checkpoint group normalization')
        labels={t.label for h in histories for t in h.tracks}|{b.background_id for h in histories for b in h.backgrounds}
        if labels & seen:raise ValueError('checkpoint object has multiple probability owners')
        seen|=labels;groups.append(Group(tuple(histories)))
    populations=[]
    for p in data['populations']:
        p=dict(p);p['plane']=GnomonicPlane(**p['plane']);p['state']=(GaussianState if p['kind']=='T' else GaussianDirection)(**p['state'])
        if p['expected_count']<0:raise ValueError('negative population mass')
        populations.append(Population(**p))
    management=tuple(ManagementState(**v) for v in data['management'])
    if {v.label for v in management}!={t.label for g in groups for h in g.histories for t in h.tracks}:
        raise ValueError('checkpoint management/probability owner mismatch')
    retired=[]
    for v in data['retired']:
        v=dict(v);v['management']=ManagementState(**v['management'])
        v['conditional_branches']=tuple((hid,w,obj(t)) for hid,w,t in v['conditional_branches'])
        retired.append(RetiredCandidate(**v))
    dormant=tuple(data.get('dormant_background_ids',()))
    background_ids={b.background_id for g in groups for h in g.histories for b in h.backgrounds}
    if len(set(dormant))!=len(dormant) or not set(dormant)<=background_ids:
        raise ValueError('checkpoint dormant background ownership mismatch')
    count=data.get('background_retired_count',0)
    if isinstance(count,bool) or not isinstance(count,int) or count<0:
        raise ValueError('checkpoint invalid retired background count')
    tracker.groups=tuple(groups);tracker.populations=tuple(populations);tracker.management=management;tracker.retired=tuple(retired)
    tracker.frame=data['frame'];tracker.timestamp_s=data['timestamp_s'];tracker.version=data['version']
    tracker.dormant_background_ids=dormant;tracker.background_retired_count=count
    tracker.consumed_frame_ids=tuple(data['consumed_frame_ids']);tracker.input_evidence_sha256=data['input_evidence_sha256']
    tracker.segment_events=tuple(data.get('segment_events',()))
