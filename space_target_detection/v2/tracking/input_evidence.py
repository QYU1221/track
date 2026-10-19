"""Canonical fingerprints of frames actually consumed by the tracker.

Future observations are excluded so the same model can resume with appended
input. Row order and equivalent integer/float coordinates do not change a
fingerprint; detection flags, covariance, features and attitude do.
"""
from collections import defaultdict
import hashlib
import json

EMPTY_DIGEST = hashlib.sha256(b'consumed-observations-1').hexdigest()


def _digest(value):
    encoded=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def frame_digest(frame,detections):
    metadata=dict(frame=frame.frame,timestamp_s=float(frame.timestamp_s),
        input_available=frame.input_available,attitude_valid=frame.attitude_valid,
        quaternion_xyzw=tuple(float(v) for v in frame.quaternion_xyzw) if frame.quaternion_xyzw is not None else None)
    points=[]
    for d in sorted(detections,key=lambda d:d.key):
        points.append(dict(frame=d.frame,detection_id=d.detection_id,x_px=float(d.x_px),y_px=float(d.y_px),
            covariance_px2=tuple(tuple(float(v) for v in row) for row in d.covariance_px2) if d.covariance_px2 is not None else None,
            covariance_status=d.covariance_status,features=sorted((k,float(v)) for k,v in d.features),
            valid=d.valid,invalid_reason=d.invalid_reason))
    return _digest(dict(metadata=metadata,detections=points))


def extend_digest(previous,number,evidence):
    return _digest((previous,number,evidence))


def packet_frame_digests(packet):
    by_frame=defaultdict(list)
    for detection in packet.detections:
        by_frame[detection.frame].append(detection)
    return {frame.frame:frame_digest(frame,by_frame[frame.frame]) for frame in packet.frames}


def validate_consumed_input(frame_ids,expected,frame_digests,*,last_frame,version):
    ids=tuple(frame_ids)
    if (any(isinstance(n,bool) or not isinstance(n,int) or n<0 for n in ids)
            or any(b<=a for a,b in zip(ids,ids[1:]))
            or len(ids)!=version or (ids[-1] if ids else None)!=last_frame):
        raise ValueError('checkpoint invalid consumed-frame history')
    actual=EMPTY_DIGEST
    for number in ids:
        if number not in frame_digests:
            raise ValueError('checkpoint consumed input frame is missing: '+str(number))
        actual=extend_digest(actual,number,frame_digests[number])
    if actual!=expected:
        raise ValueError('checkpoint consumed input differs: detections or frame metadata changed')
