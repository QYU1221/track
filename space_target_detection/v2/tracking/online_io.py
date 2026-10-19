"""Streaming artifacts for the online product of local joint posteriors."""
from dataclasses import replace,asdict
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from ..exact.io import serializable,public_navstar_catalog
from ..exact.contracts import InferenceError
from ..reporting import XStyleReporter
from .factorized import FactorizedJointTracker,MODEL_VERSION,APPROXIMATION
from . import online_checkpoint


def step_with_segmentation(tracker,frame,detections,*,geometry,detail):
    """Commit a fresh segment only after its first frame succeeds."""
    try:
        return tracker,tracker.step(frame,detections,detail=detail),None
    except InferenceError as exc:
        if exc.code!='PREDICTION_GAP_EXCEEDED':raise
    event=dict(segment_id=len(tracker.segment_events)+2,reason='PREDICTION_GAP_EXCEEDED',
        previous_frame=tracker.frame,previous_timestamp_s=tracker.timestamp_s,
        start_frame=frame.frame,start_timestamp_s=frame.timestamp_s,
        gap_s=frame.timestamp_s-tracker.timestamp_s,max_dt_s=tracker.reference.config.max_dt_s)
    fresh=FactorizedJointTracker(tracker.reference,tracker.options,geometry)
    fresh.timestamp_s=frame.timestamp_s
    # Preserve global evidence and versions; dynamic owners start afresh.
    fresh.version=tracker.version
    fresh.consumed_frame_ids=tracker.consumed_frame_ids
    fresh.input_evidence_sha256=tracker.input_evidence_sha256
    fresh.retired=tracker.retired
    fresh.background_retired_count=tracker.background_retired_count
    fresh.segment_events=tracker.segment_events+(event,)
    result=fresh.step(frame,detections,detail=detail)
    return fresh,result,event


def run_online_sequence(packet,raw,output_path,resume_path=None,checkpoint_path=None,*,
                        public_input_path,source_input_path=None,catalog=None,verbose=True,log_every=10):
    from ..reporting.progress import validate_log_every
    validate_log_every(log_every)
    from .io import build_model
    package_root=Path(__file__).parents[2]
    source_sha256={p.relative_to(package_root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package_root.rglob('*.py'))}
    folder=Path(output_path)
    if folder.exists() and any(folder.iterdir()):raise FileExistsError('tracking output must be empty')
    if raw.get('catalog_path'):
        path=Path(raw['catalog_path'])
        if not path.is_absolute():raise ValueError('online compiled catalog path must be absolute')
        if catalog is None:
            catalog=public_navstar_catalog(path,raw.get('catalog_mag_limit'))
        packet=replace(packet,catalog=catalog)
    # Validate the supplied binding before recording any derived metadata.
    model=build_model(packet,raw)
    raw=dict(raw,bound_input=dict(camera=asdict(packet.camera),time_scale=packet.time_scale))
    tracker=FactorizedJointTracker(model,raw['undetected_population'],raw['geometry'])
    if resume_path is not None:online_checkpoint.restore(tracker,resume_path)
    frames=tuple(f for f in packet.frames if ('frames' not in raw or f.frame in raw['frames']) and (tracker.frame is None or f.frame>tracker.frame))
    if not frames:raise ValueError('no requested frames after checkpoint')
    if 'frames' in raw and (len(set(raw['frames']))!=len(raw['frames']) or not set(raw['frames']).issubset({f.frame for f in packet.frames})):
        raise ValueError('requested frames missing or repeated')
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'effective_prior.json').write_text(json.dumps(raw,indent=2,allow_nan=False),encoding='utf-8')
    reporter=XStyleReporter(folder,packet,dict(raw,frames=[f.frame for f in frames]),source_input_path or public_input_path,
                            verbose=verbose,log_every=log_every)
    by_frame=reporter.detections
    rows=[];failed=None;start=time.perf_counter();cpu=time.process_time();events=[]
    with (folder/'observation_provenance.jsonl').open('w',encoding='utf-8') as ledger, \
            (folder/'background_retirement_events.jsonl').open('w',encoding='utf-8') as background_ledger, \
            (folder/'inference_audit.jsonl').open('w',encoding='utf-8') as audit_ledger:
        for frame in frames:
            frame_started=time.perf_counter()
            try:
                tracker,result,segment_event=step_with_segmentation(tracker,frame,tuple(by_frame[frame.frame]),
                    geometry=raw['geometry'],detail=raw['reporting']['detail'])
            except (InferenceError,ValueError,FloatingPointError) as exc:
                failed=dict(frame=frame.frame,error=str(exc),previous_committed_frame=tracker.frame)
                reporter.progress.failure(failed)
                break
            except KeyboardInterrupt:
                reporter.progress.write('[%04d] INTERRUPTED; previous_committed_frame=%s' % (frame.frame,tracker.frame))
                raise
            if segment_event is not None:
                reporter.progress.write('SEGMENT_RESTART '+json.dumps(segment_event,sort_keys=True))
            result['segment_id']=len(tracker.segment_events)+1
            provenance=result.pop('observation_provenance')
            ledger.write(json.dumps(serializable(dict(frame=frame.frame,groups=provenance)),allow_nan=False)+'\n')
            result['observation_provenance_reference']='observation_provenance.jsonl#frame='+str(frame.frame)
            payload=serializable(result)
            audit_ledger.write(json.dumps(dict(frame=frame.frame,**payload['inference_audit']),allow_nan=False)+'\n')
            for event in payload['background_retirement_events']:
                background_ledger.write(json.dumps(event,allow_nan=False)+'\n')
            if raw['reporting']['write_frame_json']:
                (folder/('frame_%06d.json'%frame.frame)).write_text(json.dumps(payload,allow_nan=False),encoding='utf-8')
            reporter.consume(payload,frame_started=frame_started)
            rows.append(dict(frame=frame.frame,**result['diagnostics'],input_detections=len(by_frame[frame.frame]),
                modeled_detections=sum(p['Geometry_Valid'] is True for p in result['points']),valid_attitude=frame.attitude_valid,
                valid_observation=frame.inertial_observation_available))
            events.extend(payload['lifecycle_events'])
    summary=dict(model_version=MODEL_VERSION,tracker_engine='factorized_joint_ppp',
        status='COMPLETED' if failed is None else 'STOPPED_WITH_PREVIOUS_STATE_RETAINED',failed_frame=failed,
        frames_read=len(frames),frames_committed=len(rows),detections_read=sum(len(by_frame[f.frame]) for f in frames),
        segment_count=len(tracker.segment_events)+1,segment_events=tracker.segment_events,
        segmentation_policy='REINITIALIZE_ON_PREDICTION_GAP; NO_CROSS_SEGMENT_IDENTITY',
        valid_attitude_frames=sum(f.attitude_valid for f in frames),valid_empty_frames=sum(f.inertial_observation_available and not by_frame[f.frame] for f in frames),
        elapsed_s=time.perf_counter()-start,cpu_time_s=time.process_time()-cpu,frames=rows,configuration=raw,
        labels=sorted({t.label for g in tracker.groups for h in g.histories for t in h.tracks}),
        approximation_scope=APPROXIMATION,parameter_scope=dict(mode='FIXED_DECLARED_PARAMETERS',online_learning='NOT_IMPLEMENTED'),
        real_accuracy_status='UNVERIFIED_WITH_AVAILABLE_TRUTH',retired_candidate_count=len(tracker.retired),
        retired_background_count=tracker.background_retired_count,dormant_background_count=len(tracker.dormant_background_ids),
        background_retirement_audit='background_retirement_events.jsonl',
        identity_scope='PERSISTENT_RETAINED_LABELS; ARCHIVE_REIDENTIFICATION_NOT_IMPLEMENTED',
        inference_audit='inference_audit.jsonl; observation_provenance.jsonl; optional per-frame JSON',
        frame_json_written=raw['reporting']['write_frame_json'],
        performance_quantiles_s={str(p):float(np.percentile([r['Computation_Time_s'] for r in rows],p)) for p in (50,95,99)} if rows else {},
        public_input_sha256={name:hashlib.sha256((Path(public_input_path)/name).read_bytes()).hexdigest() for name in ('frames.csv','detections.csv','catalog.csv','input_metadata.json')},
        source_sha256=source_sha256,source_hash_scope='ALL_PACKAGE_PYTHON_FILES_AT_RUN_START',
        resumed_from=str(resume_path) if resume_path is not None else None)
    for name,value in (('retired_candidates',tracker.retired),('track_event_log',events)):
        (folder/(name+'.json')).write_text(json.dumps(serializable(value),allow_nan=False),encoding='utf-8')
    if checkpoint_path is not None:
        online_checkpoint.save(tracker,checkpoint_path);summary['checkpoint_path']=str(checkpoint_path)
    summary['presentation']=reporter.finish(summary)
    (folder/'tracking_summary.json').write_text(json.dumps(serializable(summary),indent=2,allow_nan=False),encoding='utf-8')
    return serializable(summary)
