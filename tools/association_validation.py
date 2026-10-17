"""Truth-sidecar evaluation: truth is never passed into the tracker."""
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from space_target_detection.v2.data_io import load_packet
from space_target_detection.pipeline.configuration import resolve_configuration
from space_target_detection.v2.contracts import Camera,FrameInfo,Detection,ObservationPacket
from space_target_detection.v2.tracking.io import build_model
from space_target_detection.v2.tracking.factorized import FactorizedJointTracker
from space_target_detection.v2.exact.contracts import InferenceError


def assignments(groups,threshold=.7):
    points=defaultdict(lambda:defaultdict(float))
    for group in groups:
        for h in group:
            for source,key in h['targets']:
                if key is not None:points[tuple(key)][source]+=h['weight']
    result={}
    for key,weights in points.items():
        winners=[source for source,p in weights.items() if p>=threshold]
        if len(winners)==1:result[key]=winners[0]
    return result


def evaluate(truth,predictions):
    verified=[r for r in truth if r['review_status']=='VERIFIED']
    keys=[(int(r['frame']),r['detection_id']) for r in verified]
    if len(keys)!=len(set(keys)):raise ValueError('duplicate verified detection keys')
    if any(r['kind'] not in ('target','star','clutter') or (r['kind']=='target' and not r['object_id']) for r in verified):
        raise ValueError('verified truth requires a valid kind and target object ID')
    reviewed=[r for r in truth if r['review_status']=='VERIFIED' and r['kind']=='target' and r['object_id']]
    if not verified:return dict(status='AWAITING_INDEPENDENT_REVIEW',verified_target_points=0)
    paths=defaultdict(list);owners=defaultdict(set)
    for r in reviewed:
        key=(int(r['frame']),r['detection_id']);label=predictions.get(key)
        paths[r['object_id']].append((key[0],label))
        if label:owners[label].add(r['object_id'])
    switches=fragmentation=reconnects=successful=assigned=0
    for rows in paths.values():
        rows.sort();last=None;last_frame=None;seen=set()
        for frame,label in rows:
            if label:
                assigned+=1;seen.add(label)
                if last is not None:
                    switches+=label!=last
                    if frame-last_frame>1:reconnects+=1;successful+=label==last
                last=label;last_frame=frame
        fragmentation+=max(0,len(seen)-1)
    return dict(status='EVALUATED_REVIEWED_SUBSET',verified_target_points=len(reviewed),assigned_points=assigned,
                unassigned_points=len(reviewed)-assigned,identity_switches=switches,
                extra_labels_per_object_sum=fragmentation,labels_shared_by_distinct_objects=sum(len(v)>1 for v in owners.values()),
                verified_non_target_points=sum(r['kind']!='target' for r in verified),
                target_assignments_on_verified_non_targets=sum((int(r['frame']),r['detection_id']) in predictions for r in verified if r['kind']!='target'),
                reconnect_opportunities_among_assigned_points=reconnects,same_label_reconnects=successful,
                scope='Thresholded online identities on reviewed detections; unassigned points reported separately; frame-gap reconnects include unavailable assignments')


def prepare(public,out):
    packet=load_packet(public,max_rows=1000000);out.mkdir(parents=True,exist_ok=False)
    windows=[('early_exit',146,170),('crossing_and_reconnect',469,500),('later_reconnect',501,576)]
    frames={f.frame:f for f in packet.frames};rows=[]
    for name,lo,hi in windows:
        for d in packet.detections:
            if lo<=d.frame<=hi and d.valid:
                rows.append(dict(case=name,frame=d.frame,timestamp_s=frames[d.frame].timestamp_s,detection_id=d.detection_id,
                    x_px=d.x_px,y_px=d.y_px,kind='',object_id='',review_status='UNREVIEWED',reviewer='',notes=''))
    if not rows:raise ValueError('no detections in the configured HIT review windows')
    with (out/'review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    gaps=[dict(before=a.frame,after=b.frame,dt_s=b.timestamp_s-a.timestamp_s) for a,b in zip(packet.frames,packet.frames[1:]) if b.timestamp_s-a.timestamp_s>15]
    (out/'manifest.json').write_text(json.dumps(dict(status='UNREVIEWED',truth_source='HUMAN_REQUIRED; NOT_TRACKER_OUTPUT',
        input_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(public).iterdir() if p.is_file()},
        cases=windows,time_gaps_over_15s=gaps),indent=2),encoding='utf-8')
    # Self-contained visual reference. Labels shown are raw detection IDs only.
    html='''<!doctype html><meta charset="utf-8"><title>关联核验</title>
<style>body{font:16px sans-serif;margin:24px}canvas{border:1px solid #aaa}input{width:700px}</style>
<h1>原始检测核验</h1><p>拖动帧滑块检查连续性；图中编号仅为检测 ID，不是目标身份。请在 review.csv 填写 object_id、kind、reviewer，并将确认行标记 VERIFIED。不确定的行保持 UNREVIEWED。</p>
<input id="frame" type="range"><span id="number"></span><br><canvas id="plot" width="1000" height="810"></canvas>
<script>const rows=DATA,frames=[...new Set(rows.map(r=>r.frame))],slider=document.getElementById('frame'),ctx=document.getElementById('plot').getContext('2d');slider.max=frames.length-1;slider.value=0;
function draw(){let f=frames[+slider.value];document.getElementById('number').textContent='帧 '+f;ctx.clearRect(0,0,1000,810);for(let r of rows){if(r.frame>f||r.frame<f-4)continue;ctx.fillStyle=r.frame===f?'#b21f2d':'#c7c7c7';ctx.beginPath();ctx.arc(r.x_px/1288*1000,r.y_px/1040*810,4,0,7);ctx.fill();if(r.frame===f)ctx.fillText(r.detection_id,r.x_px/1288*1000+6,r.y_px/1040*810);}}slider.oninput=draw;draw();</script>'''
    (out/'review.html').write_text(html.replace('DATA',json.dumps(rows,ensure_ascii=False).replace('<','\\u003c')),encoding='utf-8')
    print(json.dumps(dict(review_points=len(rows),large_time_gaps=len(gaps))))


def prepare_experiments(public,config_path,out):
    from space_target_detection.pipeline.default_profile import bind_profile
    packet=load_packet(public,max_rows=1000000)
    raw=resolve_configuration(json.loads(Path(config_path).read_text(encoding='utf-8')))
    if 'bootstrap' in raw:raw,_=bind_profile(packet,raw,Path(config_path).absolute())
    raw['reporting'].update(track_frame_png_mode='off',write_frame_json=False)
    out.mkdir(parents=True,exist_ok=False)
    for name,mode,k in (('control','retain',None),('miss','moment_matched',None),('search','retain',64),('combined','moment_matched',64)):
        config=json.loads(json.dumps(raw));config['tracking']['miss_update_mode']=mode
        config['inference']['association_top_k']=k
        if mode=='moment_matched':config['lifecycle']['max_dormant_tracks_per_group']=32
        (out/(name+'.json')).write_text(json.dumps(config,indent=2),encoding='utf-8')


def synthetic(configs,out,seed=1729):
    out.mkdir(parents=True,exist_ok=False);all_results={}
    for scene in ('crossing','in_view_dropout','maneuver','exit_reentry','long_time_gap'):
        rng=np.random.default_rng(seed);frames=[];detections=[];truth=[]
        camera=Camera(1288,1040,3854.1666667,3854.1666667,644.,520.)
        for i in range(40):
            timestamp=2.*i+(30. if scene=='long_time_gap' and i>=20 else 0.)
            frames.append(FrameInfo(i,timestamp,True,True,(0.,0.,0.,1.)))
            a=np.array((300.+12*i,400.));b=np.array((800.-12*i,420.))
            if scene=='maneuver' and i>=18:a+=np.array((-8*(i-18),6*(i-18)))
            if scene=='exit_reentry':a=np.array((120.-18*i if i<=14 else -132.+18*(i-14),350.))
            items=[('A',a),('B',b)];rng.shuffle(items)
            for j,(identity,pixel) in enumerate(items):
                if scene=='in_view_dropout' and identity=='A' and 12<=i<=16:continue
                pixel=pixel+rng.normal(0,np.sqrt(2),2)
                if not camera.contains(pixel):continue
                d=Detection(i,str(j),*pixel,((2.,0.),(0.,2.)),'simulation_known');detections.append(d)
                truth.append(dict(frame=i,detection_id=str(j),object_id=identity,kind='target',review_status='VERIFIED'))
        packet=ObservationPacket(camera,'SYNTHETIC_SECONDS',tuple(frames),tuple(detections),origin='synthetic_validation')
        by_frame=defaultdict(list)
        for d in detections:by_frame[d.frame].append(d)
        scene_results={}
        for name,path in configs.items():
            raw=resolve_configuration(json.loads(Path(path).read_text(encoding='utf-8')))
            raw['stars']=[];raw['initial_timestamp_s']=0.
            tracker=FactorizedJointTracker(build_model(packet,raw),raw['undetected_population'],raw['geometry'])
            pred={};failure=None
            for frame in frames:
                try:result=tracker.step(frame,tuple(by_frame[frame.frame]),detail='compact')
                except InferenceError as e:
                    failure=dict(frame=frame.frame,code=e.code,previous=tracker.frame);break
                pred.update(assignments(result['observation_provenance']))
            completed_truth=[r for r in truth if tracker.frame is not None and r['frame']<=tracker.frame]
            scene_results[name]=dict(**evaluate(completed_truth,pred),failure=failure,frames_committed=tracker.version)
            (out/(scene+'_'+name+'_predictions.json')).write_text(json.dumps([dict(frame=k[0],detection_id=k[1],label=v) for k,v in pred.items()],indent=2),encoding='utf-8')
        (out/(scene+'_truth.json')).write_text(json.dumps(truth,indent=2),encoding='utf-8')
        all_results[scene]=scene_results
        (out/'summary.json').write_text(json.dumps(all_results,indent=2),encoding='utf-8')
        print(scene,json.dumps(scene_results),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('public_input');a.add_argument('output')
    a=sub.add_parser('prepare-experiments');a.add_argument('public_input');a.add_argument('config');a.add_argument('output')
    a=sub.add_parser('evaluate');a.add_argument('review_csv');a.add_argument('run_folder');a.add_argument('output')
    a=sub.add_parser('synthetic');a.add_argument('config_folder');a.add_argument('output');a.add_argument('--seed',type=int,default=1729)
    args=p.parse_args()
    if args.command=='prepare':prepare(args.public_input,Path(args.output))
    elif args.command=='prepare-experiments':prepare_experiments(args.public_input,args.config,Path(args.output))
    elif args.command=='synthetic':synthetic({n:Path(args.config_folder)/(n+'.json') for n in ('control','miss','search','combined')},Path(args.output),args.seed)
    else:
        with Path(args.review_csv).open(encoding='utf-8-sig') as f:truth=list(csv.DictReader(f))
        if any(r['review_status']=='VERIFIED' and not r['reviewer'].strip() for r in truth):raise ValueError('verified rows require a reviewer')
        pred={}
        for line in (Path(args.run_folder)/'observation_provenance.jsonl').read_text().splitlines():pred.update(assignments(json.loads(line)['groups']))
        Path(args.output).write_text(json.dumps(evaluate(truth,pred),indent=2),encoding='utf-8')
