"""Reproducible sensitivity trial; historical associations are NOT truth labels."""
import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from space_target_detection.v2.data_io import load_packet
from space_target_detection.pipeline.configuration import resolve_configuration
from space_target_detection.v2.camera_model import CameraModel
from space_target_detection.v2.geometry import quaternion_matrix
from space_target_detection.v2.gnomonic import GnomonicPlane
from space_target_detection.v2.observation_geometry import observe_detection
from space_target_detection.v2.tracking.model import cv_matrices


def calibration_observations(folder, packet, raw):
    # Exclude the previously diagnosed reconnect/crossing portions.
    segments={1:(147,156),93:(426,460),134:(470,496),136:(471,499)}
    with (folder/'trajectory_points.csv').open(encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    frames={f.frame:f for f in packet.frames}
    detections=defaultdict(list)
    for d in packet.detections:detections[d.frame].append(d)
    camera=CameraModel(packet.camera,**raw['calibration'])
    data=[]
    for tid,(lo,hi) in segments.items():
        selected=[r for r in rows if int(r['track_id'])==tid and lo<=int(r['frame'])<=hi and r['source'] in ('measured','recovered')]
        if len(selected)<8:continue
        plane=GnomonicPlane('fit',tuple(quaternion_matrix(frames[int(selected[0]['frame'])].quaternion_xyzw)[:,2]),np.radians(80))
        observations=[]
        for row in selected:
            f=frames[int(row['frame'])]
            d=min(detections[f.frame],key=lambda d:(d.x_px-float(row['x']))**2+(d.y_px-float(row['y']))**2)
            if abs(d.x_px-float(row['x']))+abs(d.y_px-float(row['y']))>1e-6:raise ValueError('trajectory/raw mismatch')
            o=observe_detection(d,f,camera,plane)
            J=np.asarray(o.jacobian_tangent_pixel)
            observations.append((f.timestamp_s,np.asarray(o.tangent_position),J@np.asarray(raw['tracking']['measurement_covariance_px2'])@J.T))
        data.append((tid,observations))
    return data


def calibrate(folder, packet, raw):
    data=calibration_observations(folder,packet,raw)
    H=np.column_stack((np.eye(2),np.zeros((2,2))))
    def score(q):
        scores=[[],[]];nis=[[],[]]
        for tid,obs in data:
            t0,z0,R0=obs[0];t1,z1,R1=obs[1];dt=t1-t0
            m=np.r_[z1,(z1-z0)/dt]
            P=np.block([[R1,R1/dt],[R1/dt,(R0+R1)/dt**2]])
            previous=t1
            for i,(t,z,R) in enumerate(obs[2:],2):
                F,Q=cv_matrices(t-previous,q);m=F@m;P=F@P@F.T+Q
                v=z-H@m;S=H@P@H.T+R
                n=float(v@np.linalg.solve(S,v));part=int(i>=max(4,int(len(obs)*.65)))
                scores[part].append(float(np.linalg.slogdet(S)[1]+n));nis[part].append(n)
                K=np.linalg.solve(S,H@P).T;m=m+K@v
                A=np.eye(4)-K@H;P=A@P@A.T+K@R@K.T;previous=t
        return dict(train_score=sum(scores[0]),validation_score=sum(scores[1]),
                    train_count=len(scores[0]),validation_count=len(scores[1]),
                    train_mean_nis=float(np.mean(nis[0])),validation_mean_nis=float(np.mean(nis[1])))
    grid=[dict(q_a=float(q),**score(q)) for q in np.logspace(-11,-4,57)]
    chosen=min(grid,key=lambda x:x['train_score'])
    return dict(scope='EXPLORATORY: historical continuity hypotheses, no independent identity truth; R fixed, not jointly calibrated',
                segments=[dict(track_id=tid,count=len(obs)) for tid,obs in data],
                selected=chosen,baseline=dict(q_a=raw['tracking']['q_a'],**score(raw['tracking']['q_a'])),grid=grid)


def main():
    p=argparse.ArgumentParser();p.add_argument('historical_run');p.add_argument('output');args=p.parse_args()
    folder=Path(args.historical_run);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    packet=load_packet(folder/'public_input',max_rows=1000000)
    raw=json.loads((folder/'effective_prior.json').read_text(encoding='utf-8'))
    raw.pop('classification',None);raw=resolve_configuration(raw)
    raw['catalog_path']=str(ROOT/'space_target_detection/NavStar_8mag.dat')
    fit=calibrate(folder,packet,raw)
    (out/'noise_sensitivity.json').write_text(json.dumps(fit,indent=2),encoding='utf-8')
    raw['reporting'].update(track_frame_png_mode='off',write_frame_json=False)
    raw.pop('frames',None)
    for name,q,budget in [('baseline',raw['tracking']['q_a'],False),('noise_only',fit['selected']['q_a'],False),('trial',fit['selected']['q_a'],True)]:
        config=json.loads(json.dumps(raw));config['tracking']['q_a']=q
        if name!='baseline':config['tracking']['parameter_provenance']=fit['scope']+'; tools/association_trial.py; held-out tail validation'
        if budget:config['inference'].update(top_k=16,max_candidates=256,max_local_hypotheses=512)
        (out/(name+'.json')).write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in fit.items() if k!='grid'},indent=2))


if __name__=='__main__':main()
