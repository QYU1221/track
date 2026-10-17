"""Joint Q/R sensitivity and an independent-seed synthetic calibration check."""
import argparse
import json
from pathlib import Path
import numpy as np
from association_trial import calibration_observations,load_packet,resolve_configuration
from space_target_detection.v2.tracking.model import cv_matrices


def innovations(observations,q,r_scale):
    H=np.column_stack((np.eye(2),np.zeros((2,2))))
    t0,z0,R0=observations[0];t1,z1,R1=observations[1];dt=t1-t0
    R0=R0*r_scale;R1=R1*r_scale
    m=np.r_[z1,(z1-z0)/dt];P=np.block([[R1,R1/dt],[R1/dt,(R0+R1)/dt**2]])
    previous=t1;rows=[]
    for i,(t,z,base_R) in enumerate(observations[2:],2):
        F,Q=cv_matrices(t-previous,q);m=F@m;P=F@P@F.T+Q;R=base_R*r_scale
        v=z-H@m;S=H@P@H.T+R;n=float(v@np.linalg.solve(S,v))
        if i>=4:rows.append(dict(index=i,score=float(np.linalg.slogdet(S)[1]+n),nis=n,
                                 whitened=np.linalg.solve(np.linalg.cholesky(S),v).tolist()))
        K=np.linalg.solve(S,H@P).T;m+=K@v;A=np.eye(4)-K@H;P=A@P@A.T+K@R@K.T;previous=t
    return rows


def metrics(rows):
    if not rows:raise ValueError('insufficient validation observations')
    nis=[r['nis'] for r in rows]
    return dict(count=len(rows),score=sum(r['score'] for r in rows),mean_nis=float(np.mean(nis)),
                fraction_inside_95_percent=float(np.mean(np.asarray(nis)<=5.9914645471)))


def grid_search(data,qs,scales,split=True):
    grid=[]
    for q in qs:
        for scale in scales:
            train=[];validation=[]
            for _,obs in data:
                rows=innovations(obs,q,scale);cut=max(5,int(len(obs)*.65))
                train.extend(r for r in rows if not split or r['index']<cut)
                validation.extend(r for r in rows if not split or r['index']>=cut)
            grid.append(dict(q_a=float(q),r_scale=float(scale),train=metrics(train),validation=metrics(validation)))
    chosen=min(grid,key=lambda r:r['train']['score'])
    return dict(selected=chosen,at_grid_boundary=chosen['q_a'] in (min(qs),max(qs)) or chosen['r_scale'] in (min(scales),max(scales)),grid=grid)


def synthetic(seed):
    rng=np.random.default_rng(seed);q=3.162277660168379e-8;R=np.eye(2)*2/3854.1666667**2
    F,Q=cv_matrices(2.,q);x=np.zeros(4);rows=[]
    for i in range(300):
        x=F@x+rng.multivariate_normal(np.zeros(4),Q)
        rows.append((i*2.,x[:2]+rng.multivariate_normal(np.zeros(2),R),R))
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('historical_run');p.add_argument('output');a=p.parse_args()
    folder=Path(a.historical_run);raw=json.loads((folder/'effective_prior.json').read_text());raw.pop('classification',None);raw=resolve_configuration(raw)
    data=calibration_observations(folder,load_packet(folder/'public_input',max_rows=1000000),raw)
    result=grid_search(data,np.logspace(-10,-6,17),(.05,.1,.25,.5,1.,2.,4.))
    result['scope']='Historical continuity hypotheses only; no independent real truth; never auto-promote Q/R'
    result['per_segment_validation']={str(tid):metrics([r for r in innovations(obs,result['selected']['q_a'],result['selected']['r_scale']) if r['index']>=max(5,int(len(obs)*.65))]) for tid,obs in data}
    syn=grid_search([(0,synthetic(710))],np.logspace(-8,-7,9),(.25,.5,1.,2.,4.),split=False)
    selected=syn['selected'];syn.pop('grid')
    syn['independent_seed_validation']=metrics(innovations(synthetic(711),selected['q_a'],selected['r_scale']))
    syn['true_q_a']=3.162277660168379e-8;syn['true_r_scale']=1.
    result['synthetic_calibration']=syn
    Path(a.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='grid'},indent=2))
