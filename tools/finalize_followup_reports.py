"""Validate completed experiments and refresh only the miss-state display label."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from summarize_association_trial import summarize


def finalize(root):
    result={}
    for name in ('control_run','search_run','miss_capacity32_run','combined_capacity32_run'):
        folder=root/'verified'/name
        summary=json.loads((folder/'tracking_summary.json').read_text(encoding='utf-8'))
        if summary['status']!='COMPLETED' or summary['frames_committed']!=430:
            raise ValueError('incomplete experiment: '+name)
        for line in (folder/'observation_provenance.jsonl').read_text().splitlines():
            for group in json.loads(line)['groups']:
                if abs(sum(h['weight'] for h in group)-1)>1e-9:raise ValueError('unnormalized group')
                for h in group:
                    keys=[tuple(k) for kind in ('targets','known_stars','unknown_stars') for _,k in h[kind] if k is not None]
                    keys.extend(tuple(k) for k in h['clutter'])
                    if len(keys)!=len(set(keys)):raise ValueError('duplicate observation ownership')
        # These experiments started before the display-label correction.
        # Inference/state/provenance files remain untouched. Record hashes.
        path=folder/'trajectory_points.csv';before=hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
        changed=0
        if summary['configuration']['tracking']['miss_update_mode']=='moment_matched':
            valid={r['frame'] for r in summary['frames'] if r['valid_observation']}
            for row in rows:
                if int(row['frame']) in valid and row['posterior_update_kind']=='Predicted_Only':
                    row['posterior_update_kind']='Miss_Conditioned_Gaussian';changed+=1
        if changed:
            with path.open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            note=dict(scope='REPORTING_ONLY: conditional-miss state label; no numerical results changed',rows=changed,
                      before_sha256=before,after_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            (folder/'report_label_correction.json').write_text(json.dumps(note,indent=2),encoding='utf-8')
        result[name]=summarize(folder)
        result[name]['maximum_target_components']=max(r['Active_Bernoulli_Component_Count'] for r in summary['frames'])
    (root/'final_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='identity_cases'} for k,v in result.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();finalize(Path(a.root))
