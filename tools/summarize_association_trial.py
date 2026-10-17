"""Compare retained-posterior diagnostics, never label these as accuracy."""
import argparse
import json
from pathlib import Path
from collections import defaultdict


def summarize(folder):
    summary=json.loads((folder/'tracking_summary.json').read_text(encoding='utf-8'))
    audits=[json.loads(s) for s in (folder/'inference_audit.jsonl').read_text(encoding='utf-8').splitlines()]
    ambiguities=0;points=0;identity_cases={}
    for line in (folder/'observation_provenance.jsonl').read_text(encoding='utf-8').splitlines():
        row=json.loads(line);beta=defaultdict(lambda:defaultdict(float))
        for group in row['groups']:
            for h in group:
                for source,key in h['targets']:
                    if key is not None:beta[tuple(key)][source]+=h['weight']
        for key,values in beta.items():
            points+=1
            ambiguities+=sum(p>.1 for p in values.values())>=2
        if row['frame'] in (156,162,163,476,477,478,479,491,493,494,495,500):
            identity_cases[str(row['frame'])]={str(key):dict(values) for key,values in beta.items() if max(values.values())>.1}
    birth=[a.get('birth_branch_discarded_mass_sum',0.) for a in audits]
    association=[a.get('group_search_summary',{}).get('association_discarded_mass_sum',0.) for a in audits]
    return dict(status=summary['status'],frames=summary['frames_committed'],failed=summary['failed_frame'],
                target_identity_ambiguous_points=ambiguities,points_with_target_support=points,
                birth_discarded_sum_max=max(birth),birth_discarded_sum_mean=sum(birth)/len(birth),
                association_discarded_sum_max=max(association),association_discarded_sum_mean=sum(association)/len(association),
                performance=summary['performance_quantiles_s'],identity_cases=identity_cases,
                caveat='Conditional retained-candidate probabilities; no identity truth and no global error bound')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();root=Path(a.folder)
    result={f.name:summarize(f) for f in sorted(root.glob('*_run')) if (f/'tracking_summary.json').exists()}
    (root/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='identity_cases'} for k,v in result.items()},indent=2))
