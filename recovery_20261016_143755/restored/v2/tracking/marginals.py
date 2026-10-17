"""Read-only marginal views of retained online joint histories."""
import math
from .model import mixture_moments


def project_track_marginals(histories, frame, confirmation_config, *, detail='full'):
    rows=[]
    indexed=tuple((history,{t.label:t for t in history.tracks},dict(history.target_assignments)) for history in histories)
    for label in sorted({t.label for h in histories for t in h.tracks}):
        components=[];associations={};beta0=0.;support=0.;confirmation=0.;first_support=None;template=None
        for history,tracks,assignments in indexed:
            track=tracks.get(label)
            if track is None:
                beta0+=history.weight;continue
            template=track
            key=assignments.get('T:'+label)
            if key is None:beta0+=history.weight
            else:associations[key]=associations.get(key,0.)+history.weight
            if track.support_keys:
                support+=history.weight*track.existence
                if track.existence>0 and history.weight>0:
                    first=min(k[0] for k in track.support_keys)
                    first_support=first if first_support is None else min(first_support,first)
                start=frame.frame-confirmation_config.confirmation_window_frames+1
                if len({k[0] for k in track.support_keys if start<=k[0]<=frame.frame})>=confirmation_config.confirmation_min_frames:
                    confirmation+=history.weight*track.existence
            if track.existence>0:components.append((history.hypothesis_id,history.weight*track.existence,track.state))
        existence=math.fsum(w for _,w,_ in components)
        moments=mixture_moments(tuple(w for _,w,_ in components),tuple(s for _,_,s in components))
        rows.append({'Track_Label':label,'Frame':frame.frame,'Timestamp':frame.timestamp_s,
            'Existence_Probability':existence,'State_Mean':moments.mean if moments else None,
            'State_Covariance':moments.covariance if moments else None,
            'Tangent_Reference_ID':template.plane.reference_id,'Tangent_Plane':template.plane,
            'Detection_Association_Probabilities':tuple(sorted(associations.items())),
            'Miss_Association_Probability':beta0,
            'Probability_Exists_And_Ever_Measured':support,'First_Real_Support_Frame':first_support,
            'Confirmation_Support_Probability':min(1.,max(0.,confirmation)),
            'Confirmation_Window_Frames':confirmation_config.confirmation_window_frames,
            'Confirmation_Min_Frames':confirmation_config.confirmation_min_frames,
            'Posterior_Representation':'conditional_history_Gaussian_mixture','Motion_Model':'CV_GNOMONIC',
            'Association_Marginals':tuple(sorted(associations.items())),
            'Measured_Or_Predicted':'Measured_Update_Mixture' if associations else 'Predicted_Only',
            'Approximation_Status':'CONTINUOUS_APPROXIMATE; JOINT HISTORIES RETAINED'})
        if detail=='full':
            rows[-1]['Conditional_Hypothesis_Metadata']=tuple({'hypothesis_id':i,'existence_weight':w} for i,w,_ in components)
            rows[-1]['State_Mixture']=tuple({'hypothesis_id':i,'weight':w/existence,'state':s} for i,w,s in components) if existence>0 else ()
    return tuple(rows)
