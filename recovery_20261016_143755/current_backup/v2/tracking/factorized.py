"""Online product of local joint histories with Poisson first detections.

Each competition group keeps conditional joint histories.
When observations couple groups, their priors are multiplied before updating.
Finite gating, bounded products/association/birth branches and Gaussian PPP
compression are explicit approximations. No shared random parameter is split:
this production implementation accepts fixed, independent sensor parameters.
This is the sole production tracking recursion in the runtime package.
"""
from dataclasses import dataclass,replace
import hashlib
import json
import math
import time
from ..numerical_policy import POSTERIOR_NORMALIZATION_ATOL
from ..geometry import quaternion_matrix
from ..gnomonic import GnomonicPlane
from ..observation_geometry import observe_detection
from ..exact.contracts import FactorTable,InferenceError
from ..exact.sources import VisibleStarProjector
from ..exact.likelihoods import build_known_star_factors
from ..exact.posterior import infer_factors
from ..efficient.config import InferenceBudgetController
from ..efficient.solver import GraphInferencePolicy
from ..efficient.graph import ProbabilisticGate
from .model import Track,JointHistory,predict_track
from .background import Background,predict_background,project_background
from .background_lifecycle import BackgroundViewIndex,retire_backgrounds
from .marginals import project_track_marginals
from .lifecycle import TrackLifecycleManager,canonical_id
from .candidates import retire_candidates
from .online_factors import affine_factors_many
from .undetected import prepare,observe
from .classification import predict_object,object_factors

MODEL_VERSION='v2.2-factorized-class-mixture-2'
APPROXIMATION='FINITE_UNION_GATES; BOUNDED_LOCAL_JOINT_HISTORIES; GAUSSIAN_PPP_MERGE; AFFINE_SENSOR; MISS_STATE_RETENTION; STATIC_CV_CLASS_MARGINALIZATION; CONFIGURED_BACKGROUND_VIEW_DORMANCY_AND_RETIREMENT'


@dataclass(frozen=True)
class Group:
    histories: tuple


def identity(parts):
    return hashlib.sha256(json.dumps(parts,separators=(',',':')).encode()).hexdigest()


def bounded_product(choices,limit):
    """Rank joint alternatives, tracking discarded mass; no marginal feedback."""
    beam=[(1.,())];discarded=0.
    for options in choices:
        pool=[(w*p,items+(item,)) for w,items in beam for p,item in options if w*p>0]
        pool.sort(key=lambda value:-value[0])  # stable input order breaks ties
        discarded+=math.fsum(w for w,_ in pool[limit:])
        beam=pool[:limit]
    return beam,discarded


class FactorizedJointTracker:
    def __init__(self,reference,options,geometry=None):
        if reference.background_config is None or reference.lifecycle_config is None or reference.inference_config is None:
            raise ValueError('online backend requires explicit background, lifecycle and inference configuration')
        required=('target_initial_expected_count','unknown_star_initial_expected_count',
            'target_birth_rate_per_second','unknown_star_birth_rate_per_second','velocity_sigma_rad_s')
        for name in required:
            value=options[name]
            if isinstance(value,bool) or not math.isfinite(value) or value<0 or name=='velocity_sigma_rad_s' and value==0:
                raise ValueError('invalid undetected population parameter: '+name)
        if not options.get('parameter_provenance'):raise ValueError('population parameter provenance required')
        self.reference=reference;self.options=dict(options)
        if reference.classification.enabled:
            if reference.config.survival_probability<=0 or reference.background_config.persistence_probability<=0:
                raise ValueError('class mixture requires positive survival probabilities')
            if not 0<reference.config.p_detection<1 or not 0<reference.background_config.p_detection<1:
                raise ValueError('class mixture requires class detection probabilities strictly between zero and one')
        if geometry is None:
            raise ValueError('online backend requires explicit geometry configuration')
        self.max_tangent_angle_rad=math.radians(geometry['max_tangent_angle_deg'])
        self.groups=();self.populations=();self.management=();self.retired=()
        self.frame=None;self.timestamp_s=reference.initial_timestamp_s;self.version=0
        self._star_projector=VisibleStarProjector()
        self._background_view_index=BackgroundViewIndex()
        self.dormant_background_ids=()
        self.background_retired_count=0

    def step(self,frame,detections,*,detail='full'):
        start=time.perf_counter();r=self.reference
        if detail not in ('compact','full'):raise ValueError('reporting detail must be compact or full')
        config=r.config;policy_config=r.inference_config;life=r.lifecycle_config
        dt=frame.timestamp_s-self.timestamp_s
        if dt<0 or self.frame is not None and (frame.frame<=self.frame or dt==0):
            raise InferenceError('NONMONOTONIC_TIME','strict frame/time ordering; no evidence replay')
        if dt>config.max_dt_s:raise InferenceError('PREDICTION_GAP_EXCEEDED','physical-time prediction budget')
        detections=tuple(sorted(detections,key=lambda d:d.key))
        if any(d.frame!=frame.frame for d in detections) or len({d.key for d in detections})!=len(detections):
            raise ValueError('one frame with unique detection keys required')
        controller=InferenceBudgetController(policy_config)
        accepted=tuple(d for d in detections if frame.inertial_observation_available and d.valid and r.clutter.window.contains((d.x_px,d.y_px)))
        populations=prepare(self.populations,frame,r.camera,config,r.background_config,self.options,dt,self.max_tangent_angle_rad,
            rebase=r.classification.enabled)
        predicted=[]
        prediction_cache={}
        # Keep only the recent confirmation window plus one older real support.
        # Full observations/ownership are stored incrementally by the runner.
        def predict_once(kind,obj):
            key=(kind,obj)
            if key not in prediction_cache:
                value=(predict_object(obj,dt,config,r.background_config) if obj.static_state is not None else
                    predict_track(obj,dt,config)) if kind=='T' else predict_background(obj,dt,r.background_config)
                keys=value.support_keys
                cutoff=frame.frame-life.confirmation_window_frames+1
                recent=tuple(k for k in keys if k[0]>=cutoff)
                older=tuple(k for k in keys if k[0]<cutoff)
                support=older[:1]+recent
                prediction_cache[key]=replace(value,support_keys=support) if support!=keys else value
            return prediction_cache[key]
        rotation=quaternion_matrix(frame.quaternion_xyzw) if frame.inertial_observation_available else None
        active_groups=(self._background_view_index.active_groups(self.groups,rotation,r.camera,r.clutter.window,r.background_config)
                       if rotation is not None else set(range(len(self.groups))))
        dormant=[]
        for group_index,group in enumerate(self.groups):
            if group_index not in active_groups:
                dormant.append(Group(tuple(replace(h,backgrounds=tuple(predict_background(b,dt,r.background_config) for b in h.backgrounds),
                    target_assignments=(),star_assignments=(),background_assignments=(),clutter_keys=()) for h in group.histories)))
                continue
            hs=[]
            for h in group.histories:
                hs.append(replace(h,tracks=tuple(predict_once('T',t) for t in h.tracks),
                    backgrounds=tuple(predict_once('S_U',b) for b in h.backgrounds)))
            predicted.append(tuple(hs))
        if not frame.inertial_observation_available:
            return self._commit_prediction(frame,detections,predicted,populations,start,controller,detail)
        remaining,intensities,birth_states,detected_population=observe(populations,frame,r.camera,r.clutter,accepted,config,r.background_config,rotation=rotation)
        factors={};states={};geometry_info={};state_factor_cache={};requests={}
        factor_objects=[];seen=set()
        for histories in predicted:
            for h in histories:
                for kind,objects in (('T',h.tracks),('S_U',h.backgrounds)):
                    for obj in objects:
                        if (kind,obj) in seen:continue
                        seen.add((kind,obj));factor_objects.append((kind,obj))
                        if kind=='T' and obj.static_state is not None:
                            continue  # Evaluate both class models in one source below.
                        source=kind+':'+(obj.label if kind=='T' else obj.background_id)
                        c=config if kind=='T' else r.background_config
                        key=(source,obj.state,obj.plane)
                        if key not in requests:
                            requests[key]=(source,kind,obj.state,obj.existence,obj.plane,c.p_detection,c.measurement_covariance_px2)
        values=affine_factors_many(tuple(requests.values()),frame,r.camera,r.clutter.window,accepted,
            policy_config.gate_probability,rotation=rotation,check=controller.check)
        state_factor_cache=dict(zip(requests,values))
        class_factor_cache={}
        for kind,obj in factor_objects:
            source=kind+':'+(obj.label if kind=='T' else obj.background_id)
            if kind=='T' and obj.static_state is not None:
                f,s,g=object_factors(obj,frame,r.camera,r.clutter.window,accepted,config,r.background_config,
                    policy_config.gate_probability,rotation,class_factor_cache)
            else:f,s,g=state_factor_cache[(source,obj.state,obj.plane)]
            if f.existence_prior!=obj.existence:f=replace(f,existence_prior=obj.existence)
            factors[(kind,obj)]=f;states[(kind,obj)]=s;geometry_info[(kind,obj)]=g
        star_sources=self._star_projector.sources(r.stars,r.catalog,frame,r.camera,r.clutter.window)
        star_factors={s.source_id:s for s in build_known_star_factors(star_sources,accepted)}
        # Union graph over every conditional state. Existing dependencies in
        # one group are never split just because current edges disconnect.
        parent={}
        def root(v):
            parent.setdefault(v,v)
            if parent[v]!=v:parent[v]=root(parent[v])
            return parent[v]
        def join(a,b):parent[root(b)]=root(a)
        for j in range(len(accepted)):root(('D',j))
        for i,histories in enumerate(predicted):
            root(('G',i))
            for h in histories:
                for kind,objects in (('T',h.tracks),('S_U',h.backgrounds)):
                    for obj in objects:
                        for j,value in enumerate(factors[(kind,obj)].log_eta):
                            if math.isfinite(value):join(('G',i),('D',j))
        for source,factor in star_factors.items():
            if not any(math.isfinite(value) for value in factor.log_eta):
                continue  # Its miss evidence is still included below.
            distances=None
            if accepted and factor.pixel_model and policy_config.gate_probability is not None:
                distances=ProbabilisticGate.distances([(d.x_px-factor.pixel_model.mean_px[0],d.y_px-factor.pixel_model.mean_px[1]) for d in accepted],factor.pixel_model.noise_px2)
            for j,value in enumerate(factor.log_eta):
                if math.isfinite(value) and (distances is None or distances[j]<=ProbabilisticGate.threshold(policy_config.gate_probability)):
                    join(('S',source),('D',j))
        components={}
        for node in parent:components.setdefault(root(node),[]).append(node)
        # Stars with no retained edge still contribute a nondetection factor
        # to the set density, although that constant cancels in posteriors.
        isolated_star_log_miss=math.fsum(
            f.log_miss_given_exists if f.log_miss_given_exists is not None else
            (math.log1p(-f.pd_bar) if f.pd_bar<1 else -math.inf)
            for source,f in star_factors.items() if ('S',source) not in parent)
        controller.check(nodes=len(parent))
        updated=[];audits=[];prior_loss=0.;birth_loss=0.;log_evidence=[]
        search_summary=dict(groups=0,evaluated=0,retained=0,
            omitted_mass_status='LOCAL_TRUNCATION_AND_FINITE_GATES; FULL_SEQUENCE_ERROR_UNKNOWN')
        for component in sorted(components.values(),key=repr):
            controller.check()
            indices=tuple(sorted(v for k,v in component if k=='D'))
            ds=tuple(accepted[j] for j in indices)
            groups=tuple(sorted(v for k,v in component if k=='G'))
            known=tuple(sorted(v for k,v in component if k=='S'))
            products,loss=bounded_product([tuple((h.weight,h) for h in predicted[i]) for i in groups],policy_config.max_candidates)
            prior_loss+=loss
            den=math.fsum(w for w,_ in products)
            priors=[];tables=[];conditionals={};gate_geometry={};source_diag=[]
            for weight,parts in products:
                hid='G:'+identity((frame.frame,[h.hypothesis_id for h in parts],indices))
                prior=JointHistory(hid,weight/den,tuple(t for h in parts for t in h.tracks),
                    backgrounds=tuple(b for h in parts for b in h.backgrounds))
                priors.append(prior);fs=[];gate_geometry[hid]={}
                for kind,objects in (('T',prior.tracks),('S_U',prior.backgrounds)):
                    for obj in objects:
                        f=factors[(kind,obj)];fs.append(replace(f,log_eta=tuple(f.log_eta[j] for j in indices)))
                        conditionals[(hid,f.source_id)]=states[(kind,obj)]
                        if geometry_info[(kind,obj)] is not None:
                            gate_geometry[hid][f.source_id]=geometry_info[(kind,obj)]
                        pre_pd=(obj.motion_probability*config.p_detection+(1-obj.motion_probability)*r.background_config.p_detection
                            if kind=='T' and obj.static_state is not None else
                            config.p_detection if kind=='T' else r.background_config.p_detection)
                        source_diag.append(dict(parent_id=hid,source_id=f.source_id,pd_bar=f.pd_bar,
                            visibility_mass_cubature=f.pd_bar/(pre_pd or 1)))
                for s in known:
                    f=star_factors[s];fs.append(replace(f,log_eta=tuple(f.log_eta[j] for j in indices)))
                    gate_geometry[hid][s]=(f.pixel_model.mean_px,f.pixel_model.noise_px2)
                kappas=tuple(r.clutter.expected_count/r.clutter.window.area+sum(intensities[d.key].values()) for d in ds)
                tables.append(FactorTable(hid,prior.weight,tuple(d.key for d in ds),tuple(fs),
                    tuple(math.log(k) if k else -math.inf for k in kappas)))
            policy=GraphInferencePolicy(policy_config,controller,ds,gate_geometry,detail=detail)
            result=infer_factors(tuple(tables),policy)
            if detail=='full':audits.append(policy.audit)
            search_summary['groups']+=1
            search_summary['evaluated']+=policy.audit['evaluated_candidates']
            search_summary['retained']+=policy.audit['retained_candidates']
            log_evidence.append(result.log_predictive_evidence)
            lookup={h.hypothesis_id:h for h in priors};conditional={s.reference:s.existence for s in result.conditional_states}
            output=[]
            for h in result.hypotheses:
                if not math.isfinite(h.log_unnormalized_weight):continue
                before=lookup[h.parent_hypothesis_id];assign=dict(h.target_assignments+h.unknown_star_assignments);refs=dict(h.conditional_state_references)
                tracks=[];backgrounds=[]
                for kind,objects in (('T',before.tracks),('S_U',before.backgrounds)):
                    for obj in objects:
                        source=kind+':'+(obj.label if kind=='T' else obj.background_id)
                        key=assign[source];existence=conditional[refs[source]]
                        if existence==0:continue
                        conditional_state=conditionals[(before.hypothesis_id,source)][key]
                        changes=dict(existence=existence,
                            support_keys=obj.support_keys+((key,) if key else ()))
                        if kind=='T' and obj.static_state is not None:
                            changes.update(zip(('state','static_state','motion_log_odds'),conditional_state))
                        else:changes['state']=conditional_state
                        if kind=='S_U':
                            pd_bar=factors[(kind,obj)].pd_bar
                            changes['visible_miss_count']=(0 if key is not None else
                                obj.visible_miss_count+int(pd_bar>=r.background_config.retirement_min_pd))
                        (tracks if kind=='T' else backgrounds).append(replace(obj,**changes))
                alternatives=[]
                for key in h.clutter_detection_keys:
                    weights=[(r.clutter.expected_count/r.clutter.window.area,('C',key))]
                    if r.classification.enabled:
                        mass=sum(intensities[key].values())
                        if mass>0:weights.append((mass,('O',key)))
                    else:weights.extend((v,(kind,key)) for kind,v in intensities[key].items() if v>0)
                    total=math.fsum(w for w,_ in weights)
                    alternatives.append(tuple((w/total,event) for w,event in weights if w>0))
                possibilities,loss=bounded_product(alternatives,policy_config.max_candidates)
                birth_loss+=h.posterior_weight*loss
                for probability,events in possibilities:
                    ts=list(tracks);bs=list(backgrounds);ta=list(h.target_assignments);ba=list(h.unknown_star_assignments);cl=[]
                    for kind,key in events:
                        if kind=='C':cl.append(key);continue
                        if kind=='O':
                            target_mass=intensities[key].get('T',0.)
                            star_mass=intensities[key].get('S_U',0.)
                            if target_mass<=0:
                                fixed,plane=birth_states[(key,'S_U')]
                                label='U:'+str(key[0])+':'+str(len(key[1]))+':'+key[1]
                                bs.append(Background(label,1.,fixed,plane,(key,)));ba.append(('S_U:'+label,key));continue
                            if star_mass<=0:
                                state,plane=birth_states[(key,'T')]
                                label='N:'+str(key[0])+':'+str(len(key[1]))+':'+key[1]
                                ts.append(Track(label,1.,state,plane,(key,)));ta.append(('T:'+label,key));continue
                            state,plane=birth_states[(key,'T')]
                            fixed,fixed_plane=birth_states[(key,'S_U')]
                            if fixed_plane!=plane:raise ValueError('birth class tangent references disagree')
                            label='N:'+str(key[0])+':'+str(len(key[1]))+':'+key[1]
                            ts.append(Track(label,1.,state,plane,(key,),fixed,
                                math.log(target_mass)-math.log(star_mass)))
                            ta.append(('T:'+label,key));continue
                        state,plane=birth_states[(key,kind)]
                        label=('N:' if kind=='T' else 'U:')+str(key[0])+':'+str(len(key[1]))+':'+key[1]
                        if kind=='T':
                            ts.append(Track(label,1.,state,plane,(key,)))
                            ta.append(('T:'+label,key))
                        else:
                            bs.append(Background(label,1.,state,plane,(key,)));ba.append(('S_U:'+label,key))
                    identifier='J:'+identity((h.hypothesis_id,events))
                    output.append(JointHistory(identifier,h.posterior_weight*probability,tuple(ts),
                        tuple(ta),h.star_assignments,tuple(cl),backgrounds=tuple(bs),background_assignments=tuple(ba)))
            output.sort(key=lambda h:(-h.weight,h.hypothesis_id))
            birth_loss+=math.fsum(h.weight for h in output[policy_config.top_k:])
            output=output[:policy_config.top_k];den=math.fsum(h.weight for h in output)
            output=tuple(replace(h,weight=h.weight/den) for h in output)
            updated.append((output,tuple(priors),tuple(source_diag)))
        return self._finish(frame,detections,accepted,updated,remaining,start,controller,
            dict(**(dict(group_audits=audits) if detail=='full' else dict(group_search_summary=search_summary)),
                prior_product_discarded_mass_sum=prior_loss,birth_branch_discarded_mass_sum=birth_loss,
                log_predictive_set_density=sum(log_evidence)+isolated_star_log_miss-r.clutter.expected_count-detected_population,
                first_detection_intensity=tuple((key,values) for key,values in intensities.items())),
            projected_star_count=len(star_factors),detail=detail,dormant=tuple(dormant))

    def _commit_prediction(self,frame,detections,predicted,populations,start,controller,detail):
        updated=[(tuple(replace(h,target_assignments=(),star_assignments=(),background_assignments=(),clutter_keys=()) for h in hs),hs,()) for hs in predicted]
        return self._finish(frame,detections,(),updated,populations,start,controller,dict(no_observation=True),detail=detail)

    def _finish(self,frame,detections,accepted,updated,populations,start,controller,audit,projected_star_count=0,*,detail='full',dormant=()):
        r=self.reference;life=r.lifecycle_config
        groups=[];marginals=[];backgrounds=[];management=[];retired=[];events=[];provenance=[]
        old_records={v.label:v for v in self.management}
        background_events=[]
        for hs,priors,diag in updated:
            rows=project_track_marginals(hs,frame,life,detail=detail)
            records,ev=TrackLifecycleManager.update(tuple(old_records[v['Track_Label']] for v in rows if v['Track_Label'] in old_records),
                rows,diag,priors,frame,life,self.version+1,self.frame)
            for row,record in zip(rows,records):
                row.update(Track_Lifecycle=record.lifecycle,Track_Maturity=record.maturity,Canonical_Track_ID=canonical_id(record.label),
                    State_Version=self.version+1,Management_Reason=record.reason)
            feedback,records,removed=retire_candidates(hs,records,frame,life,self.version+1)
            feedback,bg_removed=retire_backgrounds(feedback,frame,r.background_config)
            background_events.extend(bg_removed)
            labels={v.label for v in removed}
            for row in rows:row['Retired_After_Observation_Update']=row['Track_Label'] in labels
            if any(h.tracks or h.backgrounds for h in feedback):groups.append(Group(feedback))
            marginals.extend(rows);backgrounds.extend(project_background(hs,frame,diag,detail=detail));management.extend(records);retired.extend(removed);events.extend(ev)
            provenance.append(tuple(dict(hypothesis_id=h.hypothesis_id,weight=h.weight,
                targets=h.target_assignments,unknown_stars=h.background_assignments,known_stars=h.star_assignments,clutter=h.clutter_keys,
                object_classes=tuple(dict(label=t.label,motion_probability=t.motion_probability,
                    motion_log_odds=t.motion_log_odds) for t in h.tracks if t.static_state is not None)) for h in hs))
        groups.extend(dormant)
        dormant_ids=tuple(sorted({b.background_id for g in dormant for h in g.histories for b in h.backgrounds}))
        active_ids={b.background_id for hs,_,_ in updated for h in hs for b in h.backgrounds}
        current={v.label for v in management}
        retired_labels={v.label for v in retired}
        dropped=[v for v in self.management if v.label not in current and v.label not in retired_labels]
        for record in dropped:
            events.append(dict(frame=frame.frame,timestamp_s=frame.timestamp_s,label=record.label,kind='POSTERIOR_TRUNCATION',
                before=record.lifecycle,after='NO_RETAINED_SUPPORT',reason='LOCAL_JOINT_BRANCH_TRUNCATION; NOT_PHYSICAL_DEATH'))
        points=[]
        probability_fields=('P_Known_Star','P_Unknown_Star','P_Existing_Target','P_Birth_Target','P_Clutter')
        point_sums={d.key:{name:[] for name in probability_fields} for d in accepted}
        point_associations={d.key:{} for d in accepted}
        # Accumulate each retained assignment once, instead of scanning every
        # history again for each detection. Preserve each sum's input order.
        birth_prefix='T:N:'+str(frame.frame)+':'
        for hs,_,_ in updated:
            for h in hs:
                object_classes={'T:'+t.label:t.motion_probability for t in h.tracks}
                for kind,assignments in (('T',h.target_assignments),('S_K',h.star_assignments),('S_U',h.background_assignments)):
                    for source,key in assignments:
                        if key not in point_sums:continue
                        name='P_Known_Star' if kind=='S_K' else 'P_Unknown_Star' if kind=='S_U' else 'P_Birth_Target' if source.startswith(birth_prefix) else 'P_Existing_Target'
                        c=object_classes.get(source,1.) if kind=='T' else 1.
                        point_sums[key][name].append(h.weight*c)
                        if kind=='T':point_sums[key]['P_Unknown_Star'].append(h.weight*(1-c))
                        beta=point_associations[key]
                        beta[source]=beta.get(source,0.)+h.weight
                for key in h.clutter_keys:
                    if key in point_sums:point_sums[key]['P_Clutter'].append(h.weight)
        for d in detections:
            valid=d.key in point_sums
            values=dict(P_Known_Star=None,P_Unknown_Star=None,P_Existing_Target=None,P_Birth_Target=None,P_Clutter=None)
            beta={}
            if valid:
                sums=point_sums[d.key];beta=point_associations[d.key]
                values={name:math.fsum(v) for name,v in sums.items()}
                if abs(sum(values.values())-1)>POSTERIOR_NORMALIZATION_ATOL:raise InferenceError('POSTERIOR_NORMALIZATION','local five-source marginal')
            p=dict(Frame=frame.frame,Timestamp=frame.timestamp_s,Detection_ID=d.detection_id,Geometry_Valid=valid,
                Model_Status=MODEL_VERSION,Inference_Mode='FACTORIZED_JOINT_PPP',Approximation_Status=APPROXIMATION,
                Association_Marginals=tuple(sorted(beta.items())),**values)
            p['Association_Identity_Semantics']='T_PREFIX_IS_OBJECT_ID; SEE_PHYSICAL_CLASS_MARGINALS'
            points.append(p)
        geometry=()
        if frame.inertial_observation_available:
            rotation=quaternion_matrix(frame.quaternion_xyzw)
            plane=GnomonicPlane('output-'+str(frame.frame),tuple(rotation[:,2]),max_angle_rad=self.max_tangent_angle_rad)
            geometry=tuple(observe_detection(d,frame,r.camera,plane) for d in accepted)
        controller.check(nodes=sum(max((len(h.tracks)+len(h.backgrounds) for h in g.histories),default=0) for g in groups))
        diagnostics=dict(Computation_Time_s=time.perf_counter()-start,Parent_Hypothesis_Count=sum(len(g.histories) for g in self.groups),
            Retained_Hypothesis_Count=sum(len(g.histories) for g in groups),Active_Bernoulli_Component_Count=len(marginals),
            Detection_Count=len(accepted),KnownStar_Count=len(r.stars),
            Projected_KnownStar_Count=projected_star_count,
            Joint_Group_Count=len(groups),
            Active_Background_Count=len(active_ids),Dormant_Background_Count=len(dormant_ids),
            Class_Mixture_Object_Count=sum(row['Class_Model']=='STATIC_CV_MIXTURE' for row in marginals),
            Fixed_Source_Object_Count=sum(row['Track_Maturity']=='FIXED_SOURCE' for row in marginals),
            Dormant_Background_Group_Count=len(dormant),
            Woken_Background_Count=len(set(self.dormant_background_ids)&active_ids) if frame.inertial_observation_available else 0,
            Retired_Background_Count=len(background_events),
            Total_Retired_Background_Count=self.background_retired_count+len(background_events),
            Largest_Joint_Group_Hypotheses=max((len(g.histories) for g in groups),default=0),
            Posterior_Normalization_Error=max((abs(sum(h.weight for h in g.histories)-1) for g in groups),default=0.),
            Retired_Candidate_Count=len(retired),Total_Retired_Candidate_Count=len(self.retired)+len(retired),
            Exact_Or_Approximate=APPROXIMATION,Search_States=controller.work,
            Undetected_Population_Counts={p.kind:p.expected_count for p in populations})
        if detail=='compact' and 'group_search_summary' not in audit:
            audit=dict(audit,group_search_summary=dict(groups=0,evaluated=0,retained=0,
                omitted_mass_status='LOCAL_TRUNCATION_AND_FINITE_GATES; FULL_SEQUENCE_ERROR_UNKNOWN'))
        result=dict(frame=frame.frame,timestamp_s=frame.timestamp_s,dt_s=frame.timestamp_s-self.timestamp_s,
            status='APPROXIMATE_FACTORIZED_JOINT_POSTERIOR',raw_detections=detections,points=tuple(points),geometry=geometry,
            track_marginals=tuple(marginals),background_marginals=tuple(backgrounds),lifecycle_events=tuple(events),
            candidate_retirement_events=tuple(retired),diagnostics=diagnostics,inference_mode='FACTORIZED_JOINT_PPP',
            background_retirement_events=tuple(background_events),
            background_population_status=dict(dormant_background_ids=dormant_ids,
                marginal_scope='OBSERVATION_UPDATED_GROUPS_ONLY; DORMANT_CONDITIONAL_STATES_RETAINED_IN_CHECKPOINT',
                dormancy_enabled=r.background_config.enable_view_dormancy,retirement_enabled=r.background_config.enable_retirement),
            inference_audit=dict(approximate=True,model_version=MODEL_VERSION,scope=APPROXIMATION,**audit),
            inference_performance=dict(peak_sampled_process_rss_bytes=controller.peak_sampled_rss),
            observation_provenance=tuple(provenance),state_version=self.version+1,
            identity_scope='PERSISTENT_LABELS_WITHIN_RETAINED_JOINT_BRANCHES; ARCHIVE_REIDENTIFICATION_NOT_IMPLEMENTED')
        # One atomic commit; every computation above uses frame-local state.
        self.groups=tuple(groups);self.populations=tuple(populations);self.management=tuple(management)
        self.retired+=tuple(retired);self.frame=frame.frame;self.timestamp_s=frame.timestamp_s;self.version+=1
        self.dormant_background_ids=dormant_ids
        self.background_retired_count+=len(background_events)
        return result
