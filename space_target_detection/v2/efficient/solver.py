"""Graph enumeration and bounded candidate beams for joint normalization."""
from itertools import product
import hashlib
import json
import math
from .graph import AssociationGraphBuilder,ConditionalAssociationComponentBuilder
from ..exact.contracts import Association
from ..exact.posterior import source_log_miss_factor,normalize_log_weights


def association(table,assignments):
    assignments=tuple(sorted(assignments))
    used={j for _,j in assignments if j is not None}
    representation=[table.parent_id,[(s,table.detection_keys[j] if j is not None else None) for s,j in assignments]]
    digest=hashlib.sha256(json.dumps(representation,separators=(',',':')).encode()).hexdigest()
    return Association('H:'+digest,assignments,
        tuple(sorted((j for j in range(len(table.detection_keys)) if j not in used),key=lambda j:table.detection_keys[j])))


def assignment_key(items):
    return tuple((s,-1 if j is None else j) for s,j in items)


class ExactComponentSolver:
    @staticmethod
    def solve(component,graph,controller,limit):
        allowed={s:tuple(sorted(e.detection_index for e in graph.edges if e.source_id==s)) for s in component.sources}
        # Complete assignments with zero, one or two hits already give a
        # rigorous lower bound. Avoid an exponential count just to establish
        # that the configured local enumeration limit has been exceeded.
        controller.consume()
        choices=[set(allowed[s]) for s in component.sources]
        lower=1+sum(map(len,choices))
        if lower>limit:return None,limit+1
        if len(component.sources)==1:
            source=component.sources[0]
            # The generic count and visit each consume n+2 work units.
            controller.consume(2*len(allowed[source])+4)
            return (((source,None),),)+tuple(((source,j),) for j in allowed[source]),lower
        for i,left in enumerate(choices):
            for right in choices[i+1:]:
                controller.consume()
                lower+=len(left)*len(right)-len(left & right)
                if lower>limit:return None,limit+1
        # Count with saturation before allocating full combinations.
        memo={}
        def count(i,used):
            controller.consume()
            key=(i,used)
            if key in memo:return memo[key]
            if i==len(component.sources):return 1
            value=count(i+1,used)
            for j in allowed[component.sources[i]]:
                if not (used>>j)&1:
                    value+=count(i+1,used|(1<<j))
                    if value>limit:break
            memo[key]=min(limit+1,value);return memo[key]
        total=count(0,0)
        if total>limit:return None,total
        out=[]
        def visit(i,used,items):
            controller.consume()
            if i==len(component.sources):out.append(items);return
            s=component.sources[i]
            visit(i+1,used,items+((s,None),))
            for j in allowed[s]:
                if not (used>>j)&1:visit(i+1,used|(1<<j),items+((s,j),))
        visit(0,0,())
        return tuple(out),total


class RankedGlobalHypothesisSolver:
    @staticmethod
    def beam_component(component,graph,score,controller,width):
        beam=[()]
        for s in component.sources:
            choices=(None,)+tuple(e.detection_index for e in graph.edges if e.source_id==s)
            pool=[]
            for items in beam:
                used={j for _,j in items if j is not None}
                for j in choices:
                    if j is None or j not in used:
                        controller.consume();pool.append(items+((s,j),))
            beam=sorted(pool,key=lambda a:(-score(a,component.detections),assignment_key(a)))[:width]
        return tuple(beam)


class GlobalHypothesisCombiner:
    @staticmethod
    def combine(local,score,controller,limit):
        total=math.prod(len(x) for x in local)
        if total<=limit:
            out=[]
            for parts in product(*local):
                controller.consume();out.append(tuple(sorted(v for part in parts for v in part)))
            return tuple(out),True,total
        beam=[()]
        for choices in local:
            pool=[]
            for a in beam:
                for b in choices:
                    controller.consume();pool.append(tuple(sorted(a+b)))
            beam=sorted(pool,key=lambda a:(-score(a),assignment_key(a)))[:limit]
        return tuple(beam),False,total


class HypothesisPruner:
    @staticmethod
    def select(logs,ids,limit):
        weights,evidence=normalize_log_weights(tuple(logs))
        keep=tuple(sorted(range(len(logs)),key=lambda i:(-logs[i],ids[i]))[:limit])
        kept=set(keep)
        omitted=math.fsum(weights[i] for i in range(len(logs)) if i not in kept)
        return keep,omitted,evidence


class GraphInferencePolicy:
    def __init__(self,config,controller,detections=(),geometry=None,*,detail='full'):
        self.config=config;self.controller=controller;self.detections=detections
        self.geometry=geometry or {};self.records=[];self.complete=True;self.audit={}
        self._gate_distances={}
        self.detail=detail
        self.parent_count=1

    def proposals(self,table):
        c=self.config
        graph=AssociationGraphBuilder.build(table,c,self.detections,self.geometry.get(table.parent_id),self._gate_distances,
            record_exclusions=self.detail=='full')
        components=ConditionalAssociationComponentBuilder.build(graph)
        self.controller.check(nodes=len(graph.source_ids)+len(graph.detection_keys),edges=len(graph.edges),
            component_nodes=max((len(x.sources)+len(x.detections) for x in components),default=0))
        sources={s.source_id:s for s in table.sources}
        # These are fixed-mean proposal factors, not posterior probabilities.
        # Preserve the original summation/order while avoiding repeated logs
        # for every beam candidate.
        miss_scores={}
        hit_scores={}
        for s,source in sources.items():
            miss_scores[s]=source_log_miss_factor(source)
            log_r=source.log_existence_prior
            hit_scores[s]=tuple(log_r+v for v in source.log_eta)
        clutter_scores=table.log_kappa
        def score(items,indices=None):
            """Unnormalized fixed-mean proposal; NEVER a posterior or bound."""
            used={j for _,j in items if j is not None};terms=[]
            for s,j in items:
                terms.append(miss_scores[s] if j is None else hit_scores[s][j])
            for j in (range(len(table.detection_keys)) if indices is None else indices):
                if j in used:continue
                terms.append(clutter_scores[j])
            return -math.inf if -math.inf in terms else math.fsum(terms)
        local=[];local_complete=True;counts=[]
        per_parent=max(1,c.max_candidates//self.parent_count)
        if not graph.edges:
            # Exactly one assignment: all sources miss, all detections clutter.
            # Keep the usual true-weight selection and conditional miss update.
            counts=[1]*len(components)
            combos=(tuple((source,None) for source in graph.source_ids),)
            combo_complete=True;total=1
            # Match the work accounting of the isolated component enumeration.
            self.controller.consume(5*len(graph.source_ids)+3*len(graph.detection_keys)+1)
        else:
            for comp in components:
                values,count=ExactComponentSolver.solve(comp,graph,self.controller,c.max_local_hypotheses)
                counts.append(count)
                if values is None:
                    local_complete=False
                    values=RankedGlobalHypothesisSolver.beam_component(comp,graph,score,self.controller,per_parent)
                local.append(values)
            combos,combo_complete,total=GlobalHypothesisCombiner.combine(local,score,self.controller,per_parent)
        gate_loss=any(e[2]=='FINITE_GATE_POSSIBLE_NONZERO' for e in graph.excluded_edges)
        complete=local_complete and combo_complete and not gate_loss
        self.complete &= complete
        if self.detail=='full':self.records.append({'parent_id':table.parent_id,'nodes':len(graph.source_ids)+len(graph.detection_keys),
            'edges':len(graph.edges),'components':len(components),'max_component_nodes':max((len(x.sources)+len(x.detections) for x in components),default=0),
            'local_hypothesis_counts_capped':tuple(counts),'candidate_product_count':total,'candidates':len(combos),
            'excluded_edges':graph.excluded_edges,'gate_semantics':'ASSOCIATION_HYPOTHESIS_TRUNCATION; ORIGINAL_MISS_UNCHANGED; NOT_A_RENORMALIZED_SENSOR_MODEL',
            'gate_probability':c.gate_probability,'gate_tail_is_posterior_error_bound':False,
            'ranking':'EXHAUSTIVE' if local_complete and combo_complete else 'HEURISTIC_BEAM; GLOBAL_TOP_K_NOT_GUARANTEED',
            'complete_relative_to_original_factors':complete})
        return tuple(association(table,a) for a in combos)

    def select(self,plans,logs):
        c=self.config
        limit=getattr(c,'association_top_k',None) or c.top_k
        keep,omitted,evidence=HypothesisPruner.select(logs,[p[1].hypothesis_id for p in plans],limit)
        if self.detail=='compact':
            self.audit=dict(evaluated_candidates=len(plans),retained_candidates=len(keep),
                discarded_mass_within_evaluated_candidates=omitted,
                omitted_mass_status='CANDIDATE_CONDITIONAL_ONLY; PRE_GATED_FACTORS_AND_BEAM_LOSS_UNKNOWN')
            return keep
        truncated=len(keep)<len(plans) and omitted>0
        self.audit={'mode':'APPROXIMATE_FAST','graphs':tuple(self.records),'evaluated_candidates':len(plans),
            'retained_candidates':len(keep),'candidate_log_evidence':evidence,
            'complete_model_log_evidence':evidence if self.complete else None,
            'discarded_mass_within_evaluated_candidates':omitted,
            'omitted_mass':omitted if self.complete else None,
            'omitted_mass_status':'EXACT_FOR_THIS_FRAME_GIVEN_INPUT_PRIOR' if self.complete else 'OMITTED_MASS_UNKNOWN',
            'ranking_guarantee':'GLOBAL_TRUE_TOP_K_GIVEN_INPUT_PRIOR' if self.complete else 'ONLY_TRUE_WEIGHT_RANKING_WITHIN_CANDIDATES',
            'approximate':not self.complete or truncated,
            'search_states':self.controller.work,
            'continuous_status':'DECLARED_ONLINE_CONTINUOUS_APPROXIMATIONS'}
        return keep

