"""Conditional source-to-detection competition graph."""
from dataclasses import dataclass
import math
import numpy as np
from ..contracts import checked_covariance


class ProbabilisticGate:
    @staticmethod
    def threshold(probability):
        if not 0<probability<1: raise ValueError('0 < P_G < 1 required')
        return -2*math.log1p(-probability)  # chi-square df=2 inverse CDF

    @staticmethod
    def distances(residuals,covariance):
        S=checked_covariance(covariance)
        residuals=np.asarray(residuals,dtype=float)
        if residuals.ndim!=2 or residuals.shape[1]!=2 or not np.isfinite(residuals).all():
            raise ValueError('finite two-dimensional innovations required')
        L=np.linalg.cholesky(S)
        y=np.linalg.solve(L,residuals.T)
        return np.sum(y*y,axis=0)

    @staticmethod
    def distances_many(residuals,covariances):
        """Independent 2-D innovations; no reduction across source states."""
        matrices=np.asarray([checked_covariance(cov) for cov in covariances])
        residuals=np.asarray(residuals,dtype=float)
        if (residuals.ndim!=3 or residuals.shape[0]!=len(matrices) or
                residuals.shape[2]!=2 or not np.isfinite(residuals).all()):
            raise ValueError('finite batches of two-dimensional innovations required')
        L=np.linalg.cholesky(matrices)
        y=np.linalg.solve(L,residuals.swapaxes(1,2))
        return np.sum(y*y,axis=1)


@dataclass(frozen=True)
class AssociationEdge:
    source_id: str
    detection_index: int


@dataclass(frozen=True)
class UnifiedAssociationGraph:
    detection_keys: tuple
    source_ids: tuple
    edges: tuple
    excluded_edges: tuple


class AssociationGraphBuilder:
    @staticmethod
    def build(table,config,detections=(),geometry=None,distances_cache=None,*,record_exclusions=True):
        geometry=geometry or {};edges=[];excluded=[]
        threshold=ProbabilisticGate.threshold(config.gate_probability) if config.gate_probability is not None else None
        pixels=tuple((d.x_px,d.y_px) for d in detections)
        for s in sorted(table.sources,key=lambda x:x.source_id):
            distances=None
            info=geometry.get(s.source_id)
            if config.gate_probability is not None and info is not None:
                mean,cov=info
                key=(tuple(mean),tuple(map(tuple,cov)),pixels)
                if distances_cache is not None and key in distances_cache:
                    distances=distances_cache[key]
                else:
                    distances=ProbabilisticGate.distances([(x-mean[0],y-mean[1]) for x,y in pixels],cov) if detections else ()
                    if distances_cache is not None:distances_cache[key]=distances
                if len(distances)!=len(table.detection_keys):raise ValueError('gate detection order mismatch')
            for j,log_eta in enumerate(s.log_eta):
                if not math.isfinite(log_eta):
                    if record_exclusions:excluded.append((s.source_id,j,'ZERO_DENSITY_FACTOR',None))
                    continue
                if distances is not None and distances[j]>threshold:
                    # Restrict association hypotheses, NOT redefine sensor PD or miss.
                    # Omitted observations remain in the full set and compete elsewhere.
                    reason='FINITE_GATE_POSSIBLE_NONZERO' if s.log_existence_prior > -math.inf else 'ZERO_DENSITY_GATE'
                    # Compact output needs only whether a nonzero gated edge
                    # was omitted; detailed per-edge audit rows are optional.
                    if record_exclusions or reason=='FINITE_GATE_POSSIBLE_NONZERO' and not excluded:
                        excluded.append((s.source_id,j,reason,float(distances[j])))
                    continue
                edges.append(AssociationEdge(s.source_id,j))
        return UnifiedAssociationGraph(table.detection_keys,
            tuple(sorted(s.source_id for s in table.sources)),tuple(edges),tuple(excluded))


@dataclass(frozen=True)
class ConditionalAssociationComponent:
    sources: tuple
    detections: tuple


class ConditionalAssociationComponentBuilder:
    @staticmethod
    def build(graph):
        nodes=[('S',s) for s in graph.source_ids]+[('D',j) for j in range(len(graph.detection_keys))]
        adj={v:set() for v in nodes}
        for e in graph.edges:
            a,b=('S',e.source_id),('D',e.detection_index);adj[a].add(b);adj[b].add(a)
        seen=set();out=[]
        for node in nodes:
            if node in seen:continue
            stack=[node];group=[];seen.add(node)
            while stack:
                x=stack.pop();group.append(x)
                for y in sorted(adj[x],key=repr):
                    if y not in seen:seen.add(y);stack.append(y)
            out.append(ConditionalAssociationComponent(tuple(sorted(v for t,v in group if t=='S')),
                tuple(sorted(v for t,v in group if t=='D'))))
        return tuple(out)

