"""Computation policy and cooperative frame/resource budgets."""
from dataclasses import dataclass
import math
import time
import os
from ..exact.contracts import InferenceError


@dataclass(frozen=True)
class InferenceConfig:
    gate_probability: float
    top_k: int
    max_nodes: int
    max_edges: int
    max_component_nodes: int
    max_local_hypotheses: int
    max_candidates: int
    max_search_states: int
    max_memory_bytes: int
    max_frame_seconds: float
    association_top_k: int = None

    def __post_init__(self):
        if self.association_top_k is not None and (isinstance(self.association_top_k,bool) or
                not isinstance(self.association_top_k,int) or self.association_top_k<self.top_k):
            raise ValueError('association_top_k must be null or an integer >= top_k')
        if isinstance(self.gate_probability,bool) or not 0<self.gate_probability<1:
            raise ValueError('gate probability must lie strictly between zero and one')
        for name,value in vars(self).items():
            if name.startswith('max_') or name=='top_k':
                if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
                    raise ValueError('positive finite computation budget required: '+name)
                if name!='max_frame_seconds' and not isinstance(value,int):
                    raise ValueError('integer computation budget required: '+name)
        if self.max_memory_bytes>16*1024**3: raise ValueError('runtime memory ceiling is 16 GiB')


_windows_memory_reader=None


def process_memory_bytes():
    """Resident working set. Sampled limit, not a hard OS allocation quota."""
    if os.name=='nt':
        global _windows_memory_reader
        if _windows_memory_reader is not None:return _windows_memory_reader()
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in
                ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
                 'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        kernel=ctypes.WinDLL('kernel32'); kernel.GetCurrentProcess.restype=wintypes.HANDLE
        psapi=ctypes.WinDLL('psapi'); psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        handle=kernel.GetCurrentProcess()
        def read():
            c=Counters();c.cb=ctypes.sizeof(c)
            if not psapi.GetProcessMemoryInfo(handle,ctypes.byref(c),c.cb):raise OSError('cannot query process working set')
            return int(c.WorkingSetSize)
        _windows_memory_reader=read
        return read()
    import resource,sys
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform=='darwin' else value*1024)


class InferenceBudgetController:
    def __init__(self,config):
        self.config=config;self.started=time.perf_counter();self.peak_sampled_rss=0;self.work=0

    def fail(self,name,value,limit):
        raise InferenceError('APPROXIMATE_BUDGET_EXCEEDED','%s=%s exceeds %s; no silent model substitution'%(name,value,limit))

    def check(self,**counts):
        for name,value in counts.items():
            limit=getattr(self.config,'max_'+name)
            if value>limit:self.fail(name,value,limit)
        elapsed=time.perf_counter()-self.started
        if elapsed>self.config.max_frame_seconds:self.fail('frame_seconds',elapsed,self.config.max_frame_seconds)
        rss=process_memory_bytes();self.peak_sampled_rss=max(self.peak_sampled_rss,rss)
        if rss>self.config.max_memory_bytes:self.fail('memory_bytes',rss,self.config.max_memory_bytes)

    def consume(self,n=1):
        self.work+=n
        if self.work>self.config.max_search_states:self.fail('search_states',self.work,self.config.max_search_states)
        if self.work%128==0:self.check()



