"""Catalog loading and JSON-safe serialization used by the online runtime."""
from dataclasses import fields,is_dataclass
from pathlib import Path
import math
import numpy as np
from ..contracts import KnownStar
from ..geometry import unit


def public_navstar_catalog(path,mag_limit=None):
    if mag_limit is not None and not np.isfinite(mag_limit):raise ValueError('magnitude selection must be finite')
    result=[]
    with Path(path).open(encoding='utf-8-sig') as stream:
        for index,line in enumerate(stream):
            parts=line.split()
            if not parts:continue
            try:numbers=np.asarray([float(value) for value in parts[:6]])
            except ValueError as error:raise ValueError('invalid public catalog row '+str(index)) from error
            if len(numbers)!=6 or not np.isfinite(numbers).all() or not -90<=numbers[3]<=90:
                raise ValueError('invalid public catalog row '+str(index))
            if mag_limit is None or numbers[5]<=mag_limit:
                result.append(KnownStar('NAVSTAR:'+str(index),tuple(unit(numbers[:3]))))
    if not result:raise ValueError('public catalog contains no selected stars')
    return tuple(result)


def serializable(value):
    # Most reporting leaves are native JSON scalars. Avoid dataclass lookup
    # and NumPy ufunc dispatch for every coordinate, weight and metadata string.
    kind=type(value)
    if kind is float or kind is np.float64:
        if value==-math.inf:return '-inf'
        if not math.isfinite(value):raise ValueError('nonfinite positive value/NaN must not enter a report')
        return float(value)
    if kind in (str,int,bool,type(None)):return value
    if kind is dict:return {str(key):serializable(item) for key,item in value.items()}
    if kind is tuple or kind is list:return [serializable(item) for item in value]
    if is_dataclass(value):return {field.name:serializable(getattr(value,field.name)) for field in fields(value)}
    if isinstance(value,dict):return {str(key):serializable(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):return [serializable(item) for item in value]
    if isinstance(value,(float,np.floating)):
        if np.isneginf(value):return '-inf'
        if not np.isfinite(value):raise ValueError('nonfinite positive value/NaN must not enter a report')
        return float(value)
    if isinstance(value,np.bool_):return bool(value)
    if isinstance(value,np.integer):return int(value)
    return value
