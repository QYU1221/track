"""星表数据加载模块。

两种星图识别方法共用的基础操作：文件解析、向量归一化、星等筛选。
不包含任何 Delaunay 星对索引、描述子 KDTree 或识别逻辑。
"""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CatalogData:
    """星等筛选后的星表数据，两种识别方法共用。

    所有数组在星等筛选后长度均为 N。
    catalog_ids 保持原始星表行号，不重新编号。
    """
    catalog_ids: np.ndarray   # (N,)  原始星表行号
    unit_vectors: np.ndarray  # (N, 3) 归一化惯性单位向量
    magnitudes: np.ndarray    # (N,)   星等
    ra: np.ndarray            # (N,)   赤经 (deg)
    dec: np.ndarray           # (N,)   赤纬 (deg)


def load_star_catalog(
    catalog_path: str,
    mag_limit: Optional[float] = None,
) -> CatalogData:
    """加载星表文件，解析后按星等阈值筛选。

    支持制表符和空格分隔。每行至少包含 x, y, z, dec, ra, mag。
    原始编号使用文件行号（0-based），星等筛选后不重新编号。

    返回 CatalogData，其中所有数组均按原始行号排序。
    """
    if not os.path.isabs(catalog_path):
        catalog_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), catalog_path)

    all_ids = []
    all_vecs = []
    all_mags = []
    all_ra = []
    all_dec = []

    with open(catalog_path, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 6:
                parts = line.split()
            if len(parts) < 6:
                continue

            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                dec = float(parts[3])
                ra = float(parts[4])
                mag = float(parts[5])
            except (ValueError, IndexError):
                continue

            vec = np.array([x, y, z], dtype=np.float64)
            norm = float(np.linalg.norm(vec))
            if norm < 1e-12 or not np.isfinite(vec).all():
                continue

            all_ids.append(idx)
            all_vecs.append(vec / norm)
            all_mags.append(mag)
            all_ra.append(ra)
            all_dec.append(dec)

    if not all_ids:
        raise RuntimeError(
            f"No valid star entries found in catalog: {catalog_path}")

    ids_arr = np.array(all_ids, dtype=np.int64)
    vecs_arr = np.array(all_vecs, dtype=np.float64)
    mags_arr = np.array(all_mags, dtype=np.float64)
    ra_arr = np.array(all_ra, dtype=np.float64)
    dec_arr = np.array(all_dec, dtype=np.float64)

    if mag_limit is not None:
        mask = mags_arr <= mag_limit
        if not np.any(mask):
            raise RuntimeError(
                f"No stars brighter than mag_limit={mag_limit} in catalog")
        ids_arr = ids_arr[mask]
        vecs_arr = vecs_arr[mask]
        mags_arr = mags_arr[mask]
        ra_arr = ra_arr[mask]
        dec_arr = dec_arr[mask]

    return CatalogData(
        catalog_ids=ids_arr,
        unit_vectors=vecs_arr,
        magnitudes=mags_arr,
        ra=ra_arr,
        dec=dec_arr,
    )
