"""
DetectionLoader — 检测数据读取模块

读取 measurements/ 目录下的逐帧 CSV 文件，完成合法性校验，
转换为 trajectory_manager 兼容的 candidate 格式。
"""

# Domain implementation: detection input.
import os                                        
import numpy as np                               

from configuration.values import FRAME_W, FRAME_H


class DetectionLoader:
    """检测数据加载器。

    从 measurements/ 目录按帧号顺序读取 CSV，校验并转换格式。

    对外接口：
        load_frame(idx)          → 读取第 idx 帧，返回 (candidates, meta)
        get_energies_and_radii() → 从元数据提取能量和半径，替代图像层面的计算
        n_frames                 → 属性，返回总帧数

    参数
    ----
    measurements_dir : str        CSV 文件所在目录路径
    frame_w, frame_h : int        图像尺寸（像素），用于坐标越界检查，默认从 config 读取
    """

    def __init__(self, measurements_dir, frame_w=FRAME_W, frame_h=FRAME_H):
        self._dir = measurements_dir                # 保存 CSV 目录路径
        self._frame_w = frame_w                     # 图像宽度（像素列数）
        self._frame_h = frame_h                     # 图像高度（像素行数）
        self._files = self._scan_files()             # 扫描目录下所有 CSV，排序后作为帧列表
        self._n_frames = len(self._files)            # 总帧数 = CSV 文件数量
        self._current_meta = {}                      # 缓存最近一次 load_frame 的元数据

    def _scan_files(self):
        """扫描 measurements/ 下所有 CSV 文件，按文件名自然排序。

        排序规则：sorted() 对字符串按字典序排序，因此文件名含数字时
        "frame_2.csv" 会在 "frame_10.csv" 之前（与人直觉一致仅当数字位数相同时）。
        如果实际命名是 "frame_02.csv"、"frame_10.csv" 则无问题。
        """
        files = []
        for fname in sorted(os.listdir(self._dir)):    # 按文件名升序遍历目录
            if fname.lower().endswith('.csv'):         # 仅保留 .csv / .CSV 文件，忽略其他
                files.append(os.path.join(self._dir, fname))  # 拼接为完整路径加入列表
        return files

    @property
    def n_frames(self):
        """总帧数（只读属性）。"""
        return self._n_frames

    def load_frame(self, frame_idx):
        """按索引读取检测数据，与文件名无关。

        参数
        ----
        frame_idx : int   帧索引，范围 [0, n_frames-1]

        返回
        ----
        candidates : list[(x, y, flux)]    每项为 (x坐标, y坐标, 能量/通量)
        meta       : dict                  键为 (round(x,2), round(y,2))，值为该检测点的元数据
        """
        if frame_idx >= self._n_frames:             # 越界保护：索引超出范围直接返回空
            return [], {}

        path = self._files[frame_idx]               # 根据索引取出对应 CSV 文件路径
        raw = self._read_csv(path)                  # 逐行读取 CSV → list[dict]
        candidates, meta = self._validate_and_convert(raw)  # 合法性校验 + 格式转换
        self._current_meta = meta                   # 缓存元数据，供 get_energies_and_radii 使用
        return candidates, meta

    def _read_csv(self, path):
        """手动逐行解析 CSV，返回 list[dict]。

        不使用 csv 标准库，原因：
        - 输入格式简单，无引号转义场景
        - 避免额外的 import

        返回
        ----
        rows : list[dict]   每个元素是一行的 {列名: 值}，值均为字符串（未做类型转换）
        """
        rows = []
        with open(path, 'r') as f:                           # 文本模式打开文件
            header = f.readline().strip()                    # 读取第一行作为表头，去掉末尾换行
            cols = [c.strip().lower() for c in header.split(',')]  # 按逗号拆分列名，去空格并统一小写
            for line in f:                                   # 逐行读取剩余数据行
                line = line.strip()                          # 去掉首尾空白和换行符
                if not line:                                 # 跳过空行
                    continue
                vals = [v.strip() for v in line.split(',')]  # 按逗号拆分，每个值去空格
                if len(vals) < len(cols):                    # 列数不足的行视为残缺，丢弃
                    continue
                row = dict(zip(cols, vals))                  # 将列名与值配对成字典
                rows.append(row)                             # 加入结果列表
        return rows

    def _validate_and_convert(self, rows):
        """对 CSV 原始行做类型转换 + 合法性校验，返回 (candidates, meta)。

        校验规则：
        - (x, y) 必须为有限数值，且在 [0, frame_w) × [0, frame_h) 内
        - area 必须为正有限数
        - snr 必须为正有限数
        - confidence 必须在 [0, 1] 区间
        - energy 允许 NaN/Inf（回退为 0.0），但不会被拒绝整行
        - 任何字段解析失败（ValueError）则跳过该行

        返回
        ----
        candidates : list[(x, y, flux)]   通过校验的候选点，flux 即 energy
        meta       : dict                 键为 (round(x,2), round(y,2))，
                                          值为 {'energy', 'area', 'snr', 'confidence'}
        """
        candidates = []
        meta = {}
        for row in rows:
            try:
                # --- 类型转换：字符串 → float -------------------------------------------
                x = float(row.get('x', np.nan))                  # x 坐标，缺失则 NaN
                y = float(row.get('y', np.nan))                  # y 坐标，缺失则 NaN
                energy = float(row.get('energy', row.get('flux', 0)))  # 能量字段：优先 'energy'，其次 'flux'，都无则为 0
                area = float(row.get('area', 0))                 # 斑点面积（像素数）
                snr = float(row.get('snr', 0))                   # 信噪比
                conf = float(row.get('confidence', 0))           # 置信度 [0, 1]
            except (ValueError, TypeError):
                # 任意字段无法解析为 float 时（如空字符串、非数字文本），丢弃该行
                continue

            # --- 合法性检查 --------------------------------------------------------------

            # 坐标必须有意义（非 NaN、非 ±∞）
            if np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y):
                continue
            # 坐标必须在图像范围内（左闭右开区间）
            if x < 0 or x >= self._frame_w or y < 0 or y >= self._frame_h:
                continue
            # 面积必须为正有限数（面积 ≤ 0 或 NaN/Inf 无实际意义）
            if np.isnan(area) or np.isinf(area) or area <= 0:
                continue
            # 信噪比必须为正有限数
            if np.isnan(snr) or np.isinf(snr) or snr <= 0:
                continue
            # 置信度必须在 [0, 1] 之间
            if np.isnan(conf) or np.isinf(conf) or conf < 0 or conf > 1:
                continue
            # 能量不做拒绝处理：NaN/Inf 时回退为 0.0
            if np.isnan(energy) or np.isinf(energy):
                energy = 0.0

            # --- 数据打包 ----------------------------------------------------------------
            flux = energy                                   # flux 与 energy 等同
            candidates.append((x, y, flux))                 # 候选点格式：(x, y, flux)
            key = (round(x, 2), round(y, 2))                # 以四舍五入到 2 位小数的坐标作为键
            meta[key] = {
                'energy': energy,                            # 能量/通量值
                'area': area,                                # 斑点面积（像素）
                'snr': snr,                                  # 信噪比
                'confidence': conf,                          # 置信度 [0, 1]
            }

        return candidates, meta

    def get_energies_and_radii(self, candidates):
        """从检测元数据中提取能量和半径（替代图像计算）。

        使用场景：feature_fusion 模块中原本需要从原始图像裁剪区域计算
        候选点的"能量"和"等效半径"，本方法直接从 CSV 的 area / energy 字段
        读取并换算，避免依赖原始图像。

        参数
        ----
        candidates : list[(x, y, flux)]   与 load_frame 返回格式一致

        返回
        ----
        energies : np.ndarray  每个候选点的能量值（来自 CSV energy 字段，缺失则为 0.0）
        radii    : np.ndarray  每个候选点的等效半径 r = sqrt(area / π)
        """
        energies = []
        radii = []
        for cx, cy, _ in candidates:                     # 遍历每个候选点，只需要 x、y 坐标
            key = (round(cx, 2), round(cy, 2))           # 构造与 meta 中一致的键（保留 2 位小数）
            m = self._current_meta.get(key, {})           # 从缓存的元数据中查找，找不到返回空字典
            energies.append(m.get('energy', 0.0))         # 提取能量，缺省 0.0
            area = m.get('area', 1.0)                     # 提取面积，缺省 1.0（避免除以零）
            radii.append(np.sqrt(max(area, 0.01) / np.pi))  # r = √(area/π)，下限 0.01 防负/零
        return np.array(energies), np.array(radii)
