# 系统架构、目录职责与运行指南

本项目以逐帧 CSV 检测结果为输入，完成星图识别、姿态估计、恒星背景剔除、多目标跟踪和轨迹重连。

重构只调整代码归属和导入路径；**不改变算法、默认参数、候选点顺序、轨迹状态、阶段顺序或阶段间数据流**。

## 快速运行

在 `space_target_detection/` 目录执行：

```powershell
# 查看可用数据集
python main.py --list

# 运行 datasets/ 下指定数据集
python main.py --dataset 1

# 直接指定检测 CSV 目录
python main.py --input datasets/1/measurements

# 仅处理前 100 帧，并指定输出目录
python main.py --input datasets/1/measurements --frames 100 --output output

# 静默运行
python main.py --dataset 1 --quiet
```

完整流程也可通过明确的 runner 启动：

```powershell
python runners/run_full_pipeline.py --input datasets/1/measurements
```

参数说明：

| 参数 | 含义 |
| --- | --- |
| `--input`, `-i` | 直接指定 `measurements/` 目录 |
| `--dataset`, `-d` | 在 `datasets/` 中按名称选择数据集 |
| `--datasets-root` | 指定数据集根目录 |
| `--output`, `-o` | 指定输出根目录 |
| `--frames`, `-n` | 仅处理前 N 帧 |
| `--snapshots` | 保存各阶段的精确输入快照 |
| `--quiet`, `-q` | 不打印逐帧日志 |
| `--list`, `-l` | 列出数据集后退出 |

## 输入和输出

### 输入数据集

```text
datasets/
└── <dataset-name>/
    ├── measurements/                 # 必需：逐帧检测 CSV
    │   ├── measurements_0000.csv
    │   ├── measurements_0001.csv
    │   └── ...
    └── truth/                        # 可选：回归/指标评估真值
        ├── target_*_trajectory.npy
        └── quaternion_history.npy
```

每个 CSV 对应一帧，候选点在流程中采用 `(x, y, flux)`。存在 `truth/` 时会额外计算目标、恒星和姿态相关的评估指标。

### 输出目录

每次完整运行都创建时间戳目录，内容通常包括：

```text
<run-directory>/
├── frame_XXXX.png                   # 每帧可视化
├── per_frame_stats.csv              # 逐帧统计与耗时
├── trajectories.csv                 # 轨迹点、状态和来源
├── track_events.csv                 # 轨迹事件
├── reconnect_events.csv             # 重连事件
├── attitude.csv                     # 姿态四元数、欧拉角及可选误差
├── tracks_overlay.png               # 全轨迹叠加图
├── eval_stars.csv                   # 可选：恒星识别评估
├── eval_targets.csv                 # 可选：目标跟踪评估
└── eval_star_removal.csv            # 可选：恒星剔除评估
```

## 固定数据流

```text
DetectionLoader
  -> TrackPredictionStage
  -> CandidateSeparator.pre_separate
  -> Delaunay geometry (star and background branches)
  -> StarProcessingStage
       (identification -> attitude -> catalog projection)
  -> CandidateSeparator.refine
  -> FrameTrackingStage
  -> TrackReconnectionStage
  -> evaluation + visualization + reporting
```

1. `DetectionLoader` 读取当前帧的候选点与元数据。
2. `TrackPredictionStage` 对已有轨迹执行预测，提供可信目标位置。
3. `CandidateSeparator.pre_separate` 保守地分出星图输入和受轨迹保护的候选目标。
4. 构建星图识别和相邻帧背景判断所需的两类 Delaunay 三角形。
5. `StarProcessingStage` 完成星图识别、自适应方法选择/回退、姿态解算和星表投影。
6. `CandidateSeparator.refine` 融合姿态、星表、运动、时序、三角背景和轨迹证据，输出最终目标候选。
7. `FrameTrackingStage` 构建代价矩阵 `Phi`，执行关联、滤波更新和剪枝。
8. `TrackReconnectionStage` 处理出界归档、记忆池、再入评分和轨迹合并。
9. 评估、可视化和输出层只读取处理结果，不反向改变算法状态。

## 目录职责

| 位置 | 负责什么 | 适合单独研究什么 |
| --- | --- | --- |
| `main.py` | 整套系统的薄命令行入口 | 全流程运行 |
| `pipeline/` | 流程编排、运行状态、阶段结果、快照 | 数据流、阶段接口、回放 |
| `data_io/` | CSV 检测与数据集加载 | 输入字段、数据质量 |
| `configuration/` | 唯一配置值及按领域分组的配置视图 | 相机、阈值、开关 |
| `star_processing/` | 星图识别、姿态、星表投影 | 星点识别和姿态精度 |
| `candidate_processing/` | 预分离、时序/运动证据、三角背景、分类 | 恒星剔除和候选召回 |
| `target_tracking/` | KF/EKF/IMM、关联、生命周期 | 关联与滤波模型 |
| `track_reconnection/` | 出界记忆、再入匹配、轨迹合并 | ID 保持与重连 |
| `evaluation/` | 真值读取和指标计算 | 回归比较与量化评估 |
| `visualization/` | 每帧及全轨迹渲染 | 图像检查与展示 |
| `reporting/` | CSV、叠加图、输出目录、终端汇总 | 输出格式与结果归档 |
| `runners/` | 完整流程及单阶段命令行脚本 | 独立实验和快照回放 |
| `datasets/` | 输入样例和数据集 | 运行数据 |
| `*.dat` | 星表及静态数据资产 | 不应随代码重构删除或移动 |

### `configuration/`

`configuration/values.py` 是唯一真实配置来源。`camera.py`、`features.py`、`star_processing.py`、`candidate_processing.py`、`target_tracking.py`、`reconnection.py` 和 `evaluation.py` 只是按主题查看同一份配置，不复制参数。

```python
from configuration.camera import current_camera_config
from configuration.features import current_feature_flags
from configuration import star_processing, candidate_processing
from configuration import target_tracking, reconnection
```

### `star_processing/`

| 文件 | 职责 |
| --- | --- |
| `stage.py` | 星图处理阶段边界：识别、回退、姿态、投影 |
| `runtime.py` | 星表、索引、识别器和跨帧星图状态 |
| `delaunay_identifier.py` | Delaunay 三角形星图识别 |
| `descriptor_identifier.py` | 描述子式星图识别 |
| `attitude_solvers.py` | SVD、Davenport、QUEST 等姿态解算 |
| `projection_geometry.py` | 像点、视线向量和星表投影几何 |
| `projection_tracking.py` | 基于前帧姿态的星点投影跟踪 |

### `candidate_processing/`

| 文件 | 职责 |
| --- | --- |
| `stage.py` | 显式的 `prepare` 与 `classify` 两阶段接口 |
| `separator_impl.py` | `CandidateSeparator`：粗分离和精分离核心 |
| `temporal_filter.py` | 候选点跨帧时序证据 |
| `triangulation.py` | Delaunay 三角构建 |
| `triangle_background.py` | 三角背景匹配与恒星证据 |

### `target_tracking/`

| 文件 | 职责 |
| --- | --- |
| `linear_kf.py` | 线性卡尔曼滤波 |
| `ctrv_ekf.py` | CTRV 扩展卡尔曼滤波 |
| `imm.py` | IMM 多模型滤波 |
| `association_impl.py` | 特征代价矩阵和 Sinkhorn 软分配 |
| `trajectory_manager_impl.py` | 建轨、更新、剪枝和状态维护 |
| `stage.py` | 帧内预测、关联与更新入口 |

### 其他核心目录

- `track_reconnection/service.py`：出界记忆、再入候选、评分、ID 合并。
- `evaluation/metrics.py`：逐帧指标、统计汇总、MOTA/MOTP。
- `visualization/rendering.py`：识别和轨迹图像渲染。
- `reporting/core.py`：运行目录、重连事件 CSV、轨迹序列检查。
- `reporting/artifacts.py`：统计、姿态、轨迹、评估 CSV、叠加图和终端汇总。
- `pipeline/runtime.py`：`SequenceRuntime`、`FrameRuntime` 与具名阶段结果；只组织已有对象，不复制、转换或排序算法数据。

## 单模块精确回放

单模块实验应使用完整流程生成的受信任快照，避免自行构造上游状态。先生成快照：

```powershell
python main.py --input datasets/1/measurements --snapshots snapshots/run_001
```

再回放指定阶段：

```powershell
python runners/run_stage.py star --input <star-input.pkl> --output <star-result.pkl>
python runners/run_stage.py candidate_prepare --input <input.pkl> --output <result.pkl>
python runners/run_stage.py candidate_classify --input <input.pkl> --output <result.pkl>
python runners/run_stage.py prediction --input <input.pkl> --output <result.pkl>
python runners/run_stage.py tracking --input <input.pkl> --output <result.pkl>
python runners/run_stage.py reconnection --input <input.pkl> --output <result.pkl>
```

等价的专用入口为：

```text
runners/run_star_processing.py
runners/run_candidate_prepare.py
runners/run_candidate_classify.py
runners/run_track_prediction.py
runners/run_target_tracking.py
runners/run_track_reconnection.py
```

专用入口也需要 `--input <snapshot.pkl> --output <result.pkl>`。快照使用 `pickle`，只能读取本项目生成且可信的本地文件；它保留 NumPy 数组、滤波器实例、候选分离器状态和轨迹字典的原始结构。

## 推荐导入方式

不要使用已删除的根目录旧模块。新研究脚本应从领域包导入：

```python
from data_io import DetectionLoader
from star_processing import StarProcessingStage, StarProcessingRuntime
from star_processing.attitude_solvers import solve_svd, solve_quest
from candidate_processing.stage import CandidateProcessingStage
from candidate_processing.separator_impl import CandidateSeparator
from target_tracking import LinearKF, EKFTracker, IMMTracker
from target_tracking.association_impl import build_cost_matrix
from track_reconnection.service import reconnect_exited_tracks
```

完整流程 API 位于 `pipeline.legacy_sequence.process_sequence`；单模块研究优先使用领域入口或快照回放。

## 不可破坏的重构约束

- 不调整 `pre_separate`、星图识别、`refine`、`update_all` 与重连的调用顺序。
- 不重新排序候选点或轨迹。
- 不改变配置默认值、阈值比较方式、滤波预测次数、状态写回时机或协方差类型。
- 不将三角背景输入替换为 `star_points`。
- 结构变更后应比较轨迹、姿态、轨迹/重连事件、去除耗时字段后的逐帧统计和 PNG 哈希。
- 算法修复与结构迁移必须分开验证。
