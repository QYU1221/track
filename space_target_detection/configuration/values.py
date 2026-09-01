# 全局参数配置

# ── 相机配置 ──
# 可选值："k4"（K4 星敏）或 "standard"（项目最初使用的标准相机）
camera_profile = "k4"

CAMERA_PROFILES = {
    "standard": {
        "frame_width": 1024,
        "frame_height": 1024,
        "focal_length_m": 0.05,
        "f_number": None,          # 原始配置未提供光圈系数
        "pixel_size_m": 1.5e-05,
        "cx": 512.0,
        "cy": 512.0,
        "optical_axis": "+Z",
        "projection_model": "standard",
        "projection_u": "cx + fpx * X/Z",
        "projection_v": "cy - fpx * Y/Z",
        "inverse_X_over_Z": "(u - cx) / fpx",
        "inverse_Y_over_Z": "-(v - cy) / fpx",
        "quaternion_order": "wxyz",
        "quaternion_inverse": True,
    },
    "k4": {
        "frame_width": 1308,
        "frame_height": 982,
        "focal_length_m": 0.0163,
        "f_number": 1.83,
        "pixel_size_m": 3.985e-06,
        "cx": 668.0,
        "cy": 466.0,
        "optical_axis": "+Z",
        "projection_model": "k4",
        "projection_u": "cx - fpx * Y/Z",
        "projection_v": "cy + fpx * X/Z",
        "inverse_X_over_Z": "(v - cy) / fpx",
        "inverse_Y_over_Z": "-(u - cx) / fpx",
        "quaternion_order": "xyzw",
        "quaternion_inverse": True,
    },
}

if camera_profile not in CAMERA_PROFILES:
    raise ValueError(
        "未知相机配置 %r，可选值为：%s" %
        (camera_profile, ", ".join(sorted(CAMERA_PROFILES)))
    )

_camera = CAMERA_PROFILES[camera_profile]

# 保留项目原有变量名，使下游模块无需修改。
FRAME_W = _camera["frame_width"]
FRAME_H = _camera["frame_height"]
camera_focal_length_m = _camera["focal_length_m"]
camera_f_number = _camera["f_number"]
camera_pixel_size_m = _camera["pixel_size_m"]
camera_cx = _camera["cx"]
camera_cy = _camera["cy"]
camera_optical_axis = _camera["optical_axis"]
camera_projection_model = _camera["projection_model"]
camera_quaternion_order = _camera["quaternion_order"]
camera_quaternion_inverse = _camera["quaternion_inverse"]

# ── 跟踪器选择 ──
use_imm = True                     # True: IMM双模型; False: 单CTRV EKF

# ── EKF 参数（5D: x, y, θ, v, ω）──
Q_diag = [0.1, 0.1, 0.01, 2.0, 0.8]   # 过程噪声协方差对角线 [x, y, θ, v, ω]
R_diag = [2.0, 2.0]                     # 观测噪声协方差对角线 [x, y]
P_init_diag = [10.0, 10.0, 0.5, 5.0, 0.2]  # 初始状态协方差对角线

# ── IMM 参数 ──
imm_Pi = [[0.95, 0.05],           # 马尔可夫转移矩阵 Π[i][j]: 模型 i→j
          [0.05, 0.95]]           # CV↔CTRV 切换概率 5%
imm_mu_init = [0.5, 0.5]          # 初始模型概率（无先验，等权）
imm_mu_min = 0.01                 # 模型概率下限（防止退化）

# ── CV 模型参数（4D: x, y, vx, vy）──
cv_Q_diag = [0.1, 0.1, 2.0, 2.0]          # 过程噪声协方差对角线 [x, y, vx, vy]
cv_P_init_diag = [10.0, 10.0, 5.0, 5.0]   # 初始状态协方差对角线

# ── CTRV 模型参数（5D: x, y, θ, v, ω）──
ctrv_Q_diag = [0.1, 0.1, 0.01, 2.0, 0.8]     # 过程噪声协方差对角线 [x, y, θ, v, ω]
ctrv_P_init_diag = [10.0, 10.0, 0.5, 5.0, 0.2]  # 初始状态协方差对角线

# ── 三特征融合参数 ──
alpha = 0.6    # 边缘特征融合权重
beta = 0.2     # 区域特征融合权重
gamma = 0.2    # 纹理特征融合权重

# ── 轨迹管理参数 ──
eta = 0.2                            # 轨迹质量衰减因子（用于加权平均置信度）
prune_len = 15                       # 轨迹保留最短长度，低于此值允许剪枝
tentative_min = 3                     # 轨迹确认最少连续命中帧数（提高以抑制假目标）
prune_T_min        = 3      # 噪声轨迹最短容忍帧数
prune_T_max        = 25     # 高质量目标最长容忍帧数
in_fov_timeout     = 8      # 视场内连续未命中帧数上限，超过则判定为假轨迹
prune_n_confirm    = 10     # 达到该帧数后 S_len 饱和为 1.0
prune_hist_len     = 30     # 历史队列最大长度（只保留最近30帧）
prune_conf_weights = (0.30, 0.30, 0.20, 0.20)  # S_len/S_assoc/S_motion/S_photo

# ── 视场内漏检断连重连参数 ──
enable_in_fov_reconnect = True          # 是否启用在视场内因漏检而断连的轨迹重连
in_fov_reconnect_min_hits = 3          # 断连前轨迹至少需要的命中帧数
in_fov_reconnect_max_gap = 10          # 断连允许的最大间隔帧数
in_fov_reconnect_min_conf = 0.35       # 断连前轨迹的最小平均置信度
in_fov_gate_growth = 0.30              # 每帧断连后搜索门的扩展比例
in_fov_gate_max_scale = 3.0            # 搜索门扩展的最大倍率（相对原始门大小）
in_fov_reconnect_score_thresh = 0.60   # 重连匹配的最低融合分数
fill_missing_with_prediction = True     # 是否用预测值填充漏检帧的轨迹位置

# ── 保守兜底关联与轨迹写入保护参数 ──
enable_fallback_association = False     # 是否启用保守兜底关联（软关联失败后的硬关联补救）
fallback_mahalanobis_gate = 5.99       # 兜底关联的马氏距离门限（2 DOF 95% 置信度）
fallback_min_cosine = 0.3              # 兜底关联的最小运动方向余弦相似度
fallback_max_velocity_scale = 1.0      # 兜底关联的最大速度偏差倍率
strict_one_update_per_frame = True      # 每条轨迹每帧最多接受一个观测更新
max_record_jump_scale = 1.2            # 轨迹记录位置的最大帧间跳跃倍率（超过则拒绝写入）


# ── 飞出视场后再入重连参数 ──
enable_exit_reconnect = True                # 是否启用飞出视场后重连
exit_reconnect_max_gap = 120                # 飞出后允许的最大断连帧数
exit_reconnect_memory_ttl = 150             # 飞出轨迹在记忆池中的最大存活帧数
exit_reconnect_min_old_hits = 5             # 旧轨迹参与重连所需的最少历史命中帧数
exit_reconnect_min_old_conf = 0.45          # 旧轨迹参与重连所需的最低置信度
exit_reconnect_border_margin = 80.0         # 视场边界余量（像素），用于判定飞出/飞入
exit_reconnect_confirm_hits = 3             # 重连确认所需的新轨迹连续命中帧数
exit_reconnect_score_margin = 0.08          # 重连匹配分数相对次优候选的最小领先幅度
exit_reconnect_confirm_high = 0.72          # 高置信度直接确认的重连分数阈值
exit_reconnect_defer_low = 0.55             # 低于此分数的候选直接丢弃
exit_reconnect_defer_frames = 3             # 延迟确认等待更多观测的帧数
exit_reconnect_photo_sigma_base = 0.50      # 光度一致性评分的基础 sigma（像素亮度标准差）
exit_reconnect_size_sigma_base = 0.40       # 尺寸一致性评分的基础 sigma（像素面积标准差）
exit_reconnect_photo_sigma_gap_gain = 0.005 # 光度 sigma 随断连帧数的增量系数
exit_reconnect_size_sigma_gap_gain = 0.003  # 尺寸 sigma 随断连帧数的增量系数
exit_reconnect_maneuver_gain = 0.40         # 机动预测不确定性增益（补偿视场外可能机动）
exit_reconnect_min_measured_ratio = 0.45    # 旧轨迹中真实测量帧的最低占比
exit_reconnect_reentry_score_min = 0.35     # 再入候选接受的最低重连分数
exit_reconnect_min_new_measured_ratio = 0.70  # 再入短轨迹中真实观测点的最低占比
exit_reconnect_max_observation_gap = 3        # 再入短轨迹相邻真实观测允许的最大帧间隔
exit_reconnect_max_required_speed = 30.0      # 沿视场外边界绕行所需速度上限（像素/帧）
exit_reconnect_speed_tolerance_scale = 1.25   # 相对新旧轨迹历史速度的容差倍率
exit_reconnect_unknown_min_hits = 3           # unknown 短轨迹参与重连所需真实观测数
exit_reconnect_unknown_score_threshold = 0.65 # unknown 短轨迹的最低重连分数
exit_reconnect_pending_max_stale_frames = 3   # 延迟确认等待新观测的最长帧数

# ── 马氏距离门控参数 ──
mahalanobis_gate = 9.21          # 2 DOF 卡方 99% 置信度
max_velocity = 30.0              # 帧间位移上限（像素/帧），超过视为物理不可能

# ── Sinkhorn + JPDA-lite 软关联参数 ──
sinkhorn_epsilon = 0.05          # 熵正则化强度（越小越接近硬匹配）
sinkhorn_iters = 50              # Sinkhorn-Knopp 迭代次数
jpda_hit_threshold = 0.3         # 轨迹命中阈值：ΣP[i,j] >= 此值视为命中
jpda_new_track_threshold = 0.3   # 新轨迹阈值：ΣP[i,j] < 此值的候选视为未关联

# ── 星图识别与姿态解算参数 ──

# 启用开关
enable_star_id = True               # 是否启用星图识别
enable_trajectory_association = True  # 是否启用轨迹关联；关闭时仅运行星图识别
enable_attitude = True             # 是否启用姿态解算（依赖星图识别结果）
enable_background_subtraction = True  # 是否启用姿态反馈的恒星背景建模与未知目标提取（替换CSV标签分离）

# 识别方法选择
star_id_method = "adaptive"          # "delaunay" | "descriptor" | "adaptive"

# 星表
star_catalog_file = "NavStar.dat" 
#star_catalog_file = "SyntheticStar.dat" 
star_catalog_mag_limit = 7.0        # 星等上限，只使用亮于此值的恒星

# ── Delaunay 星图识别参数 ──
star_id_angle_tolerance = 0.003     # 角距匹配容差 (rad)，约 0.17°
star_id_vote_threshold = 3          # 最小投票数，低于此值的匹配丢弃
star_id_min_matched = 4             # 最少识别星数，低于此值不进行姿态解算
star_id_use_triangle_edges = True   # 使用 Delaunay 三角边作为候选星对（加速）
star_id_delaunay_cache_enabled = True   # 缓存排序后的星对角距和编号
star_id_delaunay_cache_rebuild = True   # 本次启动忽略旧缓存并重建；成功运行一次后改回 False
star_id_delaunay_max_triangle_candidates = 300  # 三角形匹配阶段最多保留的候选三角形数
star_id_delaunay_match_pixel_threshold = 5.0   # 三角形顶点匹配的最大像素偏差阈值
star_id_delaunay_min_hypotheses = 20           # 投票阶段最少保留的姿态假设数

# ── 自适应星图识别参数 ──
star_id_adaptive_sparse_max_points = 10     # 稀疏模式判定阈值：检测点数 <= 此值使用 Delaunay 方法
star_id_adaptive_dense_min_points = 16      # 密集模式判定阈值：检测点数 >= 此值使用描述子方法
star_id_adaptive_min_matches = 4            # 自适应识别最少成功匹配星数
star_id_adaptive_max_median_residual = 3.0  # 匹配残差中位数上限（像素），超过则拒绝该姿态
star_id_adaptive_max_residual = 6.0         # 单星匹配残差上限（像素），超过则剔除
star_id_adaptive_min_coverage = 0.20        # 匹配星数占检测星数的最小比例
star_id_adaptive_fail_frames = 2            # 连续识别失败多少帧后触发降级回退
star_id_adaptive_fallback_enabled = True    # 是否启用降级回退（失败时尝试另一种方法）
star_id_adaptive_fallback_cooldown = 3      # 降级回退后的冷却帧数，期间不切回原方法
star_id_adaptive_delaunay_max_triangles = 60  # 密集场景下 Delaunay 方法使用的最大三角形数
star_id_adaptive_prefer_previous = True     # 是否优先使用上一帧成功的识别方法
star_id_adaptive_debug = False              # 打印自适应方法选择与切换的调试信息
star_id_projection_refinement_enabled = True  # 是否启用姿态重投影精化（在位姿附近匹配更多星）
star_id_projection_refinement_radius = 3.0  # 当前姿态重投影后的精匹配半径（像素）

# ── 描述子星图识别参数 ──
star_id_descriptor_k = 8            # 观测点描述子使用的近邻数
star_id_descriptor_catalog_neighbors = 16  # 全天星表为每颗星保存的较大邻域池
star_id_descriptor_min_support = 4  # 径向角距离部分匹配所需的最少支持数
star_id_descriptor_trimmed_matches = 6  # 计算描述子分数时最多保留的最佳匹配数
star_id_descriptor_radial_tolerance = 0.0015  # 中心-邻星角距离匹配门限（rad）
star_id_descriptor_pair_tolerance = 0.0030    # 两组候选对应的夹角一致性门限（rad）
star_id_descriptor_max_observations = 50  # 密集场景参与全天识别的最大高能点数
star_id_descriptor_top_k = 15       # 部分描述子匹配后保留的全天星表候选数
star_id_descriptor_ransac_iters = 500  # 一致性图最多验证的姿态三元组数
star_id_descriptor_residual = 5.0      # 球面角残差折算到像面的门限（像素）
star_id_descriptor_min_conf = 0.35     # 单星匹配最小置信度
star_id_descriptor_debug = False       # 输出部分匹配、兼容三元组和最佳内点数
star_id_descriptor_max_top_k = 30      # 模糊观测星动态放宽后的最大候选数
star_id_descriptor_candidate_score_margin = 0.15    # 候选星得分相对次优候选的最小领先幅度
star_id_descriptor_min_pair_support_ratio = 0.60   # 候选星对一致性检验中最低支持比例
star_id_descriptor_triple_min_separation = 0.002  # 三元组最小星间角（rad）
star_id_descriptor_triple_min_shape = 0.05        # 三元组最小非退化形状质量
star_id_descriptor_sampling_grid = (4, 4)   # 像面均匀采样的网格划分 (行, 列)
star_id_descriptor_max_per_cell = 4          # 每网格单元最多保留的高能观测点数

# 姿态解算
attitude_method = "svd"             # 求解方法: "svd", "davenport", "quest"
attitude_use_mag_weight = True      # 是否用星等加权（亮星权重更高）

# ── 姿态反馈恒星背景建模与未知目标提取参数 ──
background_subtraction_match_radius = 10.0    # 投影星表与检测点匹配半径（像素）
background_subtraction_missed_radius = 20.0   # 漏检恒星判定半径（像素，通常为 match_radius 的 2 倍）

# ── 姿态与帧间运动证据参数 ──
candidate_geometry_debug = False         # 打印姿态像面预测的简洁调试信息（默认关闭）
motion_separation_match_radius = 20.0    # 帧间点匹配半径（像素）
motion_separation_consensus_threshold = 5.0  # 实际位置与姿态预测位置的残差阈值（像素），超过此值判定为目标
max_gap_frames = 5                       # 帧间位移粗分离的最大容忍缺帧数，超过则冷重启
displacement_mad_multiplier = 5.0        # 位移阈值 = 中值 + N*MAD，越大越保守（越少判为目标）
candidate_vector_motion_debug = False    # 打印粗分离匹配数、背景运动、MAD 和异常数
candidate_vector_motion_min_matches = 3  # 估计二维背景运动所需的最少一对一匹配数
candidate_vector_motion_max_mad = 5.0    # 背景运动残差 MAD 上限（像素）；超过时保守地不输出异常目标
candidate_vector_motion_min_scale = 0.5  # 运动残差鲁棒尺度下限（像素），防止 MAD≈0 时放大亚像素抖动
candidate_star_reject_threshold = 0.75   # p_star 达到该值且无证据冲突时判为高置信恒星
candidate_target_pass_threshold = 0.60   # p_target 达到该值且无证据冲突时送入现有 target_cands
candidate_no_attitude_motion_threshold = 0.75   # 无可用星表时，目标放行所需的最小相对背景运动得分
candidate_no_attitude_temporal_threshold = 0.65 # 无可用星表时，目标放行所需的最小跨帧连续性得分
candidate_catalog_quality_min_matches = 4       # 允许“远离星表”支持目标所需的最少有效背景匹配数
candidate_catalog_quality_min_ratio = 0.30      # 有效恒星匹配数/当前检测点数下限
candidate_catalog_quality_max_median_residual = 5.0  # 有效背景匹配残差中位数上限（像素）
candidate_debug = False                  # 每帧打印三类候选数量和主要证据原因
pre_separation_min_track_hits = 5        # 只有持续命中足够多的轨迹才能在星图识别前剔除候选点
pre_separation_min_track_confidence = 0.60  # 只信任高置信度 active 轨迹，防止恒星伪轨迹吞掉星点
track_star_evidence_frames = 3            # 连续累积多帧恒星运动证据后才标记为恒星轨迹
track_nonstar_evidence_frames = 3         # 连续累积多帧非恒星运动证据后才允许目标保护
track_nonstar_residual_threshold = 8.0    # 轨迹观测偏离恒星姿态预测的非恒星证据阈值（像素）

# ── 候选时序过滤参数 ──
candidate_temporal_filter = True    # 启用候选点帧间时序一致性检查（过滤差分残影）
candidate_temporal_radius = 15.0    # 候选帧间匹配半径（像素，容忍目标自身运动）
candidate_trusted_radius = 20.0     # 信任通道半径（像素，容忍EKF初始速度未收敛的偏差；>20px的偏移视为独立噪声而非目标）

# ── 帧间三角拓扑背景匹配参数 ──
triangle_background_enabled = True       # 使用相邻帧三角拓扑/仿射内点作为辅助恒星证据
triangle_use_fusion_v2 = True            # 使用最小角、边长比、面积、宽高比融合特征
triangle_feat_weights = (0.35, 0.10, 0.55)       # 旧三特征形式，仅供 triangle_match 图像接口使用
triangle_feat_weights_v2 = (0.35, 0.25, 0.10, 0.30)  # V2 四特征权重: 最小角/边长比/面积/宽高比
triangle_match_epsilon = 0.18            # 三角融合特征相对差异门限
triangle_min_matched_triangles = 3       # 估计初始仿射所需的最少三角形对
triangle_ransac_threshold = 3.0          # 仿射 RANSAC 重投影阈值（像素）
triangle_use_kdtree_refine = True        # 旧图像差分接口是否使用 KDTree 精化
triangle_match_radius = 6.0              # 仿射预测点与当前点的一对一匹配半径
triangle_target_exclusion_radius = 12.0  # 已确认目标轨迹附近点不作为背景恒星内点
triangle_min_inliers = 6                 # 背景模型可信所需的最少唯一内点数
triangle_min_inlier_ratio = 0.65         # 背景模型可信所需的最小 RANSAC 内点率
triangle_min_edge_preserve = 0.55        # 最小 Delaunay 边保持率
triangle_max_median_residual = 3.0       # 最大背景内点残差中位数（像素）
triangle_star_evidence_frames = 3        # 连续命中三角背景多少帧后把轨迹标记为 star
triangle_debug = False                   # 输出匹配三角形、内点数、残差和不可用原因

# ── 评估参数 ──
dist_thresh = 12                     # 预测位置与真值的距离阈值（像素），用于判定命中/漏检
N_pixel_norm = 1448                  # 像素归一化因子，用于将像素误差折算到像面对角线比例
