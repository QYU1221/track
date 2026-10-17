# 第二轮关联验证与候选改进

基于 `565c545` 继续，保留 CV、联合历史、PPP 新生和五源竞争骨架。
本轮分离三个问题：噪声是否可辨识、漏检是否应改变条件状态、两阶段截断是否过早。

## 实现

- `tracking.miss_update_mode` 默认 `retain`，候选模式为 `moment_matched`。
  在存在条件下，漏检状态密度正比于 `p(x) × [1 − PD × P(z∈视场|x)]`。
  新实现对仿射观测模型的窗口积分计算一、二阶矩，再投影为高斯；存在概率和漏检证据保持原公式。
  此更新可能增加协方差，也可能减小协方差并移动均值，不能等同于人工缩小门限。
  它只适用于目标条件状态；未知恒星、PPP 及投影域外退化策略保持原实现，不能称为全系统精确漏检递推。
- `inference.association_top_k` 默认 `null`（沿用 top_k），候选值 64。
  允许在出生解释展开之前保留更多关联候选，最终仍按真实联合权重裁剪至 top_k=16，不给任何身份分支加分。
  审计新增关联与出生两阶段合并后的候选条件截断质量，避免将截断从一阶段移到另一阶段误认为收益。
- 修复 Joseph 协方差表达式的浮点不对称：只将已验证输入计算出的对称表达式作 `(C+Cᵀ)/2`，不放宽输入校验，不裁剪特征值。
  真数据第 328 帧复现的不对称量约 1.9e-21，相对量约 1.4e-14，四个特征值均为正。
- 模型版本为 `v2.1-factorized-joint-ppp-5`，参数和版本均绑定检查点。

默认仍保留第一轮已验证设置。组合候选配置是
`space_target_detection/pipeline/profiles/hit_association_followup.json`，没有自动替换第一轮试验配置。

## 核验集与噪声验证

`tools/association_validation.py` 生成原始检测核验包、读取人工真值评估，以及生成合成真值回归。
真值是独立 sidecar，不传入跟踪器。评价包括未分配数、身份切换、重复标签、跨目标共享标签、重接以及非目标误关联。
没有人工确认的数据明确返回 `AWAITING_INDEPENDENT_REVIEW`，不把历史 Track_ID 转成真值。

本次核验包：`output/association_followup_20261017/review/`，包含 466 个检测点，覆盖早期出视场、交叉重接和后续片段。
`review.html` 可按帧浏览原始坐标；`review.csv` 中填写 kind（target/star/clutter）、object_id、reviewer，再把已确认行改为 VERIFIED。
不确定行保持 UNREVIEWED。manifest 保存输入哈希和全部 22 处大于 15 秒的时间断点。

联合 Q/R 网格在真实历史片段上把 R 推到搜索下界（原 R 的 0.05 倍），保留段平均 NIS 仍只有约 0.093，说明不能把这个最优点当成可信标定。
因此没有进一步调小生产 Q/R。独立合成标定能恢复设定的 q_a=3.1623e-8、R=2I，独立随机种子的验证平均 NIS 约 1.889，95% 区域覆盖约 94.9%。
这验证了校准工具的基本计算，但不提供真实数据的身份真值。

## 复现命令

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe tools/noise_validation.py 'output/20261016_211718_143421_点目标筛选哈工大.xlsx' output/noise_validation_new.json
.venv\Scripts\python.exe tools/association_validation.py prepare 'output/20261016_211718_143421_点目标筛选哈工大.xlsx/public_input' output/review_new
.venv\Scripts\python.exe tools/association_validation.py prepare-experiments 'output/20261016_211718_143421_点目标筛选哈工大.xlsx/public_input' space_target_detection/pipeline/profiles/hit_association_trial.json output/followup_new
.venv\Scripts\python.exe tools/association_validation.py synthetic output/followup_new output/synthetic_new --seed 1729
.venv\Scripts\python.exe tools/association_validation.py evaluate output/association_followup_20261017/review/review.csv output/association_followup_20261017/verified/control_run output/review_metrics_new.json
.\run.cmd --dataset 点目标筛选哈工大.xlsx --config space_target_detection/pipeline/profiles/hit_association_followup.json --frames 430
```

真实数据四组配置位于 `output/association_followup_20261017/{control,miss,search,combined}.json`。
control 为第一轮 q_a+预算方案，miss 仅启用漏检矩匹配，search 仅延迟关联截断，combined 同时启用。
首轮中止记录保留供追溯；修复对称性后的完整对照输出放在 `verified/` 下。

## 已确认的边界与验证证据

- 19 项单元/数值回归通过，包含解析半平面矩、相关二维窗口的独立蒙特卡洛比较、真实第 328 帧舍入案例、命中似然与存在证据不变、真值缺失/重复保护。
- 修复对称性后，控制组与第一轮 430 帧的所有保留分支完全一致，最大权重差为 3.33e-15。
- 两个随机种子、五类合成场景、四种配置共 40 次评估。交叉与视场内短时漏检未发生身份切换；机动场景在第二个种子出现 1 次身份切换和 22 个未可靠分配点，四种配置均未解决。
- 出视场折返场景各组均产生一次新身份，说明单一 CV 与当前在线身份保留不能解决所有出视场机动重识别。
- 大时间断点场景在第 20 帧按预期拒绝 32 秒传播，保留第 19 帧状态，不把数据断点当普通漏检跨接。
- 漏检矩匹配使更多轨迹保留在视场外，原休眠上限 8 在第 474 帧触发。继续评估使用 `*_capacity32.json`，只把实验组管理容量增加到 32；没有降低存在概率来强行腾位置。
  此容量只是硬停止保护，不参与后验计算。原控制与 search 组均未触及上限，因此它们与提高上限后的递推一致。
  组合试验配置也明确包含此实验容量；默认仍为 8。

`tools/finalize_followup_reports.py` 对完成的实验检查每个联合历史的观测所有权和组权重归一化。
同时为在显示字段修正前启动的实验更新轨迹 CSV 中的漏检状态说明，并记录修改前后哈希；不修改状态、概率、身份或观测账本。

## 最终四组结果与采用决定

四组均完成 430 帧；每个组的归一化与每个联合历史的观测唯一归属检查通过。

| 指标 | 第一轮方案 | 延迟关联截断 | 漏检矩匹配（容量32） | 两者组合（容量32） |
|---|---:|---:|---:|---:|
| 同一点两个目标标签概率均 >0.1 | 27 | 27 | 29 | 29 |
| 两阶段合并的候选条件截断质量跨组和：逐帧均值 | 0.016572 | 0.016569 | 0.016999 | 0.016999 |
| 目标分量峰值 | 25 | 25 | 35 | 35 |

结果不支持把两项候选策略升级为默认：延迟截断主要转移了截断位置，漏检矩匹配增加了失联轨迹保留量，当前片段上未减少歧义。
因此生产默认仍使用 retain 和原有阶段预算，Q/R 不再调小，休眠上限仍为 8。
保留经过测试的显式实验开关，便于补充独立真值后继续研究。
已默认合入的改进是 Joseph 数值对称性修复和更完整的审计；统一模型骨架不变。

完整对照：`output/association_followup_20261017/final_comparison.json`。
当前真实数据仍有 0 个人工核验目标点，不能给出真实身份准确率。合成结果仅覆盖所列场景与随机种子。
