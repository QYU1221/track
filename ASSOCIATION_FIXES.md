# 关联稳定性试验（2026-10-17）

修改前快照：`7faaa71`。这是本地 Git 提交；仓库未配置远程地址。

保留原有 CV 切平面状态、联合历史、五源竞争、PPP 新生和生命周期骨架。
没有添加独立重识别器，没有给预测项乘经验折扣，没有裁剪协方差。
连续时间 CV 的 F/Q 公式未变，新增测试验证 2+4+2 秒分步传播与 8 秒一次传播一致。

代码修复：

- 预测均值超出相机/畸变/切平面定义域时，保留状态并返回不可见源；批量全部不可见也不会对空协方差数组求解。其他数值错误仍抛出。
- 每帧写入 `inference_audit.jsonl`，包括关联阶段、先验组合及出生分支的截断质量。各数值是阶段条件量，跨组求和不等于全局后验误差界。
- 已在构图前做过有限门控，因此即便穷尽保留图，也不再宣称原始模型的遗漏质量已知。
- `Identity_Posterior` 填写在线检测—标签关联概率；`Final_Canonical_ID` 留空，在线关联不冒充最终重识别。
- 轨迹 CSV 增加 `posterior_update_kind` 和 `miss_association_probability`，可以辨别显示为预测点的后验是否实际包含模糊测量更新。
- 版本改为 `v2.1-factorized-joint-ppp-4`，旧模型检查点不会被静默续用。

试验配置：`space_target_detection/pipeline/profiles/hit_association_trial.json`。
只针对哈工大输入做敏感性试验，保留该数据的相机和时间元数据；不作为其他数据集的通用标定。
默认配置不变。试验把 q_a 从 3.753049352598986e-5 调整到 3.162277660168379e-8，R 仍为 2I 像素平方；top_k 从 4 到 16，max_candidates 从 32 到 256，max_local_hypotheses 从 128 到 512。

q_a 的选择来源是历史轨迹 1（147–156）、93（426–460）、134（470–496）、136（471–499）的连续测量片段。
排除了已知错误重连部分；每段后 35% 用于保留验证，前段选择网格最优值。数据不是独立身份真值，R 也没有被独立标定，因此这只是候选参数，不能据此声称真实识别率提高。
验证部分创新均方仍明显低于二维理想值 2，提示参数/输入平滑程度还有待独立标定。

复现：

直接从原始 Excel 运行候选配置（自动创建新的输出目录）：

```powershell
.\run.cmd --dataset 点目标筛选哈工大.xlsx --config space_target_detection/pipeline/profiles/hit_association_trial.json --frames 430
```

完整三组对照的配置生成和回归检查：

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe tools/association_trial.py 'output/20261016_211718_143421_点目标筛选哈工大.xlsx' output/new_trial
.venv\Scripts\python.exe -m space_target_detection.main --input 'output/20261016_211718_143421_点目标筛选哈工大.xlsx/public_input' --config output/new_trial/trial.json --output output/new_trial/trial_run --frames 430 --quiet
.venv\Scripts\python.exe tools/summarize_association_trial.py output/new_trial
```

将 trial 改为 baseline 或 noise_only 可复现相同框架下的另外两组。
对照基于当前 v2.1 骨架，并统一包含边界修复，不能与历史 v2.2 输出简单视为只有参数不同。
原始输出不覆盖。此次只验证前 430 个输入帧，后续跨小时数据断点不在本轮结论范围内。
`track_summary.csv` 中 target_posterior 等仍表示最后一次可靠检测的类别概率，当前轨迹存在概率应读取 existence_probability。

## 本次结果

三组均完成 430 帧，覆盖原先第 500 帧停止点；6 项回归测试通过。

| 指标 | 原参数 | 仅调整 q_a | q_a + 扩大预算 |
|---|---:|---:|---:|
| 同一点至少两个目标标签概率 > 0.1 的数量 | 42 | 34 | 27 |
| 出生阶段截断质量跨组和的逐帧均值 | 0.02643 | 0.00610 | 0.00201 |
| 关联阶段截断质量跨组和的逐帧均值 | 0.03005 | 0.02720 | 0.01461 |
| 单帧计算耗时中位数（秒） | 0.266 | 0.281 | 0.432 |
| 单帧计算耗时 99 分位（秒） | 0.623 | 0.529 | 3.009 |

这些是当前输入片段上的内部诊断，不是正确率。三组同时运行，耗时含资源争用影响。
结果位于 `output/association_trial_20261017/`，详细对照为 `comparison.json`，各组独立保存 Excel、轨迹、检查点和审计日志。
