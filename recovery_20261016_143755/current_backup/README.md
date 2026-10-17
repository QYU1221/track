# 空间目标在线联合概率跟踪

本包只包含当前运行系统：输入标准化、相机与观测几何、五类来源联合概率、持续泊松出生、联合关联、轨迹管理、检查点以及 X 风格报表。

## 运行

```powershell
.\run.ps1 --list
.\run.ps1 -d 点目标筛选前1040.xlsx
.\run.ps1 -i D:\path\detections.xlsx -o D:\path\output
```

常用选项：`--frames` 限制处理帧数，`--config` 指定在线参数，`--metadata` 覆盖原始表的相机与时间定义，`--checkpoint` 保存状态，`--resume` 恢复状态，`--quiet` 关闭终端进度。

Python 入口：

```python
from space_target_detection.pipeline import process_sequence
result = process_sequence('detections.xlsx', output_dir='output/run')
```

默认参数在 `pipeline/profiles/default.json`，参数说明在同目录 `README.md`。默认不生成逐帧 JSON；输出包含检测结果 Excel、轨迹/帧/事件 CSV、轨迹图、正式轨迹诊断图、完整生效参数、运行摘要和实时日志。

初始时间、观测范围和可见星列表由输入生成，写入 `effective_prior.json`。轨迹规范 ID 直接由持久标签产生，不再生成重复的 `final_identity_view.json`。

目录中的 `datasets` 是输入数据，`NavStar_8mag.dat` 是运行星表。测试、仿真、研究评估、示例、旧有限出生后端和开发命令已从运行包移除。

当前检查点格式为 `factorized-class-mixture-5`。旧格式检查点需要从原始输入重新运行。

固定源/运动源类别竞争试验：在运行命令中增加 `--config space_target_detection/pipeline/profiles/class_mixture_trial.json`。该配置保留同一物体的两种类别解释，并允许撤销当前目标确认。默认关闭此试验；详细语义与输出字段见 `pipeline/profiles/README.md`。
