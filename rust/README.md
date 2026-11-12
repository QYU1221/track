# 空间目标检测与跟踪：Rust 主流程

首次接触这个工程，可先阅读[从零认识这个 Rust 工程](QUICKSTART_ZH.md)，按运行示例、数据流、源码阅读顺序和小练习入门。

单一命令入口为 `src/main.rs`。`src/` 中的其余模块提供输入解析、配置与星表绑定、联合贝叶斯跟踪、断点恢复及 CSV/Excel/PNG 输出。运行不调用 Python。

## 编译与运行

在原工程根目录的 PowerShell 执行：

```powershell
./rust/cargo-local.ps1 build --release --offline
./rust/package-release.ps1 -SkipBuild
./rust/dist/space_target_detection.exe --input "D:/data/detections.xlsx" --frames 12
```

`--frames` 限制本次处理帧数，省略时处理全部选定帧。输出目录必须为空。运行不会生成 JSON 或 JSONL 文件。

使用 `--help` 查看配置、元数据、帧选择、数据集及进度参数。规范化输入目录需显式提供 `--config`。参数错误退出 2，Ctrl+C 在资源检查点取消并退出 130。

## 独立运行包

将 `dist/` 中的程序和 `resources/` 一起复制。随后在运行包目录执行：

```powershell
./space_target_detection.exe --input "D:/data/detections.xlsx" --output "D:/results/run_01" --frames 12
```

运行不需要 Python 或 Rust 编译环境。原始数据由调用者提供。`SPACE_TARGET_DATA_ROOT` 可指定资源目录；默认使用运行包自己的 `resources/`。可把输入表放在程序旁的 `datasets/`，默认结果写入程序旁的 `output/`（本工程为 `rust/dist/output/`）下的时间戳子目录；从 Cargo 编译目录运行时也写入 `rust/dist/output/`。也可通过 `--output` 指定空目录。

## 保留的工程文件

- `src/`：主入口及其依赖模块；`report_schema.json` 为输出列定义。
- `resources/`：主流程使用的配置、相机参数和星表。
- `Cargo.toml`、`Cargo.lock`、`build.rs`：构建及源码摘要生成。
- `cargo-local.ps1`、`.toolchain/`：本机隔离编译环境；已有 Rust 环境也可直接使用 Cargo。
- `package-release.ps1`、`dist/`：主程序运行包的生成与输出。
- `THIRD_PARTY_NOTICES_NUMPY.txt`：NumPy 兼容求和实现的许可说明。

主流程保留原 GH7 出生积分。极窄先验或近秩一投影仍可能与 Python 出现数值行为差异；不自动改用其他积分算法。PNG 的字体、抗锯齿和布局与原图像库存在差异。不承诺所有输入逐位等价。
