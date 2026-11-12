use space_target_detection::{
    error::{Error, Result},
    production::{self, Options},
};
use std::{
    io::{self, Write},
    path::{Path, PathBuf},
};
fn help() {
    println!(concat!(
        "空间目标在线联合概率跟踪（原生 Rust）\n",
        "\n",
        "用法：space_target_detection --input FILE_OR_FOLDER [选项]\n",
        "\n",
        "  --input, -i PATH          CSV/XLSX 检测表或规范化输入目录\n",
        "  --dataset, -d NAME        datasets 目录中的文件名\n",
        "  --datasets-root PATH      数据集目录\n",
        "  --output, -o PATH         空输出目录；默认程序旁 output/ 下的时间戳子目录",
        "（本工程 rust/dist/output）\n",
        "  --frames, -n N            只处理前 N 帧\n",
        "  --config PATH             在线参数 JSON；省略使用默认配置\n",
        "  --metadata PATH           原始表相机、时间和字段映射 JSON\n",
        "  --frame-metadata PATH     帧级 CSV/XLSX（保留零检测帧）\n",
        "  --resume PATH             恢复 checkpoint\n",
        "  --checkpoint PATH         检查点输出已停用\n",
        "  --quiet, -q               关闭终端进度\n",
        "  --log-every N             进度帧间隔，默认 10\n",
        "  --list, -l                列出数据集\n",
        "  --help, -h                显示帮助",
    ));
}

fn scan(root: &Path) -> Result<Vec<PathBuf>> {
    if !root.is_dir() {
        return Ok(vec![]);
    }
    let mut files = std::fs::read_dir(root)
        .map_err(|e| Error::value(e.to_string()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.is_file()
                && p.extension().is_some_and(|e| {
                    ["csv", "xlsx"].contains(&e.to_string_lossy().to_lowercase().as_str())
                })
        })
        .collect::<Vec<_>>();
    files.sort();
    Ok(files)
}

fn positive_integer(value: &str, error_message: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .ok()
        .filter(|number| *number > 0)
        .ok_or_else(|| Error::value(error_message))
}

fn apply_option(
    options: &mut Options,
    datasets_root: &mut PathBuf,
    dataset: &mut Option<String>,
    key: &str,
    value: String,
) -> Result<()> {
    match key {
        "--input" | "-i" => options.input = value.into(),
        "--output" | "-o" => options.output = Some(value.into()),
        "--config" => options.config = Some(value.into()),
        "--metadata" => options.metadata = Some(value.into()),
        "--frame-metadata" => options.frame_metadata = Some(value.into()),
        "--resume" => options.resume = Some(value.into()),
        "--checkpoint" => options.checkpoint = Some(value.into()),
        "--datasets-root" => *datasets_root = value.into(),
        "--dataset" | "-d" => *dataset = Some(value),
        "--frames" | "-n" => {
            options.frame_limit = Some(positive_integer(
                &value,
                "end_frame / --frames must be a positive number of input frames",
            )?);
        }
        "--log-every" => {
            options.log_every =
                positive_integer(&value, "log_every / --log-every must be a positive integer")?;
        }
        _ => unreachable!(),
    }

    Ok(())
}

fn execute() -> Result<i32> {
    let mut opts = Options {
        input: PathBuf::new(),
        output: None,
        config: None,
        metadata: None,
        frame_metadata: None,
        resume: None,
        checkpoint: None,
        frame_limit: None,
        verbose: true,
        log_every: 10,
    };
    let mut root = production::datasets_root();
    let mut dataset = None;
    let mut list = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let (key, supplied) = arg
            .split_once('=')
            .map(|(k, v)| (k, Some(v.to_owned())))
            .unwrap_or((&arg, None));
        match key {
            "--help" | "-h" => {
                help();
                return Ok(0);
            }
            "--quiet" | "-q" if supplied.is_none() => opts.verbose = false,
            "--list" | "-l" if supplied.is_none() => list = true,
            "--input" | "-i" | "--output" | "-o" | "--config" | "--metadata"
            | "--frame-metadata" | "--resume" | "--checkpoint" | "--datasets-root"
            | "--dataset" | "-d" | "--frames" | "-n" | "--log-every" => {
                let value = supplied.or_else(|| args.next()).ok_or_else(|| {
                    Error::value(format!("argument {key}: expected one argument"))
                })?;
                apply_option(&mut opts, &mut root, &mut dataset, key, value)?;
            }
            _ => return Err(Error::value(format!("unrecognized arguments: {arg}"))),
        }
    }
    let datasets = scan(&root)?;
    if list {
        if datasets.is_empty() {
            println!("未找到数据集：{}", root.display());
        } else {
            for p in datasets {
                println!("{}", p.file_name().unwrap().to_string_lossy());
            }
        }
        return Ok(0);
    }
    if opts.input.as_os_str().is_empty() {
        if let Some(name) = dataset {
            let path = root.join(&name);
            if !datasets.contains(&path) {
                return Err(Error::value(format!("未找到数据集：{name}")));
            }
            opts.input = path;
        } else if !datasets.is_empty() {
            println!("可用数据集：");
            for (i, path) in datasets.iter().enumerate() {
                println!(
                    "  [{}] {}",
                    i + 1,
                    path.file_name().unwrap().to_string_lossy()
                );
            }
            loop {
                print!("请输入编号，或粘贴 CSV/XLSX 路径: ");
                io::stdout()
                    .flush()
                    .map_err(|e| Error::value(e.to_string()))?;
                let mut line = String::new();
                if io::stdin()
                    .read_line(&mut line)
                    .map_err(|e| Error::value(e.to_string()))?
                    == 0
                {
                    return Ok(0);
                }
                let value = line.trim();
                let path = Path::new(value);
                if path.is_file()
                    && path.extension().is_some_and(|e| {
                        ["csv", "xlsx"].contains(&e.to_string_lossy().to_lowercase().as_str())
                    })
                {
                    opts.input = path.into();
                    break;
                }
                if let Some(index) = value
                    .parse::<usize>()
                    .ok()
                    .filter(|i| *i >= 1 && *i <= datasets.len())
                {
                    opts.input = datasets[index - 1].clone();
                    break;
                }
                println!("请输入有效编号或文件路径。");
            }
        } else {
            return Err(Error::value(
                "请使用 --input 指定 CSV/XLSX 或规范化输入目录",
            ));
        }
    }
    if !opts.input.exists()
        || (!opts.input.is_dir()
            && !opts.input.extension().is_some_and(|e| {
                ["csv", "xlsx"].contains(&e.to_string_lossy().to_lowercase().as_str())
            }))
    {
        return Err(Error::value(format!(
            "输入必须是存在的 CSV/XLSX 或规范化输入目录：{}",
            opts.input.display()
        )));
    }
    let summary = production::process_sequence(&opts)?;
    if !summary["failed_frame"].is_null() {
        return Err(Error::value(format!(
            "V2 跟踪停止：{}",
            summary["failed_frame"]["error"].as_str().unwrap()
        )));
    }
    if opts.verbose {
        println!(
            "V2 联合跟踪：{}/{} 帧，{}；输出：{}",
            summary["frames_committed"],
            summary["frames_read"],
            summary["status"].as_str().unwrap(),
            summary["output_dir"].as_str().unwrap()
        );
    }
    Ok(0)
}

fn main() {
    space_target_detection::cancellation::install();
    match execute() {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!(
                "space_target_detection: error: {}",
                if e.message.is_empty() {
                    e.kind
                } else {
                    &e.message
                }
            );
            std::process::exit(if e.kind == "KeyboardInterrupt" {
                130
            } else if e.kind == "IllegalCharacterError" {
                1
            } else {
                2
            });
        }
    }
}
