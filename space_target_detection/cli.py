"""Command-line entry point for the online production tracker."""
import argparse
from pathlib import Path


def scan_datasets(root):
    folder=Path(root)
    return tuple(path for path in sorted(folder.iterdir()) if path.is_file() and path.suffix.lower() in ('.csv','.xlsx')) if folder.is_dir() else ()


def select_dataset_interactive(initial_dir):
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as error:
        raise RuntimeError('无法打开文件选择窗口，请使用 --input 指定数据文件路径。') from error

    folder = Path(initial_dir)
    if not folder.is_dir():
        folder = Path.cwd()
    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        selected = filedialog.askopenfilename(
            parent=root,
            title='选择 CSV/XLSX 数据文件',
            initialdir=str(folder.resolve()),
            filetypes=(
                ('CSV / Excel 数据文件', '*.csv *.xlsx'),
                ('CSV 文件', '*.csv'),
                ('Excel 文件', '*.xlsx'),
            ),
        )
    except tk.TclError as error:
        raise RuntimeError('无法打开文件选择窗口，请使用 --input 指定数据文件路径。') from error
    finally:
        if root is not None:
            root.destroy()

    if not selected:
        raise SystemExit(0)
    return Path(selected)


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False,description='空间目标在线联合概率跟踪')
    parser.add_argument('--input','-i',help='CSV/XLSX 检测表或规范化输入目录')
    parser.add_argument('--dataset','-d',help='datasets 目录中的文件名')
    parser.add_argument('--datasets-root',help='数据集目录，也是文件选择窗口的默认目录')
    parser.add_argument('--output','-o',help=r'输出目录；默认保存到 E:\实验数据\实验结果数据 下的时间戳子目录')
    parser.add_argument('--frames','-n',type=int,help='只处理前 N 帧')
    parser.add_argument('--quiet','-q',action='store_true',help='关闭终端进度')
    parser.add_argument('--log-every',type=int,default=10,help='终端进度帧间隔')
    parser.add_argument('--config',help='在线参数 JSON；省略时使用默认配置')
    parser.add_argument('--metadata',help='原始表的相机、时间及字段映射 JSON')
    parser.add_argument('--frame-metadata',help='保留零检测帧的帧级 CSV/XLSX')
    parser.add_argument('--resume',help='恢复在线检查点')
    parser.add_argument('--checkpoint',help='保存在线检查点')
    parser.add_argument('--list','-l',action='store_true',help='列出数据集')
    args=parser.parse_args()
    package=Path(__file__).absolute().parent
    datasets_root=Path(args.datasets_root) if args.datasets_root else package/'datasets'
    datasets=scan_datasets(datasets_root)
    if args.list:
        print('\n'.join(path.name for path in datasets) if datasets else '未找到数据集：'+str(datasets_root))
        return
    if args.input:input_path=Path(args.input)
    elif args.dataset:
        input_path=datasets_root/args.dataset
        if input_path not in datasets:parser.error('未找到数据集：'+args.dataset)
    else:
        try:input_path=select_dataset_interactive(datasets_root)
        except RuntimeError as error:parser.error(str(error))
        if not args.quiet:print('已选择数据文件：'+str(input_path))
    if not input_path.exists() or not (input_path.is_dir() or input_path.suffix.lower() in ('.csv','.xlsx')):
        parser.error('输入必须是存在的 CSV/XLSX 或规范化输入目录：'+str(input_path))
    from .pipeline.sequence import process_sequence
    try:
        result=process_sequence(input_path,output_dir=args.output,end_frame=args.frames,verbose=not args.quiet,
            dataset_name=input_path.name,config_path=args.config,metadata_path=args.metadata,
            frames_path=args.frame_metadata,resume_path=args.resume,checkpoint_path=args.checkpoint,log_every=args.log_every)
    except (ValueError,OSError,TypeError,KeyError) as error:parser.error(str(error))
    except KeyboardInterrupt:raise SystemExit(130)
    if result['failed_frame'] is not None:
        parser.error('V2 跟踪停止：'+result['failed_frame']['error'])
