"""Public sequence entry point for the V2 joint tracker."""
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import shutil


def process_sequence(input_path, output_dir=None, end_frame=None,
                     verbose=True, dataset_name=None, *, config_path=None,
                     metadata_path=None, frames_path=None, resume_path=None,
                     checkpoint_path=None, log_every=10):
    """Return V2 output; raw tables default to the saved default profile.

    ``end_frame`` retains the public API's meaning: number of input frames.
    A limited run saves a checkpoint when additional configured frames remain.
    No prior, birth, catalog or classification is inferred from previous
    outputs.
    """
    from ..v2.reporting.progress import validate_log_every
    validate_log_every(log_every)
    automatic_profile = config_path is None
    if automatic_profile:
        if Path(input_path).is_file() and Path(input_path).suffix.lower() in ('.csv', '.xlsx'):
            from .default_profile import default_profile_path
            config_path = default_profile_path()
        else:
            raise ValueError('V2 公共输入目录需要 --config / config_path 指定先验 JSON')
    if end_frame is not None and (isinstance(end_frame, bool)
                                 or not isinstance(end_frame, int) or end_frame <= 0):
        raise ValueError('end_frame / --frames must be a positive number of input frames')
    config_path = Path(config_path).absolute()
    raw = json.loads(config_path.read_text(encoding='utf-8-sig'))
    if not isinstance(raw, dict):
        raise ValueError('V2 prior configuration must be a JSON object')
    from .configuration import resolve_configuration
    raw = resolve_configuration(raw)
    input_path = Path(input_path).absolute()
    if not input_path.exists():
        raise FileNotFoundError('V2 input does not exist: ' + str(input_path))

    from ..v2.data_io import load_packet, load_normalize_config, normalize_table, write_packet
    from ..v2.tracking.online_io import run_online_sequence

    normalization = None
    if verbose:
        print('读取输入：%s' % input_path, flush=True)
    if input_path.is_dir():
        if metadata_path is not None or frames_path is not None:
            raise ValueError('normalized public input already contains metadata and frames')
        packet = load_packet(input_path, max_rows=raw['input_metadata']['max_rows'])
    else:
        if input_path.suffix.lower() not in ('.csv', '.xlsx'):
            raise ValueError('V2 input must be a public input folder or CSV/XLSX table')
        if metadata_path is not None:
            normalization_config = load_normalize_config(metadata_path)
        else:
            from ..v2.data_io import NormalizeConfig
            from ..v2.contracts import Camera
            data = dict(raw['input_metadata'])
            data['camera'] = Camera(**data['camera'])
            normalization_config = NormalizeConfig(**data)
        from dataclasses import asdict
        raw['input_metadata'] = asdict(normalization_config)
        if verbose:
            print('正在规范化检测表和外部姿态……', flush=True)
        packet, normalization = normalize_table(input_path, normalization_config, frames_path)

    if verbose:
        print('输入读取完成：%d 帧，%d 个检测点' % (len(packet.frames), len(packet.detections)), flush=True)

    effective = dict(raw)
    bound_catalog = None
    if 'bootstrap' in raw:
        from .default_profile import bind_profile
        if verbose:
            print('正在绑定参数配置和可见星表……', flush=True)
        effective, bound_catalog = bind_profile(packet, raw, config_path)
        packet = replace(packet, catalog=bound_catalog)
        if verbose:
            period = raw.get('input_metadata', {}).get('frame_period_s')
            timing = '（名义帧周期 %.3g 秒）' % period if period is not None else ''
            print('使用默认参数配置：%s%s' % (config_path, timing), flush=True)
    if 'frames' in raw:
        requested = raw['frames']
        selected = tuple(f for f in packet.frames if f.frame in requested)
        if len(set(requested)) != len(requested) or set(requested) != {f.frame for f in selected}:
            raise ValueError('requested frames missing or repeated')
    else:
        selected = packet.frames
    if not selected:
        raise ValueError('no requested frames')
    limited = False
    if end_frame is not None:
        if resume_path is not None:
            committed = json.loads(Path(resume_path).read_text(encoding='utf-8'))['snapshot']['frame']
            if committed is not None:
                selected = tuple(f for f in selected if f.frame > committed)
        if not selected:
            raise ValueError('no frames after restored checkpoint')
        limited = len(selected) > end_frame
        effective['frames'] = [f.frame for f in selected[:end_frame]]

    if output_dir is None:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        name = Path(dataset_name or input_path.name).name
        folder = Path(__file__).absolute().parents[2] / 'output' / (stamp + '_' + name)
    else:
        folder = Path(output_dir).absolute()
    if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
        raise FileExistsError('V2 tracking output must be an empty directory: ' + str(folder))
    if limited and checkpoint_path is None:
        checkpoint_path = folder / 'checkpoint.json'
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        raise FileExistsError('checkpoint is immutable; choose a new path')

    folder.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.v2_entry_', dir=folder.parent) as temporary:
        staging = Path(temporary)
        if normalization is not None:
            public_input = staging / 'public_input'
            write_packet(public_input, packet)
        else:
            public_input = input_path
        if effective != raw:
            # Relative catalog paths remain relative to the user's prior file.
            if effective.get('catalog_path'):
                catalog = Path(effective['catalog_path'])
                effective['catalog_path'] = str(catalog if catalog.is_absolute()
                                                else config_path.parent / catalog)
        report = run_online_sequence(packet, effective, folder, resume_path, checkpoint_path,
                                     public_input_path=public_input, source_input_path=input_path,
                                     catalog=bound_catalog, verbose=verbose,log_every=log_every)
        if normalization is not None:
            shutil.copytree(public_input, folder / 'public_input')
            (folder / 'normalization_report.json').write_text(
                json.dumps(normalization, indent=2, allow_nan=False), encoding='utf-8')
    entrypoint = {
        'backend': 'v2', 'algorithm': 'V2 joint Bayesian tracking',
        'input_path': str(input_path), 'config_path': str(config_path),
        'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'metadata_path': str(Path(metadata_path).absolute()) if metadata_path is not None else None,
        'frames_path': str(Path(frames_path).absolute()) if frames_path is not None else None,
        'dataset_name': dataset_name, 'frame_limit': end_frame, 'output_dir': str(folder),
        'configuration_selection': 'automatic_default' if automatic_profile else 'explicit',
        'profile_id': raw.get('profile_id'),
        'execution': {'verbose': verbose, 'log_every': log_every},
    }
    if input_path.is_file():
        entrypoint['source_sha256'] = hashlib.sha256(input_path.read_bytes()).hexdigest()
    (folder / 'entrypoint.json').write_text(json.dumps(entrypoint, indent=2), encoding='utf-8')
    if verbose:
        print('V2 联合跟踪：%d/%d 帧，%s；输出：%s' % (
            report['frames_committed'], report['frames_read'], report['status'], folder), flush=True)
    return dict(report, output_dir=str(folder))
