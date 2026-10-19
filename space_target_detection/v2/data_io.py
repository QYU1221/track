"""Strict real-table normalization and public ObservationPacket persistence.

The inference loader opens only frames.csv, detections.csv, catalog.csv and
input_metadata.json. It never opens a truth file or the simulation manifest.
"""
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
import csv
import json
import numpy as np
from . import PUBLIC_INPUT_SCHEMA_VERSION
from .contracts import Camera, Detection, FrameInfo, KnownStar, ObservationPacket, checked_covariance
from .attitude import adapt_quaternion

FEATURES = ('Flux', 'Peak_Intensity', 'Area', 'SNR', 'Confidence',
            'Background_Mean', 'Background_Std', 'PSF_Width_X', 'PSF_Width_Y', 'Fit_Residual')
DEFAULT_MAPPING = {
    'Frame': ('Frame',), 'Timestamp': ('Timestamp', 'Timestamp_s'),
    'Detection_ID': ('Detection_ID',), 'x': ('x', 'Image_X_px', 'Pixel_X'),
    'y': ('y', 'Image_Y_px', 'Pixel_Y'), 'Sigma_X': ('Sigma_X',),
    'Sigma_Y': ('Sigma_Y',), 'Cov_XY': ('Cov_XY',),
    'Valid': ('Valid',), 'Invalid_Reason': ('Invalid_Reason',),
    'Attitude_Valid': ('Attitude_Valid',), 'Input_Available': ('Input_Available',),
    'q1': ('q1',), 'q2': ('q2',), 'q3': ('q3',), 'q4_scalar': ('q4_scalar',),
    **{name: (name,) for name in FEATURES},
    'Flux': ('Flux', 'Brightness'),
}
FORBIDDEN_PREFIXES = ('true_', 'truth_', 'object_id', 'true_track', 'canonical_track', 'track_id')
FRAME_FIELDS = ('Frame', 'Timestamp_s', 'Input_Available', 'Attitude_Valid', 'q1', 'q2', 'q3', 'q4_scalar')
DETECTION_FIELDS = ('Frame', 'Detection_ID', 'x', 'y', 'Sigma_X', 'Sigma_Y', 'Cov_XY',
                    'Covariance_Status', 'Valid', 'Invalid_Reason', *FEATURES)
CATALOG_FIELDS = ('KnownStar_ID', 'LOS_J2000_X', 'LOS_J2000_Y', 'LOS_J2000_Z')


def parse_frame(value: Any) -> int:
    """Accept non-negative integral frame numbers without truncation."""
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        raise ValueError('Frame must be a finite non-negative integer: %r' % value)
    if (not number.is_finite() or number < 0 or
            number != number.to_integral_value() or number > 2147483647):
        raise ValueError('Frame must be an integer in [0, 2147483647]: %r' % value)
    return int(number)


def read_table(path: Union[str, Path], max_rows: int) -> Tuple[tuple, List[Dict[str, Any]]]:
    """Read CSV/XLSX rows with a size limit and duplicate-header validation."""
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
        raise ValueError('max_rows must be a positive integer')
    path = Path(path)
    if path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = []
            for row in csv.reader(stream):
                if len(rows) > max_rows:
                    raise ValueError('table exceeds configured row limit')
                rows.append(row)
    elif path.suffix.lower() == '.xlsx':
        from openpyxl import load_workbook
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            rows = []
            for row in workbook.active.iter_rows(values_only=True):
                if len(rows) > max_rows:
                    raise ValueError('table exceeds configured row limit')
                rows.append(row)
        finally:
            workbook.close()
    else:
        raise ValueError('table must be CSV or XLSX')
    if not rows:
        raise ValueError('table must contain a header, even when empty')
    header = tuple(str(x).strip() if x is not None else '' for x in rows[0])
    if any(not name for name in header) or len({h.lower() for h in header}) != len(header):
        raise ValueError('table has blank or duplicate column names')
    records = []
    for row_number, row in enumerate(rows[1:], 2):
        if all(value is None or str(value).strip() == '' for value in row):
            continue
        if len(row) != len(header):
            raise ValueError('row %d has a different column count' % row_number)
        records.append(dict(zip(header, row)))
        if len(records) > max_rows:
            raise ValueError('table exceeds configured row limit')
    return header, records


def write_csv(path: Union[str, Path], fields, rows, *,
              encoding='utf-8', extrasaction='raise') -> None:
    with Path(path).open('w', encoding=encoding, newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction=extrasaction)
        writer.writeheader()
        writer.writerows(rows)


def number(value: Any, name: str, required: bool = True) -> Optional[float]:
    if isinstance(value, bool):
        raise ValueError(name + ' must be numeric, not boolean')
    if value is None or str(value).strip() == '':
        if required:
            raise ValueError(name + ' is required')
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(name + ' must be numeric') from error
    if not np.isfinite(result):
        raise ValueError(name + ' must be finite')
    return result


def flag(value: Any, name: str, default: Optional[bool] = None) -> bool:
    if value is None or str(value).strip() == '':
        if default is not None:
            return default
        raise ValueError(name + ' is required')
    values = {'1': True, '1.0': True, 'true': True, '0': False, '0.0': False, 'false': False}
    text = str(value).strip().lower()
    if text not in values:
        raise ValueError(name + ' must be 0/1 or true/false')
    return values[text]


@dataclass(frozen=True)
class NormalizeConfig:
    camera: Camera
    time_scale: str
    timestamp_unit: str
    frame_period_s: Optional[float]
    frame_zero: int
    timestamp_zero_s: float
    bounds_action: str
    column_mapping: dict
    max_rows: int
    attitude_direction: str

    def __post_init__(self):
        if not isinstance(self.time_scale, str) or not self.time_scale.strip() or self.timestamp_unit not in ('s', 'ms', 'ns', 'iso'):
            raise ValueError('declare time scale and timestamp unit s/ms/ns/iso')
        if self.frame_period_s is not None and (isinstance(self.frame_period_s, bool) or not np.isfinite(self.frame_period_s) or self.frame_period_s <= 0):
            raise ValueError('explicit frame period must be positive seconds')
        if (self.bounds_action not in ('reject', 'mark_invalid') or isinstance(self.max_rows, bool)
                or not isinstance(self.max_rows, int) or self.max_rows < 1):
            raise ValueError('invalid normalization policy')
        if not np.isfinite(self.timestamp_zero_s) or isinstance(self.frame_zero, bool) or not isinstance(self.frame_zero, int):
            raise ValueError('invalid timestamp/frame origin')
        if set(self.column_mapping) - set(DEFAULT_MAPPING):
            raise ValueError('unknown canonical field in column mapping')
        if self.attitude_direction not in ('camera_to_j2000', 'j2000_to_camera'):
            raise ValueError('declare attitude_direction camera_to_j2000 or j2000_to_camera')


def load_normalize_config(path: Union[str, Path]) -> NormalizeConfig:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    data['camera'] = Camera(**data['camera'])
    return NormalizeConfig(**data)


def mapped_rows(header: tuple, rows: List[dict], config: NormalizeConfig,
                 require_positions: bool = True) -> List[dict]:
    for name in header:
        if name.lower().startswith(FORBIDDEN_PREFIXES):
            raise ValueError('hidden truth/track field forbidden in inference input: ' + name)
    lowered = {name.lower(): name for name in header}
    mapping = {}
    for canonical, aliases in DEFAULT_MAPPING.items():
        override = config.column_mapping.get(canonical)
        aliases = ((override,) if isinstance(override, str) else tuple(override)) if override is not None else aliases
        matches = [lowered[a.lower()] for a in aliases if a.lower() in lowered]
        if len(matches) > 1:
            raise ValueError('ambiguous columns for ' + canonical)
        if matches:
            mapping[canonical] = matches[0]
    required = ('Frame', 'Detection_ID', 'x', 'y') if require_positions else ('Frame',)
    if any(name not in mapping for name in required):
        raise ValueError('missing mandatory columns: ' + ', '.join(required))
    return [{canonical: row[source] for canonical, source in mapping.items()} for row in rows]


def timestamp_seconds(value: Any, unit: str) -> Optional[float]:
    """ISO timestamps require a timezone and become UTC Unix seconds."""
    if value is None or str(value).strip() == '':
        return None
    if unit != 'iso':
        return number(value, 'Timestamp') * {'s': 1., 'ms': .001, 'ns': 1e-9}[unit]
    text = str(value).strip()
    if 'T' not in text and ' ' not in text:
        raise ValueError('ISO Timestamp requires a date, time and timezone')
    try:
        instant = datetime.fromisoformat(text[:-1]+'+00:00' if text.endswith('Z') else text)
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise ValueError('ISO Timestamp requires an explicit timezone')
        result = instant.timestamp()
    except (ValueError, OverflowError, OSError) as error:
        raise ValueError('invalid timezone-aware ISO Timestamp: %r' % value) from error
    return number(result, 'Timestamp')


def frame_from_row(row: Mapping[str, Any], config: NormalizeConfig, *, canonical_quaternion: bool = False) -> FrameInfo:
    frame = parse_frame(row.get('Frame'))
    timestamp = timestamp_seconds(row.get('Timestamp'), config.timestamp_unit)
    if timestamp is None:
        if config.frame_period_s is None:
            raise ValueError('Timestamp missing: supply an explicit frame_period_s metadata assumption')
        timestamp = config.timestamp_zero_s + (frame-config.frame_zero)*config.frame_period_s
    available = flag(row.get('Input_Available'), 'Input_Available', default=True)
    valid = flag(row.get('Attitude_Valid'), 'Attitude_Valid', default=False)
    quaternion = None
    if valid:
        values=tuple(number(row.get(k), k) for k in ('q1', 'q2', 'q3', 'q4_scalar'))
        # Public packets already contain canonical unit quaternions. Validate
        # them in FrameInfo without another rounding-changing normalization.
        quaternion = values if canonical_quaternion else adapt_quaternion(values,
            order='xyzw', direction=config.attitude_direction)
    return FrameInfo(frame, timestamp, available, valid, quaternion)


def detection_from_row(row: Mapping[str, Any], camera: Camera, bounds_action: str,
                       status: Optional[str] = None) -> Detection:
    frame = parse_frame(row.get('Frame'))
    identifier = str(row.get('Detection_ID', '')).strip()
    if row.get('Detection_ID') is None or not identifier:
        raise ValueError('Detection_ID missing')
    x, y = number(row.get('x'), 'x'), number(row.get('y'), 'y')
    sx, sy = (number(row.get(k), k, required=False) for k in ('Sigma_X', 'Sigma_Y'))
    cross = number(row.get('Cov_XY'), 'Cov_XY', required=False)
    if (sx is None) != (sy is None) or (sx is None and cross is not None):
        raise ValueError('Sigma_X/Y must both be provided with Cov_XY')
    covariance = None
    if sx is not None:
        if sx <= 0 or sy <= 0:
            raise ValueError('Sigma_X/Y must be positive')
        cross = 0. if cross is None else cross
        covariance = ((sx*sx, cross), (cross, sy*sy))
        checked_covariance(covariance)
        status = status or 'reported_pending_calibration'
    else:
        if status is not None and status != 'missing_pending_calibration':
            raise ValueError('declared covariance status disagrees with missing Sigma')
        status = 'missing_pending_calibration'
    features = tuple((name, value) for name in FEATURES
                     for value in (number(row.get(name), name, required=False),) if value is not None)
    valid = camera.contains((x, y))
    if not valid and bounds_action == 'reject':
        raise ValueError('out-of-bounds detection %s in Frame %s' % (identifier, frame))
    declared_valid = flag(row.get('Valid'), 'Valid', default=True)
    reason = ('out_of_bounds' if not valid else str(row.get('Invalid_Reason') or ''))
    if not declared_valid and not reason:
        raise ValueError('declared invalid detection requires Invalid_Reason')
    return Detection(frame, identifier, x, y, covariance, status, features,
                     valid=valid and declared_valid, invalid_reason=reason)


def _has_iso_timestamps(rows: List[dict]) -> bool:
    for row in rows:
        value = row.get('Timestamp')
        if value is None or isinstance(value, (bool, int, float)) or str(value).strip() == '':
            continue
        try:
            float(value)
        except (TypeError, ValueError):
            return True
    return False


def normalize_table(path: Union[str, Path], config: NormalizeConfig,
                     frames_path: Optional[Union[str, Path]] = None) -> Tuple[ObservationPacket, dict]:
    """Return a packet and diagnostics; no discovered truth or adjacent files.

    No Frame gaps are synthesized. Without a frames sidecar, frames with no
    detection rows are unknown, never counted as observed zero-detection frames.
    """
    header, original = read_table(path, config.max_rows)
    rows = mapped_rows(header, original, config)
    if config.timestamp_unit != 'iso' and _has_iso_timestamps(rows):
        # Timestamps are timezone-aware ISO text, which cannot be numeric s/ms/ns.
        config = replace(config, timestamp_unit='iso')
    frame_map = {}
    if frames_path is not None:
        frame_header, frame_rows = read_table(frames_path, config.max_rows)
        for row in mapped_rows(frame_header, frame_rows, config, require_positions=False):
            info = frame_from_row(row, config)
            if info.frame in frame_map:
                raise ValueError('duplicate Frame in frames sidecar')
            frame_map[info.frame] = info
    else:
        for row in rows:
            info = frame_from_row(row, config)
            if info.frame in frame_map and frame_map[info.frame] != info:
                raise ValueError('inconsistent timestamp/attitude within Frame %s' % info.frame)
            frame_map[info.frame] = info
    detections = tuple(detection_from_row(row, config.camera, config.bounds_action) for row in rows)
    if frames_path is not None:
        # Metadata provided in a detection row must agree with the sidecar.
        for row in rows:
            frame = parse_frame(row['Frame'])
            if frame not in frame_map:
                raise ValueError('detection Frame absent from frames sidecar')
            info = frame_map[frame]
            if timestamp_seconds(row.get('Timestamp'), config.timestamp_unit) is not None:
                if frame_from_row({**row, 'Attitude_Valid': 0}, config).timestamp_s != info.timestamp_s:
                    raise ValueError('timestamp conflicts with frames sidecar')
            if row.get('Input_Available') not in (None, ''):
                if flag(row['Input_Available'], 'Input_Available') != info.input_available:
                    raise ValueError('input availability conflicts with frames sidecar')
            if row.get('Attitude_Valid') not in (None, ''):
                if flag(row['Attitude_Valid'], 'Attitude_Valid') != info.attitude_valid:
                    raise ValueError('attitude validity conflicts with frames sidecar')
                if info.attitude_valid and adapt_quaternion(tuple(number(row.get(k), k) for k in ('q1','q2','q3','q4_scalar')),
                    order='xyzw', direction=config.attitude_direction) != info.quaternion_xyzw:
                    raise ValueError('quaternion conflicts with frames sidecar')
    packet = ObservationPacket(config.camera, config.time_scale,
                               tuple(frame_map[k] for k in sorted(frame_map)), detections)
    return packet, {'original_rows': len(original),
                    'invalid_bounds': sum(not config.camera.contains((d.x_px,d.y_px)) for d in detections),
                    'invalid_detections': sum(not d.valid for d in detections),
                    'covariance_missing': sum(d.covariance_px2 is None for d in detections),
                    'frames_without_detections_unknown': frames_path is None,
                    'frame_period_s_assumption': config.frame_period_s}


def write_packet(folder: Union[str, Path], packet: ObservationPacket) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    frame_rows = []
    for frame in packet.frames:
        row = dict(Frame=frame.frame, Timestamp_s=frame.timestamp_s,
                   Input_Available=int(frame.input_available), Attitude_Valid=int(frame.attitude_valid))
        row.update(dict(zip(('q1', 'q2', 'q3', 'q4_scalar'), frame.quaternion_xyzw or ('',)*4)))
        frame_rows.append(row)
    detection_rows = []
    for detection in packet.detections:
        covariance = detection.covariance_px2
        row = dict(Frame=detection.frame, Detection_ID=detection.detection_id, x=detection.x_px, y=detection.y_px,
                   Sigma_X=np.sqrt(covariance[0][0]) if covariance is not None else '',
                   Sigma_Y=np.sqrt(covariance[1][1]) if covariance is not None else '',
                   Cov_XY=covariance[0][1] if covariance is not None else '',
                   Covariance_Status=detection.covariance_status, Valid=int(detection.valid),
                   Invalid_Reason=detection.invalid_reason)
        row.update(dict(detection.features))
        detection_rows.append(row)
    write_csv(folder/'frames.csv', FRAME_FIELDS, frame_rows)
    write_csv(folder/'detections.csv', DETECTION_FIELDS, detection_rows)
    write_csv(folder/'catalog.csv', CATALOG_FIELDS,
              [dict(zip(CATALOG_FIELDS, (star.known_star_id, *star.direction_j2000))) for star in packet.catalog])
    (folder/'input_metadata.json').write_text(json.dumps({
        'model_version': PUBLIC_INPUT_SCHEMA_VERSION, 'mode': 'ANGLE_ONLY', 'camera': asdict(packet.camera),
        'time_scale': packet.time_scale, 'timestamp_unit': 's', 'origin': packet.origin,
        'quaternion': 'xyzw_camera_to_j2000', 'pixel_axes': 'camera_-X_-Y',
        'bounds': '[0,width) x [0,height)', 'measurement_error': 'D-12-A-independent-pixel',
    }, indent=2), encoding='utf-8')


def load_packet(folder: Union[str, Path], *, max_rows: int) -> ObservationPacket:
    """Inference-safe reader; explicitly opens four public files only."""
    folder = Path(folder)
    meta = json.loads((folder/'input_metadata.json').read_text(encoding='utf-8'))
    if meta['mode'] != 'ANGLE_ONLY' or meta['quaternion'] != 'xyzw_camera_to_j2000' or meta['pixel_axes'] != 'camera_-X_-Y' or meta['timestamp_unit'] != 's':
        raise ValueError('unsupported coordinate/time convention')
    if meta['model_version'] != PUBLIC_INPUT_SCHEMA_VERSION:
        raise ValueError('unsupported public input schema version')
    camera = Camera(**meta['camera'])
    _, frames = read_table(folder/'frames.csv', max_rows)
    # These values describe the fixed public-packet schema written above;
    # they are not an alternative set of user-tunable defaults.
    config = NormalizeConfig(camera=camera, time_scale=meta['time_scale'],
        timestamp_unit='s', frame_period_s=None, frame_zero=0,
        timestamp_zero_s=0., bounds_action='mark_invalid', column_mapping={},
        max_rows=max_rows, attitude_direction='camera_to_j2000')
    infos = tuple(frame_from_row({**row, 'Timestamp': row.get('Timestamp_s')}, config,
        canonical_quaternion=True) for row in frames)
    header, detections = read_table(folder/'detections.csv', max_rows)
    mapped_rows(header, [], config)  # Also reject any hidden truth columns.
    points = tuple(detection_from_row(row, camera, 'mark_invalid', row['Covariance_Status']) for row in detections)
    _, catalog = read_table(folder/'catalog.csv', max_rows)
    stars = tuple(KnownStar(row['KnownStar_ID'], tuple(number(row.get(k), k) for k in CATALOG_FIELDS[1:])) for row in catalog)
    return ObservationPacket(camera, meta['time_scale'], infos, points, stars, meta['origin'])
