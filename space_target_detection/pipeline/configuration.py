"""One resolver for online user parameters and complete effective snapshots.

The shipped default.json plus the camera profile it selects (profiles/cameras/<camera_profile>.json)
is the only source of online engineering defaults.
Runtime configuration dataclasses validate values but do not supply fallbacks.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

DEFAULT_PATH = Path(__file__).with_name('profiles') / 'default.json'
CAMERA_DIR = DEFAULT_PATH.parent / 'cameras'


def available_camera_profiles():
    return tuple(sorted(path.stem for path in CAMERA_DIR.glob('*.json')))


def _apply_camera_profile(defaults, name):
    """Overlay the selected camera profile on the shipped defaults."""
    path = CAMERA_DIR / (str(name) + '.json')
    if not isinstance(name, str) or not name or path.parent != CAMERA_DIR or not path.is_file():
        raise ValueError('unknown camera_profile %r; available: %s' % (name, ', '.join(available_camera_profiles())))
    profile = json.loads(path.read_text(encoding='utf-8'))
    if set(profile) != {'input_metadata', 'calibration'} or set(profile['input_metadata']) != {'attitude_direction', 'camera'}:
        raise ValueError('camera profile must contain exactly input_metadata.{attitude_direction,camera} and calibration: ' + path.name)
    defaults['input_metadata'].update(profile['input_metadata'])
    defaults['calibration'] = profile['calibration']


def _validate_default_completeness(defaults):
    from dataclasses import fields
    from ..v2.tracking.model import TrackingConfig
    from ..v2.tracking.background import BackgroundConfig
    from ..v2.tracking.lifecycle import LifecycleConfig
    from ..v2.efficient.config import InferenceConfig
    from ..v2.camera_model import CameraModel
    from ..v2.data_io import NormalizeConfig
    from ..v2.exact.contracts import ClutterModel
    contracts = (
        ('tracking', TrackingConfig, set()),
        ('background', BackgroundConfig, set()),
        ('clutter', ClutterModel, {'window'}),
        ('lifecycle', LifecycleConfig, set()),
        ('inference', InferenceConfig, set()),
        ('calibration', CameraModel, {'camera'}),
        ('input_metadata', NormalizeConfig, set()),
    )
    for section, contract, excluded in contracts:
        required = {field.name for field in fields(contract)} - excluded
        missing = required - set(defaults.get(section, {}))
        if missing:
            raise ValueError('default.json must explicitly declare ' + section + ': ' + ', '.join(sorted(missing)))
        extra = set(defaults.get(section, {})) - required
        if extra:
            raise ValueError('default.json contains unused ' + section + ' fields: ' + ', '.join(sorted(extra)))


def resolve_configuration(raw):
    if not isinstance(raw, dict):
        raise ValueError('configuration must be a JSON object')
    defaults = json.loads(DEFAULT_PATH.read_text(encoding='utf-8'))
    _apply_camera_profile(defaults, raw.get('camera_profile', defaults['camera_profile']))
    _validate_default_completeness(defaults)
    if raw.get('tracker_engine', defaults['tracker_engine']) != 'factorized_joint_ppp':
        raise ValueError('only factorized_joint_ppp is supported by this runtime')
    if raw.get('configuration_schema', 'online-parameters-1') != 'online-parameters-1':
        raise ValueError('unsupported online configuration schema')
    bound = 'bootstrap_definition' in raw
    allowed = set(defaults) | {'frames', 'bootstrap_definition', 'profile_binding',
                              'numerical_policy', 'bound_input', 'initial_timestamp_s', 'stars'}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError('unsupported online configuration field: ' + ', '.join(sorted(unknown)))
    if bound and 'bootstrap' in raw:
        raise ValueError('configuration cannot be both unbound (bootstrap) and bound (bootstrap_definition)')
    if not bound:
        derived = set(raw) & {'initial_timestamp_s', 'stars'}
        if isinstance(raw.get('clutter'), dict) and 'bounds_px' in raw['clutter']:
            derived.add('clutter.bounds_px')
        if derived:
            raise ValueError('parameters derived from input belong only in a bound snapshot: ' + ', '.join(sorted(derived)))
    else:
        missing = {'initial_timestamp_s', 'stars', 'bound_input'} - set(raw)
        if not isinstance(raw.get('clutter'), dict) or 'bounds_px' not in raw['clutter']:
            missing.add('clutter.bounds_px')
        if missing:
            raise ValueError('bound snapshot is missing input-derived fields: ' + ', '.join(sorted(missing)))
    if 'bootstrap' in raw:
        if not isinstance(raw['bootstrap'], dict):
            raise ValueError('configuration object required: bootstrap')
        unknown_bootstrap = set(raw['bootstrap']) - set(defaults['bootstrap'])
        if unknown_bootstrap:
            raise ValueError('unsupported online bootstrap parameter: ' + ', '.join(sorted(unknown_bootstrap)))
    result = deepcopy(defaults)
    result.update(deepcopy(raw))
    if 'bootstrap' in raw:
        result['bootstrap'] = dict(defaults['bootstrap'], **deepcopy(raw['bootstrap']))
    elif bound:
        result.pop('bootstrap', None)  # Already bound to a public input packet.
    if 'catalog_path' not in raw:
        catalog = Path(defaults['catalog_path'])
        result['catalog_path'] = str((DEFAULT_PATH.parent / catalog).absolute())
    sections = ('tracking', 'background', 'clutter', 'lifecycle', 'inference',
                'undetected_population', 'calibration', 'geometry', 'reporting', 'input_metadata')
    for section in sections:
        supplied = raw.get(section, {})
        if not isinstance(supplied, dict):
            raise ValueError('configuration object required: ' + section)
        allowed_section = set(defaults[section])
        if bound and section == 'clutter':
            allowed_section.add('bounds_px')
        unknown = set(supplied) - allowed_section
        if unknown:
            raise ValueError('unsupported online parameter: ' + section + '.' + ', '.join(sorted(unknown)))
        result[section] = dict(deepcopy(defaults[section]), **deepcopy(supplied))
    camera = raw.get('input_metadata', {}).get('camera', {})
    if not isinstance(camera, dict) or set(camera) - set(defaults['input_metadata']['camera']):
        raise ValueError('unsupported input_metadata.camera parameter')
    result['input_metadata']['camera'] = dict(defaults['input_metadata']['camera'], **camera)
    angle = result['geometry']['max_tangent_angle_deg']
    if isinstance(angle, bool) or not math.isfinite(angle) or not 0 < angle < 90:
        raise ValueError('geometry.max_tangent_angle_deg must lie between 0 and 90')
    reporting = result['reporting']
    if reporting['table_format'] not in ('compact', 'x_legacy'):
        raise ValueError('reporting.table_format must be compact or x_legacy')
    if reporting['detail'] not in ('compact', 'full') or not isinstance(reporting['write_frame_json'], bool):
        raise ValueError('invalid reporting detail or write_frame_json')
    if reporting['track_frame_png_mode'] not in ('every', 'interval', 'off'):
        raise ValueError('track_frame_png_mode must be every, interval or off')
    if (isinstance(reporting['track_frame_png_interval'], bool)
            or not isinstance(reporting['track_frame_png_interval'], int)
            or reporting['track_frame_png_interval'] < 1):
        raise ValueError('positive integer track_frame_png_interval required')
    png_scale = reporting['track_frame_png_scale']
    if (isinstance(png_scale, bool) or not isinstance(png_scale, (int, float))
            or not math.isfinite(png_scale) or not 0 < png_scale <= 1):
        raise ValueError('track_frame_png_scale must lie in (0,1]')
    period = result['tracking']['survival_reference_dt_s']
    if isinstance(period, bool) or not isinstance(period, (int, float)) or not math.isfinite(period) or period <= 0:
        raise ValueError('online tracking requires positive survival_reference_dt_s')
    result['configuration_schema'] = 'online-parameters-1'
    from ..v2.numerical_policy import snapshot
    policy = json.loads(json.dumps(snapshot()))
    if 'numerical_policy' in raw and raw['numerical_policy'] != policy:
        raise ValueError('saved numerical policy differs from this code version; it is not a runtime tuning section')
    result['numerical_policy'] = policy
    return result
