"""Construct the online production model from a bound configuration."""
from dataclasses import asdict,dataclass
from ..camera_model import CameraModel
from ..exact.contracts import StarSpec,ObservationWindow,ClutterModel
from .model import TrackingConfig
from .background import BackgroundConfig
from .lifecycle import LifecycleConfig
from ..efficient.config import InferenceConfig
from .input_evidence import packet_frame_digests


@dataclass(frozen=True)
class OnlineModel:
    camera: object
    catalog: tuple
    stars: tuple
    clutter: object
    config: TrackingConfig
    initial_timestamp_s: float
    time_scale: str
    background_config: BackgroundConfig
    lifecycle_config: LifecycleConfig
    inference_config: InferenceConfig
    input_frame_digests: dict


def validate_bound_input(packet,raw):
    """A compiled prior must retain the coordinate/time system it was bound to."""
    if 'bootstrap_definition' not in raw:
        return
    expected=dict(camera=asdict(packet.camera),time_scale=packet.time_scale)
    if raw.get('bound_input')!=expected:
        raise ValueError('bound snapshot camera/time scale differs from input; use an unbound profile to rebind')
    if tuple(raw['clutter']['bounds_px'])!=(0,packet.camera.width,0,packet.camera.height):
        raise ValueError('bound snapshot observation bounds differ from input camera')
    if packet.frames and raw['initial_timestamp_s']!=packet.frames[0].timestamp_s:
        raise ValueError('bound snapshot initial timestamp differs from input')


def build_model(packet, raw):
    validate_bound_input(packet,raw)
    stars=tuple(StarSpec(item['known_star_id'],item['p_detection'],tuple(map(tuple,item['measurement_covariance_px2'])))
        for item in raw['stars'])
    clutter=ClutterModel(raw['clutter']['expected_count'],ObservationWindow(tuple(raw['clutter']['bounds_px'])),
        raw['clutter']['parameter_provenance'])
    return OnlineModel(CameraModel(packet.camera,**raw['calibration']),packet.catalog,stars,clutter,
        TrackingConfig(**raw['tracking']),raw['initial_timestamp_s'],packet.time_scale,
        BackgroundConfig(**raw['background']),LifecycleConfig(**raw['lifecycle']),InferenceConfig(**raw['inference']),
        packet_frame_digests(packet))
