"""Construct the online production model from a bound configuration."""
from dataclasses import dataclass
from ..camera_model import CameraModel
from ..exact.contracts import StarSpec,ObservationWindow,ClutterModel
from .model import TrackingConfig
from .background import BackgroundConfig
from .lifecycle import LifecycleConfig
from ..efficient.config import InferenceConfig


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


def build_model(packet, raw):
    stars=tuple(StarSpec(item['known_star_id'],item['p_detection'],tuple(map(tuple,item['measurement_covariance_px2'])))
        for item in raw['stars'])
    clutter=ClutterModel(raw['clutter']['expected_count'],ObservationWindow(tuple(raw['clutter']['bounds_px'])),
        raw['clutter']['parameter_provenance'])
    return OnlineModel(CameraModel(packet.camera,**raw['calibration']),packet.catalog,stars,clutter,
        TrackingConfig(**raw['tracking']),raw['initial_timestamp_s'],packet.time_scale,
        BackgroundConfig(**raw['background']),LifecycleConfig(**raw['lifecycle']),InferenceConfig(**raw['inference']))
