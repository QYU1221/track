"""Ownership of star-processing resources and cross-frame adaptive state."""

from dataclasses import dataclass, field


@dataclass
class StarProcessingRuntime:
    catalog_data: object = None
    delaunay_index: object = None
    delaunay_identifier: object = None
    descriptor_catalog: object = None
    descriptor_recognizer: object = None
    camera_matrix: object = None
    descriptor_result: object = None
    adaptive: dict = field(default_factory=lambda: {
        'previous_method': None,
        'previous_attitude_R': None,
        'consecutive_failures': 0,
        'last_fallback_frame': -10 ** 9,
    })

    def reset_temporal_state(self):
        self.descriptor_result = None
        self.adaptive.update({
            'previous_method': None,
            'previous_attitude_R': None,
            'consecutive_failures': 0,
            'last_fallback_frame': -10 ** 9,
        })
