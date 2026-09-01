"""High-level star-processing stage preserving the established call order."""

from pipeline import StarProcessingResult


class StarProcessingStage:
    """Coordinate existing recognizers without reimplementing their algorithms."""

    def __init__(self, adaptive_identifier, fixed_identifier,
                 catalog_projector, recognized_star_resolver,
                 runtime_state):
        self._adaptive_identifier = adaptive_identifier
        self._fixed_identifier = fixed_identifier
        self._catalog_projector = catalog_projector
        self._recognized_star_resolver = recognized_star_resolver
        self._runtime_state = runtime_state

    def process(self, star_points, triangles, meta, frame_idx, enabled,
                method, verbose=False):
        selected = None
        selected_method = None
        identification = None
        attitude_q = None
        attitude_R = None
        attitude_ok = False
        diagnostics = {
            'selected_method': '', 'primary_method': '', 'fallback_used': 0,
            'switch_reason': 'disabled_or_insufficient_points',
            'primary_time_ms': 0.0, 'fallback_time_ms': 0.0,
            'quality_score': 0.0, 'quality_matches': 0,
            'quality_median_residual': float('inf'),
            'quality_max_residual': float('inf'), 'quality_coverage': 0.0,
            'consecutive_failures': self._runtime_state.get(
                'consecutive_failures', 0),
        }

        if enabled and len(star_points) >= 3:
            if verbose:
                print('[frame %d] star_points for ID: %d' % (
                    frame_idx, len(star_points)))
            if method == 'adaptive':
                selected, diagnostics = self._adaptive_identifier(
                    star_points, triangles, meta, frame_idx)
            else:
                selected = self._fixed_identifier(
                    method, star_points, triangles, meta)
                diagnostics.update({
                    'selected_method': method,
                    'primary_method': method,
                    'switch_reason': 'fixed_method',
                    'primary_time_ms': selected['time_ms'],
                    'quality_score': selected['quality']['score'],
                    'quality_matches': selected['quality']['matches'],
                    'quality_median_residual': selected['quality'][
                        'median_residual'],
                    'quality_max_residual': selected['quality']['max_residual'],
                    'quality_coverage': selected['quality']['coverage'],
                })
            selected_method = selected['method']
            identification = selected['star_id_result']
            attitude_q = selected['attitude_q']
            attitude_R = selected['attitude_R']
            attitude_ok = selected['attitude_ok']

        displayed_method = selected_method
        if (selected_method and
                diagnostics.get('primary_method') == 'projection_tracking'):
            displayed_method = '%s+projection' % selected_method

        projected_catalog = (self._catalog_projector(
            attitude_R, selected_method) if attitude_ok else None)
        recognized_stars = None
        if attitude_ok:
            recognized_stars = (selected.get('recognized_stars')
                                if selected else None)
            if recognized_stars is None:
                recognized_stars = self._recognized_star_resolver(
                    identification, star_points, selected_method)

        return StarProcessingResult(
            identification=identification,
            attitude_quaternion=attitude_q,
            attitude_matrix=attitude_R,
            attitude_valid=attitude_ok,
            selected_method=selected_method,
            displayed_method=displayed_method,
            diagnostics=diagnostics,
            projected_catalog=projected_catalog,
            recognized_stars=recognized_stars,
        )
