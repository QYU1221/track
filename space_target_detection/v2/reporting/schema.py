"""Compact exports and an explicit compatibility adapter for the old X column order."""

FRAME_FIELDS = ['frame', 'source_frame', 'frame_status', 'n_stars', 'n_tracked_targets_pre',
 'n_displacement_targets_pre', 'catalog_available', 'catalog_target_available',
 'catalog_matches', 'catalog_match_ratio', 'catalog_median_residual', 'n_candidates',
 'n_targets', 'n_active_tracks', 'n_lost_tracks', 'n_live_tracks', 'n_star_class_tracks',
 'n_target_class_tracks', 'n_likely_target_class_tracks', 'n_unknown_class_tracks',
 'n_exit_memory', 'n_reconnect_events', 'tangent_plane_available',
 'tangent_plane_valid_tracks', 'tangent_plane_valid_candidates', 'attitude_source',
 'catalog_projection_available', 'catalog_projection_count', 'attitude_ok', 'q_w', 'q_x',
 'q_y', 'q_z', 'roll_deg', 'pitch_deg', 'yaw_deg', 'time_ms']

TRACK_FIELDS = ['track_id', 'status', 'maturity_status', 'formal_confirmed', 'formal_confirmation_frame',
 'final_classification', 'target_posterior', 'star_posterior', 'clutter_posterior',
 'existence_probability', 'target_confirmed_ever', 'target_confirmation_frame', 'positions',
 'first_frame', 'last_frame', 'last_motion_score', 'last_attitude_residual', 'measured_frames',
 'observation_coverage', 'max_observation_gap', 'association_quality', 'track_confidence',
 'last_decision_reason', 'identity_evidence_frames', 'lifecycle_status', 'formal_target',
 'ever_formal', 'formal_entered_frame', 'formal_exit_frame', 'formal_exit_reason',
 'motion_posterior', 'formal_confidence', 'lost_since_frame', 'terminal_reason',
 'last_state_transition_reason']

TRAJECTORY_FIELDS = ['track_id', 'frame', 'x', 'y', 'source', 'status', 'classification_at_frame',
 'target_posterior', 'star_posterior', 'clutter_posterior', 'existence_probability',
 'motion_score', 'attitude_residual', 'association_quality', 'catalog_distance',
 'catalog_star', 'geometry_valid', 'measured_frames', 'observation_coverage',
 'max_observation_gap', 'track_confidence', 'decision_reason', 'inertial_observation_count',
 'inertial_speed_px_frame', 'inertial_fit_rms_px', 'inertial_stationary_rms_px',
 'inertial_sigma_px', 'inertial_moving', 'inertial_fixed',
 'posterior_update_kind', 'miss_association_probability']

EVENT_FIELDS = ['frame', 'timestamp', 'track_id', 'event_type', 'event_reason', 'related_track_id', 'value',
 'details']


DETECTION_FIELDS = [
    'Frame', 'Detection_ID', 'Attitude_Valid', 'q1', 'q2', 'q3', 'q4_scalar',
    'Image_X_px', 'Image_Y_px', 'Brightness', 'LOS_X', 'LOS_Y', 'LOS_Z',
    'Class_Group', 'Track_ID', 'Class_At_Frame', 'Final_Class', 'Geometry_Valid',
    'Classification_Valid', 'Classification_Skip_Reason', 'Target_Posterior',
    'Star_Posterior', 'Clutter_Posterior', 'Track_Confidence',
    'Existence_Probability', 'Point_Target_Probability', 'Point_Star_Probability',
    'Point_Clutter_Probability', 'Point_Observation_Quality', 'Decision_Reason',
    'Timestamp_s', 'Track_Label', 'Online_Canonical_ID', 'Final_Canonical_ID',
    'Identity_Status', 'Identity_Posterior', 'Known_Star_Probability',
    'Unknown_Star_Probability', 'Existing_Target_Probability',
    'Birth_Target_Probability', 'Association_Marginals', 'Model_Status',
    'Inference_Mode', 'Approximation_Status',
    'Association_Retained_Mass', 'Association_Support_Scope',
    'Association_PreTruncation_Support',
    'Star_Status', 'Star_Source_ID', 'Star_Evidence_Support',
    'Unknown_Star_Confirmation_Support',
]

# Canonical records contain only real values. Legacy aliases and empty columns
# are materialized at the export boundary, never stored once per observation.
_ALIASES = {
    'frame': {'source_frame': 'frame', 'n_target_class_tracks': 'n_targets'},
    'track': {'lifecycle_status': 'status', 'ever_formal': 'target_confirmed_ever',
              'target_confirmation_frame': 'formal_confirmation_frame',
              'formal_entered_frame': 'formal_confirmation_frame'},
    'trajectory': {},
    'event': {},
    'detection': {'Point_Target_Probability': 'Target_Posterior',
                  'Point_Star_Probability': 'Star_Posterior',
                  'Point_Clutter_Probability': 'Clutter_Posterior'},
}
_UNSUPPORTED = {
    'frame': {'n_tracked_targets_pre', 'n_displacement_targets_pre', 'catalog_target_available',
              'catalog_matches', 'catalog_match_ratio', 'catalog_median_residual',
              'n_star_class_tracks', 'n_likely_target_class_tracks', 'tangent_plane_available',
              'tangent_plane_valid_tracks', 'tangent_plane_valid_candidates'},
    'track': {'last_motion_score', 'last_attitude_residual', 'track_confidence',
              'identity_evidence_frames', 'motion_posterior', 'formal_confidence',
              'lost_since_frame', 'last_state_transition_reason'},
    'trajectory': {'motion_score', 'attitude_residual', 'catalog_distance', 'catalog_star',
                   'track_confidence', 'inertial_observation_count', 'inertial_speed_px_frame',
                   'inertial_fit_rms_px', 'inertial_stationary_rms_px', 'inertial_sigma_px',
                   'inertial_moving', 'inertial_fixed'},
    'event': {'related_track_id'},
    'detection': {'Track_Confidence', 'Point_Observation_Quality', 'Final_Canonical_ID'},
}
_LEGACY_FIELDS = {
    'frame': FRAME_FIELDS + ['timestamp_s', 'input_detections', 'classified_detections',
                            'model_target_components', 'inference_mode', 'model_status'],
    'track': TRACK_FIELDS + ['track_label', 'canonical_track_id', 'final_identity_reason',
                            'jump_check_status', 'jump_check_count', 'jump_count', 'jump_fraction',
                            'jump_covariance_sources'],
    'trajectory': TRAJECTORY_FIELDS + ['timestamp_s', 'track_label', 'canonical_track_id',
                                      'ra_deg', 'dec_deg', 'jump_covariance_source',
                                      'jump_mahalanobis_squared', 'jump_threshold_squared', 'jump_detected'],
    'event': EVENT_FIELDS + ['track_label'],
    'detection': DETECTION_FIELDS,
}


def table_fields(kind, format):
    if format == 'x_legacy':
        return list(_LEGACY_FIELDS[kind])
    if format != 'compact':
        raise ValueError('unknown table format: ' + str(format))
    return [name for name in _LEGACY_FIELDS[kind]
            if name not in _UNSUPPORTED[kind] and name not in _ALIASES[kind]]


def table_row(kind, row, format):
    if format == 'compact':
        return row
    if format != 'x_legacy':
        raise ValueError('unknown table format: ' + str(format))
    return dict(row, **{alias: row.get(canonical) for alias, canonical in _ALIASES[kind].items()})
