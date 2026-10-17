"""Column order retained from X; V2-specific columns are appended."""

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
 'inertial_sigma_px', 'inertial_moving', 'inertial_fixed']

EVENT_FIELDS = ['frame', 'timestamp', 'track_id', 'event_type', 'event_reason', 'related_track_id', 'value',
 'details']
