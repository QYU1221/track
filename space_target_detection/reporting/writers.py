"""Stable public result-writing API backed by the reporting domain."""


def setup_output_dirs(base_output, dataset_name=None):
    from .core import setup_output_dirs as implementation
    return implementation(base_output, dataset_name)


def save_reconnect_events(run_dir, reconnect_events):
    from .core import save_reconnect_events_csv
    return save_reconnect_events_csv(run_dir, reconnect_events)


def validate_track_sequences(tracks):
    from .core import validate_track_sequences as implementation
    return implementation(tracks)


def save_frame_statistics(run_dir, rows):
    from .artifacts import save_frame_statistics as implementation
    return implementation(run_dir, rows)


def save_attitude(run_dir, rows, input_dir=None):
    from .artifacts import save_attitude as implementation
    return implementation(run_dir, rows, input_dir)


def save_tracks(run_dir, tracks):
    from .artifacts import save_tracks as implementation
    return implementation(run_dir, tracks)


def save_track_events(run_dir, tracks):
    from .artifacts import save_track_events as implementation
    return implementation(run_dir, tracks)


def save_track_overlay(run_dir, tracks):
    from .artifacts import save_track_overlay as implementation
    return implementation(run_dir, tracks)


def save_evaluation(run_dir, star_rows, target_rows, star_removal_rows=None):
    from .artifacts import save_evaluation as implementation
    return implementation(run_dir, star_rows, target_rows, star_removal_rows)
