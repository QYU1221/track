"""Core run-directory, track validation and reconnect-event reporting."""

import csv
import os
import time


def setup_output_dirs(base_output, dataset_name=None):
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(
        base_output, '%s_%s' % (timestamp, dataset_name)
        if dataset_name else timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_reconnect_events_csv(run_dir, reconnect_events):
    path = os.path.join(run_dir, 'reconnect_events.csv')
    fields = [
        'old_id', 'temp_id', 'exit_frame', 'entry_frame', 'reconnect_frame',
        'score', 'mode', 'exit_edge', 'entry_edge', 'gap',
        'outside_path_distance', 'required_speed', 'speed_limit',
        'new_measured_hits', 'new_measured_ratio', 'new_max_observation_gap',
    ]
    with open(path, 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for event in reconnect_events or []:
            row = dict(event)
            if row.get('gap') is None:
                try:
                    row['gap'] = int(row.get('entry_frame')) - int(
                        row.get('exit_frame'))
                except Exception:
                    row['gap'] = ''
            writer.writerow(row)


def validate_track_sequences(tracks):
    valid_sources = {
        'measured', 'predicted', 'recovered', 'reconnected',
        'exit_predicted', 'extrapolated_exit',
    }
    for track in tracks:
        track_id = track.get('id', 'unknown')
        positions = track.get('positions', []) or []
        frames = track.get('frames')
        sources = track.get('point_sources')
        if frames is None:
            print('Warning: track %s has no frames field' % track_id)
            continue
        if len(frames) != len(positions):
            print('Warning: track %s length mismatch: positions=%d frames=%d' %
                  (track_id, len(positions), len(frames)))
        if sources is None:
            track['point_sources'] = ['measured'] * len(positions)
            sources = track['point_sources']
        elif len(sources) < len(positions):
            print('Warning: track %s point_sources shorter than positions: %d < %d, auto filled' %
                  (track_id, len(sources), len(positions)))
            sources.extend(['measured'] * (len(positions) - len(sources)))
        elif len(sources) > len(positions):
            print('Warning: track %s point_sources longer than positions: %d > %d, truncated' %
                  (track_id, len(sources), len(positions)))
            del sources[len(positions):]
        last_frame, seen = None, set()
        for index, frame in enumerate(frames[:len(positions)]):
            if frame is None:
                continue
            try:
                frame = int(frame)
            except (TypeError, ValueError):
                print('Warning: track %s has invalid frame value at index %d: %r' %
                      (track_id, index, frame))
                continue
            if last_frame is not None and frame < last_frame:
                print('Warning: track %s frames are not non-decreasing near index %d: %s -> %s' %
                      (track_id, index, last_frame, frame))
            if frame in seen:
                print('Warning: track %s has duplicate frame %s' %
                      (track_id, frame))
            seen.add(frame)
            last_frame = frame
        for source in sources:
            if source not in valid_sources:
                print('Warning: track %s has unknown point source: %r' %
                      (track_id, source))
