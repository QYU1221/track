"""Readable X-style tables and figures, derived solely from committed V2 output.

Numeric display IDs never replace model labels. A measured path requires the
tracker's reliable association threshold; priors and ambiguous detections do
not become measured tracks. All presentation decisions stay outside inference.
"""
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import csv
import json
import hashlib
import math
import time

import cv2
import numpy as np
from ..numerical_policy import DISPLAY_CLASS_TIE_ATOL, DISPLAY_PROJECTION_MIN_Z, DISPLAY_SPAN_FLOOR
from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from ..camera_model import CameraModel
from ..geometry import quaternion_matrix
from ..geometry_errors import GeometryError
from ..gnomonic import GnomonicPlane
from .schema import FRAME_FIELDS, TRACK_FIELDS, TRAJECTORY_FIELDS, EVENT_FIELDS
from .rendering import COLOR_PALETTE, draw_track_path, save_image
from .formal_track_plots import save_formal_track_plots
from .progress import TerminalProgress


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
]


def _write_csv(path, fields, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def _classify(point):
    """Display the unique largest physical-source marginal, with ties unknown."""
    sources = [point.get(name) for name in ('P_Existing_Target', 'P_Birth_Target',
               'P_Known_Star', 'P_Unknown_Star', 'P_Clutter')]
    if any(value is None for value in sources):
        return '未知', [None, None, None]
    values = [sources[0] + sources[1], sources[2] + sources[3], sources[4]]
    maximum = max(values)
    winners = [i for i, v in enumerate(values) if abs(v - maximum) <= DISPLAY_CLASS_TIE_ATOL]
    return (['目标', '恒星', '杂波'][winners[0]] if len(winners) == 1 else '未知'), values


def _angular_samples(rows):
    valid = [r for r in rows if r.get('_direction') is not None]
    if not valid:
        return []
    ras = np.unwrap([math.atan2(r['_direction'][1], r['_direction'][0]) for r in valid])
    start = valid[0]['timestamp_s']
    return [dict(frame=r['frame'], timestamp_s=r['timestamp_s'],
                 elapsed_seconds=r['timestamp_s'] - start, source=r['source'],
                 ra_rad=float(ra), ra_deg=float(np.degrees(ra)),
                 dec_rad=math.asin(float(np.clip(r['_direction'][2], -1, 1))),
                 dec_deg=float(np.degrees(math.asin(float(np.clip(r['_direction'][2], -1, 1))))),
                 inertial=np.asarray(r['_direction'])) for r, ra in zip(valid, ras)]


class XStyleReporter:
    """Keep compact presentation records, rather than copies of joint histories."""

    def __init__(self, folder, packet, configuration, source_input_path=None, *,
                 verbose=True, log_every=10):
        self.folder = Path(folder)
        self.packet = packet
        self.configuration = configuration
        self.source_input_path = str(source_input_path or '')
        self.camera = CameraModel(packet.camera, **configuration['calibration'])
        self.frame_info = {f.frame: f for f in packet.frames}
        self.detections = defaultdict(list)
        for detection in packet.detections:
            self.detections[detection.frame].append(detection)
        self.frames = []
        self.points = {}
        self.tracks = {}
        self.events = []
        self.label_ids = {}
        self.reliable = configuration['lifecycle']['reliable_association_probability']
        png = configuration['reporting']
        self.png_mode = png['track_frame_png_mode']
        self.png_interval = png['track_frame_png_interval']
        self.png_scale = png['track_frame_png_scale']
        self.export_seconds = 0.
        for name in ('track_frames', 'formal_tracks', 'debug'):
            if name == 'track_frames' and self.png_mode == 'off':
                continue
            (self.folder / name).mkdir(parents=True, exist_ok=True)
        self.progress = TerminalProgress(self.folder, packet, configuration,
                                        self.source_input_path, verbose=verbose, log_every=log_every)

    def _track(self, label):
        if label not in self.tracks:
            self.label_ids.setdefault(label, len(self.label_ids) + 1)
            self.tracks[label] = dict(id=self.label_ids[label], label=label, rows=[],
                                      last={}, formal_frame=None, measured_frames=0)
        return self.tracks[label]

    def _prediction(self, marginal, frame):
        state, plane = marginal.get('State_Mean'), marginal.get('Tangent_Plane')
        if state is None or plane is None:
            return None, None
        try:
            direction, _ = GnomonicPlane(**plane).inverse(state[:2])
            pixel = None
            if frame.inertial_observation_available:
                projected = self.camera.camera_to_pixel(quaternion_matrix(frame.quaternion_xyzw).T @ direction)
                if self.packet.camera.contains(projected):
                    pixel = projected.tolist()
            return pixel, direction.tolist()
        except GeometryError:
            return None, None

    def consume(self, result, *, frame_started=None):
        """Consume one JSON-safe committed frame without modifying it."""
        started = time.perf_counter()
        number = result['frame']
        frame = self.frame_info[number]
        marginals = {r['Track_Label']: r for r in result['track_marginals']}
        for label, marginal in marginals.items():
            # Reserve a stable display ID for each model label in this export.
            self.label_ids.setdefault(label, len(self.label_ids) + 1)
            if (marginal.get('Track_Maturity') == 'CONFIRMED' or
                    marginal.get('Probability_Exists_And_Ever_Measured', 0.) >= self.reliable):
                self._track(label)
        geometry = {tuple(r['detection_key']): r for r in result.get('geometry', [])}
        point_by_id = {r['Detection_ID']: r for r in result['points']}
        measured = {}
        for detection in self.detections[number]:
            point = point_by_id.get(detection.detection_id, {})
            associations = point.get('Association_Marginals', ())
            reliable = [(source[2:], probability) for source, probability in associations
                        if source.startswith('T:') and probability >= self.reliable and source[2:] in marginals]
            label, association = reliable[0] if len(reliable) == 1 else (None, None)
            item = self._detection_row(detection, frame, point, geometry.get(detection.key, {}), label)
            if label is not None:
                track = self._track(label)
                item['Track_ID'] = track['id']
                item['Existence_Probability'] = marginals[label]['Existence_Probability']
                item['Online_Canonical_ID'] = marginals[label]['Canonical_Track_ID']
                item['Identity_Status'] = 'ONLINE_ASSOCIATION_ONLY'
                item['Identity_Posterior'] = association
                measured[label] = (detection, item, association)
            self.points[detection.key] = item

        for label, marginal in marginals.items():
            if label not in self.tracks:
                continue  # Low posterior support does not establish a displayed trajectory.
            track = self.tracks[label]
            track['last'] = dict(marginal)
            if marginal.get('Track_Maturity') == 'CONFIRMED' and track['formal_frame'] is None:
                track['formal_frame'] = number
            match = measured.get(label)
            if match:
                detection, item, association = match
                pixel = [detection.x_px, detection.y_px]
                direction = geometry.get(detection.key, {}).get('j2000_los')
                source = 'measured'
                if track['rows'] and number > track['rows'][-1]['frame'] + 1:
                    source = 'recovered'
            else:
                pixel, direction = self._prediction(marginal, frame)
                item, association, source = {}, None, 'predicted'
            status = marginal['Track_Lifecycle']
            track['measured_frames'] += int(match is not None)
            row = dict(track_id=track['id'], track_label=label, frame=number,
                       timestamp_s=frame.timestamp_s, x=pixel[0] if pixel else None,
                       y=pixel[1] if pixel else None, source=source, status=status,
                       classification_at_frame='目标' if marginal.get('Track_Maturity') == 'CONFIRMED' else '未确认',
                       target_posterior=item.get('Target_Posterior'), star_posterior=item.get('Star_Posterior'),
                       clutter_posterior=item.get('Clutter_Posterior'),
                       existence_probability=marginal['Existence_Probability'],
                       association_quality=association, geometry_valid=bool(pixel is not None),
                       decision_reason=marginal['Management_Reason'],
                       measured_frames=track['measured_frames'],
                       canonical_track_id=marginal.get('Canonical_Track_ID'),
                       measured_or_predicted=source,
                       posterior_update_kind=marginal.get('Measured_Or_Predicted'),
                       miss_association_probability=marginal.get('Miss_Association_Probability'),
                       _direction=direction)
            if direction is not None:
                row.update(ra_deg=math.degrees(math.atan2(direction[1], direction[0])),
                           dec_deg=math.degrees(math.asin(float(np.clip(direction[2], -1, 1)))))
            track['rows'].append(row)

        for event in result.get('lifecycle_events', []):
            self.events.append(dict(frame=event['frame'], timestamp=event['timestamp_s'],
                                    track_label=event['label'], event_type=event['kind'],
                                    event_reason=event['reason'], details=_json(event)))
        for event in result.get('candidate_retirement_events', []):
            self.events.append(dict(frame=number, timestamp=frame.timestamp_s, track_label=event['label'],
                                    event_type='track_retired', event_reason=event['reason'],
                                    value=event['existence_probability']))
            if event['label'] in self.tracks:
                self.tracks[event['label']]['last'].update(Track_Lifecycle='RETIRED', Management_Reason=event['reason'])
        self.frames.append(self._frame_row(result, frame))
        self._save_frame(frame)
        self.export_seconds += time.perf_counter() - started
        self.progress.frame(frame, self.frames[-1], result,
                            [self.points[d.key] for d in self.detections[number]],
                            time.perf_counter() - frame_started if frame_started is not None else None)

    def _detection_row(self, detection, frame, point, geometry, label):
        group, probabilities = _classify(point)
        valid = point.get('Geometry_Valid') is True
        skip = '' if valid else (detection.invalid_reason if not detection.valid else
                'invalid_attitude' if not frame.attitude_valid else 'not_in_observation_model')
        quaternion = frame.quaternion_xyzw or (None,) * 4
        los = geometry.get('camera_los') or (None,) * 3
        return dict(zip(DETECTION_FIELDS[:13], [detection.frame, detection.detection_id,
                    int(frame.attitude_valid), *quaternion, detection.x_px, detection.y_px,
                    dict(detection.features).get('Flux'), *los]),
                    Class_Group=group, Track_ID=None, Track_Label=label,
                    Class_At_Frame=group, Final_Class=group,
                    Geometry_Valid=int(bool(geometry.get('geometry_valid', False))),
                    Classification_Valid=int(valid), Classification_Skip_Reason=skip,
                    Target_Posterior=probabilities[0], Star_Posterior=probabilities[1],
                    Clutter_Posterior=probabilities[2], Point_Target_Probability=probabilities[0],
                    Point_Star_Probability=probabilities[1], Point_Clutter_Probability=probabilities[2],
                    Existence_Probability=None,
                    Decision_Reason='unique_maximum_source_posterior' if group != '未知' else 'unavailable_or_tied_source_posterior',
                    Timestamp_s=frame.timestamp_s, Online_Canonical_ID=None,
                    Final_Canonical_ID=None,
                    Identity_Status='UNRESOLVED' if valid else 'UNAVAILABLE', Identity_Posterior=None,
                    Known_Star_Probability=point.get('P_Known_Star'),
                    Unknown_Star_Probability=point.get('P_Unknown_Star'),
                    Existing_Target_Probability=point.get('P_Existing_Target'),
                    Birth_Target_Probability=point.get('P_Birth_Target'),
                    Association_Marginals=_json(point.get('Association_Marginals', ())),
                    Model_Status=point.get('Model_Status'), Inference_Mode=point.get('Inference_Mode'),
                    Approximation_Status=point.get('Approximation_Status'))

    def _frame_row(self, result, frame):
        marginals = [r for r in result['track_marginals'] if r['Track_Label'] in self.tracks]
        point_rows = [self.points[d.key] for d in self.detections[frame.frame]]
        confirmed = [r for r in marginals if r.get('Track_Maturity') == 'CONFIRMED']
        quaternion = frame.quaternion_xyzw or (None,) * 4
        row = dict(frame=frame.frame, source_frame=frame.frame, timestamp_s=frame.timestamp_s,
                   frame_status='missing_input_frame' if not frame.input_available else
                   'skipped_invalid_attitude' if not frame.attitude_valid else 'processed',
                   n_stars=sum(p['Class_Group'] == '恒星' for p in point_rows),
                   n_candidates=sum(p['Class_Group'] == '目标' for p in point_rows),
                   n_targets=len(confirmed), n_active_tracks=sum(r.get('Track_Lifecycle') == 'ACTIVE' for r in marginals),
                   n_lost_tracks=sum(r.get('Track_Lifecycle') == 'LOST' for r in marginals),
                   n_live_tracks=len(marginals), n_target_class_tracks=len(confirmed),
                   n_unknown_class_tracks=len(marginals) - len(confirmed),
                   n_exit_memory=sum(r.get('Track_Lifecycle') == 'DORMANT' for r in marginals),
                   n_reconnect_events=0,
                   attitude_source='public_frame_metadata', attitude_ok=int(frame.attitude_valid),
                   q_w=quaternion[3], q_x=quaternion[0], q_y=quaternion[1], q_z=quaternion[2],
                   catalog_available=int(bool(self.packet.catalog)), catalog_projection_available=int(frame.inertial_observation_available),
                   catalog_projection_count=result['diagnostics']['Projected_KnownStar_Count'],
                   input_detections=len(point_rows), classified_detections=sum(p['Classification_Valid'] for p in point_rows),
                   model_target_components=len(result['track_marginals']),
                   time_ms=result['diagnostics']['Computation_Time_s'] * 1000,
                   inference_mode=result.get('inference_mode'), model_status=result.get('status'))
        if frame.attitude_valid:
            x, y, z, w = quaternion
            row.update(roll_deg=math.degrees(math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))),
                       pitch_deg=math.degrees(math.asin(float(np.clip(2*(w*y-z*x), -1, 1)))),
                       yaw_deg=math.degrees(math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))))
        return row

    def _save_frame(self, frame):
        if self.png_mode == 'off':
            return
        if self.png_mode == 'interval' and len(self.frames) % self.png_interval != 0:
            return
        camera = self.packet.camera
        canvas = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
        for detection in self.detections[frame.frame]:
            if camera.contains((detection.x_px, detection.y_px)):
                cv2.circle(canvas, (int(round(detection.x_px)), int(round(detection.y_px))),
                           2, (180, 180, 180), -1, cv2.LINE_AA)
        cv2.putText(canvas, 'All measurements; colored lines: tracks', (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, 'Frame %d%s' % (frame.frame, '' if frame.inertial_observation_available else ' - observation unavailable'),
                    (12, 44), cv2.FONT_HERSHEY_SIMPLEX, .55, (180, 180, 180), 1, cv2.LINE_AA)
        for track in self.tracks.values():
            rows = track['rows']
            if rows and rows[-1]['frame'] == frame.frame:
                self._draw(canvas, track, rows)
        if self.png_scale != 1.0:
            canvas = cv2.resize(canvas, None, fx=self.png_scale, fy=self.png_scale,
                                interpolation=cv2.INTER_AREA)
        save_image(canvas, str(self.folder / 'track_frames' / ('frame_%04d.png' % frame.frame)))

    def _draw(self, canvas, track, rows, coordinates=None):
        if coordinates is None:
            coordinates = [(int(round(r['x'])), int(round(r['y'])))
                           if r.get('x') is not None and r.get('y') is not None else None for r in rows]
        draw_track_path(canvas, SimpleNamespace(id=track['id']), coordinates,
                        frames=[r['frame'] for r in rows], sources=[r['source'] for r in rows])
        if len(coordinates) == 1 and coordinates[0] is not None:
            cv2.circle(canvas, coordinates[0], 4, COLOR_PALETTE[track['id'] % len(COLOR_PALETTE)], -1)

    def _save_overlays(self, formal):
        camera = self.packet.camera
        image = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
        for track in formal:
            self._draw(image, track, track['rows'])
        if not formal:
            cv2.putText(image, 'No confirmed target tracks', (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        .7, (180, 180, 180), 1, cv2.LINE_AA)
        save_image(image, str(self.folder / 'tracks_overlay.png'))
        image = np.zeros((1000, 1600, 3), dtype=np.uint8)
        valid_frames = [self.frame_info[r['frame']] for r in self.frames
                        if self.frame_info[r['frame']].inertial_observation_available]
        paths = []
        if valid_frames:
            reference = valid_frames[len(valid_frames)//2]
            rotation = quaternion_matrix(reference.quaternion_xyzw)
            for track in formal:
                path = []
                for row in track['rows']:
                    direction = row.get('_direction')
                    if direction is None:
                        path.append(None)
                        continue
                    vector = rotation.T @ np.asarray(direction)
                    path.append(-vector[:2] / vector[2] if vector[2] > DISPLAY_PROJECTION_MIN_Z else None)
                paths.append((track, path))
        points = [p for _, path in paths for p in path if p is not None]
        if points:
            points = np.asarray(points)
            lower, upper = points.min(axis=0), points.max(axis=0)
            span = np.maximum(upper-lower, DISPLAY_SPAN_FLOOR)
            scale = min(1480/span[0], 880/span[1])
            center = (lower+upper)/2
            for track, path in paths:
                coords = [tuple(np.rint((p-center)*scale + [800, 500]).astype(int)) if p is not None else None for p in path]
                self._draw(image, track, track['rows'], coords)
        else:
            cv2.putText(image, 'No confirmed target tracks with valid inertial samples', (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (180, 180, 180), 1, cv2.LINE_AA)
        save_image(image, str(self.folder / 'tracks_inertial_overlay.png'))

    def finish(self, summary):
        started = time.perf_counter()
        self.progress.write('Exporting Excel/CSV, track overlays and formal-track diagnostics...',
                            console_message='正在保存汇总结果……')
        track_rows, trajectories, formal = [], [], []
        for track in self.tracks.values():
            rows, last = track['rows'], track['last']
            observed = [r for r in rows if r['source'] != 'predicted']
            frames = [r['frame'] for r in observed]
            count = len(observed)
            coverage = count/(frames[-1]-frames[0]+1) if frames else None
            gap = max((b-a-1 for a, b in zip(frames, frames[1:])), default=0)
            confirmed = track['formal_frame'] is not None
            for row in rows:
                row.update(observation_coverage=coverage, max_observation_gap=gap)
            trajectories.extend(rows)
            item = dict(track_id=track['id'], track_label=track['label'], status=last['Track_Lifecycle'],
                        maturity_status=last.get('Track_Maturity'), formal_confirmed=int(confirmed),
                        formal_confirmation_frame=track['formal_frame'], final_classification='目标' if confirmed else '未确认',
                        existence_probability=last.get('Existence_Probability'), target_confirmed_ever=int(confirmed),
                        target_confirmation_frame=track['formal_frame'], positions=len(rows),
                        first_frame=rows[0]['frame'], last_frame=rows[-1]['frame'], measured_frames=count,
                        observation_coverage=coverage, max_observation_gap=gap,
                        lifecycle_status=last.get('Track_Lifecycle'), formal_target=int(confirmed and last.get('Track_Lifecycle') == 'ACTIVE'),
                        ever_formal=int(confirmed), formal_entered_frame=track['formal_frame'],
                        last_decision_reason=last.get('Management_Reason'),
                        canonical_track_id=last.get('Canonical_Track_ID'))
            if observed:
                for key in ('target_posterior', 'star_posterior', 'clutter_posterior', 'association_quality'):
                    item[key] = observed[-1].get(key)
            track_rows.append(item)
            if confirmed:
                formal.append(track)
        for point in self.points.values():
            label = point.get('Track_Label')
            if label in self.tracks:
                point['Final_Class'] = '目标' if self.tracks[label]['formal_frame'] is not None else '未确认'
        _write_csv(self.folder/'frame_summary.csv', FRAME_FIELDS + ['timestamp_s', 'input_detections', 'classified_detections', 'model_target_components', 'inference_mode', 'model_status'], self.frames)
        _write_csv(self.folder/'track_summary.csv', TRACK_FIELDS + ['track_label', 'canonical_track_id'], track_rows)
        _write_csv(self.folder/'trajectory_points.csv', TRAJECTORY_FIELDS + ['timestamp_s', 'track_label', 'canonical_track_id', 'ra_deg', 'dec_deg'], trajectories)
        for event in self.events:
            track = self.tracks.get(event.get('track_label'))
            event['track_id'] = track['id'] if track else None
        _write_csv(self.folder/'track_events.csv', EVENT_FIELDS + ['track_label'], self.events)
        self._save_workbook()
        self._save_overlays(formal)
        plots = [SimpleNamespace(id=t['id'], formal_confirmed=True,
                                 angular_samples=_angular_samples(t['rows'])) for t in formal]
        paths = save_formal_track_plots(self.folder, plots)
        self.export_seconds += time.perf_counter()-started
        requested_ids = set(self.configuration.get('frames', self.frame_info))
        # A resumed report contains this chunk only, matching frame_summary.csv.
        detections = [d for d in self.packet.detections if d.frame in requested_ids]
        report = dict(input_file=self.source_input_path, total_frames=len(self.frames), input_frames=len(self.frames),
                      processed_frames=sum(r['frame_status'] == 'processed' for r in self.frames),
                      skipped_frames=sum(r['frame_status'] == 'skipped_invalid_attitude' for r in self.frames),
                      missing_input_frames=sum(r['frame_status'] == 'missing_input_frame' for r in self.frames),
                      total_detections=len(detections), classified_detections=sum(p['Classification_Valid'] for p in self.points.values()),
                      valid_detections=sum(d.valid for d in detections),
                      out_of_bounds_count=sum(not self.packet.camera.contains((d.x_px, d.y_px)) for d in detections),
                      total_tracks=len(self.tracks), formal_tracks=len(formal), confirmed_targets=len(formal),
                      likely_targets=0, unknown_tracks=len(self.tracks)-len(formal),
                      status=summary['status'], failed_frame=summary.get('failed_frame'),
                      requested_frames=summary['frames_read'], frames_committed=summary['frames_committed'],
                      output_style='X', model_version=summary['model_version'], reporting_elapsed_s=self.export_seconds,
                      report_scope='current committed chunk; unprocessed input rows are explicitly marked',
                      track_id_scope='numeric display ID for this export; track_label and canonical_track_id retain V2 identity',
                      classification_rule='unique maximum physical source posterior; ties/unavailable are unknown; formal tracks require V2 CONFIRMED maturity',
                      unsupported_reference_scores='blank; no replacement scores are manufactured',
                      reporting_source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                               for p in sorted(Path(__file__).parent.glob('*.py'))},
                      figures=['tracks_overlay.png', 'tracks_inertial_overlay.png'] + [str(Path(p).relative_to(self.folder)) for p in paths])
        (self.folder/'run_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        self.progress.finish(summary, report)
        return report

    def _save_workbook(self):
        workbook = Workbook(write_only=True)
        sheet = workbook.create_sheet('Sheet')
        sheet.freeze_panes = 'C2'
        header = []
        for index, field in enumerate(DETECTION_FIELDS, 1):
            width = min(36, max(12, len(field)+2))
            if field == 'Decision_Reason':
                width = 44
            sheet.column_dimensions[get_column_letter(index)].width = width
            cell = WriteOnlyCell(sheet, value=field)
            cell.font = Font(name='Calibri', size=11, bold=True, color='FFFFFF')
            cell.fill = PatternFill('solid', fgColor='203864')
            header.append(cell)
        sheet.append(header)
        row_count = 1
        requested = set(self.configuration.get('frames', self.frame_info))
        for detection in self.packet.detections:
            if detection.frame not in requested:
                continue
            row = self.points.get(detection.key)
            if row is None:
                frame = self.frame_info[detection.frame]
                row = self._detection_row(detection, frame, {}, {}, None)
                row.update(Classification_Skip_Reason='not_processed', Model_Status='NOT_PROCESSED')
            # Preserve identifiers and reasons as literal strings, even when they start with '='.
            cells = []
            for key in DETECTION_FIELDS:
                value = row.get(key)
                if isinstance(value, str):
                    cell = WriteOnlyCell(sheet, value=value)
                    cell.data_type = 's'
                    cells.append(cell)
                else:
                    cells.append(value)
            sheet.append(cells)
            row_count += 1
        sheet.auto_filter.ref = 'A1:%s%d' % (get_column_letter(len(DETECTION_FIELDS)), row_count)
        workbook.save(self.folder/'detection_results.xlsx')
        workbook.close()
