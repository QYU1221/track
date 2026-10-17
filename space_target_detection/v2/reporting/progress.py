"""Live terminal diagnostics derived from public input and committed results."""
from collections import Counter
from pathlib import Path
import sys
import time


def validate_log_every(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError('log_every / --log-every must be a positive integer')


class TerminalProgress:
    def __init__(self, folder, packet, configuration, source_input_path, *,
                 verbose=True, log_every=10):
        validate_log_every(log_every)
        self.verbose = verbose
        self.log_every = log_every
        self.path = Path(folder) / 'debug' / 'terminal_output.log'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text('', encoding='utf-8')
        self.started = time.perf_counter()
        self.committed = 0
        self.previous_state = None
        selected = set(configuration.get('frames', (f.frame for f in packet.frames)))
        frames = [f for f in packet.frames if f.frame in selected]
        detections = [d for d in packet.detections if d.frame in selected]
        self.total = len(frames)
        self.write('Input: %d frames, %d detections (external attitude)' %
                   (self.total, len(detections)), console_message='开始处理：%d 帧' % self.total)
        self.write('Source: %s' % source_input_path, console=False)
        self.write('Output: %s' % Path(folder).absolute(), console=False)
        self.write('Engine: %s; inference=%s' % (
            configuration.get('tracker_engine', 'factorized_joint_ppp'),
            'APPROXIMATE_FAST'), console=False)
        camera = packet.camera
        self.write('Input diagnostics: camera=%dx%d valid_detections=%d invalid_detections=%d '
                   'out_of_bounds=%d invalid_attitude_frames=%d missing_input_frames=%d' % (
                       camera.width, camera.height, sum(d.valid for d in detections),
                       sum(not d.valid for d in detections),
                       sum(not camera.contains((d.x_px, d.y_px)) for d in detections),
                       sum(not f.attitude_valid for f in frames),
                       sum(not f.input_available for f in frames)), console=False)
        reasons = Counter(d.invalid_reason or 'unspecified' for d in detections if not d.valid)
        if reasons:
            self.write('Invalid detection reasons: ' + self._counts(reasons), console=False)
        self.write('Progress: console every %d frames and on validity/track-count changes or invalid frames; '
                   'log saves every frame. pts=input, valid=modeled, cat=known-star priors, '
                   'cand=target-class points, tracks=displayed tracks, confirmed=confirmed tracks, '
                   'components=target model components, bg=unknown-star components, '
                   'step=tracking time, wall=step plus frame export time' % log_every, console=False)

    @staticmethod
    def _counts(values):
        return ', '.join('%s=%s' % item for item in sorted(values.items()))

    def write(self, message, *, console=True, console_message=None):
        # Closing each append makes the last message readable during the run,
        # including if inference or artifact export subsequently fails.
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(message + '\n')
        if self.verbose and console:
            print(message if console_message is None else console_message, file=sys.stderr, flush=True)

    def frame(self, frame, row, result, point_rows=(), wall_seconds=None):
        self.committed += 1
        diagnostics = result['diagnostics']
        counts = (row['n_candidates'], row['n_live_tracks'], row['n_targets'])
        events = result.get('lifecycle_events', ())
        retired = result.get('candidate_retirement_events', ())
        state = (frame.inertial_observation_available, row['n_live_tracks'])
        interesting = state != self.previous_state or not frame.inertial_observation_available
        console = (self.committed == 1 or self.committed == self.total or
                   frame.frame % self.log_every == 0 or interesting)
        self.previous_state = state
        status = ('prediction-only: missing input frame' if not frame.input_available else
                  'prediction-only: external attitude invalid' if not frame.attitude_valid else
                  '+att; valid empty frame' if not row['input_detections'] else '+att')
        message = ('[%04d]  pts=%d  valid=%d  cat=%d  stars=%d  cand=%d  tracks=%d  '
                   'confirmed=%d  components=%d  bg=%d  hyp=%d  step=%.0fms' % (
                       frame.frame, row['input_detections'], row['classified_detections'],
                       diagnostics.get('KnownStar_Count', 0), row['n_stars'], *counts,
                       row['model_target_components'], len(result.get('background_marginals', ())),
                       diagnostics.get('Retained_Hypothesis_Count', 0), row['time_ms']))
        if wall_seconds is not None:
            message += '  wall=%.0fms' % (wall_seconds * 1000)
        if 'Joint_Group_Count' in diagnostics:
            message += '  groups=%d' % diagnostics['Joint_Group_Count']
        if 'Dormant_Background_Count' in diagnostics:
            message += '  bg_sleep=%d  bg_retired=%d' % (diagnostics['Dormant_Background_Count'],diagnostics['Retired_Background_Count'])
        message += '  %s  (%d/%d)' % (status, self.committed, self.total)
        validity = ('无效（输入缺失）' if not frame.input_available else
                    '无效（姿态无效）' if not frame.attitude_valid else
                    '有效（无检测）' if not row['input_detections'] else '有效')
        milliseconds = wall_seconds * 1000 if wall_seconds is not None else row['time_ms']
        stars = str(row['n_stars']) if frame.inertial_observation_available else '--'
        concise = '[%04d] %s  总耗时=%.0fms  轨迹=%d  恒星=%s  (%d/%d)' % (
            frame.frame, validity, milliseconds, row['n_live_tracks'], stars,
            self.committed, self.total)
        self.write(message, console=console, console_message=concise)
        reasons = Counter(p.get('Classification_Skip_Reason') or 'not_in_observation_model'
                          for p in point_rows if not p['Classification_Valid'])
        if reasons:
            self.write('[%04d] excluded: %s' % (frame.frame, self._counts(reasons)), console=False)
        if events or retired:
            self.write('[%04d] events: %s; reasons: %s; retired=%d' % (
                frame.frame, self._counts(Counter(e['kind'] for e in events)),
                self._counts(Counter(e['reason'] for e in events)), len(retired)), console=False)
        for event in events:
            self.write('[%04d] event: %s label=%s %s->%s reason=%s' % (
                frame.frame, event['kind'], event['label'], event.get('before', ''),
                event.get('after', ''), event['reason']),
                console=False)
        for event in retired:
            self.write('[%04d] retired: label=%s existence=%.6g reason=%s' % (
                frame.frame, event['label'], event['existence_probability'], event['reason']),
                console=False)

    def failure(self, failed):
        self.write('[%04d] FAILED: %s; previous_committed_frame=%s' % (
            failed['frame'], failed['error'], failed['previous_committed_frame']))

    def finish(self, summary, report):
        elapsed = time.perf_counter() - self.started
        status = '处理完成' if summary['status'] == 'COMPLETED' else '处理停止'
        self.write('%s | Committed frames: %d/%d | tracks=%d confirmed=%d | elapsed=%.2fs' % (
            summary['status'], summary['frames_committed'], summary['frames_read'],
            report['total_tracks'], report['formal_tracks'], elapsed),
            console_message='%s：%d/%d 帧，总用时=%.2fs' % (
                status, summary['frames_committed'], summary['frames_read'], elapsed))
        if summary.get('checkpoint_path'):
            self.write('Checkpoint: ' + str(summary['checkpoint_path']), console=False)
        if summary.get('failed_frame'):
            self.write('Stopped: ' + str(summary['failed_frame']), console=False)
        self.write('Debug log: ' + str(self.path.absolute()), console=False)
