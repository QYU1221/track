"""Angular overview and gap diagnostics for formally confirmed tracks."""

import math
import os

import numpy as np
from ..numerical_policy import DISPLAY_DIRECTION_SINE_FLOOR

formal_track_plot_dpi = 180


OBSERVED_SOURCES = {'measured', 'recovered', 'reconnected'}
RECONNECTED_SOURCES = {'recovered', 'reconnected'}


def _load_pyplot():
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    return plt, LineCollection


def _local_offsets(samples):
    if not samples:
        return np.zeros((0, 2), dtype=np.float64)
    # Spherical logarithmic map onto the first sample's local tangent basis.
    # This retains directions at the celestial poles, unlike delta-RA*cos(Dec).
    ra0 = samples[0]['ra_rad']
    dec0 = samples[0]['dec_rad']
    origin = np.array([math.cos(dec0)*math.cos(ra0),
                       math.cos(dec0)*math.sin(ra0), math.sin(dec0)])
    east = np.array([-math.sin(ra0), math.cos(ra0), 0.0])
    north = np.cross(origin, east)
    offsets = []
    for item in samples:
        ra, dec = item['ra_rad'], item['dec_rad']
        vector = np.array([math.cos(dec)*math.cos(ra),
                           math.cos(dec)*math.sin(ra), math.sin(dec)])
        components = np.array([vector @ east, vector @ north])
        sine = float(np.linalg.norm(components))
        cosine = float(np.clip(vector @ origin, -1, 1))
        if sine < DISPLAY_DIRECTION_SINE_FLOOR:
            offset = np.zeros(2) if cosine > 0 else np.full(2, np.nan)
        else:
            offset = components * (math.atan2(sine, cosine) / sine)
        offsets.append(np.degrees(offset) * 60.0)
    return np.asarray(offsets, dtype=np.float64)


def _observed_segments(samples):
    observed = [item for item in samples
                if item['source'] in OBSERVED_SOURCES]
    segments = []
    current = []
    for item in observed:
        starts_new = bool(
            current and (
                item['frame'] > current[-1]['frame'] + 1 or
                (item['source'] in RECONNECTED_SOURCES and
                 current[-1]['source'] not in RECONNECTED_SOURCES)))
        if starts_new:
            segments.append(current)
            current = []
        current.append(item)
    if current:
        segments.append(current)
    return segments


def _predict_gap(samples, previous_segment, next_segment):
    """Plot-only angular linear extrapolation; no tracker-state feedback."""
    previous = previous_segment[-1]
    observed_next = next_segment[0]
    target_frame = observed_next['frame']
    gap_samples = [
        item for item in samples
        if previous['frame'] < item['frame'] < target_frame and
        item['source'] == 'predicted'
    ]
    history = list(previous_segment[-2:]) + gap_samples
    if len(history) >= 2:
        first, last = history[-2], history[-1]
        dt = last['timestamp_s'] - first['timestamp_s']
        velocity_ra = (last['ra_rad'] - first['ra_rad']) / float(dt)
        velocity_dec = (last['dec_rad'] - first['dec_rad']) / float(dt)
        remaining = observed_next['timestamp_s'] - last['timestamp_s']
        predicted_ra = last['ra_rad'] + velocity_ra * remaining
        predicted_dec = last['dec_rad'] + velocity_dec * remaining
    else:
        predicted_ra = previous['ra_rad']
        predicted_dec = previous['dec_rad']

    predicted_dec = float(np.clip(
        predicted_dec, -math.pi / 2.0, math.pi / 2.0))
    predicted_vector = np.array([
        math.cos(predicted_dec) * math.cos(predicted_ra),
        math.cos(predicted_dec) * math.sin(predicted_ra),
        math.sin(predicted_dec),
    ], dtype=np.float64)
    dot = float(np.clip(
        np.dot(predicted_vector, observed_next['inertial']), -1.0, 1.0))
    residual_deg = float(np.degrees(math.acos(dot)))
    return {
        'previous': previous,
        'observed_next': observed_next,
        'predicted_ra_deg': float(np.degrees(predicted_ra)),
        'predicted_dec_deg': float(np.degrees(predicted_dec)),
        'residual_deg': residual_deg,
        'gap_frames': max(target_frame - previous['frame'] - 1, 0),
        'delta_seconds': float(
            observed_next['timestamp_s'] - previous['timestamp_s']),
        'prediction_samples': gap_samples,
    }


def _gap_diagnostics(samples, segments):
    return [
        _predict_gap(samples, previous, following)
        for previous, following in zip(segments[:-1], segments[1:])
    ]


def _plot_time_colored_path(axis, samples, line_collection_cls):
    offsets = _local_offsets(samples)
    elapsed = np.asarray(
        [item['elapsed_seconds'] for item in samples], dtype=np.float64)
    observed_mask = np.asarray(
        [item['source'] in OBSERVED_SOURCES for item in samples], dtype=bool)
    predicted_mask = np.asarray(
        [item['source'] == 'predicted' for item in samples], dtype=bool)
    if len(offsets) >= 2:
        consecutive = np.asarray([b['frame'] == a['frame'] + 1
                                  for a, b in zip(samples[:-1], samples[1:])])
        finite = np.isfinite(offsets).all(axis=1)
        keep = consecutive & finite[:-1] & finite[1:]
        segments = np.stack([offsets[:-1], offsets[1:]], axis=1)[keep]
        predicted_segments = (predicted_mask[:-1] | predicted_mask[1:])[keep]
        collection = line_collection_cls(
            segments, cmap='viridis', linewidth=2.0,
            linestyles=['dashed' if predicted else 'solid'
                        for predicted in predicted_segments])
        collection.set_array(np.asarray(elapsed[:-1][keep], dtype=np.float64))
        axis.add_collection(collection)
    if observed_mask.any():
        axis.scatter(
            offsets[observed_mask, 0], offsets[observed_mask, 1],
            c=elapsed[observed_mask], cmap='viridis',
            s=20, zorder=3, label='Observed')
    if predicted_mask.any():
        axis.scatter(
            offsets[predicted_mask, 0], offsets[predicted_mask, 1],
            marker='x', s=46, color='#e8641b', linewidth=1.6,
            zorder=4, label='Predicted (no reliable association)')
    axis.scatter(
        offsets[0, 0], offsets[0, 1], marker='o', s=95,
        facecolor='limegreen', edgecolor='black', linewidth=1.2,
        zorder=5, label='Start')
    axis.scatter(
        offsets[-1, 0], offsets[-1, 1], marker='^', s=105,
        facecolor='#ef3340', edgecolor='black', linewidth=1.2,
        zorder=5, label='End')
    axis.legend(fontsize=8, loc='best')
    axis.autoscale()


def _save_angular_overview(output_dir, formal_tracks, observed_only=False):
    plt, line_collection_cls = _load_pyplot()
    columns = min(2, len(formal_tracks))
    rows = int(math.ceil(len(formal_tracks) / float(columns)))
    figure, axes = plt.subplots(
        rows, columns, figsize=(8.0 * columns, 5.2 * rows), squeeze=False)
    figure.suptitle(
        ('Confirmed tracks: reliable observations only ' if observed_only else
         'Confirmed tracks: observations and supported predicted states ')+
        '(green: start, red: end, color: time, '
        'orange x: predicted)', fontsize=17)

    for axis, (track, samples) in zip(axes.flat, formal_tracks):
        _plot_time_colored_path(axis, samples, line_collection_cls)
        duration = samples[-1]['elapsed_seconds'] - samples[0]['elapsed_seconds']
        n_predicted = sum(
            item['source'] == 'predicted' for item in samples)
        axis.set_title(
            'T%06d | %d observed, %d predicted | %.1f s | %s' %
            (int(getattr(track, 'id', 0)), len(samples)-n_predicted, n_predicted, duration,
             getattr(track, 'status', 'UNKNOWN')),
            fontsize=14)
        axis.set_xlabel('Eastward offset (arcmin)')
        axis.set_ylabel('Northward offset (arcmin)')
        axis.grid(True, alpha=0.25)

    for axis in axes.flat[len(formal_tracks):]:
        axis.set_visible(False)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    path = os.path.join(output_dir, 'observed_trajectory_angular_overview.png' if observed_only
                        else 'formal_trajectory_angular_overview.png')
    figure.savefig(path, dpi=formal_track_plot_dpi, bbox_inches='tight')
    plt.close(figure)
    return path


def _plot_series_panel(axis, segments, gaps, field, ylabel, title):
    colors = ['#5b8db8', '#4aa24a', '#b276a2', '#d18f3f', '#6576b8']
    for index, segment in enumerate(segments):
        color = colors[index % len(colors)]
        axis.plot(
            [item['elapsed_seconds'] for item in segment],
            [item[field] for item in segment],
            '-o', color=color, linewidth=1.5, markersize=3.0,
            label='Retained observation' if index == 0 else None)
    for index, gap in enumerate(gaps):
        previous = gap['previous']
        observed = gap['observed_next']
        predicted = (gap['predicted_ra_deg'] if field == 'ra_deg'
                     else gap['predicted_dec_deg'])
        axis.plot(
            [previous['elapsed_seconds'], observed['elapsed_seconds']],
            [previous[field], predicted], '--', color='#666666',
            linewidth=1.2,
            label='Auxiliary angular linear extrapolation' if index == 0 else None)
        axis.plot(
            observed['elapsed_seconds'], predicted, marker='x',
            color='#d64f4f', markersize=6,
            label='Extrapolated next start' if index == 0 else None)
        axis.plot(
            observed['elapsed_seconds'], observed[field], marker='s',
            markerfacecolor='white', markeredgecolor='#2776a0',
            markersize=6, linestyle='None',
            label='Observed next segment start' if index == 0 else None)
    axis.set_title(title)
    axis.set_xlabel('Elapsed time from first attitude-valid sample (s)')
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.22)
    axis.legend(fontsize=8, ncol=2, loc='best')


def _plot_radec_panel(axis, segments, gaps):
    colors = ['#5b8db8', '#4aa24a', '#b276a2', '#d18f3f', '#6576b8']
    for index, segment in enumerate(segments):
        color = colors[index % len(colors)]
        axis.plot(
            [item['ra_deg'] for item in segment],
            [item['dec_deg'] for item in segment],
            '-o', color=color, linewidth=1.5, markersize=3.0,
            label='Retained observation' if index == 0 else None)
        axis.plot(
            segment[0]['ra_deg'], segment[0]['dec_deg'], marker='^',
            color=color, markersize=6, linestyle='None',
            label='Segment start' if index == 0 else None)
        axis.plot(
            segment[-1]['ra_deg'], segment[-1]['dec_deg'], marker='s',
            color=color, markersize=5, linestyle='None',
            label='Segment end' if index == 0 else None)
        midpoint = segment[len(segment) // 2]
        axis.annotate(
            'P%d' % (index + 1),
            (midpoint['ra_deg'], midpoint['dec_deg']),
            xytext=(4, 4), textcoords='offset points', fontsize=8)

    for index, gap in enumerate(gaps):
        previous = gap['previous']
        observed = gap['observed_next']
        predicted_ra = gap['predicted_ra_deg']
        predicted_dec = gap['predicted_dec_deg']
        axis.plot(
            [previous['ra_deg'], predicted_ra],
            [previous['dec_deg'], predicted_dec], '--',
            color='#666666', linewidth=1.2,
            label='Auxiliary angular linear extrapolation' if index == 0 else None)
        axis.plot(
            predicted_ra, predicted_dec, marker='x', color='#d64f4f',
            markersize=6, linestyle='None',
            label='Extrapolated next start' if index == 0 else None)
        axis.plot(
            observed['ra_deg'], observed['dec_deg'], marker='s',
            markerfacecolor='white', markeredgecolor='#2776a0',
            markersize=6, linestyle='None',
            label='Observed next segment start' if index == 0 else None)
        axis.plot(
            [predicted_ra, observed['ra_deg']],
            [predicted_dec, observed['dec_deg']], '-.',
            color='#d64f4f', linewidth=1.0,
            label='Auxiliary extrapolation residual' if index == 0 else None)
        if observed['source'] in RECONNECTED_SOURCES:
            axis.plot(
                observed['ra_deg'], observed['dec_deg'], marker='*',
                color='#d9902f', markersize=10, linestyle='None',
                label='Recovery/re-entry segment start' if index == 0 else None)

    axis.set_title('(c) Complete RA/Dec trajectory: observed segments and auxiliary angular extrapolations')
    axis.set_xlabel('Right ascension, RA (deg)')
    axis.set_ylabel('Declination, Dec (deg)')
    axis.grid(True, alpha=0.22)
    axis.legend(fontsize=7, ncol=2, loc='best')


def _save_track_diagnostics(output_dir, track, samples):
    plt, _ = _load_pyplot()
    segments = _observed_segments(samples)
    if not segments:
        return None
    gaps = _gap_diagnostics(samples, segments)
    figure, axes = plt.subplots(2, 2, figsize=(16, 11))
    track_label = 'T%06d' % int(getattr(track, 'id', 0))
    figure.suptitle(
        '%s Complete Track Continuity and Auxiliary Angular Linear Extrapolation Diagnostics' %
        track_label, fontsize=17)

    _plot_series_panel(
        axes[0, 0], segments, gaps, 'ra_deg',
        'Right ascension, RA (deg)',
        '(a) Complete RA evolution: observations and auxiliary angular extrapolations')
    _plot_series_panel(
        axes[0, 1], segments, gaps, 'dec_deg',
        'Declination, Dec (deg)',
        '(b) Complete Dec evolution: observations and auxiliary angular extrapolations')
    _plot_radec_panel(axes[1, 0], segments, gaps)

    labels = ['P%d -> P%d' % (index + 1, index + 2)
              for index in range(len(gaps))]
    residuals = [gap['residual_deg'] for gap in gaps]
    axes[1, 1].bar(labels, residuals, color='#7297bd', width=0.65)
    axes[1, 1].set_title(
        '(d) Great-circle auxiliary extrapolation residuals for all observation gaps')
    axes[1, 1].set_xlabel('Observed subsegment transition')
    axes[1, 1].set_ylabel('Great-circle angular residual (deg)')
    axes[1, 1].grid(True, axis='y', alpha=0.22)
    axes[1, 1].margins(y=0.22)
    for index, gap in enumerate(gaps):
        axes[1, 1].text(
            index, gap['residual_deg'],
            '%.4f deg\ndt=%.1f s, miss=%d' %
            (gap['residual_deg'], gap['delta_seconds'], gap['gap_frames']),
            ha='center', va='bottom', fontsize=8)
    if not gaps:
        axes[1, 1].text(
            0.5, 0.5, 'No observation gaps', ha='center', va='center',
            transform=axes[1, 1].transAxes)

    reconnected = sum(segment[0]['source'] in RECONNECTED_SOURCES
                      for segment in segments)
    figure.text(
        0.5, 0.015,
        '%d retained observations | %d observed subsegments | %d gaps\n'
        'Recovery/re-entry segment starts: %d' %
        (sum(len(segment) for segment in segments), len(segments),
         len(gaps), reconnected),
        ha='center', fontsize=10)
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))
    path = os.path.join(
        output_dir,
        '%s_complete_track_prediction_diagnostics.png' % track_label)
    figure.savefig(path, dpi=formal_track_plot_dpi, bbox_inches='tight')
    plt.close(figure)
    return path


def save_formal_track_plots(run_dir, tracks):
    """Reuse X's layout with V2 LOS and declared timestamps, never a fixed period."""
    formal = [(track, track.angular_samples) for track in tracks
              if track.formal_confirmed and track.angular_samples]
    output_dir = os.path.join(run_dir, 'formal_tracks')
    os.makedirs(output_dir, exist_ok=True)
    if not formal:
        plt, _ = _load_pyplot()
        figure, axis = plt.subplots(figsize=(10, 5.2))
        axis.set_title('Formal trajectory angular overview')
        axis.text(.5, .5, 'No confirmed target tracks with valid angular samples',
                  ha='center', va='center', transform=axis.transAxes)
        axis.set_axis_off()
        path = os.path.join(output_dir, 'formal_trajectory_angular_overview.png')
        figure.savefig(path, dpi=formal_track_plot_dpi, bbox_inches='tight')
        plt.close(figure)
        return [path]
    paths = [_save_angular_overview(output_dir, formal)]
    observed = [(track, [s for s in samples if s['source'] in OBSERVED_SOURCES])
                for track, samples in formal]
    observed = [(track, samples) for track, samples in observed if samples]
    if observed:
        paths.append(_save_angular_overview(output_dir, observed, observed_only=True))
    for track, samples in formal:
        path = _save_track_diagnostics(output_dir, track, samples)
        if path is not None:
            paths.append(path)
    return paths
