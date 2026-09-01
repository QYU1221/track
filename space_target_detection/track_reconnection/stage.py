"""Pipeline stage for archiving exited tracks and reconnecting re-entries."""

from dataclasses import dataclass
from typing import Any

from .service import (
    build_exit_memory_item,
    prune_exit_memory,
    reconnect_exited_tracks,
)


@dataclass
class TrackReconnectionResult:
    tracks: Any
    archived_tracks: Any
    trajectory_memory: Any
    pending_reconnect: Any
    frame_events: Any


class TrackReconnectionStage:
    """Archive exited tracks and run long-gap re-entry reconnection."""

    @staticmethod
    def run(tracks, pruned_tracks, archived_tracks, trajectory_memory,
            pending_reconnect, frame_idx, enabled=True):
        for track in tracks:
            if 'start_frame' not in track:
                track['start_frame'] = frame_idx
        for track in pruned_tracks:
            if track.get('status') == 'exited':
                archived_tracks.append(track)
                if not track.get('exit_memory_built'):
                    memory_item = build_exit_memory_item(track, frame_idx)
                    if memory_item is not None:
                        trajectory_memory.append(memory_item)

        if enabled:
            trajectory_memory = prune_exit_memory(
                trajectory_memory, frame_idx)
            tracks, trajectory_memory, pending_reconnect, frame_events = (
                reconnect_exited_tracks(
                    tracks, trajectory_memory, frame_idx, pending_reconnect))
        else:
            frame_events = []

        return TrackReconnectionResult(
            tracks=tracks,
            archived_tracks=archived_tracks,
            trajectory_memory=trajectory_memory,
            pending_reconnect=pending_reconnect,
            frame_events=frame_events,
        )
