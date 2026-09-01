"""Exact local snapshots for replaying an individual stateful stage.

Snapshots use pickle because track filters and separator state must retain their
exact Python/NumPy types.  Load only snapshots produced by this project; pickle
is not a safe interchange format for untrusted files.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import os
import pickle
import tempfile


SNAPSHOT_SCHEMA_VERSION = 1


@dataclass
class StageSnapshot:
    stage: str
    payload: object
    schema_version: int = SNAPSHOT_SCHEMA_VERSION
    created_utc: str = ''

    def __post_init__(self):
        if not self.created_utc:
            self.created_utc = datetime.now(timezone.utc).isoformat()


def save_stage_snapshot(path, stage, payload):
    """Atomically save exact stage input, output or state."""
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    snapshot = StageSnapshot(stage=stage, payload=payload)
    fd, temporary_path = tempfile.mkstemp(
        prefix='.stage-snapshot-', suffix='.tmp', dir=parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            pickle.dump(snapshot, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, path)
    except Exception:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise
    return path


def load_stage_snapshot(path, expected_stage=None):
    """Load a trusted local snapshot and validate its schema and stage."""
    with open(path, 'rb') as stream:
        snapshot = pickle.load(stream)
    if not isinstance(snapshot, StageSnapshot):
        raise ValueError('Not a StageSnapshot: %s' % path)
    if snapshot.schema_version != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError('Unsupported snapshot schema: %s' %
                         snapshot.schema_version)
    if expected_stage is not None and snapshot.stage != expected_stage:
        raise ValueError('Expected stage %r, got %r' %
                         (expected_stage, snapshot.stage))
    return snapshot
