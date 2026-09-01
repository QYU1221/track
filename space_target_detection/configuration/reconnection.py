"""Exit and re-entry reconnection configuration namespace."""

from .values import *  # noqa: F401,F403

RECONNECTION_KEYS = tuple(name for name in globals() if (
    name.startswith('exit_reconnect_') or name == 'enable_exit_reconnect'
))
