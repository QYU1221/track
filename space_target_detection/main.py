"""Thin command-line and compatibility entry point."""

from pipeline.legacy_sequence import main, process_sequence

__all__ = ["main", "process_sequence"]


if __name__ == '__main__':
    main()
