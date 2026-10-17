def main():
    from space_target_detection.cli import main as _main
    return _main()


__all__ = ["main"]


if __name__ == '__main__':
    if not __package__:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).absolute().parent.parent))
    main()
