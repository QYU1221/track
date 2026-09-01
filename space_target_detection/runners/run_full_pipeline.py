"""Run the complete, behavior-compatible processing pipeline.

This named entry point makes the execution scope explicit.  It currently
delegates argument parsing to the legacy-compatible ``main`` function.
"""

import os
import sys


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from main import main


if __name__ == '__main__':
    main()
