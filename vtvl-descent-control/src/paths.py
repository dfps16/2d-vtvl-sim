"""Centralised filesystem paths.

Single source of truth for where outputs live, so scripts and notebooks never
hardcode machine-specific absolute paths. Paths are derived relative to this
file, so the repo works from any clone location.
"""

import os

# Project root = the vtvl-descent-control directory (this file lives in src/).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')


def result_path(filename):
    """Absolute path to ``filename`` inside the results directory.

    Ensures the results directory exists so a fresh clone can save straight away.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, filename)
