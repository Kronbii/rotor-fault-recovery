"""Run a single experiment; see rotorfault/experiments/e1_hover_loe.py for the definition.

Computes the FDI calibration first if results/calibration.json does not have it.
"""

import sys

from rotorfault.experiments.common import default_paths
from rotorfault.experiments.e1_hover_loe import run

if __name__ == "__main__":
    run(default_paths(), quick="--quick" in sys.argv)
