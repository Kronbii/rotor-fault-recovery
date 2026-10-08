"""Run a single experiment; see rotorfault/experiments/e4_complete_loss.py for the definition.

Computes the FDI calibration first if results/calibration.json does not have it.
"""

import sys

from rotorfault.experiments.common import default_paths
from rotorfault.experiments.e4_complete_loss import run_e5

if __name__ == "__main__":
    run_e5(default_paths(), quick="--quick" in sys.argv)
