"""Run a single experiment; see rotorfault/experiments/e6_robustness.py for the definition.

Requires the FDI calibration (`rotorfault calibrate`) to exist in results/.
"""

import sys

from rotorfault.experiments.common import default_paths
from rotorfault.experiments.e6_robustness import run

if __name__ == "__main__":
    run(default_paths(), quick="--quick" in sys.argv)
