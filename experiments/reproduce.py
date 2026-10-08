"""Regenerate every number, table and figure of the study (same as `make reproduce`).

python experiments/reproduce.py           # full study
python experiments/reproduce.py --quick   # smoke test, about a minute
"""

import sys

from rotorfault.experiments.run_all import reproduce

if __name__ == "__main__":
    reproduce(quick="--quick" in sys.argv)
