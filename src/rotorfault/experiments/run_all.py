"""Run every experiment (and the calibration it depends on), then figures and tables."""

from __future__ import annotations

import time

from . import (
    e1_hover_loe,
    e2_fdi_montecarlo,
    e3_trajectory,
    e4_complete_loss,
    e6_robustness,
    e7_fdi_ablation,
)
from .calibration import calibrate
from .common import Paths, default_paths, save_json

EXPERIMENTS = {
    "e1": e1_hover_loe.run,
    "e2": e2_fdi_montecarlo.run,
    "e3": e3_trajectory.run,
    "e4": e4_complete_loss.run,
    "e5": e4_complete_loss.run_e5,
    "e6": e6_robustness.run,
    "e7": e7_fdi_ablation.run,
}


def run_experiments(
    paths: Paths | None = None, quick: bool = False, only=None, recalibrate: bool = True
) -> dict:
    paths = paths or default_paths()
    t0 = time.time()
    store = paths.results / "calibration.json"
    if recalibrate and not only and store.exists():
        store.unlink()  # every calibration is recomputed from its seeds
    print(f"calibration (quick={quick})")
    calibrate(paths, 1.0, quick=quick)
    timings = {}
    for key, fn in EXPERIMENTS.items():
        if only and key not in only:
            continue
        t = time.time()
        print(f"{key}: running")
        fn(paths, quick=quick)
        timings[key] = round(time.time() - t, 1)
        print(f"{key}: done in {timings[key]} s")
    timings["total"] = round(time.time() - t0, 1)
    if not only and not quick:
        save_json(paths.results / "timings.json", timings)
    return timings


def reproduce(paths: Paths | None = None, quick: bool = False) -> None:
    from .figures import make_all_figures
    from .report import write_reports

    paths = paths or default_paths()
    run_experiments(paths, quick=quick)
    make_all_figures(paths)
    write_reports(paths)


if __name__ == "__main__":  # pragma: no cover
    import sys

    reproduce(quick="--quick" in sys.argv)
