"""E7 -- FDI ablation: what each design element of the diagnoser is for.

Variants (each calibrated on its own fault-free flights, same procedure and seeds):

* ``full``             -- the proposed FDI (taken from E2's calibration),
* ``no_baseline``      -- detection against the nominal model (eta_ref = 1) instead of
                          the healthy baseline learned in the initial hover,
* ``no_adaptive``      -- constant residual scale (no inertia / motor-lag terms),
* ``single_cusum``     -- one CUSUM designed for a 15 % loss instead of the bank.

Each variant: 60 faulty runs (LOE ~ U[0.2, 0.9], hover and figure-eight) and 30
fault-free runs (hover, figure-eight, aggressive steps), closed loop with FTC.
"""

from __future__ import annotations

from .calibration import calibrate
from .common import Paths, pmap, save_csv, save_json
from .e2_fdi_montecarlo import _fault_tasks, _free_tasks, fdi_summary
from .runner import run_task

NAME = "e7_fdi_ablation"
SEED0 = 700_000
VARIANTS = {
    "full": None,
    "no_baseline": {"use_baseline": False},
    "no_adaptive": {"unc_inertia": 0.0, "unc_tau": 0.0},
    "single_cusum": {"deltas": [0.15]},
}


def run(paths: Paths, quick: bool = False) -> dict:
    n_f, n_0 = (6, 3) if quick else (60, 30)
    out = {}
    rows_all = []
    for name, var in VARIANTS.items():
        if var is None:
            c = calibrate(paths, 1.0, quick=quick)
        else:
            c = calibrate(paths, 1.0, n_runs=30, duration=10.0, quick=quick, variant=var, tag=name)
        cal = dict(sigma=c.sigma.tolist(), thresholds=list(c.thresholds), variant=var or {})
        ts = _fault_tasks(n_f, SEED0, cal, tag=name) + _free_tasks(n_0, SEED0 + 500, cal, tag=name)
        rows = pmap(run_task, ts, desc=f"{NAME} {name}")
        rows_all += rows
        s = fdi_summary(
            [r for r in rows if r["magnitude"] > 0], [r for r in rows if r["magnitude"] == 0]
        )
        s["calibration"] = cal
        out[name] = s
    result = dict(variants=out, config=dict(n_fault=n_f, n_free=n_0, seeds_from=SEED0))
    save_json(paths.results / f"{NAME}.json", result)
    save_csv(paths.results / f"{NAME}.csv", rows_all)
    return result
