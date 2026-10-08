"""Calibration of the FDI residual scale and CUSUM threshold on fault-free flights.

Procedure (seeds disjoint from every evaluation seed):

1. Fly ``n`` fault-free runs (half hover, half figure-eight) with a provisional scale
   and detection disabled; set the per-row floor ``sigma`` to the RMS of the residual
   (roll and pitch rows pooled, since the airframe is nearly symmetric).
2. Fly the same runs again with the calibrated ``sigma`` and record, for each CUSUM of
   the bank, the largest statistic reached by any rotor in any run; set that test's
   alarm threshold to ``safety x`` this value. Because the large-magnitude tests
   often never leave zero on fault-free data, the threshold is floored at the
   evidence accumulated in ``N_MIN`` samples (20 ms) by a fault of exactly the design
   magnitude at hover thrust, ``N_MIN mu^2 / 2``.

The aggressive waypoint-step manoeuvre is deliberately *not* part of the calibration
set, so the false-alarm rate measured on it (E2) tests generalisation.
"""

from __future__ import annotations

import numpy as np

from ..fdi import FDIConfig, hover_signature_norm
from ..params import nominal_params
from ..scenarios import make_scenario
from ..sim import simulate
from .common import SEED_CALIBRATION, Paths, load_json, pmap, save_json

SAFETY = 2.0
N_MIN = 10  # samples (20 ms) of evidence at the design magnitude, threshold floor


def _cal_run(args):
    seed, traj, gyro_scale, sigma, duration = args
    cfg = FDIConfig(sigma=np.array(sigma), thresholds=np.full(3, np.inf))
    sc = make_scenario(traj, duration, magnitude=0.0, gyro_scale=gyro_scale)
    r = simulate(sc, "passive", seed, cfg)
    m = r.t >= 0.0
    res = r.residual[m]
    return {"ms": (res**2).mean(axis=0), "gmax": r.cusum_bank[m].max(axis=0).tolist()}


def calibrate(
    paths: Paths,
    gyro_scale: float = 1.0,
    n_runs: int = 60,
    duration: float = 12.0,
    force: bool = False,
    quick: bool = False,
) -> FDIConfig:
    key = f"gyro_x{gyro_scale:g}" + ("_quick" if quick else "")
    store = paths.results / "calibration.json"
    db = load_json(store) if store.exists() else {}
    if key in db and not force:
        e = db[key]
        return FDIConfig(sigma=np.array(e["sigma"]), thresholds=np.array(e["thresholds"]))
    if quick:
        n_runs, duration = 6, 6.0
    sigma0 = FDIConfig().sigma
    tasks = [
        (
            SEED_CALIBRATION + int(1000 * gyro_scale) + i,
            ("hover", "fig8")[i % 2],
            gyro_scale,
            sigma0.tolist(),
            duration,
        )
        for i in range(n_runs)
    ]
    out = pmap(_cal_run, tasks, desc=f"calibration pass 1 (gyro x{gyro_scale:g})")
    ms = np.mean([o["ms"] for o in out], axis=0)
    rms = np.sqrt(ms)
    rp = np.sqrt(0.5 * (ms[1] + ms[2]))
    sigma = np.array([rms[0], rp, rp, rms[3]])
    tasks2 = [(s, tr, g, sigma.tolist(), d) for (s, tr, g, _, d) in tasks]
    out2 = pmap(_cal_run, tasks2, desc=f"calibration pass 2 (gyro x{gyro_scale:g})")
    gmax = np.array([o["gmax"] for o in out2])  # (runs, bank)
    mu = np.asarray(FDIConfig().deltas) * hover_signature_norm(nominal_params(), sigma)
    floor = N_MIN * mu**2 / 2
    thresholds = np.maximum(floor, SAFETY * gmax.max(axis=0))
    db[key] = {
        "gyro_scale": gyro_scale,
        "n_runs": n_runs,
        "duration_s": duration,
        "seeds": [t[0] for t in tasks],
        "sigma": sigma.tolist(),
        "residual_rms_per_row": rms.tolist(),
        "deltas": list(FDIConfig().deltas),
        "h0_cusum_max_per_run": gmax.tolist(),
        "safety_factor": SAFETY,
        "threshold_floor": floor.tolist(),
        "thresholds": thresholds.tolist(),
    }
    save_json(store, db)
    return FDIConfig(sigma=sigma, thresholds=thresholds)
