"""E6 -- where does complete-loss recovery break? One-at-a-time robustness sweeps.

The full estimated pipeline (detection -> isolation -> yaw-relaxed control) is run on a
complete loss in hover while one property of the *true* system is changed away from
the model the flight stack uses:

* actuator transport delay (the stack still assumes 2 ms),
* motor time constant (the stack assumes 25 ms),
* rotational damping, which sets the equilibrium spin rate (the stack assumes nominal),
* body inertia (scaled as a whole),
* turbulence intensity (1 = the default light turbulence).

Each point: four rotors x two seeds.
"""

from __future__ import annotations

from .common import Paths, load_json, nan_stats, pmap, save_csv, save_json, wilson_interval
from .runner import run_task

NAME = "e6_robustness"
SEED0 = 600_000
SWEEPS = {
    "actuator_delay_ms": [0, 2, 4, 8, 12, 16, 20, 24],
    "motor_tau_ms": [15, 25, 40, 60, 80, 100],
    "damping_scale": [0.25, 0.5, 1.0, 1.5, 2.0],
    "inertia_scale": [0.7, 0.85, 1.0, 1.2, 1.4],
    "turbulence_scale": [0.0, 1.0, 2.0, 3.0, 4.0],
}


def _task(param, value, rot, s, cal):
    t = dict(
        exp=NAME,
        sweep=param,
        value=value,
        traj="hover",
        duration=10.0,
        baseline="estimated",
        seed=SEED0 + 10 * rot + s,
        rotor=rot,
        magnitude=1.0,
        onset=2.0,
        cal=cal,
    )
    if param == "actuator_delay_ms":
        t["actuator_delay_steps"] = int(round(value / 2.0))
    elif param == "motor_tau_ms":
        t["true_scale"] = {"tau_motor": value / 25.0}
    elif param == "damping_scale":
        t["true_scale"] = {"rot_damping": value}
    elif param == "inertia_scale":
        t["true_scale"] = {"inertia": value}
    elif param == "turbulence_scale":
        t["turbulence"] = value
    return t


def run(paths: Paths, quick: bool = False) -> dict:
    cal = load_json(paths.results / "calibration.json")["gyro_x1" + ("_quick" if quick else "")]
    sweeps = {k: (v[:2] if quick else v) for k, v in SWEEPS.items()}
    rotors = (0,) if quick else range(4)
    n_seeds = 1 if quick else 2
    tasks = [
        _task(p, v, rot, s, cal)
        for p, vals in sweeps.items()
        for v in vals
        for rot in rotors
        for s in range(n_seeds)
    ]
    rows = pmap(run_task, tasks, desc=NAME)
    out = {}
    for p, vals in sweeps.items():
        pts = []
        for v in vals:
            rs = [r for r in rows if r["sweep"] == p and r["value"] == v]
            ok = [r for r in rs if not r["lost"]]
            n_lost = len(rs) - len(ok)
            pts.append(
                dict(
                    value=v,
                    n=len(rs),
                    lost=n_lost,
                    lost_ci95=wilson_interval(n_lost, len(rs)),
                    max_dev=nan_stats([r["max_dev"] for r in ok]),
                    settled_err=nan_stats([r["settled_err"] for r in ok]),
                    final_spin=nan_stats([r["final_spin"] for r in ok]),
                    max_tilt_deg=nan_stats([r["max_tilt_deg"] for r in ok]),
                )
            )
        out[p] = pts
    result = dict(sweeps=out, config=dict(case="complete loss in hover, estimated pipeline"))
    save_json(paths.results / f"{NAME}.json", result)
    save_csv(paths.results / f"{NAME}.csv", rows)
    return result
