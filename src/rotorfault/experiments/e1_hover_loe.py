"""E1 -- hover with a step loss of effectiveness on one rotor at t = 2 s.

Magnitudes 0-90 % in 10 % steps; passive vs oracle vs estimated; several seeds per
point (paired: the same vehicle draw, sensor noise and turbulence for every baseline
and magnitude). Rotor 1 (front-right) is faulted; the airframe is symmetric up to the
inertia difference between roll and pitch.
"""

from __future__ import annotations

import numpy as np

from ..stack import BASELINES
from .common import SEED_E1, Paths, load_json, nan_stats, pmap, save_csv, save_json, wilson_interval
from .runner import run_task

NAME = "e1_hover_loe"
ONSET = 2.0
DURATION = 8.0
ROTOR = 0
REPRESENTATIVE = (0.5, 0.8)


def tasks(paths: Paths, cal: dict, quick: bool) -> list[dict]:
    mags = [0.0, 0.5, 0.9] if quick else [round(0.1 * k, 1) for k in range(10)]
    n_seeds = 2 if quick else 8
    out = []
    for m in mags:
        for b in BASELINES:
            for s in range(n_seeds):
                t = dict(
                    exp=NAME,
                    traj="hover",
                    duration=DURATION,
                    baseline=b,
                    seed=SEED_E1 + s,
                    rotor=ROTOR,
                    magnitude=m,
                    onset=ONSET,
                    cal=cal,
                )
                if s == 0 and m in REPRESENTATIVE:
                    t["npz"] = str(paths.cache / f"{NAME}_m{int(m * 100)}_{b}.npz")
                out.append(t)
    return out


def summarise(rows: list[dict]) -> dict:
    mags = sorted({r["magnitude"] for r in rows})
    summary = []
    for m in mags:
        for b in BASELINES:
            rs = [r for r in rows if r["magnitude"] == m and r["baseline"] == b]
            ok = [r for r in rs if not r["lost"]]
            n_lost = len(rs) - len(ok)
            summary.append(
                dict(
                    magnitude=m,
                    baseline=b,
                    n=len(rs),
                    lost=n_lost,
                    lost_rate=n_lost / len(rs),
                    lost_ci95=wilson_interval(n_lost, len(rs)),
                    max_dev=nan_stats([r["max_dev"] for r in ok]),
                    settled_err=nan_stats([r["settled_err"] for r in ok]),
                    recovery_time=nan_stats([r.get("recovery_time") for r in ok]),
                    final_spin=nan_stats([r["final_spin"] for r in ok]),
                    used_relaxed=sum(bool(r.get("used_relaxed")) for r in rs),
                )
            )
    return {"rows": summary}


def run(paths: Paths, quick: bool = False) -> dict:
    cal = load_json(paths.results / "calibration.json")["gyro_x1" + ("_quick" if quick else "")]
    rows = pmap(run_task, tasks(paths, cal, quick), desc=NAME)
    summary = summarise(rows)
    summary["config"] = dict(
        onset_s=ONSET,
        duration_s=DURATION,
        rotor=ROTOR + 1,
        seeds=sorted({r["seed"] for r in rows}),
        lost_definition="ground contact, tilt > 90 deg, or position error > 3 m",
        settled_definition="mean position error over the last 2 s (non-lost runs)",
    )
    save_json(paths.results / f"{NAME}.json", summary)
    save_csv(paths.results / f"{NAME}.csv", rows)
    return summary


def headline(summary: dict) -> dict:
    """A few numbers quoted in the README."""
    out = {}
    for r in summary["rows"]:
        key = f"{r['baseline']}_{int(round(r['magnitude'] * 100))}"
        out[key] = dict(
            lost=r["lost"],
            n=r["n"],
            max_dev_median=r["max_dev"]["median"],
            settled_median=r["settled_err"]["median"],
        )
    return out


if __name__ == "__main__":  # pragma: no cover
    from .common import default_paths

    s = run(default_paths())
    print(np.array([[r["magnitude"], r["lost"]] for r in s["rows"]]))
