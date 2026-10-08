"""E3 -- figure-eight tracking (peak 3.3 m/s) with a 50 % LOE on rotor 1 at t = 10 s."""

from __future__ import annotations

from ..stack import BASELINES
from .common import SEED_E3, Paths, load_json, nan_stats, pmap, save_csv, save_json
from .runner import run_task

NAME = "e3_trajectory"
ONSET = 10.0
DURATION = 20.0
MAGNITUDE = 0.5
ROTOR = 0
WINDOWS = {
    "before": (3.0, ONSET),
    "after": (ONSET, DURATION),
    "transient": (ONSET, ONSET + 3.0),
    "steady": (ONSET + 3.0, DURATION),
}


def tasks(paths: Paths, cal: dict, quick: bool) -> list[dict]:
    n_seeds = 2 if quick else 10
    out = []
    for b in BASELINES:
        for s in range(n_seeds):
            t = dict(
                exp=NAME,
                traj="fig8",
                duration=DURATION,
                baseline=b,
                seed=SEED_E3 + s,
                rotor=ROTOR,
                magnitude=MAGNITUDE,
                onset=ONSET,
                cal=cal,
                rmse_windows=WINDOWS,
            )
            if s == 0:
                t["npz"] = str(paths.cache / f"{NAME}_{b}.npz")
                t["replay"] = str(paths.replays / f"e3_fig8_loe50_{b}.json")
                t["replay_extra"] = {"experiment": "E3 figure-eight, 50 % LOE rotor 1 at 10 s"}
            out.append(t)
    return out


def run(paths: Paths, quick: bool = False) -> dict:
    cal = load_json(paths.results / "calibration.json")["gyro_x1" + ("_quick" if quick else "")]
    rows = pmap(run_task, tasks(paths, cal, quick), desc=NAME)
    summary = []
    for b in BASELINES:
        rs = [r for r in rows if r["baseline"] == b]
        summary.append(
            dict(
                baseline=b,
                n=len(rs),
                lost=sum(r["lost"] for r in rs),
                **{f"rmse_{w}": nan_stats([r[f"rmse_{w}"] for r in rs]) for w in WINDOWS},
                max_dev=nan_stats([r["max_dev"] for r in rs if not r["lost"]]),
                delay_ms=nan_stats([1e3 * r["delay"] for r in rs if r["detected"]]),
            )
        )
    result = dict(
        rows=summary,
        config=dict(
            trajectory="figure-eight x = 3 sin(wt), y = 1.5 sin(2wt), period 8 s, z = 2 m",
            peak_speed_m_s=3.33,
            onset_s=ONSET,
            magnitude=MAGNITUDE,
            rotor=ROTOR + 1,
            windows_s=WINDOWS,
        ),
    )
    save_json(paths.results / f"{NAME}.json", result)
    save_csv(paths.results / f"{NAME}.csv", rows)
    return result
