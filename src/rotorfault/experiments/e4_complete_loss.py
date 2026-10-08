"""E4 -- complete loss of one rotor in hover (and E5: during figure-eight flight).

Passive (nominal controller) vs the yaw-relaxed controller engaged by the oracle and
by the estimated pipeline. Every rotor is failed in turn, several seeds each.
"""

from __future__ import annotations

from ..stack import BASELINES
from .calibration import calibration_dict
from .common import (
    SEED_E4,
    SEED_E5,
    Paths,
    nan_stats,
    pmap,
    save_csv,
    save_json,
    wilson_interval,
)
from .runner import run_task

CASES = {
    "e4_complete_loss": dict(traj="hover", onset=2.0, duration=10.0, seed0=SEED_E4, n_seeds=8),
    "e5_loss_in_flight": dict(traj="fig8", onset=6.0, duration=16.0, seed0=SEED_E5, n_seeds=6),
}


def tasks(name: str, paths: Paths, cal: dict, quick: bool) -> list[dict]:
    c = CASES[name]
    n_seeds = 1 if quick else c["n_seeds"]
    rotors = (0, 2) if quick else range(4)
    out = []
    for b in BASELINES:
        for rot in rotors:
            for s in range(n_seeds):
                t = dict(
                    exp=name,
                    traj=c["traj"],
                    duration=c["duration"],
                    baseline=b,
                    seed=c["seed0"] + 10 * rot + s,
                    rotor=rot,
                    magnitude=1.0,
                    onset=c["onset"],
                    cal=cal,
                    rmse_windows={"after_settle": (c["onset"] + 3.0, c["duration"])},
                )
                if s == 0 and rot == 0:
                    t["npz"] = str(paths.cache / f"{name}_{b}.npz")
                    tag = "e4_hover_loss" if c["traj"] == "hover" else "e5_fig8_loss"
                    t["replay"] = str(paths.replays / f"{tag}_rotor1_{b}.json")
                    t["replay_extra"] = {
                        "experiment": f"{name}: complete loss of rotor 1 at {c['onset']} s "
                        f"({c['traj']})"
                    }
                out.append(t)
    return out


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    for b in BASELINES:
        rs = [r for r in rows if r["baseline"] == b]
        ok = [r for r in rs if not r["lost"]]
        n_lost = len(rs) - len(ok)
        out.append(
            dict(
                baseline=b,
                n=len(rs),
                lost=n_lost,
                lost_ci95=wilson_interval(n_lost, len(rs)),
                lost_reasons={
                    k: sum(r["lost_reason"] == k for r in rs)
                    for k in ("ground", "flip", "position")
                },
                time_to_loss=nan_stats([r["lost_time"] - r["onset"] for r in rs if r["lost"]]),
                max_dev=nan_stats([r["max_dev"] for r in ok]),
                settled_err=nan_stats([r["settled_err"] for r in ok]),
                rmse_after_settle=nan_stats([r["rmse_after_settle"] for r in ok]),
                recovery_time=nan_stats([r["recovery_time"] for r in ok]),
                final_spin=nan_stats([r["final_spin"] for r in ok]),
                max_tilt_deg=nan_stats([r["max_tilt_deg"] for r in ok]),
                switch_delay_ms=nan_stats(
                    [1e3 * (r["switch_relaxed"] - r["onset"]) for r in rs if r["used_relaxed"]]
                ),
                detection_delay_ms=nan_stats([1e3 * r["delay"] for r in rs if r["detected"]]),
            )
        )
    return out


def run_case(name: str, paths: Paths, quick: bool = False) -> dict:
    cal = calibration_dict(paths, quick)
    rows = pmap(run_task, tasks(name, paths, cal, quick), desc=name)
    c = CASES[name]
    result = dict(
        rows=summarise(rows),
        config=dict(
            trajectory=c["traj"],
            onset_s=c["onset"],
            duration_s=c["duration"],
            rotors=sorted({r["rotor"] + 1 for r in rows}),
            seeds_per_rotor=1 if quick else c["n_seeds"],
        ),
    )
    save_json(paths.results / f"{name}.json", result)
    save_csv(paths.results / f"{name}.csv", rows)
    return result


def run(paths: Paths, quick: bool = False) -> dict:
    return run_case("e4_complete_loss", paths, quick)


def run_e5(paths: Paths, quick: bool = False) -> dict:
    return run_case("e5_loss_in_flight", paths, quick)
