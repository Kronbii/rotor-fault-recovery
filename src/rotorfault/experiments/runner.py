"""Generic single-run task used by every experiment (picklable, for multiprocessing)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .. import metrics as M
from ..fdi import FDIConfig
from ..params import SimConfig
from ..scenarios import make_scenario
from ..sim import simulate
from ..stack import PARTIAL, RELAXED
from .common import export_replay, save_run_npz


def fdi_config(cal: dict) -> FDIConfig:
    return FDIConfig(
        sigma=np.array(cal["sigma"]),
        thresholds=np.array(cal["thresholds"]),
        **cal.get("variant", {}),
    )


def run_task(task: dict) -> dict:
    """Run one simulation described by ``task`` and return scalar metrics.

    Keys: traj, duration, baseline, seed, rotor, magnitude, onset, gyro_scale,
    turbulence, cal (dict with sigma/thresholds), true_scale, actuator_delay_steps,
    npz (path or None), replay (path or None), plus any extra keys which are echoed.
    """
    sc = make_scenario(
        task["traj"],
        task["duration"],
        rotor=task.get("rotor", 0),
        magnitude=task.get("magnitude", 0.0),
        onset=task.get("onset", 2.0),
        gyro_scale=task.get("gyro_scale", 1.0),
        turbulence=task.get("turbulence", 1.0),
        true_scale=task.get("true_scale"),
    )
    cfg = SimConfig(actuator_delay_steps=task.get("actuator_delay_steps", 1))
    r = simulate(sc, task["baseline"], task["seed"], fdi_config(task["cal"]), cfg=cfg)
    if task.get("npz"):
        save_run_npz(Path(task["npz"]), r)
    if task.get("replay"):
        export_replay(Path(task["replay"]), r, extra=task.get("replay_extra"))
    return summarize(r, task)


def summarize(r, task: dict) -> dict:
    onset = task.get("onset", 2.0) if task.get("magnitude", 0.0) > 0 else math.nan
    mag = task.get("magnitude", 0.0)
    rotor = task.get("rotor", 0)
    t_alarm = r.alarm_time
    has_fault = mag > 0
    out = {k: v for k, v in task.items() if k not in ("cal", "npz", "replay", "replay_extra")}
    if "true_scale" in out:
        out["true_scale"] = str(out["true_scale"])
    out.update(
        lost=bool(r.crashed),
        lost_reason=r.crash_reason,
        lost_time=r.crash_time,
        alarm_time=t_alarm,
        isolated=int(r.isolated),
        switch_partial=r.switch_partial,
        switch_relaxed=r.switch_relaxed,
        final_mode=int(r.mode[-1]),
        flight_time=float(r.t[-1]),
    )
    t_eval0 = onset if has_fault else 0.0
    out["max_dev"] = M.max_deviation(r, t_eval0)
    out["settled_err"] = M.settled_error(r)
    out["final_spin"] = M.spin_rate(r)
    out["max_tilt_deg"] = M.max_tilt_deg(r, t_eval0)
    if has_fault:
        out["recovery_time"] = M.recovery_time(r, onset)
        out["detected"] = bool(np.isfinite(t_alarm) and t_alarm >= onset)
        out["false_alarm_pre_onset"] = bool(np.isfinite(t_alarm) and t_alarm < onset)
        out["delay"] = t_alarm - onset if out["detected"] else math.nan
        out["isolation_correct"] = bool(out["detected"] and r.isolated == rotor)
        # LOE estimate of the true rotor, 0.25 s after the alarm or at the switch to
        # yaw-relaxed control if earlier (afterwards the rotor idles: unobservable)
        if out["detected"]:
            t_ev = t_alarm + 0.25
            if np.isfinite(r.switch_relaxed):
                t_ev = min(t_ev, r.switch_relaxed)
            k = min(int(np.searchsorted(r.t, t_ev)), len(r.t) - 1)
            loe_hat = 1.0 - r.eta_hat[k, rotor] / r.eta_ref[rotor]
            out["loe_hat"] = float(loe_hat)
            out["loe_err"] = float(loe_hat - mag)
            out["loe_eval_dt"] = float(r.t[k] - t_alarm)
        else:
            out["loe_hat"] = out["loe_err"] = out["loe_eval_dt"] = math.nan
        out["used_relaxed"] = bool(np.isfinite(r.switch_relaxed))
        out["used_partial"] = bool(np.any(r.mode == PARTIAL) or np.any(r.mode == RELAXED))
    else:
        out["false_alarm"] = bool(np.isfinite(t_alarm))
    # windows for trajectory-tracking RMSE (E3)
    for name, (a, b) in task.get("rmse_windows", {}).items():
        out[f"rmse_{name}"] = M.rmse(r, a, b) if not r.crashed or r.crash_time > b else math.nan
    return out
