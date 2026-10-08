"""E2 -- fault detection and isolation Monte Carlo.

* 200 faulty runs: random rotor, LOE magnitude ~ U[0.2, 0.9], onset ~ U[2, 5] s,
  half in hover and half on the figure-eight, closed loop with the proposed pipeline.
* 100 fault-free runs of 12 s (hover, figure-eight, and aggressive waypoint steps that
  were not used for calibration) to measure false alarms.
* Gyro-noise sensitivity: noise x {0.5, 1, 2, 4, 8}; residual scale and thresholds are
  re-calibrated for every level (as one would when tuning on a vehicle), then 50 faulty
  and 30 fault-free runs per level.
"""

from __future__ import annotations

import numpy as np

from .calibration import calibrate
from .common import (
    SEED_E2_FAULT,
    SEED_E2_FREE,
    SEED_E2_SWEEP,
    Paths,
    nan_stats,
    pmap,
    save_csv,
    save_json,
    wilson_interval,
)
from .runner import run_task

NAME = "e2_fdi"
GYRO_SCALES = (0.5, 1.0, 2.0, 4.0, 8.0)
MAG_BINS = ((0.2, 0.35), (0.35, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 0.9 + 1e-9))
ETA_RELAX = 0.35


def _fault_tasks(n, seed0, cal, gyro_scale=1.0, trajs=("hover", "fig8"), tag="main"):
    out = []
    for i in range(n):
        rng = np.random.default_rng(seed0 + i)
        onset = float(rng.uniform(2.0, 5.0))
        out.append(
            dict(
                exp=NAME,
                set=tag,
                traj=trajs[i % len(trajs)],
                duration=onset + 2.0,
                baseline="estimated",
                seed=seed0 + i,
                rotor=int(rng.integers(4)),
                magnitude=float(rng.uniform(0.2, 0.9)),
                onset=onset,
                gyro_scale=gyro_scale,
                cal=cal,
            )
        )
    return out


def _free_tasks(n, seed0, cal, gyro_scale=1.0, tag="main", duration=12.0):
    trajs = ("hover", "fig8", "steps")
    return [
        dict(
            exp=NAME,
            set=tag,
            traj=trajs[i % 3],
            duration=duration,
            baseline="estimated",
            seed=seed0 + i,
            magnitude=0.0,
            gyro_scale=gyro_scale,
            cal=cal,
        )
        for i in range(n)
    ]


def fdi_summary(fault_rows: list[dict], free_rows: list[dict]) -> dict:
    n_f = len(fault_rows)
    det = [r for r in fault_rows if r["detected"]]
    pre = sum(r["false_alarm_pre_onset"] for r in fault_rows)
    iso = sum(r["isolation_correct"] for r in det)
    fa_runs = sum(r["false_alarm"] for r in free_rows)
    free_time = sum(r["flight_time"] for r in free_rows)
    # pre-onset flight time of faulty runs also counts as fault-free exposure
    pre_time = sum(r["onset"] for r in fault_rows)
    by_bin = []
    for lo, hi in MAG_BINS:
        rs = [r for r in fault_rows if lo <= r["magnitude"] < hi]
        d = [r for r in rs if r["detected"]]
        by_bin.append(
            dict(
                lo=lo,
                hi=min(hi, 0.9),
                n=len(rs),
                detected=len(d),
                delay_ms=nan_stats([1e3 * r["delay"] for r in d]),
                isolation=(sum(r["isolation_correct"] for r in d) / len(d)) if d else None,
                loe_abs_err=nan_stats([abs(r["loe_err"]) for r in d]),
            )
        )
    should_relax = [r for r in det if (1 - r["magnitude"]) < ETA_RELAX]
    should_not = [r for r in det if (1 - r["magnitude"]) >= ETA_RELAX]
    return dict(
        n_fault_runs=n_f,
        detected=len(det),
        detection_rate=len(det) / n_f if n_f else None,
        detection_ci95=wilson_interval(len(det), n_f),
        missed=n_f - len(det) - pre,
        false_alarms_before_onset=pre,
        isolation_accuracy=iso / len(det) if det else None,
        isolation_ci95=wilson_interval(iso, len(det)),
        delay_ms=nan_stats([1e3 * r["delay"] for r in det]),
        loe_err=nan_stats([r["loe_err"] for r in det]),
        loe_abs_err=nan_stats([abs(r["loe_err"]) for r in det]),
        mode_decision=dict(
            relaxed_when_eta_below=ETA_RELAX,
            should_relax=len(should_relax),
            did_relax=sum(r["used_relaxed"] for r in should_relax),
            should_not_relax=len(should_not),
            wrongly_relaxed=sum(r["used_relaxed"] for r in should_not),
        ),
        lost_fault_runs=sum(r["lost"] for r in fault_rows),
        n_free_runs=len(free_rows),
        false_alarm_runs=fa_runs,
        false_alarm_run_ci95=wilson_interval(fa_runs, len(free_rows)),
        fault_free_flight_time_s=free_time + pre_time,
        false_alarms_total=fa_runs + pre,
        false_alarms_per_hour=(fa_runs + pre) / ((free_time + pre_time) / 3600.0),
        lost_free_runs=sum(r["lost"] for r in free_rows),
        by_magnitude=by_bin,
        false_alarms_by_traj={
            tr: sum(r["false_alarm"] for r in free_rows if r["traj"] == tr)
            for tr in ("hover", "fig8", "steps")
        },
    )


def run(paths: Paths, quick: bool = False) -> dict:
    n_fault, n_free = (16, 6) if quick else (200, 100)
    cal_cfg = calibrate(paths, 1.0, quick=quick)
    cal = dict(sigma=cal_cfg.sigma.tolist(), thresholds=list(cal_cfg.thresholds))
    tasks = _fault_tasks(n_fault, SEED_E2_FAULT, cal) + _free_tasks(n_free, SEED_E2_FREE, cal)
    rows = pmap(run_task, tasks, desc=f"{NAME} main")
    fault_rows = [r for r in rows if r["magnitude"] > 0]
    free_rows = [r for r in rows if r["magnitude"] == 0]
    main = fdi_summary(fault_rows, free_rows)
    main["calibration"] = cal

    # ---------------- gyro-noise sensitivity ----------------
    scales = (1.0, 4.0) if quick else GYRO_SCALES
    n_sf, n_s0 = (6, 3) if quick else (50, 30)
    sweep = []
    sweep_rows = []
    for k, g in enumerate(scales):
        c = calibrate(paths, g, n_runs=40, quick=quick)
        cd = dict(sigma=c.sigma.tolist(), thresholds=list(c.thresholds))
        seed0 = SEED_E2_SWEEP + 1000 * k
        ts = _fault_tasks(n_sf, seed0, cd, g, tag=f"gyro_x{g:g}") + _free_tasks(
            n_s0, seed0 + 500, cd, g, tag=f"gyro_x{g:g}"
        )
        rs = pmap(run_task, ts, desc=f"{NAME} gyro x{g:g}")
        sweep_rows += rs
        s = fdi_summary(
            [r for r in rs if r["magnitude"] > 0], [r for r in rs if r["magnitude"] == 0]
        )
        s["gyro_scale"] = g
        s["gyro_noise_rad_s"] = 0.03 * g
        s["calibration"] = cd
        sweep.append(s)

    result = dict(
        main=main,
        sweep=sweep,
        config=dict(
            n_fault_runs=n_fault,
            n_free_runs=n_free,
            magnitude_range=[0.2, 0.9],
            onset_range_s=[2.0, 5.0],
            post_onset_s=2.0,
            free_duration_s=12.0,
            delay_definition="first CUSUM alarm time minus fault onset",
            loe_error_definition=(
                "LOE estimate of the faulty rotor (1 - eta_hat/eta_ref) minus true LOE, "
                "0.25 s after the alarm or at the switch to yaw-relaxed control if earlier"
            ),
        ),
    )
    save_json(paths.results / f"{NAME}.json", result)
    save_csv(paths.results / f"{NAME}_runs.csv", rows)
    save_csv(paths.results / f"{NAME}_sweep_runs.csv", sweep_rows)
    return result
