"""Replay the rotorfault diagnoser on the PX4 SITL logs and compare with PX4.

    .venv/bin/python sitl/analyze.py          # reads build/sitl/npz/*.npz

Writes ``sitl/results/sitl_runs.csv`` (one row per run and diagnoser variant),
``sitl/results/sitl_summary.json``, ``sitl/results/sitl_calibration.json`` and the
figures ``sitl/figures/*.png``, then refreshes the generated README / paper blocks
(``rotorfault report``). Runs offline; does not need PX4.

Protocol
--------
* Diagnoser = ``rotorfault.fdi`` unchanged (same filter, Kalman filter, CUSUM design
  sizes, adaptive-scale uncertainties, baseline length), driven by the x500 airframe
  file read from PX4's model files. Two variants: as designed, and without the adaptive
  residual scale (the element the simulation ablation could not justify).
* The healthy baseline is learned over the 2.5 s of hover before the manoeuvre command;
  detection runs from the manoeuvre command to the end of the flight.
* Transport delay, residual floor and CUSUM thresholds come only from the fault-free
  ``calibration`` group (seeds disjoint from every evaluation run).
* True onset: first IMU sample at which the simulated ESC output of the faulty motor
  departs from the nominal map (``esc``), or the motor command becomes NaN (``px4_off``).
* PX4: first raised ``failure_detector_status`` flag after the onset; vehicle lost when
  the ground-truth tilt exceeds 90 deg or the ground-truth height drops below 0.5 m.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NPZ = ROOT / "build" / "sitl" / "npz"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

from rotorfault.airframe import Airframe  # noqa: E402
from rotorfault.experiments.common import nan_stats, pmap, wilson_interval  # noqa: E402
from rotorfault.fdi import FDIConfig  # noqa: E402
from rotorfault.logs.flightlog import FlightLog  # noqa: E402
from rotorfault.replay import (  # noqa: E402
    LogSegment,
    best_delay,
    calibrate_on_logs,
    replay,
    save_calibration,
)

RATE_HZ = 250.0
BASELINE_S = 2.5
LOE_EVAL_S = 0.25  # LOE estimate read 0.25 s after the alarm (as in the simulation)
VARIANTS = {
    "proposed": {},
    "no_adaptive": {"unc_inertia": 0.0, "unc_tau": 0.0},
}
FD_FLAGS = ("fd_roll", "fd_pitch", "fd_alt", "fd_ext", "fd_motor", "fd_imbalanced_prop")
X500_MIN, X500_MAX = 150.0, 1000.0


def load_runs() -> dict[str, FlightLog]:
    runs = {}
    for p in sorted(NPZ.glob("*.npz")):
        log = FlightLog.load(p)
        man = log.meta.get("manifest", {})
        if not man.get("ok"):
            print(f"  skipping {p.stem}: {man.get('error')}")
            continue
        runs[p.stem] = log
    return runs


def true_onset(log: FlightLog) -> float:
    man = log.meta["manifest"]
    spec, ev = man["spec"], man["events"]
    j = spec["motor"] - 1
    t0 = ev["inject_sent_s"] - 0.05
    after = log.t >= t0
    if spec["fault"] == "px4_off":
        k = np.flatnonzero(after & np.isnan(log.motor[:, j]))
    else:
        u = log.motor[:, j]
        expected = X500_MIN + (X500_MAX - X500_MIN) * np.clip(np.nan_to_num(u), 0, 1)
        dev = np.abs(log.extra["esc_output"][:, j] - expected) / expected
        k = np.flatnonzero(after & (dev > 0.02))
    return float(log.t[k[0]]) if len(k) else math.nan


def true_loe(log: FlightLog) -> float:
    """Thrust loss of the faulty rotor at the hover command (integer ESC limits)."""
    man = log.meta["manifest"]
    spec = man["spec"]
    if spec["loe"] >= 1.0:
        return 1.0
    n = spec["motor"]
    fp = man["fault_params"]
    w_lo, w_hi = fp[f"SIM_GZ_EC_MIN{n}"], fp[f"SIM_GZ_EC_MAX{n}"]
    u = 0.728  # x500 hover command in these runs
    w = X500_MIN + (X500_MAX - X500_MIN) * u
    w_f = w_lo + (w_hi - w_lo) * u
    return 1.0 - (w_f / w) ** 2


def loss_time(log: FlightLog, t0: float, t1: float) -> float:
    """First time in [t0, t1] with ground-truth tilt > 90 deg or height < 0.5 m."""
    m = (log.t >= t0) & (log.t <= t1)
    tilt = log.extra["gt_tilt_deg"]
    h = -log.extra["gt_pos_ned"][:, 2]
    k = np.flatnonzero(m & ((tilt > 90.0) | (h < 0.5)))
    return float(log.t[k[0]]) if len(k) else math.nan


def px4_response(log: FlightLog, onset: float, t_end: float) -> dict:
    first = {}
    for f in FD_FLAGS:
        if f not in log.extra:
            continue
        k = np.flatnonzero((log.t >= onset) & (log.t <= t_end) & (log.extra[f] > 0.5))
        first[f] = float(log.t[k[0]] - onset) if len(k) else None
    raised = {k: v for k, v in first.items() if v is not None}
    flag = min(raised, key=raised.get) if raised else None
    return {
        "px4_flag": flag,
        "px4_delay": raised[flag] if flag else math.nan,
        "px4_fd_motor": first.get("fd_motor") is not None,
    }


def evaluate(name: str, log: FlightLog, af: Airframe, cfg: FDIConfig, delay: int) -> dict:
    man = log.meta["manifest"]
    spec, ev = man["spec"], man["events"]
    t_det = ev["manoeuvre_s"]
    group = name.split("-")[0]
    faulty = spec["fault"] != "none"
    row = {
        "run": name,
        "group": group,
        "condition": f"{spec['traj']}-{'wind' if spec['world'] == 'windy' else 'calm'}",
        "seed": spec["seed"],
        "fault": spec["fault"],
        "motor": spec["motor"] if faulty else None,
    }
    if not faulty:
        r = replay(log, af, cfg, t_det, BASELINE_S, t_end=ev["end_s"], delay_steps=delay)
        row.update(
            flight_s=ev["end_s"] - t_det,
            false_alarm=r.alarm,
            alarm_after_s=r.alarm_time - t_det if r.alarm else math.nan,
        )
        return row
    onset = true_onset(log)
    if not math.isfinite(onset):  # injection without effect (PX4 failure injection, CA mode 0)
        j = spec["motor"] - 1
        m = (log.t >= ev["inject_sent_s"]) & (log.t <= ev["end_s"])
        tilt = log.extra["gt_tilt_deg"][m]
        row.update(
            injection_effective=False,
            min_esc_output_after=float(np.nanmin(log.extra["esc_output"][m, j])),
            max_tilt_deg=float(np.max(tilt)),
            false_alarm_after_injection=bool(
                replay(log, af, cfg, t_det, BASELINE_S, t_end=ev["end_s"], delay_steps=delay).alarm
            ),
        )
        return row
    row["injection_effective"] = True
    t_end = min(onset + spec["post_s"], ev["end_s"])
    t_loss = loss_time(log, onset, t_end)
    r = replay(log, af, cfg, t_det, BASELINE_S, t_end=t_end, delay_steps=delay)
    loe = true_loe(log)
    detected = r.alarm and r.alarm_time >= onset
    row.update(
        loe=loe,
        onset=onset,
        flight_s=onset - t_det,
        false_alarm_pre_onset=r.alarm and r.alarm_time < onset,
        detected=detected,
        delay=r.alarm_time - onset if detected else math.nan,
        isolated=r.isolated + 1 if r.alarm else None,
        isolation_correct=bool(detected and r.isolated == spec["motor"] - 1),
        lost=math.isfinite(t_loss),
        time_to_loss=t_loss - onset if math.isfinite(t_loss) else math.nan,
        alarm_before_loss=bool(detected and (not math.isfinite(t_loss) or r.alarm_time < t_loss)),
    )
    if detected:
        t_ev = min(r.alarm_time + LOE_EVAL_S, t_end, t_loss if math.isfinite(t_loss) else t_end)
        lh = float(r.loe_hat(t_ev)[spec["motor"] - 1])
        row.update(loe_hat=lh, loe_err=lh - loe)
    else:
        row.update(loe_hat=math.nan, loe_err=math.nan)
    m = (log.t >= onset) & (log.t <= t_end)
    row["max_tilt_deg"] = float(np.max(log.extra["gt_tilt_deg"][m]))
    h = -log.extra["gt_pos_ned"][:, 2]
    row["alt_loss_m"] = float(np.max(h[log.t < onset][-50:].mean() - h[m]))
    row["max_yaw_rate"] = float(np.max(np.abs(log.gyro[m, 2])))
    # PX4's outcome within the post-fault window: held altitude (< 1 m lost), descended
    # (more than 1 m lost, never past 90 deg of tilt), or flipped (tilt > 90 deg)
    row["px4_outcome"] = (
        "flipped"
        if row["max_tilt_deg"] > 90
        else "descended"
        if row["alt_loss_m"] > 1.0
        else "held"
    )
    row.update(px4_response(log, onset, t_end))
    return row


def _evaluate_job(job):
    name, log, af, cfg, delay, variant = job
    row = evaluate(name, log, af, cfg, delay)
    row["variant"] = variant
    return row


def summarize(rows: list[dict]) -> dict:
    out = {}
    free = [r for r in rows if r["group"] == "free"]
    fa = sum(r["false_alarm"] for r in free)
    minutes = sum(r["flight_s"] for r in free) / 60
    pre = [r for r in rows if r["group"] in ("partial", "complete")]
    fa_pre = sum(r["false_alarm_pre_onset"] for r in pre)
    pre_min = sum(r["flight_s"] for r in pre) / 60
    out["fault_free"] = {
        "n_runs": len(free),
        "minutes": minutes,
        "false_alarm_runs": fa,
        "pre_onset_minutes": pre_min,
        "false_alarms_pre_onset": fa_pre,
        "false_alarms_per_min": (fa + fa_pre) / (minutes + pre_min) if minutes else None,
        "by_condition": {
            c: {
                "n": sum(1 for r in free if r["condition"] == c),
                "false_alarms": sum(r["false_alarm"] for r in free if r["condition"] == c),
                "minutes": sum(r["flight_s"] for r in free if r["condition"] == c) / 60,
            }
            for c in sorted({r["condition"] for r in free})
        },
    }

    def block(sel):
        n = len(sel)
        det = [r for r in sel if r["detected"]]
        iso = sum(r["isolation_correct"] for r in sel)
        px4 = [r for r in sel if r["px4_flag"]]
        flags = sorted({r["px4_flag"] for r in px4})
        return {
            "n": n,
            "detected": len(det),
            "detected_ci95": wilson_interval(len(det), n),
            "delay_ms": nan_stats([1000 * r["delay"] for r in det]),
            "isolation_correct": iso,
            "isolation_accuracy": iso / len(det) if det else None,
            "loe_abs_err": nan_stats([abs(r["loe_err"]) for r in det]),
            "false_alarm_pre_onset": sum(r["false_alarm_pre_onset"] for r in sel),
            "px4_detected": len(px4),
            "px4_flags": flags,
            "px4_delay_ms": nan_stats([1000 * r["px4_delay"] for r in px4]),
            "px4_fd_motor": sum(r["px4_fd_motor"] for r in sel),
            "lost": sum(r["lost"] for r in sel),
            "px4_held": sum(r["px4_outcome"] == "held" for r in sel),
            "px4_descended": sum(r["px4_outcome"] == "descended" for r in sel),
            "px4_flipped": sum(r["px4_outcome"] == "flipped" for r in sel),
            "time_to_loss_s": nan_stats([r["time_to_loss"] for r in sel if r["lost"]]),
            "alarm_before_loss": sum(r["alarm_before_loss"] for r in sel if r["lost"]),
            "px4_after_loss": sum(
                1 for r in sel if r["lost"] and r["px4_flag"] and r["px4_delay"] > r["time_to_loss"]
            ),
            "max_tilt_deg": nan_stats([r["max_tilt_deg"] for r in sel if not r["lost"]]),
            "alt_loss_m": nan_stats([r["alt_loss_m"] for r in sel if not r["lost"]]),
            "max_yaw_rate": nan_stats([r["max_yaw_rate"] for r in sel if not r["lost"]]),
            "loe_range": [min(r["loe"] for r in sel), max(r["loe"] for r in sel)] if sel else None,
        }

    noeff = [r for r in rows if r["group"] == "px4_off_default"]
    if noeff:
        out["px4_off_default"] = {
            "n": len(noeff),
            "injection_effective": sum(r["injection_effective"] for r in noeff),
            "min_esc_output_after": min(r["min_esc_output_after"] for r in noeff),
            "max_tilt_deg": max(r["max_tilt_deg"] for r in noeff),
            "diagnoser_alarms": sum(r["false_alarm_after_injection"] for r in noeff),
        }
    for g in ("partial", "complete", "px4_off"):
        sel = [r for r in rows if r["group"] == g]
        if not sel:
            continue
        out[g] = block(sel)
        out[g]["by_condition"] = {
            c: block([r for r in sel if r["condition"] == c])
            for c in sorted({r["condition"] for r in sel})
        }
    part = [r for r in rows if r["group"] == "partial"]
    if part:
        edges = [0.15, 0.25, 0.35, 0.5]
        out["partial"]["by_loe"] = [
            dict(
                lo=a,
                hi=b,
                **block([r for r in part if a <= r["loe"] < b or (b == 0.5 and r["loe"] == b)]),
            )
            for a, b in zip(edges[:-1], edges[1:])
        ]
    return out


def main():
    runs = load_runs()
    if not runs:
        sys.exit(f"no converted SITL logs in {NPZ}; run sitl/campaign.py first")
    runs = {k: v.resample(RATE_HZ) for k, v in runs.items()}
    af = Airframe.from_yaml(HERE / "config" / "x500.yaml")
    cal_names = [k for k in runs if k.startswith("calibration-")]
    segs = [
        LogSegment(
            runs[k],
            runs[k].meta["manifest"]["events"]["manoeuvre_s"],
            runs[k].meta["manifest"]["events"]["end_s"],
        )
        for k in cal_names
    ]
    print(f"{len(runs)} runs, {len(segs)} calibration flights")
    delay, scores = best_delay(segs, af, 4, BASELINE_S, mapper=pmap)
    print(
        f"  transport delay: {delay} samples ({1000 * delay / RATE_HZ:.0f} ms), "
        f"scores {np.round(scores, 3)}"
    )
    cal = {"delay_steps": delay, "delay_scores": scores, "rate_hz": RATE_HZ, "runs": cal_names}
    cfgs = {}
    for v, kw in VARIANTS.items():
        cfg, info = calibrate_on_logs(
            segs, af, FDIConfig(**kw), BASELINE_S, delay_steps=delay, mapper=pmap
        )
        cfgs[v] = cfg
        cal[v] = info
        print(
            f"  {v}: sigma {np.round(info['sigma'], 5)}, "
            f"thresholds {np.round(info['thresholds'], 0)}"
        )
    jobs = [
        (k, log, af, cfg, delay, v)
        for v, cfg in cfgs.items()
        for k, log in runs.items()
        if not k.startswith("calibration-")
    ]
    rows = pmap(_evaluate_job, jobs, desc="replay of the evaluation flights")
    m = re.search(r"PX4-Autopilot (\S+);", af.source)
    summary = {
        "px4_version": runs[cal_names[0]].meta.get("ver_sw"),
        "px4_describe": m.group(1) if m else None,
        "rate_hz": RATE_HZ,
        "delay_steps": delay,
        "n_runs_total": len(runs),
        "n_calibration": len(cal_names),
        "calibration_minutes": cal["proposed"]["fault_free_s"] / 60,
        "variants": {v: summarize([r for r in rows if r["variant"] == v]) for v in VARIANTS},
    }
    from figures_sitl import make_figures

    make_figures(runs, af, cfgs, delay, rows, summary, FIGURES)
    RESULTS.mkdir(parents=True, exist_ok=True)
    from rotorfault.experiments.common import save_json

    save_json(RESULTS / "sitl_calibration.json", cal)
    for v, cfg in cfgs.items():  # in the format of `rotorfault replay --calibration`
        save_calibration(RESULTS / f"sitl_calibration_{v}.json", cfg, None, delay)
    save_json(RESULTS / "sitl_summary.json", summary)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "variant", k != "run", k))
    with open(RESULTS / "sitl_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    k: (round(v, 6) if isinstance(v, float) and math.isfinite(v) else v)
                    for k, v in r.items()
                }
            )
    print(json.dumps(summary["variants"]["proposed"]["fault_free"], indent=1, default=float))
    for g in ("partial", "complete", "px4_off"):
        for v in VARIANTS:
            s = summary["variants"][v].get(g)
            if s:
                print(
                    f"  {g:8s} {v:12s} det {s['detected']}/{s['n']} delay med "
                    f"{s['delay_ms']['median']} p95 {s['delay_ms'].get('p95')} iso "
                    f"{s['isolation_correct']} loe_err {s['loe_abs_err']['median']} "
                    f"px4 {s['px4_detected']} {s['px4_flags']} {s['px4_delay_ms']['median']} "
                    f"lost {s['lost']}"
                )
    from rotorfault.experiments.report import write_reports

    write_reports()


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
