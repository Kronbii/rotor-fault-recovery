"""Command-line interface: ``rotorfault <command>`` or ``python -m rotorfault <command>``."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np


def _cmd_simulate(a):
    from pathlib import Path

    from . import metrics as M
    from .experiments.common import default_paths, export_replay, load_json
    from .experiments.runner import fdi_config
    from .fdi import FDIConfig
    from .scenarios import make_scenario
    from .sim import simulate

    paths = default_paths()
    cal_file = paths.results / "calibration.json"
    cfg = FDIConfig()
    if cal_file.exists() and "gyro_x1" in load_json(cal_file):
        cfg = fdi_config(load_json(cal_file)["gyro_x1"])
    sc = make_scenario(
        a.traj,
        a.duration,
        rotor=a.rotor - 1,
        magnitude=a.loe,
        onset=a.onset,
        gyro_scale=a.gyro_noise,
        turbulence=a.turbulence,
    )
    t0 = time.time()
    r = simulate(sc, a.baseline, a.seed, cfg)
    onset = a.onset if a.loe > 0 else 0.0
    out = dict(
        baseline=a.baseline,
        lost=r.crashed,
        lost_reason=r.crash_reason or None,
        alarm_delay_s=(r.alarm_time - a.onset) if math.isfinite(r.alarm_time) else None,
        isolated_rotor=(r.isolated + 1) if r.isolated >= 0 else None,
        relaxed_switch_s=r.switch_relaxed if math.isfinite(r.switch_relaxed) else None,
        max_deviation_m=M.max_deviation(r, onset),
        settled_error_m=M.settled_error(r),
        final_spin_rad_s=M.spin_rate(r),
        wall_time_s=time.time() - t0,
    )
    print(
        json.dumps(
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}, indent=1
        )
    )
    if a.replay:
        export_replay(Path(a.replay), r)
        print(f"replay written to {a.replay}")
    if a.plot:
        _quick_plot(r, a.plot)
        print(f"plot written to {a.plot}")


def _quick_plot(r, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axs = plt.subplots(4, 1, figsize=(7, 8), sharex=True)
    axs[0].plot(r.t, r.position - r.p_ref)
    axs[0].set_ylabel("position error (m)")
    axs[0].legend(["x", "y", "z"], fontsize=7)
    axs[1].plot(r.t, r.omega)
    axs[1].set_ylabel("body rates (rad/s)")
    axs[2].plot(r.t, r.thrust)
    axs[2].set_ylabel("rotor thrust (N)")
    axs[3].plot(r.t, r.eta_hat)
    axs[3].plot(r.t, r.eta_true, "k", lw=0.6)
    axs[3].set_ylabel("eta_hat")
    axs[3].set_xlabel("time (s)")
    for ax in axs:
        ax.grid(alpha=0.3)
        if np.isfinite(r.fault_onset):
            ax.axvline(r.fault_onset, color="0.5", lw=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def _cmd_equilibrium(a):
    import numpy as np

    from .ftc import solve_relaxed_hover
    from .params import nominal_params

    P = nominal_params()
    eq = solve_relaxed_hover(P, a.rotor - 1, a.tilt)
    print(f"failed rotor {a.rotor}, primary-axis tilt {a.tilt} deg")
    print(f"  rotor thrusts (N): {np.round(eq.f, 3).tolist()}  (limit {P.f_max:.2f} N)")
    print(f"  body rate (rad/s): {np.round(eq.omega, 3).tolist()}  |omega| = {eq.spin:.2f}")
    print(f"  primary axis n:    {np.round(eq.n, 4).tolist()}")
    print(
        f"  total thrust {eq.total_thrust:.3f} N vs weight {P.weight:.3f} N, "
        f"residual {eq.residual:.1e}"
    )


def _cmd_calibrate(a):
    from .experiments.calibration import calibrate
    from .experiments.common import default_paths

    c = calibrate(default_paths(), a.gyro_noise, force=True, quick=a.quick)
    print("sigma", c.sigma.tolist(), "thresholds", list(c.thresholds))


def _cmd_experiments(a):
    from .experiments.run_all import run_experiments

    only = a.only.split(",") if a.only else None
    print(run_experiments(quick=a.quick, only=only))


def _cmd_figures(a):
    from .experiments.figures import make_all_figures

    make_all_figures()


def _cmd_report(a):
    from .experiments.report import write_reports

    write_reports()


def _cmd_reproduce(a):
    from .experiments.run_all import reproduce

    t0 = time.time()
    reproduce(quick=a.quick)
    print(f"reproduce finished in {time.time() - t0:.0f} s")


def _load_for_replay(path, af, rate):
    from .logs.flightlog import load_any

    log = load_any(path, af)
    target = rate or min(500.0, round(log.rate_hz))
    if abs(log.rate_hz - target) > 1.0 or np.ptp(np.diff(log.t)) > 0.5 * log.dt:
        log = log.resample(target)
    return log


def _cmd_replay(a):
    from .airframe import Airframe
    from .fdi import FDIConfig
    from .replay import auto_detect_start, load_baseline, load_calibration, replay

    af = Airframe.from_yaml(a.airframe)
    eta_ref = None
    if a.calibration:
        cfg, delay = load_calibration(a.calibration)
        if a.stored_baseline:
            eta_ref = load_baseline(a.calibration)
            if eta_ref is None:
                sys.exit("the calibration file has no stored baseline (eta_ref_mean)")
    else:
        cfg, delay = FDIConfig(), None
        print(
            "warning: no --calibration given; using the residual floor and thresholds of the "
            "simulated 5-inch quad (calibrate on fault-free logs with `rotorfault calibrate-logs`)",
            file=sys.stderr,
        )
    out = []
    for path in a.logs:
        log = _load_for_replay(path, af, a.rate)
        t_det = (
            a.detect_from
            if a.detect_from is not None
            else auto_detect_start(log, af, 1.0, a.baseline)
        )
        r = replay(
            log,
            af,
            cfg,
            t_det,
            a.baseline,
            t_end=a.until,
            measured_speed=a.rpm,
            delay_steps=delay,
            eta_ref=eta_ref,
        )
        k_end = len(r.t) - 1
        res = {
            "log": str(path),
            "samples_hz": round(log.rate_hz, 1),
            "detection_from_s": round(t_det, 3),
            "evaluated_s": round(float(r.t[-1] - t_det), 2) if len(r.t) else 0.0,
            "alarm": r.alarm,
            "alarm_time_s": round(r.alarm_time, 4) if r.alarm else None,
            "isolated_motor": r.isolated + 1 if r.alarm else None,
            "loe_estimate_at_alarm_plus_0p25s": (
                [round(float(v), 3) for v in r.loe_hat(r.alarm_time + 0.25)] if r.alarm else None
            ),
            "loe_estimate_at_end": [round(float(v), 3) for v in 1 - r.eta_hat[k_end] / r.eta_ref],
            "healthy_baseline_eta": [round(float(v), 4) for v in r.eta_ref],
            "max_normalised_cusum": [round(float(v), 3) for v in r.cusum.max(axis=0)],
            "used_rpm": bool(a.rpm),
        }
        out.append(res)
        if a.plot:
            _replay_plot(log, r, a.plot if len(a.logs) == 1 else f"{a.plot}.{len(out)}.png")
    txt = json.dumps(out if len(out) > 1 else out[0], indent=1)
    print(txt)
    if a.json:
        with open(a.json, "w") as f:
            f.write(txt + "\n")


def _replay_plot(log, r, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    m = (log.t >= r.t[0]) & (log.t <= r.t[-1])
    axs[0].plot(log.t[m], log.motor[m])
    axs[0].set_ylabel("motor command")
    axs[0].legend([f"motor {i + 1}" for i in range(4)], fontsize=7, ncol=4)
    axs[1].plot(r.t, np.minimum(r.cusum, 3))
    axs[1].axhline(1.0, color="k", lw=0.6, ls=":")
    axs[1].set_ylabel("CUSUM / threshold")
    axs[2].plot(r.t, 100 * (1 - r.eta_hat / r.eta_ref))
    axs[2].set_ylabel("estimated loss (%)")
    axs[2].set_xlabel("log time (s)")
    for ax in axs:
        ax.axvline(r.t_detect, color="0.5", lw=0.8)
        if r.alarm:
            ax.axvline(r.alarm_time, color="r", lw=0.8, ls="--")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"plot written to {path}", file=sys.stderr)


def _cmd_calibrate_logs(a):
    from .airframe import Airframe
    from .fdi import FDIConfig
    from .replay import (
        LogSegment,
        auto_detect_start,
        best_delay,
        calibrate_on_logs,
        save_calibration,
    )

    af = Airframe.from_yaml(a.airframe)
    segs = []
    for path in a.logs:
        log = _load_for_replay(path, af, a.rate)
        t_det = (
            a.detect_from
            if a.detect_from is not None
            else auto_detect_start(log, af, 1.0, a.baseline)
        )
        segs.append(LogSegment(log, t_det, a.until))
    delay = None
    if not a.rpm:
        delay, scores = best_delay(segs, af, 4, a.baseline)
        print(f"transport delay: {delay} samples (scores {np.round(scores, 3).tolist()})")
    base = FDIConfig(unc_inertia=0.0, unc_tau=0.0) if a.no_adaptive else FDIConfig()
    cfg, info = calibrate_on_logs(
        segs, af, base, a.baseline, measured_speed=a.rpm, delay_steps=delay
    )
    save_calibration(a.out, cfg, info, delay)
    print(
        f"{len(segs)} logs, {info['fault_free_s']:.0f} s fault-free; sigma "
        f"{np.round(info['sigma'], 5).tolist()}; thresholds "
        f"{np.round(info['thresholds'], 1).tolist()} -> {a.out}"
    )


def _cmd_identify(a):
    from . import identify as I
    from .airframe import Airframe
    from .logs.flightlog import load_any

    af = Airframe.from_yaml(a.airframe)
    log = load_any(a.log, af)
    res = {"log": a.log, "samples_hz": round(log.rate_hz, 1)}

    def attempt(name, fn):
        try:
            v = fn()
            res[name] = v.__dict__ if hasattr(v, "__dict__") else v
        except ValueError as e:
            res[name] = f"not identified: {e}"

    attempt("hover", lambda: I.hover_thrust_coefficient(log, af))
    if log.rotor_speed is not None:
        attempt(
            "speed_map",
            lambda: dict(zip(("speed_at_zero", "speed_at_full", "rms"), I.speed_map_from_rpm(log))),
        )
        attempt("motor_lag", lambda: I.motor_time_constant(log, af))
    else:
        attempt("motor_lag", lambda: I.motor_time_constant_from_accel(log, af))
    attempt("inertia_damping", lambda: I.inertia_and_damping(log, af))
    print(json.dumps(res, indent=1, default=lambda o: np.round(o, 7).tolist()))


def main(argv=None):
    p = argparse.ArgumentParser(prog="rotorfault", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="run one closed-loop simulation")
    s.add_argument("--traj", choices=["hover", "fig8", "steps"], default="hover")
    s.add_argument("--baseline", choices=["passive", "oracle", "estimated"], default="estimated")
    s.add_argument("--rotor", type=int, default=1, choices=[1, 2, 3, 4], help="1=FR 2=FL 3=RL 4=RR")
    s.add_argument("--loe", type=float, default=0.5, help="loss of effectiveness 0..1")
    s.add_argument("--onset", type=float, default=2.0)
    s.add_argument("--duration", type=float, default=8.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--gyro-noise", type=float, default=1.0, help="gyro noise scale")
    s.add_argument("--turbulence", type=float, default=1.0, help="turbulence scale")
    s.add_argument("--plot", help="write a diagnostic PNG")
    s.add_argument("--replay", help="write a replay JSON")
    s.set_defaults(fn=_cmd_simulate)

    s = sub.add_parser("equilibrium", help="relaxed-hover equilibrium after a rotor loss")
    s.add_argument("--rotor", type=int, default=1, choices=[1, 2, 3, 4])
    s.add_argument("--tilt", type=float, default=15.0)
    s.set_defaults(fn=_cmd_equilibrium)

    s = sub.add_parser("calibrate", help="calibrate the FDI on fault-free flights")
    s.add_argument("--gyro-noise", type=float, default=1.0)
    s.add_argument("--quick", action="store_true")
    s.set_defaults(fn=_cmd_calibrate)

    s = sub.add_parser("experiments", help="run experiments E1-E6")
    s.add_argument("--quick", action="store_true", help="small smoke-test version")
    s.add_argument("--only", help="comma-separated subset, e.g. e1,e4")
    s.set_defaults(fn=_cmd_experiments)

    s = sub.add_parser("figures", help="regenerate all figures from results/")
    s.set_defaults(fn=_cmd_figures)

    s = sub.add_parser("report", help="regenerate tables and README/paper blocks")
    s.set_defaults(fn=_cmd_report)

    s = sub.add_parser("reproduce", help="experiments + figures + report, end to end")
    s.add_argument("--quick", action="store_true")
    s.set_defaults(fn=_cmd_reproduce)

    s = sub.add_parser("replay", help="run the diagnoser offline on flight logs")
    s.add_argument("logs", nargs="+", help=".ulg (PX4), .csv (Betaflight blackbox_decode) or .npz")
    s.add_argument("--airframe", required=True, help="airframe YAML, e.g. config/5inch.yaml")
    s.add_argument("--calibration", help="JSON from `rotorfault calibrate-logs`")
    s.add_argument(
        "--detect-from",
        type=float,
        help="log time (s) at which detection starts; "
        "the baseline is learned over the preceding --baseline seconds",
    )
    s.add_argument("--until", type=float, help="log time (s) at which to stop")
    s.add_argument("--baseline", type=float, default=2.5)
    s.add_argument("--rate", type=float, help="resample to this rate (default: min(log, 500 Hz))")
    s.add_argument(
        "--rpm", action="store_true", help="use RPM telemetry instead of the motor model"
    )
    s.add_argument(
        "--stored-baseline",
        action="store_true",
        help="use the healthy baseline stored in --calibration instead of learning it "
        "(for faults present from take-off, e.g. a weakened propeller)",
    )
    s.add_argument("--plot", help="write a diagnostic PNG")
    s.add_argument("--json", help="write the result as JSON")
    s.set_defaults(fn=_cmd_replay)

    s = sub.add_parser("calibrate-logs", help="residual floor and thresholds from fault-free logs")
    s.add_argument("logs", nargs="+")
    s.add_argument("--airframe", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--detect-from", type=float)
    s.add_argument("--until", type=float)
    s.add_argument("--baseline", type=float, default=2.5)
    s.add_argument("--rate", type=float)
    s.add_argument("--rpm", action="store_true")
    s.add_argument("--no-adaptive", action="store_true", help="no model-uncertainty term")
    s.set_defaults(fn=_cmd_calibrate_logs)

    s = sub.add_parser("identify", help="identify airframe parameters from a flight log")
    s.add_argument("log")
    s.add_argument("--airframe", required=True)
    s.set_defaults(fn=_cmd_identify)

    a = p.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
