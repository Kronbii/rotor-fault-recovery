"""Command-line interface: ``rotorfault <command>`` or ``python -m rotorfault <command>``."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time


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

    a = p.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
