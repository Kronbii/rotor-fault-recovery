"""All figures, generated from ``results/*.json`` and the cached representative runs.

Style: one palette for the whole repository. The three baselines always use the same
categorical slots (validated palette, first three slots pass all-pairs CVD checks) and
also differ in line style, so identity never relies on colour alone.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from ..ftc import _equilibrium_for_axis, solve_relaxed_hover  # noqa: E402
from ..params import nominal_params  # noqa: E402
from ..rotations import quat_to_rot, tilt_angle  # noqa: E402
from . import e1_hover_loe, e3_trajectory, e4_complete_loss  # noqa: E402
from .common import CachedRun, Paths, default_paths, load_json  # noqa: E402
from .runner import run_task  # noqa: E402

DPI = 200
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
BASE_STYLE = {
    "estimated": dict(color=SERIES[0], ls="-", marker="o", label="estimated (proposed)"),
    "passive": dict(color=SERIES[1], ls="--", marker="s", label="passive (no FTC)"),
    "oracle": dict(color=SERIES[2], ls=":", marker="^", label="oracle fault knowledge"),
}
ORDER = ("passive", "oracle", "estimated")
ROTOR_NAMES = ["rotor 1 (FR)", "rotor 2 (FL)", "rotor 3 (RL)", "rotor 4 (RR)"]
ROTOR_LS = ["-", "--", "-.", ":"]


def _style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.facecolor": SURFACE,
            "figure.facecolor": "white",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "legend.frameon": False,
            "legend.fontsize": 7.5,
            "lines.linewidth": 1.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
        }
    )


def _save(fig, paths: Paths, name: str) -> Path:
    out = paths.figures / name
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {out.relative_to(paths.root)}")
    return out


def _panel(ax, letter):
    ax.text(-0.12, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=9.5, color=INK)


def _events(ax, r, show=("onset", "alarm", "relaxed"), label=False):
    marks = []
    if "onset" in show and np.isfinite(r.fault_onset):
        marks.append((r.fault_onset, MUTED, "-", "fault"))
    if "alarm" in show and np.isfinite(r.alarm_time):
        marks.append((r.alarm_time, SERIES[0], (0, (1, 1)), "alarm"))
    if "relaxed" in show and np.isfinite(r.switch_relaxed):
        marks.append((r.switch_relaxed, SERIES[6], (0, (3, 1, 1, 1)), "yaw-relaxed on"))
    for t, c, ls, name in marks:
        ax.axvline(t, color=c, ls=ls, lw=0.9, label=name if label else None, zorder=0)


# ======================================================================================
# cached representative runs
# ======================================================================================
def _ensure(paths: Paths, tasks: list[dict]) -> None:
    for t in tasks:
        if t.get("npz") and not Path(t["npz"]).exists():
            print(f"  re-simulating missing cache {Path(t['npz']).name}")
            t = dict(t)
            t.pop("replay", None)
            run_task(t)


def _load(paths: Paths, name: str) -> CachedRun:
    return CachedRun(paths.cache / name)


def _cal(paths: Paths) -> dict:
    db = load_json(paths.results / "calibration.json")
    return db["gyro_x1"] if "gyro_x1" in db else db["gyro_x1_quick"]


# ======================================================================================
# Figure: architecture
# ======================================================================================
def fig_architecture(paths: Paths):
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.set_xlim(0, 120)
    ax.set_ylim(-4, 35)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef4fc", ec=SERIES[0]):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=fc, ec=ec, lw=1.0
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.6, color=INK)

    def arrow(p0, p1, text=None, at=None, ha="center"):
        ax.add_patch(
            FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, color=INK2, lw=0.9)
        )
        if text:
            ax.text(at[0], at[1], text, ha=ha, va="center", fontsize=6.0, color=INK2)

    grey = dict(fc="#f3f2ef", ec=MUTED)
    green = dict(fc="#eaf7f1", ec=SERIES[2])
    X = {1: 1, 2: 24, 3: 51, 4: 78, 5: 104}
    W = {1: 17, 2: 20, 3: 20, 4: 20, 5: 15}
    top, mid, bot, H = 25, 13, 1, 8
    box(X[1], top, W[1], H, "reference\nhover / figure-8", **grey)
    box(X[2], top, W[2], H, "SE(3) position and\nattitude control")
    box(X[3], top, W[3], H, "prioritised allocation\nB diag(eta) f")
    box(X[4], top, W[4], H, "5-inch quad: 6-DOF,\nrotor lag, turbulence", **grey)
    box(X[5], top, W[5], H, "rotor faults\neta_i(t)", fc="#fdf0ea", ec=SERIES[1])
    box(X[2], mid, W[2], H, "supervisor: nominal /\npartial / relaxed", **green)
    box(X[3], mid, W[3], H, "yaw-relaxed control\n(LQR, spin about n)")
    box(X[2], bot, W[2], H, "FDI: regression,\nKF on eta, CUSUM bank", **green)
    box(X[3], bot, W[3], H, "state estimator\n(attitude, p, v)", **grey)
    box(X[4], bot, W[4], H, "gyro, accelerometer,\nmocap 100 Hz, 10 ms", **grey)
    yt = top + H / 2
    arrow((18.6, yt), (23.4, yt))
    arrow((44.6, yt), (50.4, yt), "T, tau", at=(47.5, yt + 5.6))
    arrow((71.6, yt), (77.4, yt), "f_cmd", at=(74.5, yt + 5.6))
    arrow((103.4, yt), (98.6, yt))
    arrow((88, 24.4), (88, 9.6), "measurements", at=(89.2, 17), ha="left")
    arrow((77.4, 5), (71.6, 5))
    arrow((50.4, 5), (44.6, 5), "rates, accel, f_cmd", at=(47.5, -1.4))
    arrow((34, 9.6), (34, 12.4), "alarm, rotor j, eta_hat", at=(35.2, 11.0), ha="left")
    arrow((40, 21.6), (55, 24.4), "eta_hat (partial loss)", at=(39, 23.4), ha="right")
    arrow((44.6, 17), (50.4, 17), "eta_j\n< 0.35", at=(47.5, 14.3))
    arrow((71.6, 18), (82, 24.4), "f (rotor j idle)", at=(75.5, 22.6), ha="right")
    arrow((61, 9.6), (61, 12.4))
    return _save(fig, paths, "fig_architecture.png")


# ======================================================================================
# E1
# ======================================================================================
def _band(ax, rows, base, key, scale=1.0, **kw):
    rs = sorted([r for r in rows if r["baseline"] == base], key=lambda r: r["magnitude"])
    x = np.array([100 * r["magnitude"] for r in rs])
    med = np.array([np.nan if r[key]["median"] is None else r[key]["median"] for r in rs]) * scale
    lo = np.array([np.nan if r[key]["p25"] is None else r[key]["p25"] for r in rs]) * scale
    hi = np.array([np.nan if r[key]["p75"] is None else r[key]["p75"] for r in rs]) * scale
    st = BASE_STYLE[base]
    ax.plot(x, med, color=st["color"], ls=st["ls"], marker=st["marker"], ms=4, label=st["label"])
    ax.fill_between(x, lo, hi, color=st["color"], alpha=0.15, lw=0)


def fig_e1(paths: Paths):
    s = load_json(paths.results / "e1_hover_loe.json")
    rows = s["rows"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for b in ORDER:
        _band(axs[0], rows, b, "max_dev", 100)
        _band(axs[1], rows, b, "settled_err", 100)
        rs = sorted([r for r in rows if r["baseline"] == b], key=lambda r: r["magnitude"])
        st = BASE_STYLE[b]
        axs[2].plot(
            [100 * r["magnitude"] for r in rs],
            [100 * r["lost_rate"] for r in rs],
            color=st["color"],
            ls=st["ls"],
            marker=st["marker"],
            ms=4,
            label=st["label"],
        )
    for ax, ticks in ((axs[0], [3, 5, 10, 20, 50, 100]), (axs[1], [2, 3, 5, 10, 20, 30])):
        ax.set_yscale("log")
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(t) for t in ticks])
        ax.minorticks_off()
    axs[0].set_ylabel("max deviation after fault (cm)")
    axs[1].set_ylabel("settled error, last 2 s (cm)")
    axs[2].set_ylabel("runs lost (%)")
    axs[2].set_ylim(-5, 105)
    for ax, letter in zip(axs, "abc"):
        ax.set_xlabel("loss of effectiveness, rotor 1 (%)")
        ax.set_xticks(range(0, 100, 20))
        _panel(ax, letter)
    axs[0].legend(loc="upper left")
    fig.text(
        0.5,
        -0.05,
        "Median with inter-quartile band over non-lost runs; lost = ground contact, "
        "tilt > 90 deg or error > 3 m.",
        ha="center",
        fontsize=7,
        color=INK2,
    )
    fig.tight_layout(w_pad=1.2)
    return _save(fig, paths, "fig_e1_hover_loe.png")


def fig_e1_timeseries(paths: Paths):
    cal = _cal(paths)
    _ensure(paths, e1_hover_loe.tasks(paths, cal, quick=False))
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.0), sharex=True)
    for col, m in enumerate(e1_hover_loe.REPRESENTATIVE):
        ax = axs[0, col]
        for b in ORDER:
            r = _load(paths, f"e1_hover_loe_m{int(m * 100)}_{b}.npz")
            e = np.linalg.norm(r.position - r.p_ref, axis=1)
            st = BASE_STYLE[b]
            ax.plot(r.t, 100 * e, color=st["color"], ls=st["ls"], label=st["label"])
            if b == "passive" and r.crashed:
                ax.plot(r.t[-1], 100 * e[-1], "x", color=st["color"], ms=7, mew=1.5)
                ax.annotate(
                    "lost",
                    (r.t[-1], 100 * e[-1]),
                    xytext=(4, -10),
                    textcoords="offset points",
                    fontsize=7,
                    color=INK2,
                )
        ax.set_ylim(0, None)
        ax.set_title(f"{int(m * 100)} % loss of effectiveness on rotor 1 at t = 2 s")
        ax.set_ylabel("position error (cm)")
        if m >= 0.65:
            ax.set_ylim(0, 120)
        ax2 = axs[1, col]
        r = _load(paths, f"e1_hover_loe_m{int(m * 100)}_estimated.npz")
        for j in range(4):
            ax2.plot(
                r.t, r.eta_hat[:, j], color=SERIES[j], ls=ROTOR_LS[j], lw=1.2, label=ROTOR_NAMES[j]
            )
        ax2.plot(
            r.t, r.eta_true[:, 0], color=INK, lw=0.8, ls="-", alpha=0.6, label="true eta, rotor 1"
        )
        _events(ax2, r, label=(col == 0))
        _events(ax, r)
        ax2.set_ylim(-0.05, 1.15)
        ax2.set_ylabel("effectiveness estimate eta_hat")
        ax2.set_xlabel("time (s)")
        ax2.set_xlim(1.5, 6.0)
        if np.isfinite(r.switch_relaxed):
            ax2.text(
                r.switch_relaxed + 0.05,
                0.55,
                "rotor 1 idled:\nestimate frozen",
                fontsize=6.5,
                color=INK2,
            )
    axs[0, 0].legend(loc="upper right")
    axs[1, 0].legend(loc="lower center", ncol=3, fontsize=6.0)
    for ax, letter in zip(axs.ravel(), "abcd"):
        _panel(ax, letter)
    fig.tight_layout()
    return _save(fig, paths, "fig_e1_timeseries.png")


# ======================================================================================
# E2
# ======================================================================================
def _csv(path: Path) -> list[dict]:
    import csv

    with open(path) as f:
        rows = list(csv.DictReader(f))

    def conv(v):
        if v in ("", "None"):
            return math.nan
        if v in ("True", "False"):
            return v == "True"
        try:
            return float(v)
        except ValueError:
            return v

    return [{k: conv(v) for k, v in r.items()} for r in rows]


def fig_e2(paths: Paths):
    res = load_json(paths.results / "e2_fdi.json")
    rows = _csv(paths.results / "e2_fdi_runs.csv")
    fr = [r for r in rows if r["magnitude"] > 0]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
    # (a) delay vs magnitude
    ax = axs[0]
    for traj, mk, c in (("hover", "o", SERIES[0]), ("fig8", "^", SERIES[1])):
        rs = [r for r in fr if r["traj"] == traj and r["detected"]]
        ax.scatter(
            [100 * r["magnitude"] for r in rs],
            [1e3 * r["delay"] for r in rs],
            s=9,
            marker=mk,
            color=c,
            alpha=0.75,
            lw=0,
            label="hover" if traj == "hover" else "figure-eight",
        )
    ax.set_yscale("log")
    ax.set_yticks([20, 30, 50, 100])
    ax.set_yticklabels(["20", "30", "50", "100"])
    ax.minorticks_off()
    ax.set_xlabel("true loss of effectiveness (%)")
    ax.set_ylabel("detection delay (ms)")
    ax.legend(loc="upper right")
    # (b) LOE estimate vs truth
    ax = axs[1]
    det = [r for r in fr if r["detected"]]
    ok = [r for r in det if r["isolation_correct"]]
    bad = [r for r in det if not r["isolation_correct"]]
    relax = [r for r in ok if r["used_relaxed"]]
    part = [r for r in ok if not r["used_relaxed"]]
    ax.plot([0, 100], [0, 100], color=MUTED, lw=0.8, ls="--")
    ax.scatter(
        [100 * r["magnitude"] for r in part],
        [100 * r["loe_hat"] for r in part],
        s=9,
        color=SERIES[0],
        lw=0,
        alpha=0.8,
        label="partial-loss FTC",
    )
    ax.scatter(
        [100 * r["magnitude"] for r in relax],
        [100 * r["loe_hat"] for r in relax],
        s=10,
        marker="D",
        color=SERIES[6],
        lw=0,
        alpha=0.8,
        label="switched to yaw-relaxed",
    )
    if bad:
        ax.scatter(
            [100 * r["magnitude"] for r in bad],
            [100 * r["loe_hat"] for r in bad],
            s=22,
            marker="x",
            color=SERIES[1],
            label="wrong rotor isolated",
        )
    ax.axhline(65, color=MUTED, lw=0.6, ls=":")
    ax.text(17, 91, "dotted: switch to yaw-relaxed\n(eta_hat < 0.35)", fontsize=6.0, color=INK2)
    ax.set_xlabel("true loss of effectiveness (%)")
    ax.set_ylabel("estimated loss (%)")
    ax.set_xlim(15, 95)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=6.5)
    # (c) ECDF of delay by magnitude bin
    ax = axs[2]
    for k, b in enumerate(res["main"]["by_magnitude"]):
        rs = [1e3 * r["delay"] for r in det if b["lo"] <= r["magnitude"] < b["hi"] + 1e-9]
        if not rs:
            continue
        xs = np.sort(rs)
        ax.step(
            xs,
            np.arange(1, len(xs) + 1) / len(xs),
            where="post",
            color=SERIES[k],
            ls=ROTOR_LS[k % 4],
            label=f"{int(100 * b['lo'])}-{int(round(100 * b['hi']))} %",
        )
    ax.set_xscale("log")
    ax.set_xticks([20, 30, 50, 100])
    ax.set_xticklabels(["20", "30", "50", "100"])
    ax.minorticks_off()
    ax.set_xlabel("detection delay (ms)")
    ax.set_ylabel("fraction detected")
    ax.legend(title="LOE", fontsize=6.5, title_fontsize=6.5, loc="lower right")
    for ax, letter in zip(axs, "abc"):
        _panel(ax, letter)
    fig.tight_layout(w_pad=1.0)
    return _save(fig, paths, "fig_e2_fdi.png")


def fig_e2_noise(paths: Paths):
    res = load_json(paths.results / "e2_fdi.json")
    sw = res["sweep"]
    g = np.array([s["gyro_noise_rad_s"] for s in sw]) * 180 / np.pi
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3))
    med = [s["delay_ms"]["median"] for s in sw]
    p25 = [s["delay_ms"]["p25"] for s in sw]
    p75 = [s["delay_ms"]["p75"] for s in sw]
    axs[0].plot(g, med, color=SERIES[0], marker="o", ms=4)
    axs[0].fill_between(g, p25, p75, color=SERIES[0], alpha=0.15, lw=0)
    axs[0].set_ylabel("detection delay (ms)")
    fr_own = [100 * s["false_alarm_runs"] / s["n_free_runs"] for s in sw]
    fr_fix = [
        np.nan
        if s.get("false_alarms_fixed_cal") is None
        else 100 * s["false_alarms_fixed_cal"] / s["n_free_fixed_cal"]
        for s in sw
    ]
    axs[1].plot(g, fr_own, color=SERIES[0], marker="o", ms=4, label="thresholds re-calibrated")
    axs[1].plot(
        g, fr_fix, color=SERIES[1], ls="--", marker="s", ms=4, label="thresholds from 1x noise"
    )
    axs[1].set_ylabel("fault-free runs with an alarm (%)")
    axs[1].set_ylim(-5, 105)
    axs[1].legend(loc="upper left")
    axs[2].plot(
        g,
        [s["loe_abs_err"]["median"] * 100 for s in sw],
        color=SERIES[0],
        marker="o",
        ms=4,
        label="median |error|",
    )
    axs[2].plot(
        g,
        [s["loe_abs_err"]["p95"] * 100 for s in sw],
        color=SERIES[0],
        ls=":",
        marker="^",
        ms=4,
        label="95th percentile",
    )
    axs[2].set_ylabel("LOE estimate error (points)")
    axs[2].legend(loc="upper left")
    for ax, letter in zip(axs, "abc"):
        ax.set_xscale("log")
        ax.set_xticks(g)
        ax.set_xticklabels([f"{v:.1f}" for v in g], fontsize=7)
        ax.minorticks_off()
        ax.set_xlabel("gyro noise per sample (deg/s)")
        _panel(ax, letter)
    det = ", ".join(f"{s_['detected']}/{s_['n_fault_runs']}" for s_ in sw)
    iso = ", ".join(f"{100 * s_['isolation_accuracy']:.0f}" for s_ in sw)
    fig.text(
        0.5,
        -0.08,
        f"Faults detected per level: {det}; isolation accuracy (%): {iso}. "
        "Median and inter-quartile range in (a).",
        ha="center",
        fontsize=6.5,
        color=INK2,
    )
    fig.tight_layout(w_pad=1.0)
    return _save(fig, paths, "fig_e2_noise.png")


def fig_e2_example(paths: Paths):
    """Zoom on detection: residual, decision statistics and estimate around onset."""
    cal = _cal(paths)
    task = dict(
        exp="fig",
        traj="hover",
        duration=3.0,
        baseline="estimated",
        seed=123_456,
        rotor=2,
        magnitude=0.3,
        onset=2.0,
        cal=cal,
        npz=str(paths.cache / "fig_e2_example.npz"),
    )
    _ensure(paths, [task])
    r = _load(paths, "fig_e2_example.npz")
    m = (r.t >= 1.9) & (r.t <= 2.4)
    t = 1e3 * (r.t[m] - 2.0)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3))
    sig = np.array(cal["sigma"])
    names = ["collective", "roll", "pitch", "yaw"]
    for k in range(4):
        axs[0].plot(
            t, r.residual[m, k] / sig[k], color=SERIES[k], ls=ROTOR_LS[k], lw=1.1, label=names[k]
        )
    axs[0].set_ylabel("residual / floor sigma_0")
    axs[0].legend(fontsize=5.8, loc="lower left", ncol=2, bbox_to_anchor=(0.0, -0.02))
    axs[0].set_ylim(-28, 4)
    for j in range(4):
        axs[1].plot(t, r.cusum[m, j], color=SERIES[j], ls=ROTOR_LS[j], label=ROTOR_NAMES[j])
    axs[1].axhline(1.0, color=INK2, lw=0.8)
    axs[1].text(150, 1.05, "alarm threshold", fontsize=6.5, color=INK2)
    axs[1].set_ylabel("CUSUM / threshold")
    axs[1].set_ylim(0, 2.0)

    for j in range(4):
        axs[2].plot(t, r.eta_hat[m, j], color=SERIES[j], ls=ROTOR_LS[j], label=ROTOR_NAMES[j])
    axs[2].plot(t, r.eta_true[m, 2] * r.eta_ref[2], color=INK, lw=0.8, alpha=0.6, label="truth")
    axs[2].set_ylabel("eta_hat")
    axs[2].legend(fontsize=5.8, loc="lower right", ncol=3, columnspacing=0.8)
    axs[2].set_ylim(0.25, 1.1)
    axs[2].set_yticks([0.6, 0.8, 1.0])
    for ax, letter in zip(axs, "abc"):
        ax.axvline(0, color=MUTED, lw=0.9)
        if np.isfinite(r.alarm_time):
            ax.axvline(1e3 * (r.alarm_time - 2.0), color=SERIES[0], ls=(0, (1, 1)), lw=0.9)
        ax.set_xlabel("time from fault onset (ms)")
        _panel(ax, letter)
    fig.suptitle(
        f"30 % LOE on rotor 3 in hover: alarm after {1e3 * (r.alarm_time - 2.0):.0f} ms, "
        f"isolated rotor {int(r.isolated) + 1}",
        fontsize=8.5,
        y=1.02,
    )
    fig.tight_layout(w_pad=1.0)
    return _save(fig, paths, "fig_e2_example.png")


# ======================================================================================
# E3
# ======================================================================================
def fig_e3(paths: Paths):
    cal = _cal(paths)
    _ensure(paths, e3_trajectory.tasks(paths, cal, quick=False))
    s = load_json(paths.results / "e3_trajectory.json")
    fig = plt.figure(figsize=(7.2, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.5, 0.8])
    ax = fig.add_subplot(gs[0])
    on = e3_trajectory.ONSET
    for b in ORDER:
        r = _load(paths, f"e3_trajectory_{b}.npz")
        m = r.t >= on
        st = BASE_STYLE[b]
        ax.plot(r.position[m, 0], r.position[m, 1], color=st["color"], ls=st["ls"], lw=1.0)
    ax.plot(r.p_ref[:, 0], r.p_ref[:, 1], color=INK, lw=0.6, alpha=0.5)
    k = np.searchsorted(r.t, on)
    ax.plot(*r.p_ref[k, :2], "o", color=INK, ms=4)
    ax.annotate("fault", r.p_ref[k, :2], xytext=(6, 6), textcoords="offset points", fontsize=7)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_ylim(-2.2, 2.2)
    ax.text(
        0.0, 2.0, "top view, t = 10-20 s (grey: reference)", ha="center", fontsize=6.5, color=INK2
    )
    ax2 = fig.add_subplot(gs[1])
    for b in ORDER:
        r = _load(paths, f"e3_trajectory_{b}.npz")
        e = np.linalg.norm(r.position - r.p_ref, axis=1)
        st = BASE_STYLE[b]
        ax2.plot(r.t, 100 * e, color=st["color"], ls=st["ls"], lw=1.1, label=st["label"])
    _events(ax2, _load(paths, "e3_trajectory_estimated.npz"), show=("onset", "alarm"))
    ax2.set_xlim(3, 20)
    ax2.set_ylim(0, None)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("position error (cm)")
    ax2.legend(fontsize=6.3, loc="upper right")
    ax3 = fig.add_subplot(gs[2])
    xs = np.arange(2)
    w = 0.26
    for i, b in enumerate(ORDER):
        row = next(x for x in s["rows"] if x["baseline"] == b)
        vals = [100 * row["rmse_before"]["median"], 100 * row["rmse_steady"]["median"]]
        st = BASE_STYLE[b]
        ax3.bar(
            xs + (i - 1) * w,
            vals,
            width=w * 0.92,
            color=st["color"],
            hatch=["", "///", ".."][i],
            edgecolor="white",
            lw=0,
        )
    ax3.set_xticks(xs)
    ax3.set_xticklabels(["before\n3-10 s", "after\n13-20 s"])
    ax3.set_ylabel("RMSE (cm), median of 10")
    for a, letter in zip((ax, ax2, ax3), "abc"):
        _panel(a, letter)
    fig.suptitle(
        "Figure-eight at up to 3.3 m/s with a 50 % LOE on rotor 1 at t = 10 s", fontsize=8.5, y=1.02
    )
    fig.tight_layout(w_pad=1.0)
    return _save(fig, paths, "fig_e3_trajectory.png")


# ======================================================================================
# E4 / E5
# ======================================================================================
def _axis_angle_deg(r, n_body):
    """Angle between the body primary axis and world vertical (deg)."""
    out = np.empty(len(r.t))
    for k, q in enumerate(r.quat):
        out[k] = math.degrees(math.acos(np.clip((quat_to_rot(q) @ n_body)[2], -1, 1)))
    return out


def fig_e4(paths: Paths):
    cal = _cal(paths)
    _ensure(paths, e4_complete_loss.tasks("e4_complete_loss", paths, cal, quick=False))
    runs = {b: _load(paths, f"e4_complete_loss_{b}.npz") for b in ORDER}
    eq = solve_relaxed_hover(nominal_params(), 0, 15.0)
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.4), sharex=True)
    ax = axs[0, 0]
    for b in ORDER:
        r = runs[b]
        e = np.linalg.norm(r.position - r.p_ref, axis=1)
        st = BASE_STYLE[b]
        ax.plot(r.t, 100 * e, color=st["color"], ls=st["ls"], label=st["label"])
        if r.crashed:
            ax.plot(r.t[-1], min(100 * e[-1], 150), "x", color=st["color"], ms=7, mew=1.5)
            ax.annotate(
                f"lost ({r.crash_reason}) at {r.crash_time:.2f} s",
                (r.t[-1], min(100 * e[-1], 150)),
                xytext=(6, -2),
                textcoords="offset points",
                fontsize=6.8,
                color=INK2,
            )
    ax.set_ylim(0, 160)
    ax.set_ylabel("position error (cm)")
    ax.legend(loc="upper right")
    ax = axs[0, 1]
    for b in ORDER:
        r = runs[b]
        st = BASE_STYLE[b]
        ax.plot(
            r.t, np.degrees([tilt_angle(q) for q in r.quat]), color=st["color"], ls=st["ls"], lw=1.1
        )
    r = runs["estimated"]
    ax.plot(
        r.t,
        _axis_angle_deg(r, eq.n),
        color=SERIES[6],
        lw=1.1,
        label="primary axis n from vertical (estimated)",
    )
    ax.set_ylabel("tilt of body z (deg)")
    ax.set_ylim(0, 95)
    ax.legend(loc="upper right", fontsize=6.5)
    ax = axs[1, 0]
    for b in ORDER:
        r = runs[b]
        st = BASE_STYLE[b]
        ax.plot(r.t, np.linalg.norm(r.omega, axis=1), color=st["color"], ls=st["ls"], lw=1.1)
    ax.axhline(np.deg2rad(2000), color=INK2, lw=0.8, ls="-")
    ax.text(2.2, np.deg2rad(2000) + 0.8, "gyro range +-2000 deg/s", fontsize=6.5, color=INK2)
    ax.axhline(eq.spin, color=SERIES[6], lw=0.7, ls=":")
    ax.text(6.0, eq.spin - 3.5, f"design equilibrium {eq.spin:.1f} rad/s", fontsize=6.5, color=INK2)
    ax.set_ylabel("|omega| (rad/s)")
    ax.set_ylim(0, 40)
    ax.set_xlabel("time (s)")
    ax = axs[1, 1]
    r = runs["estimated"]
    for j in range(4):
        ax.plot(r.t, r.thrust[:, j], color=SERIES[j], ls=ROTOR_LS[j], lw=1.1, label=ROTOR_NAMES[j])
    ax.axhline(nominal_params().f_max, color=INK2, lw=0.8)
    ax.text(5.2, nominal_params().f_max - 0.45, "rotor thrust limit", fontsize=6.5, color=INK2)
    ax.set_ylabel("rotor thrust (N)")
    ax.set_xlabel("time (s)")
    ax.legend(loc="center right", fontsize=6.5, title="proposed pipeline", title_fontsize=6.5)
    for a, letter in zip(axs.ravel(), "abcd"):
        _events(a, runs["estimated"])
        a.set_xlim(1.0, 10.0)
        _panel(a, letter)
    fig.suptitle("Complete loss of rotor 1 in hover at t = 2 s", fontsize=9, y=1.0)
    fig.tight_layout()
    return _save(fig, paths, "fig_e4_complete_loss.png")


def fig_e4_3d(paths: Paths):
    runs = {b: _load(paths, f"e4_complete_loss_{b}.npz") for b in ("passive", "estimated")}
    fig = plt.figure(figsize=(7.2, 3.2))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for b in ("passive", "estimated"):
        r = runs[b]
        m = r.t >= 1.5
        st = BASE_STYLE[b]
        ax.plot(
            r.position[m, 0],
            r.position[m, 1],
            r.position[m, 2],
            color=st["color"],
            ls=st["ls"],
            lw=1.0,
            label=st["label"],
        )
    r = runs["estimated"]
    # body z axis "whiskers" every 50 ms show the spinning, tilted attitude
    for k in range(np.searchsorted(r.t, 2.0), len(r.t), 25):
        R = quat_to_rot(r.quat[k])
        p = r.position[k]
        d = 0.08 * R[:, 2]
        ax.plot(
            [p[0], p[0] + d[0]],
            [p[1], p[1] + d[1]],
            [p[2], p[2] + d[2]],
            color=SERIES[6],
            lw=0.5,
            alpha=0.6,
        )
    ax.set_xlabel("x (m)", labelpad=-2)
    ax.set_ylabel("y (m)", labelpad=-2)
    ax.set_zlabel("z (m)", labelpad=-2)
    ax.tick_params(labelsize=6.5, pad=-1)
    ax.set_title("3-D path (violet: body z-axis every 50 ms)", fontsize=8)
    ax.legend(fontsize=6.5, loc="upper left")
    ax2 = fig.add_subplot(1, 2, 2)
    m = r.t >= 2.0
    ax2.plot(
        100 * (r.position[m, 0] - r.p_ref[m, 0]),
        100 * (r.position[m, 1] - r.p_ref[m, 1]),
        color=SERIES[0],
        lw=0.8,
    )
    ax2.plot(0, 0, "+", color=INK, ms=10)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x error (cm)")
    ax2.set_ylabel("y error (cm)")
    ax2.set_title("estimated pipeline: horizontal error from t = 2 s", fontsize=8)
    fig.tight_layout()
    return _save(fig, paths, "fig_e4_3d.png")


def fig_e5(paths: Paths):
    cal = _cal(paths)
    _ensure(paths, e4_complete_loss.tasks("e5_loss_in_flight", paths, cal, quick=False))
    runs = {b: _load(paths, f"e5_loss_in_flight_{b}.npz") for b in ORDER}
    on = e4_complete_loss.CASES["e5_loss_in_flight"]["onset"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(width_ratios=[1.2, 1.4]))
    ax = axs[0]
    for b in ORDER:
        r = runs[b]
        m = r.t >= on - 1.0
        st = BASE_STYLE[b]
        ax.plot(
            r.position[m, 0],
            r.position[m, 1],
            color=st["color"],
            ls=st["ls"],
            lw=1.0,
            label=st["label"],
        )
        if r.crashed:
            ax.plot(r.position[-1, 0], r.position[-1, 1], "x", color=st["color"], ms=7, mew=1.5)
    r = runs["estimated"]
    ax.plot(r.p_ref[:, 0], r.p_ref[:, 1], color=INK, lw=0.6, alpha=0.5, label="reference")
    k = np.searchsorted(r.t, on)
    ax.plot(*r.p_ref[k, :2], "o", color=INK, ms=4)
    ax.annotate(
        "rotor 1 lost", r.p_ref[k, :2], xytext=(5, -12), textcoords="offset points", fontsize=7
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax = axs[1]
    for b in ORDER:
        r = runs[b]
        e = np.linalg.norm(r.position - r.p_ref, axis=1)
        st = BASE_STYLE[b]
        ax.plot(r.t, 100 * e, color=st["color"], ls=st["ls"], lw=1.1)
    _events(ax, runs["estimated"], label=True)
    ax.set_ylim(0, 60)
    ax.set_xlim(3, 16)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("position error (cm)")
    ax.legend(fontsize=6.5, loc="upper right")
    for a, letter in zip(axs, "ab"):
        _panel(a, letter)
    fig.suptitle(
        "Complete loss of rotor 1 at t = 6 s while flying the figure-eight (up to 3.3 m/s)",
        fontsize=8.5,
        y=1.02,
    )
    fig.tight_layout()
    return _save(fig, paths, "fig_e5_loss_in_flight.png")


# ======================================================================================
# E6 and design figures
# ======================================================================================
def fig_e6(paths: Paths):
    res = load_json(paths.results / "e6_robustness.json")["sweeps"]
    titles = {
        "actuator_delay_ms": ("actuator delay (ms)", 2),
        "motor_tau_ms": ("true motor time constant (ms)", 25),
        "damping_scale": ("rotational damping x", 1.0),
        "inertia_scale": ("inertia x", 1.0),
        "turbulence_scale": ("turbulence intensity x", 1.0),
    }
    fig, axs = plt.subplots(2, 5, figsize=(7.4, 3.2), sharey="row")
    for i, (key, (xl, nom)) in enumerate(titles.items()):
        pts = res[key]
        x = [p["value"] for p in pts]
        lost = [100 * p["lost"] / p["n"] for p in pts]
        ax = axs[0, i]
        ax.bar(range(len(x)), lost, color=SERIES[1], width=0.7, edgecolor="white")
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels([f"{v:g}" for v in x], fontsize=6.5)
        ax.set_ylim(0, 105)
        k = x.index(nom) if nom in x else None
        if k is not None:
            ax.get_xticklabels()[k].set_fontweight("bold")
        ax2 = axs[1, i]
        se = [
            np.nan if p["settled_err"]["median"] is None else 100 * p["settled_err"]["median"]
            for p in pts
        ]
        sp = [
            np.nan if p["final_spin"]["median"] is None else p["final_spin"]["median"] for p in pts
        ]
        ax2.plot(range(len(x)), se, color=SERIES[0], marker="o", ms=3.5, label="settled error (cm)")
        ax2.plot(
            range(len(x)),
            sp,
            color=SERIES[6],
            marker="D",
            ms=3.5,
            ls="--",
            label="spin rate (rad/s)",
        )
        ax2.set_xticks(range(len(x)))
        ax2.set_xticklabels([f"{v:g}" for v in x], fontsize=6.5)
        ax2.set_xlabel(xl, fontsize=7)
        if k is not None:
            ax2.get_xticklabels()[k].set_fontweight("bold")
    axs[0, 0].set_ylabel("runs lost (%)")
    axs[1, 0].set_ylabel("median value")
    axs[1, 0].legend(fontsize=6, loc="upper left")
    axs[1, 0].set_ylim(0, 50)
    fig.suptitle(
        "Complete loss in hover, estimated pipeline: one true-system property changed at a time "
        "(bold = value assumed by the controller); 8 runs per point",
        fontsize=7.5,
        y=1.02,
    )
    fig.tight_layout(w_pad=0.4, h_pad=0.8)
    return _save(fig, paths, "fig_e6_robustness.png")


def fig_relaxed_family(paths: Paths):
    P = nominal_params()
    tilts = np.linspace(2, 30, 15)
    spin, fmax, fmin, ftot = [], [], [], []
    for th in tilts:
        eq = solve_relaxed_hover(P, 0, float(th))
        spin.append(eq.spin)
        fa = eq.f[eq.active]
        fmax.append(fa.max() / P.f_max)
        fmin.append(fa.min() / P.f_max)
        ftot.append(eq.total_thrust / P.weight)
    # azimuth dependence at the chosen tilt
    th = math.radians(15.0)
    phis = np.linspace(0, 360, 181)
    sp_phi = []
    for ph in np.radians(phis):
        n = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
        sol = _equilibrium_for_axis(P, 0, n)
        ok = sol is not None and sol[1][[1, 2, 3]].min() > 0 and sol[1].max() < P.f_max
        sp_phi.append(sol[0] if ok else np.nan)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3))
    axs[0].plot(tilts, spin, color=SERIES[6], marker="o", ms=3)
    axs[0].axhline(np.deg2rad(2000), color=INK2, lw=0.8)
    axs[0].text(3, np.deg2rad(2000) + 0.7, "gyro range", fontsize=6.5, color=INK2)
    axs[0].axvline(15, color=MUTED, ls=":", lw=0.8)
    axs[0].set_ylabel("equilibrium spin |omega| (rad/s)")
    axs[0].set_xlabel("primary-axis tilt from body z (deg)")
    axs[1].plot(tilts, 100 * np.array(fmax), color=SERIES[0], marker="o", ms=3, label="largest")
    axs[1].plot(
        tilts, 100 * np.array(fmin), color=SERIES[1], ls="--", marker="s", ms=3, label="smallest"
    )
    axs[1].plot(
        tilts,
        100 * (np.array(ftot) - 1),
        color=SERIES[2],
        ls=":",
        marker="^",
        ms=3,
        label="extra total thrust",
    )
    axs[1].axvline(15, color=MUTED, ls=":", lw=0.8)
    axs[1].set_ylabel("% of rotor limit / of weight")
    axs[1].set_xlabel("primary-axis tilt from body z (deg)")
    axs[1].legend(fontsize=6.5)
    axs[2].plot(phis, sp_phi, color=SERIES[6])
    axs[2].set_xlabel("primary-axis azimuth at 15 deg tilt (deg)")
    axs[2].set_ylabel("spin (rad/s)")
    eq = solve_relaxed_hover(P, 0, 15.0)
    az = math.degrees(math.atan2(eq.n[1], eq.n[0])) % 360
    axs[2].axvline(az, color=MUTED, ls=":", lw=0.8)
    axs[2].text(az + 5, max(np.nanmin(sp_phi) + 1, 5), "chosen", fontsize=6.5, color=INK2)
    for a, letter in zip(axs, "abc"):
        _panel(a, letter)
    fig.suptitle(
        "Relaxed-hover equilibria with rotor 1 failed (design choice: 15 deg tilt, "
        "minimum-spin azimuth)",
        fontsize=8,
        y=1.03,
    )
    fig.tight_layout(w_pad=1.0)
    return _save(fig, paths, "fig_relaxed_family.png")


def fig_e7(paths: Paths):
    from .report import VARIANT_LABEL

    res = load_json(paths.results / "e7_fdi_ablation.json")["variants"]
    keys = list(VARIANT_LABEL)
    labels = [VARIANT_LABEL[k].replace(" (", "\n(") for k in keys]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
    y = np.arange(len(keys))[::-1]
    fa = [100 * res[k]["false_alarm_runs"] / res[k]["n_free_runs"] for k in keys]
    pre = [100 * res[k]["false_alarms_before_onset"] / res[k]["n_fault_runs"] for k in keys]
    axs[0].barh(y + 0.18, fa, height=0.34, color=SERIES[1], label="fault-free runs")
    axs[0].barh(
        y - 0.18,
        pre,
        height=0.34,
        color=SERIES[1],
        alpha=0.5,
        hatch="///",
        edgecolor="white",
        label="faulty runs, before onset",
    )
    axs[0].set_xlabel("runs with a false alarm (%)")
    axs[0].legend(fontsize=6.3, loc="lower right")
    med = [res[k]["delay_ms"]["median"] or np.nan for k in keys]
    p95 = [res[k]["delay_ms"]["p95"] or np.nan for k in keys]
    axs[1].barh(y + 0.18, med, height=0.34, color=SERIES[0], label="median")
    axs[1].barh(
        y - 0.18,
        p95,
        height=0.34,
        color=SERIES[0],
        alpha=0.5,
        hatch="///",
        edgecolor="white",
        label="95th percentile",
    )
    axs[1].set_xlabel("detection delay (ms)")
    axs[1].set_xlim(0, 125)
    axs[1].legend(fontsize=6.3, loc="upper right")
    iso = [100 * (res[k]["isolation_accuracy"] or 0) for k in keys]
    axs[2].barh(y, iso, height=0.5, color=SERIES[2])
    axs[2].set_xlim(0, 105)
    axs[2].set_xlabel("isolation accuracy (%)")
    for i, a in enumerate(axs):
        a.set_yticks(y)
        a.set_yticklabels(labels if i == 0 else [], fontsize=7)
        a.grid(axis="y", visible=False)
        _panel(a, "abc"[i])
    fig.tight_layout(w_pad=0.8)
    return _save(fig, paths, "fig_e7_ablation.png")


def fig_summary(paths: Paths):
    """Key figure: E1 loss rate vs magnitude and E4 complete-loss recovery."""
    s = load_json(paths.results / "e1_hover_loe.json")
    e4 = load_json(paths.results / "e4_complete_loss.json")
    runs = {b: _load(paths, f"e4_complete_loss_{b}.npz") for b in ORDER}
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.5), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    for b in ORDER:
        rs = sorted([r for r in s["rows"] if r["baseline"] == b], key=lambda r: r["magnitude"])
        st = BASE_STYLE[b]
        x = [100 * r["magnitude"] for r in rs]
        axs[0].plot(
            x,
            [100 * r["max_dev"]["median"] if r["max_dev"]["median"] else np.nan for r in rs],
            color=st["color"],
            ls=st["ls"],
            marker=st["marker"],
            ms=3.5,
            label=st["label"],
        )
        lost = [100 * r["lost_rate"] for r in rs]
        x4 = x + [100]
        row4 = next(r for r in e4["rows"] if r["baseline"] == b)
        lost4 = lost + [100 * row4["lost"] / row4["n"]]
        axs[1].plot(x4, lost4, color=st["color"], ls=st["ls"], marker=st["marker"], ms=3.5)
    axs[0].set_yscale("log")
    axs[0].set_yticks([3, 5, 10, 20, 50, 100])
    axs[0].set_yticklabels(["3", "5", "10", "20", "50", "100"])
    axs[0].minorticks_off()
    axs[0].set_xlabel("LOE on rotor 1 in hover (%)")
    axs[0].set_ylabel("median max deviation (cm)")
    axs[0].legend(fontsize=6.3, loc="upper left")
    axs[1].set_xlabel("LOE on rotor 1 in hover (%)")
    axs[1].set_ylabel("runs lost (%)")
    axs[1].set_ylim(-5, 105)
    axs[1].text(97, 50, "100 %:\nall 4 rotors\n(E4)", fontsize=6, color=INK2, ha="right")
    ax = axs[2]
    for b in ORDER:
        r = runs[b]
        e = np.linalg.norm(r.position - r.p_ref, axis=1)
        st = BASE_STYLE[b]
        ax.plot(r.t, 100 * e, color=st["color"], ls=st["ls"], lw=1.2)
        if r.crashed:
            ax.plot(r.t[-1], min(100 * e[-1], 100), "x", color=st["color"], ms=7, mew=1.5)
    _events(ax, runs["estimated"], label=True)
    ax.set_xlim(1.5, 10)
    ax.set_ylim(0, 60)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("position error (cm)")
    ax.set_title("complete loss of rotor 1 at t = 2 s", fontsize=8)
    ax.legend(fontsize=6.3, loc="upper right")
    for a, letter in zip(axs, "abc"):
        _panel(a, letter)
    fig.tight_layout(w_pad=1.0)
    return _save(fig, paths, "fig_summary.png")


def make_all_figures(paths: Paths | None = None) -> None:
    paths = paths or default_paths()
    _style()
    print("figures")
    fig_architecture(paths)
    fig_relaxed_family(paths)
    fig_e1(paths)
    fig_e1_timeseries(paths)
    fig_e2(paths)
    fig_e2_noise(paths)
    fig_e2_example(paths)
    fig_e3(paths)
    fig_e4(paths)
    fig_e4_3d(paths)
    fig_e5(paths)
    fig_e6(paths)
    fig_e7(paths)
    fig_summary(paths)


if __name__ == "__main__":  # pragma: no cover
    make_all_figures()
