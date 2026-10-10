"""Figures of the PX4 SITL validation (called by ``sitl/analyze.py``)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rotorfault.experiments.figures import DPI, INK, INK2, MUTED, SERIES, _style  # noqa: E402
from rotorfault.replay import replay  # noqa: E402

MOTOR_NAMES = ["motor 1 (FR)", "motor 2 (RL)", "motor 3 (FL)", "motor 4 (RR)"]
MOTOR_LS = ["-", "--", "-.", ":"]
COND_ORDER = ["hover-calm", "orbit-calm", "hover-wind", "orbit-wind"]
COND_STYLE = {
    "hover-calm": dict(color=SERIES[0], marker="o"),
    "orbit-calm": dict(color=SERIES[1], marker="s"),
    "hover-wind": dict(color=SERIES[2], marker="^"),
    "orbit-wind": dict(color=SERIES[3], marker="D"),
}


def _pick(rows, group, condition, loe_target):
    sel = [
        r
        for r in rows
        if r["variant"] == "proposed" and r["group"] == group and r["condition"] == condition
    ]
    sel = [r for r in sel if r["detected"]] or sel
    return min(sel, key=lambda r: (abs(r["loe"] - loe_target), r["run"]))


def _timeline(axs, log, af, cfg, delay, row, window, title, legend_loc="lower left"):
    onset = row["onset"]
    t_det = log.meta["manifest"]["events"]["manoeuvre_s"]
    r = replay(log, af, cfg, t_det, 2.5, t_end=onset + window[1] + 0.2, delay_steps=delay)
    j = row["motor"] - 1
    m = (log.t >= onset + window[0]) & (log.t <= onset + window[1])
    tt = log.t[m] - onset
    mr = (r.t >= onset + window[0]) & (r.t <= onset + window[1])
    tr = r.t[mr] - onset
    for i in range(4):
        lw = 1.8 if i == j else 1.0
        col = SERIES[0] if i == j else MUTED
        axs[0].plot(tt, log.motor[m, i], color=col, ls=MOTOR_LS[i], lw=lw, label=MOTOR_NAMES[i])
        axs[1].plot(tr, np.clip(r.cusum[mr, i], 1e-3, 1e3), color=col, ls=MOTOR_LS[i], lw=lw)
        loe = 1.0 - r.eta_hat[mr, i] / r.eta_ref[i]
        axs[2].plot(tr, 100 * loe, color=col, ls=MOTOR_LS[i], lw=lw)
    axs[1].axhline(1.0, color=INK2, lw=0.8, ls=":")
    axs[1].text(window[0] + 0.02, 1.3, "alarm threshold", fontsize=6.5, color=INK2)
    axs[1].set_yscale("log")
    axs[2].plot(
        [0, window[1]], [100 * row["loe"]] * 2, color=INK, lw=0.8, ls=(0, (4, 2)), label="true loss"
    )
    axs[3].plot(tt, log.extra["gt_tilt_deg"][m], color=INK, lw=1.2)
    axs[0].set_ylabel("motor command")
    axs[1].set_ylabel("CUSUM / threshold")
    axs[2].set_ylabel("estimated loss (%)")
    axs[3].set_ylabel("tilt, ground truth (deg)")
    axs[3].set_xlabel("time from fault onset (s)")
    axs[1].set_ylim(1e-3, 1.5e3)
    axs[2].set_ylim(-10, 110)
    for ax in axs:
        ax.axvline(0.0, color=MUTED, lw=0.9)
        if row["detected"]:
            ax.axvline(row["delay"], color=SERIES[0], lw=0.9, ls="--")
        if row.get("px4_flag"):
            ax.axvline(row["px4_delay"], color=SERIES[1], lw=0.9, ls="-.")
        if row["lost"] and row["time_to_loss"] <= window[1]:
            ax.axvline(row["time_to_loss"], color=INK, lw=0.9, ls=":")
        ax.set_xlim(*window)
    axs[0].set_title(title, loc="left")
    y = axs[3].get_ylim()[1]
    if row["detected"]:
        axs[3].text(
            row["delay"] + 0.02,
            0.92 * y,
            f"diagnoser\n{1000 * row['delay']:.0f} ms",
            fontsize=6.5,
            color=SERIES[0],
            va="top",
        )
    if row.get("px4_flag"):
        axs[3].text(
            row["px4_delay"] + 0.02,
            0.55 * y,
            f"PX4 {row['px4_flag']}\n{1000 * row['px4_delay']:.0f} ms",
            fontsize=6.5,
            color=SERIES[1],
            va="top",
        )
    axs[0].legend(loc=legend_loc, ncol=2, fontsize=6.5)


def fig_timeline(runs, af, cfgs, delay, rows, out: Path):
    part = _pick(rows, "partial", "orbit-wind", 0.3)
    comp = _pick(rows, "complete", "hover-calm", 1.0)
    fig, axs = plt.subplots(4, 2, figsize=(7.2, 7.0), sharex="col")
    _timeline(
        axs[:, 0],
        runs[part["run"]],
        af,
        cfgs["proposed"],
        delay,
        part,
        (-0.5, 2.0),
        f"(a) {100 * part['loe']:.0f} % loss, motor {part['motor']}, orbit in wind",
    )
    _timeline(
        axs[:, 1],
        runs[comp["run"]],
        af,
        cfgs["proposed"],
        delay,
        comp,
        (-0.3, 1.5),
        f"(b) complete loss, motor {comp['motor']}, hover",
        legend_loc="center right",
    )
    fig.align_ylabels(axs[:, 0])
    handles = [
        plt.Line2D([], [], color=MUTED, lw=0.9, label="fault onset"),
        plt.Line2D([], [], color=SERIES[0], lw=0.9, ls="--", label="diagnoser alarm"),
        plt.Line2D([], [], color=SERIES[1], lw=0.9, ls="-.", label="PX4 failure detector"),
        plt.Line2D([], [], color=INK, lw=0.9, ls=":", label="tilt past 90 deg"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return part["run"], comp["run"]


def fig_delay(rows, summary, out: Path):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    for c in COND_ORDER:
        st = COND_STYLE[c]
        for v, fill in (("proposed", True), ("no_adaptive", False)):
            sel = [
                r
                for r in rows
                if r["variant"] == v
                and r["condition"] == c
                and r["group"] == "partial"
                and r["detected"]
            ]
            ax.scatter(
                [100 * r["loe"] for r in sel],
                [1000 * r["delay"] for r in sel],
                s=16,
                marker=st["marker"],
                facecolors=st["color"] if fill else "none",
                edgecolors=st["color"],
                linewidths=0.9,
                label=c if v == "proposed" else None,
            )
            bad = [r for r in sel if not r["isolation_correct"]]
            ax.scatter(
                [100 * r["loe"] for r in bad],
                [1000 * r["delay"] for r in bad],
                s=40,
                marker="+",
                color=INK,
                linewidths=1.0,
                zorder=5,
            )
    ax.scatter([], [], s=40, marker="+", color=INK, linewidths=1.0, label="wrong motor isolated")
    ax.set_xlabel("true thrust loss of the faulty motor (%)")
    ax.set_ylabel("detection delay (ms)")
    ax.set_ylim(0, 170)
    ax.text(
        0.98,
        0.02,
        "filled: as designed\nopen: without adaptive scale",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color=INK2,
    )
    ax.set_title("(a) partial losses (PX4 raised no flag)", loc="left")
    ax.legend(fontsize=6.3, loc="center right", ncol=2, handletextpad=0.2, columnspacing=0.8)
    ax = axs[1]
    comp = {
        v: sorted(
            [r for r in rows if r["variant"] == v and r["group"] == "complete"],
            key=lambda r: (COND_ORDER.index(r["condition"]), r["motor"]),
        )
        for v in ("proposed", "no_adaptive")
    }
    labels = []
    for i, (r, q) in enumerate(zip(comp["proposed"], comp["no_adaptive"])):
        if r["detected"]:
            ax.plot(1000 * r["delay"], i, "o", color=SERIES[0], ms=4)
        if q["detected"]:
            ax.plot(1000 * q["delay"], i, "o", mfc="none", mec=SERIES[0], ms=4)
            if not q["isolation_correct"]:
                ax.plot(1000 * q["delay"], i, "+", color=INK, ms=7, mew=1.0)
        if r["px4_flag"]:
            ax.plot(1000 * r["px4_delay"], i, "x", color=INK, ms=4)
        if r["lost"]:
            ax.plot(1000 * r["time_to_loss"], i, "|", color=SERIES[1], ms=7, mew=1.5)
        labels.append(f"{r['condition']}, m{r['motor']}")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.set_xlim(0, 720)
    ax.set_xlabel("time after the complete loss (ms)")
    ax.set_title("(b) complete losses", loc="left")
    ax.plot([], [], "o", color=SERIES[0], ms=4, label="diagnoser, as designed")
    ax.plot([], [], "o", mfc="none", mec=SERIES[0], ms=4, label="diagnoser, no adaptive scale")
    ax.plot([], [], "|", color=SERIES[1], ms=7, mew=1.5, label="tilt past 90 deg")
    ax.plot([], [], "x", color=INK, ms=4, label="PX4 failure detector")
    ax.plot([], [], "+", color=INK, ms=7, mew=1.0, label="wrong motor isolated")
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.grid(axis="y", alpha=0.0)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def make_figures(runs, af, cfgs, delay, rows, summary, outdir: Path):
    _style()
    outdir.mkdir(parents=True, exist_ok=True)
    picked = fig_timeline(runs, af, cfgs, delay, rows, outdir / "fig_sitl_timeline.png")
    fig_delay(rows, summary, outdir / "fig_sitl_delay.png")
    summary.setdefault("figures", {})["timeline_runs"] = list(picked)
    print(f"  wrote figures to {outdir} (timeline: {picked})")
