"""Portfolio plates for the rotor-fault study, drawn from the repository's committed results.

Square hero (1:1) and gallery plates (4:3) sized for the portfolio's frames, so nothing is cropped.
Run from the repository root after the experiments, with the output directory as the argument:

    python experiments/portfolio_plates.py <outdir>
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rotorfault.experiments.common import Paths, load_json  # noqa: E402
from rotorfault.experiments.figures import (  # noqa: E402
    BASE_STYLE,
    INK,
    ORDER,
    SERIES,
    _events,
    _load,
    _style,
)

OUT = Path(sys.argv[1])
paths = Paths(Path.cwd())
_style()
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10.5,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 2.0,
        "savefig.bbox": None,
    }
)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("wrote", OUT / name)


# 1. Hero, 1:1: complete loss of one rotor in simulated hover.
runs = {b: _load(paths, f"e4_complete_loss_{b}.npz") for b in ORDER}


def hero(figsize, adjust, name):
    fig, ax = plt.subplots(figsize=figsize)
    for b in ORDER:
        r = runs[b]
        e = np.linalg.norm(r.position - r.p_ref, axis=1)
        st = BASE_STYLE[b]
        label = {
            "passive": "nominal controller (flips)",
            "oracle": "told the true fault",
            "estimated": "this pipeline",
        }[b]
        ax.plot(r.t, 100 * e, color=st["color"], ls=st["ls"], label=label)
        if r.crashed:
            ax.plot(r.t[-1], min(100 * e[-1], 100), "x", color=st["color"], ms=11, mew=2.2)
    _events(ax, runs["estimated"], label=True)
    ax.set_xlim(1.5, 8)
    ax.set_ylim(0, 60)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("position error (cm)")
    ax.set_title(
        "Complete loss of one rotor at t = 2 s\n(simulation, 5-inch quadrotor in hover)", loc="left"
    )
    ax.legend(loc="upper right")
    fig.subplots_adjust(**adjust)
    save(fig, name)


hero((5.4, 5.4), dict(left=0.14, right=0.96, bottom=0.12, top=0.86), "plate.png")
hero((6.4, 4.8), dict(left=0.12, right=0.97, bottom=0.13, top=0.84), "complete-loss.png")

# 2. Gallery, 4:3: PX4 SITL complete losses, diagnoser vs the flip vs PX4's detector.
rows = list(csv.DictReader(open(paths.root / "sitl/results/sitl_runs.csv")))
comp = [r for r in rows if r["variant"] == "proposed" and r["group"] == "complete"]
order = ["hover-calm", "hover-wind", "orbit-calm", "orbit-wind"]
comp.sort(key=lambda r: (order.index(r["condition"]), int(r["motor"])))
fig, ax = plt.subplots(figsize=(6.4, 4.8))
for i, r in enumerate(comp):
    if r["detected"] == "True":
        ax.plot(1000 * float(r["delay"]), i, "o", color=SERIES[0], ms=7)
    if r["lost"] == "True" and r["time_to_loss"]:
        ax.plot(1000 * float(r["time_to_loss"]), i, "|", color=SERIES[1], ms=12, mew=2.4)
    if r["px4_flag"] not in ("", "False"):
        ax.plot(1000 * float(r["px4_delay"]), i, "x", color=INK, ms=7, mew=1.8)
ax.set_yticks(range(len(comp)))
ax.set_yticklabels([f"{r['condition']}, motor {r['motor']}" for r in comp], fontsize=8.5)
ax.set_xlim(0, 760)
ax.set_xlabel("time after the rotor stops (ms)")
ax.grid(axis="y", alpha=0.0)
ax.plot([], [], "o", color=SERIES[0], ms=7, label="diagnoser alarm")
ax.plot([], [], "|", color=SERIES[1], ms=12, mew=2.4, label="vehicle tilts past 90°")
ax.plot([], [], "x", color=INK, ms=7, mew=1.8, label="PX4 failure detector")
fig.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 0.915),
    ncol=3,
    fontsize=9.5,
    handletextpad=0.3,
    columnspacing=1.2,
)
fig.suptitle(
    "PX4 software-in-the-loop: 16 complete rotor losses", x=0.03, y=0.975, ha="left", fontsize=13
)
fig.subplots_adjust(left=0.27, right=0.97, bottom=0.12, top=0.83)
save(fig, "sitl-complete.png")

# 3. Gallery, 4:3: partial loss in hover, peak deviation against loss size.
s = load_json(paths.results / "e1_hover_loe.json")
fig, ax = plt.subplots(figsize=(6.4, 4.8))
for b in ORDER:
    rs = sorted([r for r in s["rows"] if r["baseline"] == b], key=lambda r: r["magnitude"])
    st = BASE_STYLE[b]
    label = {
        "passive": "nominal controller",
        "oracle": "told the true fault",
        "estimated": "this pipeline",
    }[b]
    ax.plot(
        [100 * r["magnitude"] for r in rs],
        [100 * r["max_dev"]["median"] if r["max_dev"]["median"] else np.nan for r in rs],
        color=st["color"],
        ls=st["ls"],
        marker=st["marker"],
        ms=6,
        label=label,
    )
ax.set_yscale("log")
ax.set_yticks([3, 5, 10, 20, 50, 100])
ax.set_yticklabels(["3", "5", "10", "20", "50", "100"])
ax.minorticks_off()
ax.set_xlabel("effectiveness lost on one rotor (%)")
ax.set_ylabel("median peak deviation (cm)")
ax.set_title("Partial rotor loss in hover (simulation)", loc="left")
ax.legend(loc="upper left")
fig.subplots_adjust(left=0.13, right=0.97, bottom=0.13, top=0.9)
save(fig, "partial-sweep.png")
