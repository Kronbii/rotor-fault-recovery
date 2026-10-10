"""Tables and headline numbers generated from ``results/*.json``.

Writes ``results/summary.md``, ``results/headline.json``, LaTeX tables and number
macros under ``paper/generated/``, and refreshes the auto-generated blocks of
``README.md`` and ``paper/report.md`` (between ``<!-- BEGIN:x -->`` / ``<!-- END:x -->``
markers). Nothing in those blocks should be edited by hand.
"""

from __future__ import annotations

import math
import re

from .common import Paths, default_paths, load_json, save_json

BASES = ("passive", "oracle", "estimated")
BASE_LABEL = {"passive": "Passive", "oracle": "Oracle", "estimated": "Estimated (proposed)"}


def _f(x, nd=1, scale=1.0, dash="--"):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return dash
    return f"{x * scale:.{nd}f}"


def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _tex_table(header, rows, caption, label, colspec=None, wide=False):
    colspec = colspec or ("l" + "c" * (len(header) - 1))
    env = "table*" if wide else "table"

    def esc(s):
        return str(s).replace("%", r"\%").replace("--", "--").replace("_", r"\_")

    lines = [
        r"\begin{" + env + "}[t]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{adjustbox}{max width=\linewidth}",
        r"\begin{tabular}{" + colspec + "}",
        r"\toprule",
        " & ".join(esc(h) for h in header) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{" + env + "}"]
    return "\n".join(lines)


# ======================================================================================
def table_e1(e1):
    have = sorted({x["magnitude"] for x in e1["rows"]})
    mags = [m for m in have if any(abs(m - w) < 1e-9 for w in (0.2, 0.4, 0.6, 0.7, 0.8, 0.9))]
    mags = mags or [m for m in have if m > 0]
    rows = []
    for m in mags:
        row = [f"{int(m * 100)} %"]
        for b in BASES:
            r = next(x for x in e1["rows"] if abs(x["magnitude"] - m) < 1e-9 and x["baseline"] == b)
            row.append(
                f"{_f(r['max_dev']['median'], 1, 100)} / {_f(r['settled_err']['median'], 1, 100)}"
                f" ({r['lost']}/{r['n']})"
            )
        rows.append(row)
    header = ["LOE rotor 1"] + [f"{BASE_LABEL[b]}" for b in BASES]
    note = (
        "Hover, fault at t = 2 s. Each cell: median max deviation / median settled error "
        "(cm) over non-lost runs, (lost/total)."
    )
    return header, rows, note


def table_e2(e2):
    m = e2["main"]
    rows = []
    for b in m["by_magnitude"]:
        d = b["delay_ms"]
        rows.append(
            [
                f"{int(round(100 * b['lo']))}-{int(round(100 * b['hi']))} %",
                f"{b['detected']}/{b['n']}",
                f"{_f(d['median'], 0)} ({_f(d['p95'], 0)})",
                _f(b["isolation"], 1, 100) + " %",
                _f(b["loe_abs_err"]["median"], 1, 100)
                + f" ({_f(b['loe_abs_err']['p95'], 1, 100)})",
            ]
        )
    d = m["delay_ms"]
    rows.append(
        [
            "all 20-90 %",
            f"{m['detected']}/{m['n_fault_runs']}",
            f"{_f(d['median'], 0)} ({_f(d['p95'], 0)})",
            _f(m["isolation_accuracy"], 1, 100) + " %",
            _f(m["loe_abs_err"]["median"], 1, 100) + f" ({_f(m['loe_abs_err']['p95'], 1, 100)})",
        ]
    )
    header = [
        "LOE",
        "detected",
        "delay ms: median (p95)",
        "isolation",
        "abs. LOE error, pts: median (p95)",
    ]
    note = (
        f"{m['n_fault_runs']} faulty runs (hover and figure-eight). False alarms: "
        f"{m['false_alarms_total']} in {m['fault_free_flight_time_s'] / 60:.0f} min of "
        f"fault-free flight ({m['n_free_runs']} fault-free runs incl. unseen aggressive steps, "
        "plus pre-onset time of the faulty runs)."
    )
    return header, rows, note


def table_e2_noise(e2):
    rows = []
    for s in e2["sweep"]:
        d = s["delay_ms"]
        rows.append(
            [
                f"{s['gyro_scale']:g}x ({math.degrees(s['gyro_noise_rad_s']):.1f})",
                f"{s['detected']}/{s['n_fault_runs']}",
                f"{_f(d['median'], 0)} ({_f(d['p95'], 0)})",
                _f(s["isolation_accuracy"], 1, 100) + " %",
                _f(s["loe_abs_err"]["median"], 1, 100),
                f"{s['false_alarms_total']}",
                "--"
                if s.get("false_alarms_fixed_cal") is None
                else f"{s['false_alarms_fixed_cal']}/{s['n_free_fixed_cal']} runs",
                f"{s['lost_fault_runs']}",
            ]
        )
    header = [
        "gyro noise (deg/s)",
        "detected",
        "delay ms: median (p95)",
        "isolation",
        "abs. LOE error, pts",
        "false alarms",
        "false alarms, 1x thresholds",
        "lost",
    ]
    note = (
        "1x = main experiment (200 + 100 runs); other levels 40 faulty + 20 fault-free runs with "
        "residual floor and thresholds re-calibrated for that level; the second false-alarm column "
        "re-flies the 20 fault-free runs with the thresholds calibrated at 1x."
    )
    return header, rows, note


def table_e3(e3):
    rows = []
    for b in BASES:
        r = next(x for x in e3["rows"] if x["baseline"] == b)
        rows.append(
            [
                BASE_LABEL[b],
                _f(r["rmse_before"]["median"], 1, 100),
                _f(r["rmse_transient"]["median"], 1, 100),
                _f(r["rmse_steady"]["median"], 1, 100),
                _f(r["max_dev"]["median"], 1, 100),
                f"{r['lost']}/{r['n']}",
            ]
        )
    header = [
        "Baseline",
        "RMSE before",
        "RMSE 0-3 s after",
        "RMSE 3-10 s after",
        "max dev.",
        "lost",
    ]
    note = (
        "Figure-eight (peak 3.3 m/s), 50 % LOE on rotor 1 at t = 10 s; medians over 10 seeds, cm."
    )
    return header, rows, note


def table_e4(e4, e5):
    rows = []
    for name, res in (("hover", e4), ("figure-eight", e5)):
        for b in BASES:
            r = next(x for x in res["rows"] if x["baseline"] == b)
            rows.append(
                [
                    name,
                    BASE_LABEL[b],
                    f"{r['lost']}/{r['n']}",
                    _f(r["time_to_loss"]["median"], 2) if r["lost"] else "--",
                    _f(r["max_dev"]["median"], 1, 100),
                    _f(r["settled_err"]["median"], 1, 100),
                    _f(r["final_spin"]["median"], 1),
                    _f(r["switch_delay_ms"]["median"], 0),
                ]
            )
    header = [
        "case",
        "baseline",
        "lost",
        "time to loss (s)",
        "max dev. (cm)",
        "settled (cm)",
        "spin (rad/s)",
        "switch delay (ms)",
    ]
    note = (
        "Complete loss of one rotor (all four rotors in turn, several seeds each). "
        "Spin = mean |omega| over the last 2 s; switch delay = onset to yaw-relaxed control."
    )
    return header, rows, note


def table_e6(e6):
    names = {
        "actuator_delay_ms": "actuator delay (ms)",
        "motor_tau_ms": "motor time constant (ms)",
        "damping_scale": "rotational damping (x)",
        "inertia_scale": "inertia (x)",
        "turbulence_scale": "turbulence (x)",
    }
    rows = []
    for k, label in names.items():
        pts = e6["sweeps"][k]
        rows.append([label, ", ".join(f"{p['value']:g}: {p['lost']}/{p['n']}" for p in pts)])
    header = ["true-system change", "runs lost per value"]
    note = (
        "Complete loss in hover, estimated pipeline; the controller always assumes the "
        "nominal model."
    )
    return header, rows, note


VARIANT_LABEL = {
    "full": "proposed (all elements)",
    "no_baseline": "no healthy baseline",
    "no_adaptive": "no adaptive residual scale",
    "single_cusum": "single CUSUM (15 %)",
}


def table_e7(e7):
    rows = []
    for k, label in VARIANT_LABEL.items():
        s = e7["variants"][k]
        d = s["delay_ms"]
        rows.append(
            [
                label,
                f"{s['detected']}/{s['n_fault_runs']}",
                f"{_f(d['median'], 0)} ({_f(d['p95'], 0)})",
                _f(s["isolation_accuracy"], 1, 100) + " %",
                _f(s["loe_abs_err"]["median"], 1, 100),
                f"{s['false_alarm_runs']}/{s['n_free_runs']} (+{s['false_alarms_before_onset']})",
                f"{s['lost_fault_runs'] + s['lost_free_runs']}",
            ]
        )
    header = [
        "FDI variant",
        "detected",
        "delay ms: median (p95)",
        "isolation",
        "abs. LOE error, pts",
        "false-alarm runs (+ before onset)",
        "lost",
    ]
    note = (
        "60 faulty + 30 fault-free runs per variant (same seeds), each variant calibrated "
        "with the same procedure; detected counts alarms after onset."
    )
    return header, rows, note


# ======================================================================================
def headline(paths: Paths) -> dict:
    e1 = load_json(paths.results / "e1_hover_loe.json")
    e2 = load_json(paths.results / "e2_fdi.json")
    e3 = load_json(paths.results / "e3_trajectory.json")
    e4 = load_json(paths.results / "e4_complete_loss.json")
    e5 = load_json(paths.results / "e5_loss_in_flight.json")
    e6 = load_json(paths.results / "e6_robustness.json")
    e7 = load_json(paths.results / "e7_fdi_ablation.json")

    def e1row(m, b):
        return next(x for x in e1["rows"] if abs(x["magnitude"] - m) < 1e-9 and x["baseline"] == b)

    def e4row(res, b):
        return next(x for x in res["rows"] if x["baseline"] == b)

    m = e2["main"]
    bins = m["by_magnitude"]
    h = dict(
        e1_passive_maxdev_50_cm=100 * e1row(0.5, "passive")["max_dev"]["median"],
        e1_est_maxdev_50_cm=100 * e1row(0.5, "estimated")["max_dev"]["median"],
        e1_oracle_maxdev_50_cm=100 * e1row(0.5, "oracle")["max_dev"]["median"],
        e1_passive_lost_70=e1row(0.7, "passive")["lost"],
        e1_passive_lost_90=e1row(0.9, "passive")["lost"],
        e1_n=e1row(0.9, "passive")["n"],
        e1_est_lost_any=sum(x["lost"] for x in e1["rows"] if x["baseline"] == "estimated"),
        e1_est_total=sum(x["n"] for x in e1["rows"] if x["baseline"] == "estimated"),
        e1_est_maxdev_90_cm=100 * e1row(0.9, "estimated")["max_dev"]["median"],
        e2_n_fault=m["n_fault_runs"],
        e2_detected=m["detected"],
        e2_isolation_pct=100 * m["isolation_accuracy"],
        e2_delay_median_ms=m["delay_ms"]["median"],
        e2_delay_p95_ms=m["delay_ms"]["p95"],
        e2_delay_low_bin_median_ms=bins[0]["delay_ms"]["median"],
        e2_delay_high_bin_median_ms=bins[-1]["delay_ms"]["median"],
        e2_loe_abs_err_median_pts=100 * m["loe_abs_err"]["median"],
        e2_loe_abs_err_p95_pts=100 * m["loe_abs_err"]["p95"],
        e2_false_alarms=m["false_alarms_total"],
        e2_free_minutes=m["fault_free_flight_time_s"] / 60,
        e2_n_free=m["n_free_runs"],
        e2_noise_max_scale=e2["sweep"][-1]["gyro_scale"],
        e2_noise_max_deg=math.degrees(e2["sweep"][-1]["gyro_noise_rad_s"]),
        e2_noise_max_delay_ms=e2["sweep"][-1]["delay_ms"]["median"],
        e2_noise_max_isolation_pct=100 * (e2["sweep"][-1]["isolation_accuracy"] or 0),
        e2_noise_max_detected=e2["sweep"][-1]["detected"],
        e2_noise_max_n=e2["sweep"][-1]["n_fault_runs"],
        e2_noise_max_fa=e2["sweep"][-1]["false_alarms_total"],
        e3_rmse_before_cm=100 * e4row(e3, "estimated")["rmse_before"]["median"],
        e3_passive_transient_cm=100 * e4row(e3, "passive")["rmse_transient"]["median"],
        e3_est_transient_cm=100 * e4row(e3, "estimated")["rmse_transient"]["median"],
        e3_passive_maxdev_cm=100 * e4row(e3, "passive")["max_dev"]["median"],
        e3_est_maxdev_cm=100 * e4row(e3, "estimated")["max_dev"]["median"],
        e4_n=e4row(e4, "estimated")["n"],
        e4_passive_lost=e4row(e4, "passive")["lost"],
        e4_passive_ttl_s=e4row(e4, "passive")["time_to_loss"]["median"],
        e4_est_lost=e4row(e4, "estimated")["lost"],
        e4_oracle_lost=e4row(e4, "oracle")["lost"],
        e4_est_maxdev_cm=100 * e4row(e4, "estimated")["max_dev"]["median"],
        e4_oracle_maxdev_cm=100 * e4row(e4, "oracle")["max_dev"]["median"],
        e4_est_settled_cm=100 * e4row(e4, "estimated")["settled_err"]["median"],
        e4_est_spin=e4row(e4, "estimated")["final_spin"]["median"],
        e4_est_switch_ms=e4row(e4, "estimated")["switch_delay_ms"]["median"],
        e4_est_tilt_deg=e4row(e4, "estimated")["max_tilt_deg"]["median"],
        e5_n=e4row(e5, "estimated")["n"],
        e5_est_lost=e4row(e5, "estimated")["lost"],
        e5_passive_lost=e4row(e5, "passive")["lost"],
        e5_est_maxdev_cm=100 * e4row(e5, "estimated")["max_dev"]["median"],
        e5_est_settled_cm=100 * e4row(e5, "estimated")["settled_err"]["median"],
    )
    for k in ("no_baseline", "no_adaptive", "single_cusum"):
        v = e7["variants"][k]
        h[f"e7_{k}_fa_runs"] = v["false_alarm_runs"]
        h[f"e7_{k}_fa_pre"] = v["false_alarms_before_onset"]
        h[f"e7_{k}_delay_ms"] = v["delay_ms"]["median"]
        h[f"e7_{k}_isolation_pct"] = 100 * (v["isolation_accuracy"] or 0)
    h["e7_n_free"] = e7["variants"]["full"]["n_free_runs"]
    h["e7_n_fault"] = e7["variants"]["full"]["n_fault_runs"]
    # robustness limits: first swept value at which any run is lost
    lim = {}
    for k, pts in e6["sweeps"].items():
        lost = [p for p in pts if p["lost"] > 0]
        lim[k] = [(p["value"], p["lost"], p["n"]) for p in lost]
    h["e6_failures"] = lim
    return h


ABSTRACT = (
    "We study rotor-fault tolerance for a simulated 5-inch FPV-class quadrotor "
    "(0.64 kg, thrust-to-weight 3.6) with sensor noise, a 10 ms motion-capture latency, "
    "parameter uncertainty and light turbulence. An online estimator of the four rotor "
    "effectiveness factors (filtered rigid-body regression, Kalman filter) feeds a bank of "
    "CUSUM tests for detection and isolation; a supervisor then either re-allocates "
    "thrust with the estimated effectiveness (partial loss) or, when a rotor can no longer "
    "carry its share of the weight, switches to a yaw-relaxed controller in the spirit of "
    "Mueller and D'Andrea (2014) that spins the vehicle about a tilted primary axis "
    "(gain-scheduled LQR on the reduced attitude). In {e2_n_fault} Monte-Carlo runs with "
    "20-90 % loss of effectiveness, {e2_detect_phrase} (median delay "
    "{e2_delay_median_ms:.0f} ms, 95th percentile {e2_delay_p95_ms:.0f} ms) and "
    "{e2_iso_phrase}, with a median magnitude error of {e2_loe_abs_err_median_pts:.1f} "
    "percentage points; {e2_fa_phrase} in {e2_free_minutes:.0f} min of fault-free "
    "flight. With a 50 % loss in hover the median peak deviation drops from "
    "{e1_passive_maxdev_50_cm:.0f} cm (passive) to {e1_est_maxdev_50_cm:.1f} cm. After a "
    "complete rotor loss in hover the passive controller lost the vehicle in "
    "{e4_passive_lost}/{e4_n} runs (median {e4_passive_ttl_s:.2f} s), while the proposed "
    "pipeline recovered {e4_n_ok}/{e4_n} with a median peak deviation of "
    "{e4_est_maxdev_cm:.0f} cm and a spin of {e4_est_spin:.0f} rad/s; it also recovered "
    "{e5_n_ok}/{e5_n} complete losses during a 3.3 m/s figure-eight. These are simulation "
    "results with assumed, not identified, parameters; in the robustness sweeps "
    "complete-loss recovery started to fail at {e6_limits}."
)


def _e6_limits(h: dict) -> str:
    names = {
        "actuator_delay_ms": "an actuator delay of {:g} ms (2 ms assumed)",
        "motor_tau_ms": "a motor time constant of {:g} ms (25 ms assumed)",
        "damping_scale": "{:g}x the rotational damping",
        "inertia_scale": "{:g}x the inertia",
        "turbulence_scale": "{:g}x the default turbulence",
    }
    parts = [names[k].format(v[0][0]) for k, v in h["e6_failures"].items() if v]
    if not parts:
        return "none of the tested values"
    return ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1]


def abstract_text(h: dict) -> str:
    d = dict(h)
    d["e6_limits"] = _e6_limits(h)
    n, k = h["e2_n_fault"], h["e2_detected"]
    d["e2_detect_phrase"] = (
        "all faults were detected" if k == n else f"{k}/{n} faults were detected"
    )
    iso = h["e2_isolation_pct"]
    d["e2_iso_phrase"] = (
        "all were isolated to the correct rotor"
        if iso >= 100.0 - 1e-9
        else f"{iso:.1f} % were isolated to the correct rotor"
    )
    fa = h["e2_false_alarms"]
    d["e2_fa_phrase"] = (
        "no false alarm occurred"
        if fa == 0
        else f"{fa} false alarm{'s' if fa > 1 else ''} occurred"
    )
    d["e4_n_ok"] = h["e4_n"] - h["e4_est_lost"]
    d["e5_n_ok"] = h["e5_n"] - h["e5_est_lost"]
    return ABSTRACT.format(**d)


E6_NAMES = {
    "actuator_delay_ms": ("actuator delay", "ms", 2),
    "motor_tau_ms": ("motor time constant", "ms", 25),
    "damping_scale": ("rotational damping", "x nominal", 1.0),
    "inertia_scale": ("inertia", "x nominal", 1.0),
    "turbulence_scale": ("turbulence", "x default", 1.0),
}


def e6_text(e6) -> str:
    fails, ok = [], []
    for k, (label, unit, nom) in E6_NAMES.items():
        pts = e6["sweeps"][k]
        bad = [p for p in pts if p["lost"] > 0]
        vals = [p["value"] for p in pts]
        rng = f"{min(vals):g}-{max(vals):g} {unit}"
        if bad:
            lost = ", ".join(
                "{:g} {}: {}/{} lost".format(p["value"], unit, p["lost"], p["n"]) for p in bad
            )
            fails.append(f"{label} ({lost}; tested {rng}, nominal {nom:g})")
        else:
            ok.append(f"{label} ({rng})")
    txt = "Generated from `results/e6_robustness.json`. "
    if ok:
        joined = ", ".join(ok[:-1]) + (" and " if len(ok) > 1 else "") + ok[-1]
        txt += "Every run recovered over the whole tested range of " + joined + ". "
    if fails:
        txt += "Runs were lost for: " + "; ".join(fails) + "."
    return txt


# ======================================================================================
# PX4 SITL validation (sitl/analyze.py writes sitl/results/sitl_summary.json)
COND_LABEL = {
    "hover-calm": "hover, calm",
    "orbit-calm": "orbit, calm",
    "hover-wind": "hover, wind",
    "orbit-wind": "orbit, wind",
}


def _px4_outcome(b) -> str:
    parts = [
        f"{name} {b[k]}"
        for k, name in (
            ("px4_held", "held"),
            ("px4_descended", "descended"),
            ("px4_flipped", "flipped"),
        )
        if b[k]
    ]
    return ", ".join(parts)


def table_sitl(s):
    P, N = s["variants"]["proposed"], s["variants"]["no_adaptive"]

    def row(label, cond, b, nb):
        d, dn = b["delay_ms"], nb["delay_ms"]
        px4 = "none"
        if b["px4_detected"]:
            px4 = (
                f"{'/'.join(b['px4_flags'])} {b['px4_detected']}/{b['n']}, "
                f"{_f(b['px4_delay_ms']['median'], 0)} ms"
            )
        return [
            label,
            cond,
            f"{b['n']}",
            f"{b['detected']}/{b['n']}",
            f"{b['isolation_correct']}/{b['n']}",
            f"{_f(d['median'], 0)} ({_f(d['p95'], 0)})",
            f"{_f(dn['median'], 0)} ({_f(dn['p95'], 0)}), {nb['isolation_correct']}/{nb['n']}",
            _f(b["loe_abs_err"]["median"], 1, 100),
            px4,
            _px4_outcome(b),
        ]

    lo, hi = P["partial"]["loe_range"]
    rows = []
    for c in COND_LABEL:
        if c in P["partial"]["by_condition"]:
            rows.append(
                row(
                    "partial",
                    COND_LABEL[c],
                    P["partial"]["by_condition"][c],
                    N["partial"]["by_condition"][c],
                )
            )
    rows.append(row("partial", "all", P["partial"], N["partial"]))
    rows.append(row("complete", "all four", P["complete"], N["complete"]))
    header = [
        "fault",
        "condition",
        "runs",
        "detected",
        "isolated",
        "delay ms: median (p95)",
        "no adaptive scale: delay (p95), isolated",
        "abs. loss error, pts",
        "PX4 failure detector",
        "PX4 after the fault",
    ]
    ff = P["fault_free"]
    fa_n = ff["false_alarm_runs"] + ff["false_alarms_pre_onset"]
    ffn = N["fault_free"]
    fa_nn = ffn["false_alarm_runs"] + ffn["false_alarms_pre_onset"]
    ver = s.get("px4_describe") or s["px4_version"][:10]
    note = (
        f"PX4 {ver} SITL, x500 in Gazebo Harmonic; the diagnoser is replayed offline on the "
        f"ULogs. Partial: thrust loss of {100 * lo:.0f}-{100 * hi:.0f} % on a random motor, "
        "onset 8-14 s into the manoeuvre; complete: each motor in each condition (the loss "
        f"estimate is clipped at 100 %). False alarms: {fa_n} (as designed) and {fa_nn} (no "
        "adaptive "
        f"scale) in {ff['minutes'] + ff['pre_onset_minutes']:.0f} min of fault-free flight "
        f"({ff['n_runs']} held-out flights plus the time before each onset); calibration on "
        f"{s['n_calibration']} other fault-free flights. PX4 after the fault, within 8 s "
        "(6 s for complete losses): held altitude (less than 1 m lost), descended (more than "
        "1 m lost, tilt below 90 deg) or flipped."
    )
    return header, rows, note


def sitl_headline(s) -> dict:
    P, N = s["variants"]["proposed"], s["variants"]["no_adaptive"]
    pa, co, ff = P["partial"], P["complete"], P["fault_free"]
    bins = pa["by_loe"]
    h = dict(
        sitl_n_total=s["n_runs_total"],
        sitl_n_cal=s["n_calibration"],
        sitl_n_free=ff["n_runs"],
        sitl_free_min=ff["minutes"] + ff["pre_onset_minutes"],
        sitl_fa=ff["false_alarm_runs"] + ff["false_alarms_pre_onset"],
        sitl_fa_noad=N["fault_free"]["false_alarm_runs"]
        + N["fault_free"]["false_alarms_pre_onset"],
        sitl_part_n=pa["n"],
        sitl_part_det=pa["detected"],
        sitl_part_iso=pa["isolation_correct"],
        sitl_part_loe_lo=100 * pa["loe_range"][0],
        sitl_part_loe_hi=100 * pa["loe_range"][1],
        sitl_part_delay_med=pa["delay_ms"]["median"],
        sitl_part_delay_p95=pa["delay_ms"]["p95"],
        sitl_part_loe_err=100 * pa["loe_abs_err"]["median"],
        sitl_part_loe_err_p95=100 * pa["loe_abs_err"]["p95"],
        sitl_part_delay_noad=N["partial"]["delay_ms"]["median"],
        sitl_part_iso_noad=N["partial"]["isolation_correct"],
        sitl_part_px4=pa["px4_detected"],
        sitl_part_held=pa["px4_held"],
        sitl_part_descended=pa["px4_descended"],
        sitl_part_flipped=pa["px4_flipped"],
        sitl_part_held_low=bins[0]["px4_held"] + bins[1]["px4_held"],
        sitl_part_n_low=bins[0]["n"] + bins[1]["n"],
        sitl_part_held_high=bins[2]["px4_held"],
        sitl_part_n_high=bins[2]["n"],
        sitl_bin_edge=100 * bins[2]["lo"],
        sitl_comp_n=co["n"],
        sitl_comp_det=co["detected"],
        sitl_comp_iso=co["isolation_correct"],
        sitl_comp_delay_med=co["delay_ms"]["median"],
        sitl_comp_delay_p95=co["delay_ms"]["p95"],
        sitl_comp_delay_noad=N["complete"]["delay_ms"]["median"],
        sitl_comp_iso_noad=N["complete"]["isolation_correct"],
        sitl_comp_px4=co["px4_detected"],
        sitl_comp_px4_ms=co["px4_delay_ms"]["median"],
        sitl_comp_flip_ms=1000 * co["time_to_loss_s"]["median"],
        sitl_comp_flipped=co["px4_flipped"],
        sitl_comp_before=co["alarm_before_loss"],
        sitl_comp_px4_after=co.get("px4_after_loss", 0),
        sitl_delay_ms=1000 * s["delay_steps"] / s["rate_hz"],
    )
    if "px4_off" in P:
        h.update(
            sitl_pxoff_n=P["px4_off"]["n"],
            sitl_pxoff_flipped=P["px4_off"]["px4_flipped"],
            sitl_pxoff_px4_ms=P["px4_off"]["px4_delay_ms"]["median"],
        )
    if "px4_off_default" in P:
        d = P["px4_off_default"]
        h.update(sitl_pxdef_n=d["n"], sitl_pxdef_effective=d["injection_effective"])
    return h


SITL_TEXT = (
    "Generated from `sitl/results/sitl_summary.json`. In {sitl_n_total} PX4 SITL flights "
    "({sitl_n_cal} of them for calibration), the diagnoser detected {sitl_part_det}/"
    "{sitl_part_n} partial losses of {sitl_part_loe_lo:.0f}-{sitl_part_loe_hi:.0f} % "
    "(median delay {sitl_part_delay_med:.0f} ms, 95th percentile {sitl_part_delay_p95:.0f} "
    "ms), isolated {sitl_part_iso}/{sitl_part_n} to the right motor and estimated the loss "
    "with a median error of {sitl_part_loe_err:.1f} points; it detected {sitl_comp_det}/"
    "{sitl_comp_n} complete losses after a median {sitl_comp_delay_med:.0f} ms, "
    "{sitl_comp_before_phrase} the vehicle tilted past 90 deg (median {sitl_comp_flip_ms:.0f} "
    "ms). {sitl_fa_phrase} "
    "in {sitl_free_min:.0f} min of fault-free flight. PX4's failure detector raised "
    "{sitl_part_px4_phrase} for the partial losses, and for the complete losses its "
    "attitude-limit check fired {sitl_comp_px4}/{sitl_comp_n} times after a median "
    "{sitl_comp_px4_ms:.0f} ms, in {sitl_comp_px4_after} of them after the tilt had passed "
    "90 deg. Without the adaptive residual scale the median delays drop "
    "to {sitl_part_delay_noad:.0f} ms (partial) and {sitl_comp_delay_noad:.0f} ms "
    "(complete), but {sitl_iso_err_noad} of {sitl_n_faulty} faults are then isolated to the "
    "wrong motor. After a partial loss PX4 held altitude in {sitl_part_held_low}/"
    "{sitl_part_n_low} runs below {sitl_bin_edge:.0f} % loss and in {sitl_part_held_high}/"
    "{sitl_part_n_high} runs above.{sitl_pxoff_phrase}"
)


def sitl_text(h) -> str:
    d = dict(h)
    fa = h["sitl_fa"]
    d["sitl_fa_phrase"] = "No false alarm occurred" if fa == 0 else f"{fa} false alarms occurred"
    d["sitl_part_px4_phrase"] = (
        "no flag" if h["sitl_part_px4"] == 0 else f"a flag in {h['sitl_part_px4']} runs"
    )
    d["sitl_n_faulty"] = h["sitl_part_n"] + h["sitl_comp_n"]
    nb, nc = h["sitl_comp_before"], h["sitl_comp_flipped"]
    d["sitl_comp_before_phrase"] = "all before" if nb == nc else f"{nb} of {nc} before"
    d["sitl_iso_err_noad"] = d["sitl_n_faulty"] - h["sitl_part_iso_noad"] - h["sitl_comp_iso_noad"]
    d["sitl_pxoff_phrase"] = ""
    if "sitl_pxdef_n" in h and "sitl_pxoff_n" in h:
        n0, eff = h["sitl_pxdef_n"], h["sitl_pxdef_effective"]
        d["sitl_pxoff_phrase"] = (
            f" PX4's own `failure motor off` injection changed nothing in {n0 - eff}/{n0} "
            "flights with the default CA_FAILURE_MODE = 0; with CA_FAILURE_MODE = 1 it stopped "
            f"the motor and the vehicle flipped in {h['sitl_pxoff_flipped']}/{h['sitl_pxoff_n']} "
            f"(attitude check after a median {h['sitl_pxoff_px4_ms']:.0f} ms)."
        )
    return SITL_TEXT.format(**d)


ABSTRACT_SITL = (
    " In {sitl_n_total} PX4 software-in-the-loop flights of a Gazebo x500, replayed offline "
    "with parameters read from the simulator's model files, the same diagnoser detected and "
    "isolated {sitl_part_iso_word} {sitl_part_n} injected partial losses "
    "({sitl_part_loe_lo:.0f}-{sitl_part_loe_hi:.0f} %, median delay {sitl_part_delay_med:.0f} "
    "ms) and {sitl_comp_iso_word} {sitl_comp_n} complete losses (median "
    "{sitl_comp_delay_med:.0f} ms), with {sitl_fa_word} in {sitl_free_min:.0f} min of "
    "fault-free flight; PX4's own failure detector flagged {sitl_part_px4_word} of the "
    "partial losses and the complete losses only after {sitl_comp_px4_ms:.0f} ms{sitl_flip_phrase}."
)


def abstract_sitl(h) -> str:
    d = dict(h)

    def word(k, n):
        return "all" if k == n else f"{k} of"

    d["sitl_part_iso_word"] = word(h["sitl_part_iso"], h["sitl_part_n"])
    d["sitl_comp_iso_word"] = word(h["sitl_comp_iso"], h["sitl_comp_n"])
    d["sitl_fa_word"] = "no false alarm" if h["sitl_fa"] == 0 else f"{h['sitl_fa']} false alarms"
    d["sitl_part_px4_word"] = "none" if h["sitl_part_px4"] == 0 else str(h["sitl_part_px4"])
    n_after, n = h["sitl_comp_px4_after"], h["sitl_comp_px4"]
    d["sitl_flip_phrase"] = (
        ", once the vehicle had flipped"
        if n and n_after == n
        else f" (after the vehicle had flipped in {n_after} of {n})"
        if n_after
        else ""
    )
    return ABSTRACT_SITL.format(**d)


def load_sitl(paths: Paths):
    f = paths.root / "sitl" / "results" / "sitl_summary.json"
    return load_json(f) if f.exists() else None


def _inject_inline(path, key: str, content: str) -> None:
    if not path.exists():
        return
    s = path.read_text()
    s = re.sub(
        rf"(<!-- BEGIN:{key} -->)(.*?)(<!-- END:{key} -->)",
        lambda m: m.group(1) + content + m.group(3),
        s,
        flags=re.S,
    )
    path.write_text(s)


def _inject(path, key: str, content: str) -> bool:
    if not path.exists():
        return False
    s = path.read_text()
    pat = re.compile(rf"(<!-- BEGIN:{key} -->\n)(.*?)(<!-- END:{key} -->)", re.S)
    if not pat.search(s):
        return False
    s = pat.sub(lambda m: m.group(1) + content.rstrip() + "\n" + m.group(3), s)
    path.write_text(s)
    return True


def _tex_macros(h: dict) -> str:
    def name(k):
        k = re.sub(r"[^a-zA-Z0-9]", " ", k).title().replace(" ", "")
        for dgt, w in zip(
            "0123456789",
            ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"],
        ):
            k = k.replace(dgt, w)
        return "res" + k

    lines = ["% generated by rotorfault.experiments.report -- do not edit"]
    for k, v in h.items():
        if isinstance(v, (int, float)):
            if isinstance(v, int) or float(v).is_integer() and abs(v) > 3:
                txt = f"{int(round(v))}"
            elif abs(v) >= 10:
                txt = f"{v:.0f}"
            else:
                txt = f"{v:.1f}"
            lines.append(rf"\newcommand{{\{name(k)}}}{{{txt}}}")
    return "\n".join(lines) + "\n"


def write_reports(paths: Paths | None = None) -> dict:
    paths = paths or default_paths()
    e1 = load_json(paths.results / "e1_hover_loe.json")
    e2 = load_json(paths.results / "e2_fdi.json")
    e3 = load_json(paths.results / "e3_trajectory.json")
    e4 = load_json(paths.results / "e4_complete_loss.json")
    e5 = load_json(paths.results / "e5_loss_in_flight.json")
    e6 = load_json(paths.results / "e6_robustness.json")
    e7 = load_json(paths.results / "e7_fdi_ablation.json")
    tables = {
        "table_e1": (
            table_e1(e1),
            "E1: hover with a step loss of effectiveness on rotor 1.",
            "tab:e1",
        ),
        "table_e2": (table_e2(e2), "E2: fault detection and isolation Monte Carlo.", "tab:e2"),
        "table_e2_noise": (table_e2_noise(e2), "E2: sensitivity to gyro noise.", "tab:e2noise"),
        "table_e3": (table_e3(e3), "E3: figure-eight tracking with a 50 % loss.", "tab:e3"),
        "table_e4": (table_e4(e4, e5), "E4/E5: complete loss of one rotor.", "tab:e4"),
        "table_e6": (table_e6(e6), "E6: robustness of complete-loss recovery.", "tab:e6"),
        "table_e7": (table_e7(e7), "E7: FDI ablation.", "tab:e7"),
    }
    gen = paths.paper / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    md_all = ["# Results summary (generated)\n"]
    for key, ((header, rows, note), cap, label) in tables.items():
        md = _md_table(header, rows) + f"\n\n*{note}*\n"
        md_all.append(f"## {cap}\n\n{md}")
        for doc in (paths.root / "README.md", paths.paper / "report.md"):
            _inject(doc, key, md)
        wide = key in ("table_e2_noise", "table_e4", "table_e7", "table_e1")
        cap_tex = cap.replace("%", r"\%") + " " + note.replace("%", r"\%")
        cap_tex = cap_tex.replace("|omega|", r"$|\omega|$")
        (gen / f"{key}.tex").write_text(_tex_table(header, rows, cap_tex, label, wide=wide) + "\n")
    sitl = load_sitl(paths)
    sh = {}
    if sitl is not None:
        header, rows, note = table_sitl(sitl)
        md = _md_table(header, rows) + f"\n\n*{note}*\n"
        md_all.append(f"## PX4 SITL validation.\n\n{md}")
        for doc in (paths.root / "README.md", paths.paper / "report.md"):
            _inject(doc, "table_sitl", md)
        cap_tex = ("PX4 SITL validation. " + note).replace("%", r"\%")
        (gen / "table_sitl.tex").write_text(
            _tex_table(header, rows, cap_tex, "tab:sitl", wide=True) + "\n"
        )
        sh = sitl_headline(sitl)
        st = sitl_text(sh)
        for doc in (paths.root / "README.md", paths.paper / "report.md"):
            _inject(doc, "sitl_text", st)
        st_tex = st.split(". ", 1)[1] if st.startswith("Generated from") else st
        st_tex = st_tex.replace("%", r"\%").replace("`", "").replace("_", r"\_")
        (gen / "sitl_text.tex").write_text(st_tex + "\n")
    et = e6_text(e6)
    for doc in (paths.root / "README.md", paths.paper / "report.md"):
        _inject(doc, "e6_text", et)
    et_tex = et.split(". ", 1)[1] if et.startswith("Generated from") else et
    (gen / "e6_text.tex").write_text(et_tex.replace("%", r"\%").replace("`", "") + "\n")
    (paths.results / "summary.md").write_text("\n".join(md_all))
    try:
        h = headline(paths)
    except (StopIteration, KeyError, IndexError, TypeError):
        # the quick smoke version lacks some of the points quoted in the abstract
        print("  headline numbers need the full study (not --quick); abstract not updated")
        return {}
    h.update(sh)
    ab = abstract_text(h) + (abstract_sitl(h) if sh else "")
    for doc in (paths.root / "README.md", paths.paper / "report.md"):
        _inject(doc, "abstract", ab)
    (gen / "abstract.tex").write_text(ab.replace("%", r"\%") + "\n")
    tf = paths.results / "timings.json"
    if tf.exists():
        minutes = load_json(tf)["total"] / 60.0
        _inject_inline(paths.root / "README.md", "runtime", f"{max(1, round(minutes + 0.5))}")
    (gen / "numbers.tex").write_text(_tex_macros(h))
    save_json(paths.results / "headline.json", h)
    print("  wrote results/summary.md, results/headline.json, paper/generated/*.tex")
    return h


if __name__ == "__main__":  # pragma: no cover
    write_reports()
