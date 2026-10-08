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


def _tex_table(header, rows, caption, label, colspec=None):
    colspec = colspec or ("l" + "c" * (len(header) - 1))

    def esc(s):
        return str(s).replace("%", r"\%").replace("--", "--").replace("_", r"\_")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{" + colspec + "}",
        r"\toprule",
        " & ".join(esc(h) for h in header) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ======================================================================================
def table_e1(e1):
    mags = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9]
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
        "|LOE error| pts: median (p95)",
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
        "|LOE err| pts",
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


# ======================================================================================
def headline(paths: Paths) -> dict:
    e1 = load_json(paths.results / "e1_hover_loe.json")
    e2 = load_json(paths.results / "e2_fdi.json")
    e3 = load_json(paths.results / "e3_trajectory.json")
    e4 = load_json(paths.results / "e4_complete_loss.json")
    e5 = load_json(paths.results / "e5_loss_in_flight.json")
    e6 = load_json(paths.results / "e6_robustness.json")

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
    "20-90 % loss of effectiveness, all faults were detected (median delay "
    "{e2_delay_median_ms:.0f} ms, 95th percentile {e2_delay_p95_ms:.0f} ms) and "
    "{e2_isolation_pct:.1f} % were isolated to the correct rotor, with a median "
    "magnitude error of {e2_loe_abs_err_median_pts:.1f} percentage points; "
    "{e2_false_alarms} false alarms occurred in {e2_free_minutes:.0f} min of fault-free "
    "flight. With a 50 % loss in hover the median peak deviation drops from "
    "{e1_passive_maxdev_50_cm:.0f} cm (passive) to {e1_est_maxdev_50_cm:.1f} cm. After a "
    "complete rotor loss in hover the passive controller lost the vehicle in "
    "{e4_passive_lost}/{e4_n} runs (median {e4_passive_ttl_s:.2f} s), while the proposed "
    "pipeline recovered {e4_n_ok}/{e4_n} with a median peak deviation of "
    "{e4_est_maxdev_cm:.0f} cm and a spin of {e4_est_spin:.0f} rad/s; it also recovered "
    "{e5_n_ok}/{e5_n} complete losses during a 3.3 m/s figure-eight. These are simulation "
    "results with assumed, not identified, parameters; the robustness sweeps show where "
    "the yaw-relaxed controller stops working (actuator delay, slower motors, lower yaw "
    "damping)."
)


def abstract_text(h: dict) -> str:
    d = dict(h)
    d["e4_n_ok"] = h["e4_n"] - h["e4_est_lost"]
    d["e5_n_ok"] = h["e5_n"] - h["e5_est_lost"]
    return ABSTRACT.format(**d)


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
    tables = {
        "table_e1": (
            table_e1(e1),
            "E1: hover with a step loss of effectiveness on rotor 1.",
            "tab:e1",
        ),
        "table_e2": (table_e2(e2), "E2: fault detection and isolation Monte Carlo.", "tab:e2"),
        "table_e2_noise": (table_e2_noise(e2), "E2: sensitivity to gyro noise.", "tab:e2noise"),
        "table_e3": (table_e3(e3), "E3: figure-eight tracking with a 50\\,\\% loss.", "tab:e3"),
        "table_e4": (table_e4(e4, e5), "E4/E5: complete loss of one rotor.", "tab:e4"),
        "table_e6": (table_e6(e6), "E6: robustness of complete-loss recovery.", "tab:e6"),
    }
    gen = paths.paper / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    md_all = ["# Results summary (generated)\n"]
    for key, ((header, rows, note), cap, label) in tables.items():
        md = _md_table(header, rows) + f"\n\n*{note}*\n"
        md_all.append(f"## {cap}\n\n{md}")
        for doc in (paths.root / "README.md", paths.paper / "report.md"):
            _inject(doc, key, md)
        (gen / f"{key}.tex").write_text(
            _tex_table(
                header, rows, cap.replace("%", r"\%") + " " + note.replace("%", r"\%"), label
            )
            + "\n"
        )
    h = headline(paths)
    ab = abstract_text(h)
    for doc in (paths.root / "README.md", paths.paper / "report.md"):
        _inject(doc, "abstract", ab)
    (gen / "abstract.tex").write_text(ab.replace("%", r"\%") + "\n")
    (gen / "numbers.tex").write_text(_tex_macros(h))
    (paths.results / "summary.md").write_text("\n".join(md_all))
    save_json(paths.results / "headline.json", h)
    print("  wrote results/summary.md, results/headline.json, paper/generated/*.tex")
    return h


if __name__ == "__main__":  # pragma: no cover
    write_reports()
