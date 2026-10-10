"""Betaflight blackbox logs -> :class:`FlightLog`.

Input is the CSV written by ``blackbox_decode`` from betaflight/blackbox-tools, run with

    blackbox_decode --unit-rotation rad/s --unit-acceleration m/s2 --unit-frame-time s \\
                    --save-headers LOG.BBL

which gives columns ``time (s)``, ``gyroADC[0..2] (rad/s)``, ``accSmooth[0..2] (m/s/s)``,
``motor[0..3]`` and, with bidirectional DShot, ``eRPM[0..3]``. Raw units are also
accepted when the scale factors are given in the airframe file. A minimal documented
CSV (``FORMAT_MINIMAL`` below) is accepted too, for logs from other sources.

Conventions come from the ``log`` section of the airframe file (defaults below). They
are *assumptions until checked on the first real log* (``docs/tethered-test.md``, bench
checks): Betaflight's gyro and accelerometer axes are taken as x forward, y left, z up
(level accelerometer reads +1 g on z), motor outputs as DShot values between
``motor_range`` (48..2047), and ``eRPM[i]`` as electrical RPM / 100.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import numpy as np

from .flightlog import FlightLog

DEFAULTS = {
    "motor_range": [48.0, 2047.0],  # DShot command range -> normalised [0, 1]
    "erpm_units": 100.0,  # value in the CSV x this = electrical RPM
    "motor_poles": 14,  # 2306-class motors: 12N14P
    "gyro_axes": ["+x", "+y", "+z"],  # CSV axis 0, 1, 2 -> body FLU axis (with sign)
    "acc_axes": ["+x", "+y", "+z"],
    "gyro_raw_scale": None,  # rad/s per raw unit, only for raw gyro columns
    "acc_1g": None,  # raw accelerometer units per g, only for raw columns
    "g": 9.80665,
}

# rotorfault minimal CSV: t_s, gyro_x/y/z (rad/s, FLU), acc_x/y/z (m/s^2, FLU),
# motor_1..4 (normalised 0..1), optional rpm_1..4 (rotor speed, rad/s)
FORMAT_MINIMAL = ["t_s", "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"] + [
    f"motor_{i}" for i in range(1, 5)
]

_UNIT = re.compile(r"^\s*(?P<name>[^()]+?)\s*(\((?P<unit>[^)]*)\))?\s*$")


def conventions(airframe=None) -> dict:
    c = dict(DEFAULTS)
    if airframe is not None:
        c.update({k: v for k, v in airframe.log.items() if k in DEFAULTS})
    return c


def _axes(spec: list[str]) -> tuple[list[int], np.ndarray]:
    idx, sign = [], []
    for s in spec:
        s = s.strip().lower()
        sign.append(-1.0 if s.startswith("-") else 1.0)
        idx.append("xyz".index(s[-1]))
    order = np.argsort(idx)  # CSV column feeding body axis 0, 1, 2
    if sorted(idx) != [0, 1, 2]:
        raise ValueError(f"axis mapping must be a permutation of x, y, z: {spec}")
    return list(order), np.array(sign)[order]


WANTED = (
    {"time", "t_s"}
    | {f"gyroADC[{i}]" for i in range(3)}
    | {f"accSmooth[{i}]" for i in range(3)}
    | {f"motor[{i}]" for i in range(4)}
    | {f"eRPM[{i}]" for i in range(4)}
    | set(FORMAT_MINIMAL)
    | {f"rpm_{i}" for i in range(1, 5)}
)


def _read_columns(path: Path) -> tuple[dict, dict]:
    """Numeric columns needed by the reader (keyed by bare field name) and their units.
    Other columns (flight-mode flags as text, debug fields, ...) are ignored."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        names, units = [], {}
        for h in header:
            m = _UNIT.match(h)
            name = m.group("name").strip() if m else h.strip()
            names.append(name)
            units[name] = ((m.group("unit") if m else None) or "raw").strip()
        idx = [i for i, n in enumerate(names) if n in WANTED]
        vals = [[] for _ in idx]
        for row in reader:
            if len(row) < len(names):
                continue  # truncated last line
            for j, i in enumerate(idx):
                v = row[i].strip()
                vals[j].append(float(v) if v not in ("", "nan") else np.nan)
    cols = {names[i]: np.array(v, dtype=float) for i, v in zip(idx, vals)}
    return cols, units


def read_blackbox_csv(path: str | Path, airframe=None) -> FlightLog:
    path = Path(path)
    cols, units = _read_columns(path)
    c = conventions(airframe)
    if "t_s" in cols:  # minimal documented format
        t = cols["t_s"]
        gyro = np.column_stack([cols[f"gyro_{a}"] for a in "xyz"])
        acc = np.column_stack([cols[f"acc_{a}"] for a in "xyz"])
        motor = np.column_stack([cols[f"motor_{i}"] for i in range(1, 5)])
        rpm = np.column_stack([cols[f"rpm_{i}"] for i in range(1, 5)]) if "rpm_1" in cols else None
        source = "rotorfault-minimal-csv"
    else:
        if "time" not in cols:
            raise ValueError("no time column: decode with blackbox_decode (see module doc)")
        tu = units["time"]
        t = cols["time"] * (1.0 if tu == "s" else 1e-6)
        order, sign = _axes(c["gyro_axes"])
        gu = units["gyroADC[0]"]
        g_raw = np.column_stack([cols[f"gyroADC[{i}]"] for i in range(3)])
        if gu == "rad/s":
            gs = 1.0
        elif gu == "deg/s":
            gs = math.pi / 180.0
        elif c["gyro_raw_scale"]:
            gs = float(c["gyro_raw_scale"])
        else:
            raise ValueError(
                "raw gyro units: decode with --unit-rotation rad/s or set gyro_raw_scale"
            )
        gyro = g_raw[:, order] * sign * gs
        order, sign = _axes(c["acc_axes"])
        if "accSmooth[0]" not in cols:
            raise ValueError("no accSmooth columns: enable accelerometer logging on the FC")
        au = units["accSmooth[0]"]
        a_raw = np.column_stack([cols[f"accSmooth[{i}]"] for i in range(3)])
        if au in ("m/s/s", "m/s2"):
            asc = 1.0
        elif au == "g":
            asc = c["g"]
        elif c["acc_1g"]:
            asc = c["g"] / float(c["acc_1g"])
        else:
            raise ValueError("raw accelerometer units: decode with --unit-acceleration m/s2")
        acc = a_raw[:, order] * sign * asc
        lo, hi = c["motor_range"]
        motor = (np.column_stack([cols[f"motor[{i}]"] for i in range(4)]) - lo) / (hi - lo)
        motor = np.clip(motor, 0.0, 1.0)
        rpm = None
        if "eRPM[0]" in cols:
            erpm = np.column_stack([cols[f"eRPM[{i}]"] for i in range(4)]) * c["erpm_units"]
            rpm = erpm / (c["motor_poles"] / 2.0) * 2.0 * math.pi / 60.0
        source = "betaflight-blackbox-csv"
    keep = np.concatenate([[True], np.diff(t) > 0])  # drop repeated / backwards stamps
    t, gyro, acc, motor = t[keep], gyro[keep], acc[keep], motor[keep]
    rpm = rpm[keep] if rpm is not None else None
    return FlightLog(
        t=t - t[0],
        gyro=gyro,
        accel=acc,
        motor=motor,
        v_body=None,
        rotor_speed=rpm,
        meta={"source": source, "file": path.name, "conventions": c, "t0_log": float(t[0])},
    )


def write_blackbox_csv(log: FlightLog, path: str | Path, airframe=None) -> None:
    """Inverse of :func:`read_blackbox_csv` in blackbox_decode's layout (used to build a
    synthetic Betaflight log from the simulator, for the round-trip test)."""
    c = conventions(airframe)
    order, sign = _axes(c["gyro_axes"])
    g = np.empty_like(log.gyro)
    g[:, order] = log.gyro * sign
    order, sign = _axes(c["acc_axes"])
    a = np.empty_like(log.accel)
    a[:, order] = log.accel * sign
    lo, hi = c["motor_range"]
    mot = np.round(lo + (hi - lo) * np.clip(np.nan_to_num(log.motor), 0, 1)).astype(int)
    header = ["loopIteration", "time (s)"]
    header += [f"gyroADC[{i}] (rad/s)" for i in range(3)]
    header += [f"accSmooth[{i}] (m/s/s)" for i in range(3)]
    header += [f"motor[{i}]" for i in range(4)]
    erpm = None
    if log.rotor_speed is not None:
        header += [f"eRPM[{i}]" for i in range(4)]
        erpm = np.round(
            log.rotor_speed * 60.0 / (2 * math.pi) * (c["motor_poles"] / 2.0) / c["erpm_units"]
        ).astype(int)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for k in range(len(log.t)):
            row = [k, f"{log.t[k]:.6f}"]
            row += [f"{v:.5f}" for v in g[k]]
            row += [f"{v:.4f}" for v in a[k]]
            row += [str(v) for v in mot[k]]
            if erpm is not None:
                row += [str(v) for v in erpm[k]]
            w.writerow(row)
