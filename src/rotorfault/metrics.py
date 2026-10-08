"""Performance metrics computed from a :class:`~rotorfault.sim.SimResult`.

Definitions used throughout the report
--------------------------------------
* **lost** (crash / flip / loss of control): the simulation is stopped and the run
  counted as lost as soon as the vehicle touches the ground (z <= 0), its body z-axis
  tilts more than 90 deg from vertical (inverted), or its position error exceeds 3 m.
* **max deviation**: max ||p - p_ref|| from fault onset to the end of the run.
* **settled error**: mean ||p - p_ref|| over the last 2 s of the run.
* **recovery time**: time after onset after which ||p - p_ref|| stays below 0.15 m.
"""

from __future__ import annotations

import math

import numpy as np

from .rotations import tilt_angle

RECOVERY_BAND = 0.15  # m


def position_error(r) -> np.ndarray:
    return np.linalg.norm(r.position - r.p_ref, axis=1)


def window(r, t0: float, t1: float | None = None) -> np.ndarray:
    t1 = r.t[-1] + 1e-9 if t1 is None else t1
    return (r.t >= t0) & (r.t <= t1)


def max_deviation(r, t0: float) -> float:
    e = position_error(r)
    m = window(r, t0)
    return float(e[m].max()) if m.any() else math.nan


def settled_error(r, last: float = 2.0) -> float:
    if r.crashed:
        return math.nan
    e = position_error(r)
    m = r.t >= r.t[-1] - last
    return float(e[m].mean())


def rmse(r, t0: float, t1: float | None = None) -> float:
    e = position_error(r)
    m = window(r, t0, t1)
    return float(np.sqrt((e[m] ** 2).mean())) if m.any() else math.nan


def recovery_time(r, t0: float, band: float = RECOVERY_BAND) -> float:
    if r.crashed:
        return math.nan
    e = position_error(r)
    m = r.t >= t0
    t, e = r.t[m], e[m]
    outside = np.nonzero(e > band)[0]
    if len(outside) == 0:
        return 0.0
    k = outside[-1]
    if k == len(e) - 1:
        return math.nan
    return float(t[k + 1] - t0)


def max_tilt_deg(r, t0: float) -> float:
    m = r.t >= t0
    return float(np.degrees(max(tilt_angle(q) for q in r.quat[m]))) if m.any() else math.nan


def spin_rate(r, last: float = 2.0) -> float:
    """Mean |omega| over the last ``last`` seconds (rad/s)."""
    m = r.t >= r.t[-1] - last
    return float(np.linalg.norm(r.omega[m], axis=1).mean())
