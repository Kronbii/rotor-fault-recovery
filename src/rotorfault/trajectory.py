"""Reference trajectories: hover and a horizontal figure-eight (lemniscate of Gerono)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Reference:
    p: np.ndarray
    v: np.ndarray
    a: np.ndarray
    yaw: float = 0.0


class Hover:
    def __init__(self, position=(0.0, 0.0, 2.0)):
        self.p = np.array(position, dtype=float)

    def __call__(self, t: float) -> Reference:
        return Reference(self.p.copy(), np.zeros(3), np.zeros(3))


class FigureEight:
    """x = A sin(w t'),  y = (A/2) sin(2 w t'), z = const, with a smooth start.

    ``t' `` is a time warp that ramps the speed up over ``ramp`` seconds so that the
    reference starts at rest. With the defaults (A = 3 m, period 8 s) the peak speed
    is 3.3 m/s and the peak acceleration 3.1 m/s^2.
    """

    def __init__(self, amplitude=3.0, period=8.0, height=2.0, t0=1.0, ramp=2.0):
        self.A = amplitude
        self.w = 2 * np.pi / period
        self.h = height
        self.t0 = t0
        self.ramp = ramp

    def _phase(self, t):
        """Phase s(t), its rate and acceleration (C2 smooth ramp-up)."""
        tau = t - self.t0
        if tau <= 0:
            return 0.0, 0.0, 0.0
        T = self.ramp
        if tau < T:
            u = tau / T
            # rate goes 0 -> 1 with a smoothstep profile, s = integral of rate
            rate = 3 * u**2 - 2 * u**3
            s = T * (u**3 - 0.5 * u**4)
            acc = (6 * u - 6 * u**2) / T
            return s, rate, acc
        return tau - 0.5 * T, 1.0, 0.0

    def __call__(self, t: float) -> Reference:
        s, sd, sdd = self._phase(t)
        A, w = self.A, self.w
        th = w * s
        x = A * np.sin(th)
        y = 0.5 * A * np.sin(2 * th)
        dx = A * w * np.cos(th)
        dy = A * w * np.cos(2 * th)
        ddx = -A * w**2 * np.sin(th)
        ddy = -2 * A * w**2 * np.sin(2 * th)
        p = np.array([x, y, self.h])
        v = np.array([dx * sd, dy * sd, 0.0])
        a = np.array([ddx * sd**2 + dx * sdd, ddy * sd**2 + dy * sdd, 0.0])
        return Reference(p, v, a)


class WaypointSteps:
    """Aggressive position steps (used only to test false alarms on manoeuvres that
    were *not* part of the FDI calibration set)."""

    def __init__(self, waypoints=None, dwell=2.0, t0=1.0):
        self.wps = np.array(
            waypoints
            if waypoints is not None
            else [
                (0.0, 0.0, 2.0),
                (1.5, 0.0, 2.0),
                (1.5, 1.5, 2.6),
                (0.0, 1.5, 2.0),
                (-1.0, 0.0, 1.6),
                (0.0, 0.0, 2.0),
            ],
            dtype=float,
        )
        self.dwell = dwell
        self.t0 = t0

    def __call__(self, t: float) -> Reference:
        k = 0 if t < self.t0 else int((t - self.t0) // self.dwell) + 1
        p = self.wps[k % len(self.wps)]
        return Reference(p.copy(), np.zeros(3), np.zeros(3))
