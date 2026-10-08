"""Turbulence-like external disturbances.

Each component of the disturbance force (world frame) and torque (body frame) is an
independent Ornstein-Uhlenbeck process (first-order Gauss-Markov) with standard
deviation ``sigma`` and correlation time ``tau``, discretised exactly at the control
rate and held constant over each control period. This is a deliberately simple
stand-in for gusts and unmodelled aerodynamic moments: it is *not* a Dryden model, and
its intensity is an assumption ("light turbulence").

It matters for the study in two ways: the controllers must reject it, and it is the
dominant nuisance in the fault-detection residual, i.e. it sets the trade-off between
detection delay and false alarms (without it the simulated residual would be
unrealistically clean).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DisturbanceParams:
    force_sigma: float = 0.15  # N (about 2.4 % of weight) per axis
    torque_sigma: float = 4.0e-3  # N m per axis (about 1 % of max roll torque)
    yaw_torque_sigma: float = 1.0e-3  # N m
    tau: float = 0.3  # s correlation time

    @classmethod
    def calm(cls) -> DisturbanceParams:
        return cls(0.0, 0.0, 0.0, 0.3)

    def scaled(self, k: float) -> DisturbanceParams:
        return DisturbanceParams(
            self.force_sigma * k, self.torque_sigma * k, self.yaw_torque_sigma * k, self.tau
        )


class Disturbance:
    def __init__(self, dp: DisturbanceParams, rng: np.random.Generator, dt: float):
        self.dp = dp
        self.rng = rng
        self.a = float(np.exp(-dt / dp.tau))
        self.sig = np.array([dp.force_sigma] * 3 + [dp.torque_sigma] * 2 + [dp.yaw_torque_sigma])
        self.b = self.sig * np.sqrt(1.0 - self.a**2)
        self.x = self.sig * rng.standard_normal(6)
        self._buf = np.empty((0, 6))
        self._k = 0

    def step(self) -> tuple:
        if self._k >= len(self._buf):
            self._buf = self.rng.standard_normal((4096, 6))
            self._k = 0
        self.x = self.a * self.x + self.b * self._buf[self._k]
        self._k += 1
        return tuple(self.x.tolist())
