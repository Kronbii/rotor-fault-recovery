"""Sensor models: gyro (white noise + random-walk bias + range limit), accelerometer,
and a motion-capture system providing position and attitude at 100 Hz."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import SensorParams
from .rotations import np_clip, quat_exp, quat_mul


@dataclass
class ImuSample:
    gyro: np.ndarray
    accel: np.ndarray


@dataclass
class MocapSample:
    position: np.ndarray
    quat: np.ndarray


class Sensors:
    """All noise comes from one seeded generator, so runs are reproducible."""

    _BLOCK = 4096

    def __init__(self, sp: SensorParams, rng: np.random.Generator, dt_imu: float):
        self.sp = sp
        self.rng = rng
        self.dt = dt_imu
        self.gyro_bias = rng.normal(0.0, sp.gyro_bias_init, 3)
        self.accel_bias = rng.normal(0.0, sp.accel_bias, 3)
        self._mocap_period = 1.0 / sp.mocap_rate
        self._next_mocap = 0.0
        self._buf = np.empty((0, 9))
        self._k = 0

    def _noise(self) -> np.ndarray:
        if self._k >= len(self._buf):
            self._buf = self.rng.standard_normal((self._BLOCK, 9))
            self._k = 0
        n = self._buf[self._k]
        self._k += 1
        return n

    def imu(self, omega_true: np.ndarray, specific_force: np.ndarray) -> ImuSample:
        sp = self.sp
        n = self._noise()
        self.gyro_bias = self.gyro_bias + sp.gyro_bias_walk * np.sqrt(self.dt) * n[6:9]
        g = omega_true + self.gyro_bias + sp.gyro_noise * n[0:3]
        g = np_clip(g, -sp.gyro_range, sp.gyro_range)
        a = specific_force + self.accel_bias + sp.accel_noise * n[3:6]
        return ImuSample(gyro=g, accel=a)

    def mocap(self, t: float, p_true: np.ndarray, q_true: np.ndarray) -> MocapSample | None:
        """Returns a sample when one is due (100 Hz), otherwise ``None``."""
        if t + 1e-9 < self._next_mocap:
            return None
        self._next_mocap += self._mocap_period
        sp = self.sp
        n = self.rng.standard_normal(6)
        p = p_true + sp.mocap_pos_noise * n[0:3]
        q = quat_mul(q_true, quat_exp(sp.mocap_att_noise * n[3:6]))
        return MocapSample(position=p, quat=q / np.linalg.norm(q))
