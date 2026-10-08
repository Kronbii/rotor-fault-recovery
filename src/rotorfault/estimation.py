"""State estimation from the noisy sensors.

* Attitude: complementary filter (Mahony-style) propagating the bias-corrected gyro at
  500 Hz and correcting with motion-capture attitude at 100 Hz; also estimates the gyro
  bias.
* Translation: per-axis steady-state Kalman filter with state (p, v), driven by the
  rotated accelerometer and corrected with motion-capture position at 100 Hz.

Controllers and the fault estimator only ever see the outputs of this module.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are

from .params import G
from .rotations import quat_conj, quat_exp, quat_log, quat_mul, quat_to_rot


class StateEstimator:
    def __init__(
        self,
        dt: float,
        mocap_dt: float = 0.01,
        k_p: float = 10.0,
        k_i: float = 5.0,
        accel_sigma: float = 0.6,
        pos_sigma: float = 0.001,
    ):
        self.dt = dt
        self.mocap_dt = mocap_dt
        self.k_p = k_p
        self.k_i = k_i
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.R = np.eye(3)
        self.bias = np.zeros(3)
        self.omega = np.zeros(3)
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        # steady-state Kalman gain of the 100 Hz position update
        T = mocap_dt
        A = np.array([[1.0, T], [0.0, 1.0]])
        Q = accel_sigma**2 * np.array([[T**4 / 4, T**3 / 2], [T**3 / 2, T**2]])
        H = np.array([[1.0, 0.0]])
        Rm = np.array([[pos_sigma**2]])
        P = solve_discrete_are(A.T, H.T, Q, Rm)
        self.L = (P @ H.T @ np.linalg.inv(H @ P @ H.T + Rm)).ravel()
        self.initialised = False

    def reset(self, p, q, v=None):
        self.p = np.array(p, dtype=float)
        self.v = np.zeros(3) if v is None else np.array(v, dtype=float)
        self.q = np.array(q, dtype=float)
        self.R = quat_to_rot(self.q)
        self.initialised = True

    def predict(self, gyro: np.ndarray, accel: np.ndarray):
        self.omega = gyro - self.bias
        self.q = quat_mul(self.q, quat_exp(self.omega * self.dt))
        self.q /= np.linalg.norm(self.q)
        self.R = quat_to_rot(self.q)
        a_w = self.R @ accel - np.array([0.0, 0.0, G])
        self.p = self.p + self.v * self.dt + 0.5 * a_w * self.dt**2
        self.v = self.v + a_w * self.dt

    def correct(self, p_meas: np.ndarray, q_meas: np.ndarray):
        e = quat_log(quat_mul(quat_conj(self.q), q_meas))
        self.q = quat_mul(self.q, quat_exp(self.k_p * self.mocap_dt * e))
        self.q /= np.linalg.norm(self.q)
        self.R = quat_to_rot(self.q)
        self.bias = self.bias - self.k_i * self.mocap_dt * e
        r = p_meas - self.p
        self.p = self.p + self.L[0] * r
        self.v = self.v + self.L[1] * r
