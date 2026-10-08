"""State estimation from the noisy sensors.

* Attitude: complementary filter (Mahony-style) propagating the bias-corrected gyro at
  500 Hz and correcting with motion-capture attitude at 100 Hz; also estimates the gyro
  bias. Motion-capture samples arrive with a fixed latency (10 ms by default) and are
  fused against the estimate stored for their time stamp.
* Translation: per-axis steady-state Kalman filter with state (p, v), driven by the
  rotated accelerometer and corrected with motion-capture position at 100 Hz.

Controllers and the fault estimator only ever see the outputs of this module.
"""

from __future__ import annotations

import math
from collections import deque

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
        latency_steps: int = 0,
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
        # history of past estimates for delayed (time-stamped) mocap corrections
        self.latency_steps = latency_steps
        self._hist = deque(maxlen=latency_steps + 1)

    def reset(self, p, q, v=None):
        self.p = np.array(p, dtype=float)
        self.v = np.zeros(3) if v is None else np.array(v, dtype=float)
        self.q = np.array(q, dtype=float)
        self.R = quat_to_rot(self.q)
        self.initialised = True
        self._hist.clear()

    def predict(self, gyro: np.ndarray, accel: np.ndarray):
        self.omega = gyro - self.bias
        q = quat_mul(self.q, quat_exp(self.omega * self.dt))
        self.q = q / math.sqrt(float(q @ q))
        self.R = quat_to_rot(self.q)
        a_w = self.R @ accel
        a_w[2] -= G
        self.p = self.p + self.v * self.dt + 0.5 * a_w * self.dt**2
        self.v = self.v + a_w * self.dt
        self._hist.append((self.q.copy(), self.R.copy(), self.p.copy()))

    def correct(self, p_meas: np.ndarray, q_meas: np.ndarray):
        """Mocap correction. The measurement is ``latency_steps`` old: the innovation
        is formed against the estimate at the measurement time and the attitude error
        is transported to the current body frame before being applied."""
        if self._hist:
            q_then, R_then, p_then = self._hist[0]
        else:
            q_then, R_then, p_then = self.q, self.R, self.p
        e_then = quat_log(quat_mul(quat_conj(q_then), q_meas))
        e = self.R.T @ (R_then @ e_then)
        q = quat_mul(self.q, quat_exp(self.k_p * self.mocap_dt * e))
        self.q = q / math.sqrt(float(q @ q))
        self.R = quat_to_rot(self.q)
        self.bias = self.bias - self.k_i * self.mocap_dt * e
        r = p_meas - p_then
        self.p = self.p + self.L[0] * r
        self.v = self.v + self.L[1] * r
