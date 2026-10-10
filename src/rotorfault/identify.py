"""Parameter identification helpers for a real airframe, from flight logs.

These turn the "assumed" parameters of the airframe file into measured ones:

* :func:`hover_thrust_coefficient` -- ``k_f`` from steady hover: ``m a_z = k_f sum w_i^2``
  (rotor speeds from RPM telemetry, or from the command map when there is none; with
  only the hover command, this is ``k_f = m g / (4 w(u_hover)^2)``).
* :func:`speed_map_from_rpm` -- rotor speed versus normalised command (linear fit).
* :func:`motor_time_constant` -- first-order rotor lag and transport delay from command
  steps, against RPM telemetry (``tau_up`` / ``tau_down`` separately).
* :func:`motor_time_constant_from_accel` -- the same lag without RPM telemetry, from the
  collective response (body-z accelerometer).
* :func:`inertia_and_damping` -- per-axis ``J_i`` and rotational damping ``K_i`` from
  ``tau_i = J_i dw_i/dt + K_i w_i`` (gyroscopic coupling neglected: valid for moderate
  rates), with torques predicted from the rotor speeds (thrust moments, yaw drag torque
  and the reaction of accelerating rotors, which needs ``rotor_inertia``).

All of them assume the body frame and motor order of the airframe file, and use only
samples where the vehicle is flying (mean command above ``min_command``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, lfilter, lfiltic

from .airframe import Airframe
from .logs.flightlog import FlightLog

TAU_GRID = np.arange(0.005, 0.1001, 0.0025)  # s, candidate motor time constants


def _lowpass(x: np.ndarray, fc: float, fs: float) -> np.ndarray:
    b, a = butter(2, fc / (0.5 * fs))
    return filtfilt(b, a, x, axis=0)


def _lag(w_cmd: np.ndarray, dt: float, tau_up: float, tau_down: float, delay: int) -> np.ndarray:
    """Asymmetric first-order lag of rotor-speed commands (N, 4), with a delay in samples."""
    if delay > 0:
        w_cmd = np.vstack([np.repeat(w_cmd[:1], delay, axis=0), w_cmd[:-delay]])
    a_up = math.exp(-dt / tau_up) if tau_up > 0 else 0.0
    a_dn = math.exp(-dt / tau_down) if tau_down > 0 else 0.0
    if a_up == a_dn:  # symmetric lag: linear filter
        zi = np.array([lfiltic([1 - a_up], [1, -a_up], [w], [w]) for w in w_cmd[0]])
        return lfilter([1 - a_up], [1, -a_up], w_cmd, axis=0, zi=zi.T)[0]
    w = np.empty_like(w_cmd)
    w[0] = w_cmd[0]
    for k in range(1, len(w_cmd)):
        a = np.where(w_cmd[k] > w[k - 1], a_up, a_dn)
        w[k] = w_cmd[k] + (w[k - 1] - w_cmd[k]) * a
    return w


def flying(log: FlightLog, min_command: float = 0.15) -> np.ndarray:
    return np.nan_to_num(log.motor).mean(axis=1) > min_command


def hover_samples(log: FlightLog, max_rate: float = 0.3, max_accel_dev: float = 0.8, g=9.81):
    """Steady near-hover samples: low body rates, specific force close to 1 g."""
    acc = _lowpass(log.accel, 5.0, log.rate_hz)
    rate = _lowpass(log.gyro, 5.0, log.rate_hz)
    return (
        flying(log)
        & (np.linalg.norm(rate, axis=1) < max_rate)
        & (np.abs(np.linalg.norm(acc, axis=1) - g) < max_accel_dev)
    )


@dataclass
class HoverFit:
    k_f: float  # N s^2
    hover_command: float  # mean normalised command in the hover samples
    hover_speed: float  # rad/s, mean rotor speed in the hover samples
    n_samples: int
    relative_rms: float  # RMS of (m a_z - k_f sum w^2) / (m g)
    used_rpm: bool


def hover_thrust_coefficient(
    log: FlightLog, airframe: Airframe, use_rpm: bool | None = None
) -> HoverFit:
    p = airframe.params
    use_rpm = log.rotor_speed is not None if use_rpm is None else use_rpm
    m = hover_samples(log, g=p.g)
    if m.sum() < 10:
        raise ValueError("no steady hover found in the log")
    w = log.rotor_speed if use_rpm else airframe.motor.speed(log.motor)
    s = np.sum(w[m] ** 2, axis=1)
    y = p.mass * log.accel[m, 2]
    k_f = float(np.dot(s, y) / np.dot(s, s))
    rr = float(np.sqrt(np.mean((y - k_f * s) ** 2)) / (p.mass * p.g))
    return HoverFit(
        k_f,
        float(np.nanmean(log.motor[m])),
        float(np.mean(w[m])),
        int(m.sum()),
        rr,
        use_rpm,
    )


def speed_map_from_rpm(log: FlightLog, max_accel: float = 2000.0) -> tuple[float, float, float]:
    """(speed at command 0, speed at command 1, RMS error) from steady samples."""
    if log.rotor_speed is None:
        raise ValueError("needs RPM telemetry")
    w = log.rotor_speed
    dw = np.gradient(_lowpass(w, 10.0, log.rate_hz), log.dt, axis=0)
    m = flying(log)[:, None] & (np.abs(dw) < max_accel) & np.isfinite(log.motor)
    u = log.motor[m]
    ws = w[m]
    A = np.column_stack([np.ones_like(u), u])
    coef, *_ = np.linalg.lstsq(A, ws, rcond=None)
    rms = float(np.sqrt(np.mean((A @ coef - ws) ** 2)))
    return float(coef[0]), float(coef[0] + coef[1]), rms


@dataclass
class LagFit:
    tau_up: float
    tau_down: float
    delay_s: float
    rms: float  # rad/s (RPM fit) or N (accelerometer fit)
    method: str


def motor_time_constant(
    log: FlightLog,
    airframe: Airframe,
    taus=None,
    max_delay_s: float = 0.01,
) -> LagFit:
    """Grid search of (delay, tau_up, tau_down) between the command map and RPM telemetry."""
    taus = TAU_GRID if taus is None else np.asarray(taus)
    if log.rotor_speed is None:
        raise ValueError("needs RPM telemetry; see motor_time_constant_from_accel")
    m = flying(log)
    w_cmd = airframe.motor.speed(log.motor)
    dt = log.dt
    best = (math.inf, None)
    for d in range(int(round(max_delay_s / dt)) + 1):
        for tu in taus:
            e = (_lag(w_cmd, dt, tu, tu, d) - log.rotor_speed)[m]
            if (s := float(np.mean(e**2))) < best[0]:
                best = (s, (d, tu))
    d, tu0 = best[1]
    best = (math.inf, None)
    near = taus[(taus > 0.5 * tu0) & (taus < 2 * tu0)][::2]
    for tu in near:
        for td in near:
            e = (_lag(w_cmd, dt, tu, td, d) - log.rotor_speed)[m]
            if (s := float(np.mean(e**2))) < best[0]:
                best = (s, (tu, td))
    tu, td = best[1]
    return LagFit(float(tu), float(td), d * dt, math.sqrt(best[0]), "rpm")


def motor_time_constant_from_accel(
    log: FlightLog, airframe: Airframe, taus=None, delay_steps: int = 0
) -> LagFit:
    """Symmetric rotor lag from the collective: m a_z vs k_f sum lag(w_cmd)^2."""
    taus = TAU_GRID if taus is None else np.asarray(taus)
    p = airframe.params
    m = flying(log)
    w_cmd = airframe.motor.speed(log.motor)
    y = _lowpass(p.mass * log.accel[:, 2], 30.0, log.rate_hz)
    best = (math.inf, None)
    for tu in taus:
        f = airframe.motor.k_f * np.sum(_lag(w_cmd, log.dt, tu, tu, delay_steps) ** 2, axis=1)
        s = float(np.mean((_lowpass(f, 30.0, log.rate_hz) - y)[m] ** 2))
        if s < best[0]:
            best = (s, tu)
    return LagFit(float(best[1]), float(best[1]), delay_steps * log.dt, math.sqrt(best[0]), "accel")


@dataclass
class InertiaFit:
    inertia: np.ndarray  # (3,) kg m^2
    damping: np.ndarray  # (3,) N m s/rad
    r2: np.ndarray  # (3,) coefficient of determination per axis


def inertia_and_damping(
    log: FlightLog, airframe: Airframe, use_rpm: bool | None = None, fc: float = 15.0
) -> InertiaFit:
    """Least squares per axis of tau_i = J_i dw_i/dt + K_i w_i (both sides low-passed)."""
    p, mm = airframe.params, airframe.motor
    use_rpm = log.rotor_speed is not None if use_rpm is None else use_rpm
    if use_rpm:
        w = log.rotor_speed
    else:
        w = _lag(
            mm.speed(log.motor), log.dt, mm.tau_up, mm.tau_down, int(round(mm.delay_s / log.dt))
        )
    tau = (p.mixer()[1:4] @ (mm.k_f * w**2).T).T
    fs = log.rate_hz
    # reaction torque of accelerating rotors about z
    dw = np.gradient(_lowpass(w, fc, fs), log.dt, axis=0)
    tau[:, 2] += p.rotor_inertia * dw @ p.spin_dir
    om = _lowpass(log.gyro, fc, fs)
    dom = np.gradient(om, log.dt, axis=0)
    tau = _lowpass(tau, fc, fs)
    m = flying(log)
    J, K, r2 = np.zeros(3), np.zeros(3), np.zeros(3)
    for i in range(3):
        A = np.column_stack([dom[m, i], om[m, i]])
        y = tau[m, i]
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        J[i], K[i] = c
        r2[i] = 1.0 - np.var(y - A @ c) / np.var(y)
    return InertiaFit(J, K, r2)
