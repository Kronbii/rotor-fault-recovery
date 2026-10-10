"""Synthetic flight logs from the simulator (for round-trip tests of the log pipeline).

The simulator's measured gyro and accelerometer samples, its motor commands (converted
to normalised commands with the simulator's own ESC map) and its true rotor speeds
(standing in for RPM telemetry) are written in the motor order of a flight controller.
"""

from __future__ import annotations

import dataclasses
import math

from ..airframe import Airframe, simulator_airframe
from ..dynamics import thrust_to_speed
from ..fdi import FDIConfig
from ..params import nominal_params
from ..scenarios import make_scenario
from ..sim import SimResult, simulate
from .flightlog import FlightLog

# Betaflight quad-X motor order: 1 rear-right, 2 front-right, 3 rear-left, 4 front-left,
# as simulator rotor indices (0 FR, 1 FL, 2 RL, 3 RR)
BETAFLIGHT_ORDER = (3, 0, 2, 1)


def flight_log_from_sim(
    r: SimResult, airframe: Airframe, motor_order=BETAFLIGHT_ORDER
) -> FlightLog:
    if r.gyro_meas is None:
        raise ValueError("simulate(..., record_sensors=True) is needed for a flight log")
    order = list(motor_order)
    P = nominal_params()
    w_cmd = thrust_to_speed(r.f_cmd, P)
    u = airframe.motor.command(w_cmd)[:, order]
    faulty = r.fault_rotor >= 0 and r.fault_magnitude > 0
    events = {
        "fault_onset": r.fault_onset if faulty else None,
        "fault_motor": order.index(r.fault_rotor) + 1 if faulty else None,
        "fault_loe": r.fault_magnitude,
        "sim_alarm": r.alarm_time if math.isfinite(r.alarm_time) else None,
        "sim_isolated_motor": order.index(r.isolated) + 1 if r.isolated >= 0 else None,
    }
    return FlightLog(
        t=r.t.copy(),
        gyro=r.gyro_meas.copy(),
        accel=r.accel_meas.copy(),
        motor=u,
        v_body=r.v_body_est.copy(),
        rotor_speed=r.x[:, 13:17][:, order].copy(),
        extra={"eta_true": r.eta_true[:, order].copy(), "eta_hat_sim": r.eta_hat[:, order].copy()},
        events=events,
        meta={"source": "rotorfault-sim", "seed": r.seed, "baseline": r.baseline},
    )


def synthetic_flight(
    traj: str = "hover",
    duration: float = 6.0,
    rotor: int = 0,
    loe: float = 0.4,
    onset: float = 4.0,
    seed: int = 7,
    motor_order=BETAFLIGHT_ORDER,
    baseline: str = "passive",
    turbulence: float = 1.0,
    uncertainty: bool = True,
    cfg: FDIConfig | None = None,
) -> tuple[FlightLog, Airframe, SimResult]:
    """Simulate one flight and return it as a flight log in ``motor_order``.

    The in-simulation diagnoser learns its baseline over [0.5, 3.0) s and detects from
    3.0 s, so that a replay of the log can use the same windows (the log starts after a
    0.5 s settling phase that is not recorded).
    """
    sc = make_scenario(
        traj,
        duration,
        rotor=rotor,
        magnitude=loe,
        onset=onset,
        turbulence=turbulence,
        uncertainty=uncertainty,
    )
    sc.settle_time = 0.5
    cfg = cfg or FDIConfig()
    cfg = dataclasses.replace(cfg, baseline_start=0.5, detect_after=3.0)
    r = simulate(sc, baseline, seed, cfg, record_sensors=True)
    af = simulator_airframe(motor_order)
    return flight_log_from_sim(r, af, motor_order), af, r
