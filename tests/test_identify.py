"""Parameter identification recovers the simulator's (known) parameters from its logs."""

import numpy as np
import pytest

from rotorfault import identify as I
from rotorfault.logs.synthetic import synthetic_flight
from rotorfault.params import nominal_params

P = nominal_params()


@pytest.fixture(scope="module")
def steps():
    # nominal vehicle (no parameter spread), waypoint steps for excitation, light turbulence
    log, af, _ = synthetic_flight("steps", 8.0, loe=0.0, seed=3, uncertainty=False, turbulence=0.3)
    return log, af


def test_hover_thrust_coefficient(steps):
    log, af = steps
    for use_rpm in (True, False):
        fit = I.hover_thrust_coefficient(log, af, use_rpm=use_rpm)
        assert fit.k_f == pytest.approx(P.k_f, rel=0.02)
        assert fit.n_samples > 500


def test_motor_time_constant_from_rpm_and_accel(steps):
    log, af = steps
    lag = I.motor_time_constant(log, af)
    assert lag.tau_up == pytest.approx(P.tau_motor, rel=0.1)
    assert lag.tau_down == pytest.approx(P.tau_motor, rel=0.1)
    assert 0.0 <= lag.delay_s <= 0.006
    lag2 = I.motor_time_constant_from_accel(log, af, delay_steps=1)
    assert lag2.tau_up == pytest.approx(P.tau_motor, rel=0.2)


def test_speed_map_and_inertia(steps):
    log, af = steps
    w0, w1, _ = I.speed_map_from_rpm(log)
    assert w0 == pytest.approx(P.omega_min, abs=60) and w1 == pytest.approx(P.omega_max, rel=0.05)
    fit = I.inertia_and_damping(log, af)
    J = np.diag(P.inertia)
    assert fit.inertia[:2] == pytest.approx(J[:2], rel=0.05)  # roll / pitch: well excited
    assert fit.r2[:2].min() > 0.95
    assert fit.inertia[2] == pytest.approx(J[2], rel=0.3)  # yaw: weakly excited here
