"""Physics sanity checks of the true-vehicle model."""

import math

import numpy as np

from rotorfault.dynamics import QuadDynamics, hover_state
from rotorfault.params import MassModel, nominal_params
from rotorfault.rotations import quat_to_rot


def _no_drag(P):
    return P.copy(drag_lin=np.zeros(3), drag_quad=0.0, rot_damping=np.zeros(3), rotor_inertia=0.0)


def test_parameters_are_in_the_documented_ranges():
    P = nominal_params()
    assert 0.6 <= P.mass <= 0.7
    assert abs(P.arm_length - 0.11) < 1e-12
    assert 3.0 <= P.thrust_to_weight <= 4.0
    assert 0.01 <= P.kappa <= 0.02
    assert 0.020 <= P.tau_motor <= 0.040
    J = np.diag(P.inertia)
    assert np.all(J > 0) and J[2] > J[0] and J[2] > J[1]
    assert abs(MassModel().total_mass - P.mass) < 1e-12


def test_hover_equilibrium_thrust_equals_weight():
    P = nominal_params()
    dyn = QuadDynamics(P)
    x = hover_state(P)
    w = x[13:17]
    assert math.isclose(dyn.thrusts(w, np.ones(4)).sum(), P.weight, rel_tol=1e-9)
    d = np.array(dyn.deriv(x.tolist(), w.tolist(), [1.0] * 4))
    assert np.allclose(d[3:6], 0.0, atol=1e-9)  # no linear acceleration
    assert np.allclose(d[10:13], 0.0, atol=1e-9)  # no angular acceleration
    assert np.allclose(d[13:17], 0.0, atol=1e-9)  # rotor speeds steady


def test_free_fall_accelerates_at_g():
    P = _no_drag(nominal_params())
    dyn = QuadDynamics(P)
    x = hover_state(P, (0.0, 0.0, 100.0))
    x[13:17] = 0.0
    dt, T = 0.001, 1.0
    for _ in range(int(T / dt)):
        x = dyn.step(x, [0.0] * 4, [1.0] * 4, dt)
    assert math.isclose(x[2], 100.0 - 0.5 * P.g * T**2, abs_tol=1e-6)
    assert math.isclose(x[5], -P.g * T, abs_tol=1e-6)


def test_quaternion_stays_unit_during_aggressive_rotation():
    P = nominal_params()
    dyn = QuadDynamics(P)
    x = hover_state(P)
    x[10:13] = [8.0, -5.0, 25.0]
    rng = np.random.default_rng(0)
    for _ in range(3000):
        w = rng.uniform(P.omega_min, P.omega_max, 4)
        x = dyn.step(x, w, [1.0] * 4, 0.001)
        assert abs(np.linalg.norm(x[6:10]) - 1.0) < 1e-12


def test_torque_free_spin_conserves_angular_momentum_and_energy():
    P = _no_drag(nominal_params()).copy(g=0.0)
    dyn = QuadDynamics(P)
    x = hover_state(P)
    x[13:17] = 0.0  # rotors stopped: no thrust, no rotor momentum
    x[10:13] = [3.0, 0.5, 8.0]  # near the intermediate axis -> rich tumbling
    J = P.inertia

    def momentum(x):
        return quat_to_rot(x[6:10]) @ (J @ x[10:13])

    def energy(x):
        w = x[10:13]
        return 0.5 * w @ J @ w

    L0, E0 = momentum(x), energy(x)
    for _ in range(5000):  # 5 s at 1 kHz
        x = dyn.step(x, [0.0] * 4, [1.0] * 4, 0.001)
    assert np.linalg.norm(momentum(x) - L0) / np.linalg.norm(L0) < 1e-8
    assert abs(energy(x) - E0) / E0 < 1e-8


def test_loss_of_effectiveness_scales_thrust_and_yaw_torque():
    P = nominal_params()
    dyn = QuadDynamics(P)
    x = hover_state(P)
    w = x[13:17]
    eta = np.array([0.5, 1.0, 1.0, 1.0])
    f = dyn.thrusts(w, eta)
    assert math.isclose(f[0], 0.5 * f[1])
    d = np.array(dyn.deriv(x.tolist(), w.tolist(), eta.tolist()))
    # front-right rotor weakened: the right side drops (positive roll about x_b) and the
    # nose drops (positive pitch about y_b, which is to the left)
    assert d[10] > 0 and d[11] > 0
    assert d[12] < 0  # its yaw reaction torque (+kappa f_1) is reduced
