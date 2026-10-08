"""Control allocation: exact round trip and saturation priorities."""

import numpy as np

from rotorfault.allocation import PriorityAllocator
from rotorfault.params import nominal_params

P = nominal_params()
B = P.mixer()


def _alloc():
    return PriorityAllocator(B, P.f_min, P.f_max)


def test_round_trip_when_feasible():
    a = _alloc()
    rng = np.random.default_rng(1)
    for _ in range(200):
        f_true = rng.uniform(0.5, 4.5, 4)
        v = B @ f_true
        res = a.allocate(v[0], v[1:], None)
        assert np.allclose(B @ res.f, v, atol=1e-9)
        assert res.alpha == 1.0 and res.beta == 1.0


def test_round_trip_with_effectiveness():
    a = _alloc()
    eta = np.array([0.6, 1.0, 0.9, 1.0])
    f_cmd = np.array([2.5, 1.6, 1.8, 1.6])
    v = B @ (eta * f_cmd)
    res = a.allocate(v[0], v[1:], eta)
    assert np.allclose(B @ (eta * res.f), v, atol=1e-9)


def test_yaw_is_sacrificed_first():
    a = _alloc()
    T, tau = P.weight, np.array([0.05, -0.03, 0.2])  # huge yaw demand
    res = a.allocate(T, tau, None)
    v = B @ res.f
    assert res.alpha < 1.0
    assert np.all(res.f >= P.f_min - 1e-12) and np.all(res.f <= P.f_max + 1e-12)
    assert np.isclose(v[0], T, atol=1e-9)  # thrust kept
    assert np.allclose(v[1:3], tau[:2], atol=1e-9)  # roll/pitch kept
    assert abs(v[3]) < abs(tau[2])  # yaw reduced


def test_roll_pitch_kept_by_shifting_thrust():
    a = _alloc()
    T = 4 * P.f_max * 0.95  # near full throttle
    tau = np.array([0.15, 0.0, 0.0])
    res = a.allocate(T, tau, None)
    v = B @ res.f
    assert np.isclose(v[1], 0.15, atol=1e-9)  # roll torque achieved
    assert v[0] < T  # by lowering collective thrust
    assert res.thrust_shift < 0


def test_infeasible_roll_pitch_is_scaled_not_clipped_per_rotor():
    a = _alloc()
    tau = np.array([2.0, 1.0, 0.0])  # impossible
    res = a.allocate(P.weight, tau, None)
    v = B @ res.f
    assert 0.0 < res.beta < 1.0
    assert np.all(res.f >= P.f_min - 1e-12) and np.all(res.f <= P.f_max + 1e-12)
    # direction of the roll/pitch torque is preserved
    d = v[1:3] / np.linalg.norm(v[1:3])
    assert np.allclose(d, tau[:2] / np.linalg.norm(tau[:2]), atol=1e-6)
