"""Fault estimation and detection: synthetic data and closed-loop checks."""

import numpy as np

from rotorfault.fdi import CusumBank, EffectivenessKF, FDIConfig
from rotorfault.params import nominal_params
from rotorfault.scenarios import make_scenario
from rotorfault.sim import simulate

P = nominal_params()
B = P.mixer()


def test_kalman_estimator_converges_on_synthetic_regression():
    rng = np.random.default_rng(3)
    eta_true = np.array([1.0, 0.55, 1.0, 0.97])
    sigma = np.array([0.05, 1e-3, 1e-3, 5e-4])
    kf = EffectivenessKF(4, q=0.2)
    for _ in range(2000):
        f = 1.6 + 0.3 * rng.standard_normal(4)
        Phi = B * f[None, :]
        y = Phi @ eta_true + sigma * rng.standard_normal(4)
        kf.update(Phi, y, 2 * sigma, 0.002)
    assert np.max(np.abs(kf.eta - eta_true)) < 0.02


def test_kalman_estimate_is_projected_and_frozen_when_unexcited():
    kf = EffectivenessKF(4, q=1.0, lo=0.0, hi=1.1)
    Phi = B * np.array([1.6, 1.6, 1.6, 0.0])[None, :]
    y = Phi @ np.array([1.0, 1.0, 1.0, 1.0]) + np.array([5.0, 0, 0, 0])  # absurd data
    excited = np.array([True, True, True, False])
    for _ in range(100):
        kf.update(Phi, y, np.full(4, 0.1), 0.002, excited)
    assert np.all(kf.eta <= 1.1) and np.all(kf.eta >= 0.0)
    assert kf.eta[3] == 1.0  # unexcited rotor untouched


def _signatures(sigma):
    return -(B * P.hover_thrust) / sigma[:, None]


def test_cusum_detects_synthetic_step_within_bound_and_isolates():
    rng = np.random.default_rng(4)
    sigma = np.array([0.17, 4.5e-3, 4.5e-3, 2.4e-3])
    D = _signatures(sigma)
    bank = CusumBank((0.15, 0.3, 0.6), (1500.0, 740.0, 2960.0))
    k_fault, delta, rotor = 1000, 0.3, 2
    alarm_k = None
    for k in range(2000):
        r = rng.standard_normal(4)
        if k >= k_fault:
            r = r + delta * D[:, rotor]
        tj, dn = bank.statistics(r, D)
        if bank.update(tj, dn) and alarm_k is None:
            alarm_k = k
    assert alarm_k is not None and alarm_k >= k_fault
    assert (alarm_k - k_fault) * 0.002 < 0.05  # within 50 ms on white noise
    assert bank.isolated == rotor


def test_cusum_no_detection_when_fault_free():
    rng = np.random.default_rng(5)
    sigma = np.array([0.17, 4.5e-3, 4.5e-3, 2.4e-3])
    D = _signatures(sigma)
    bank = CusumBank((0.15, 0.3, 0.6), (1500.0, 740.0, 2960.0))
    for _ in range(50_000):  # 100 s at 500 Hz of white residual
        tj, dn = bank.statistics(rng.standard_normal(4), D)
        bank.update(tj, dn)
    assert not bank.alarm


def test_closed_loop_detection_isolation_and_estimate():
    sc = make_scenario("hover", 4.0, rotor=1, magnitude=0.4, onset=2.0)
    r = simulate(sc, "estimated", 7, FDIConfig())
    assert np.isfinite(r.alarm_time)
    assert 2.0 <= r.alarm_time < 2.15
    assert r.isolated == 1
    k = np.searchsorted(r.t, r.alarm_time + 0.25)
    loe = 1 - r.eta_hat[k, 1] / r.eta_ref[1]
    assert abs(loe - 0.4) < 0.08


def test_closed_loop_no_alarm_without_fault():
    for traj in ("hover", "fig8"):
        r = simulate(make_scenario(traj, 6.0), "estimated", 8, FDIConfig())
        assert not np.isfinite(r.alarm_time)
