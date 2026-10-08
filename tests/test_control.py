"""Closed-loop behaviour: nominal hover, partial-loss FTC, complete-loss FTC."""

import numpy as np

from rotorfault import metrics as M
from rotorfault.fdi import FDIConfig
from rotorfault.ftc import design_relaxed, solve_relaxed_hover
from rotorfault.params import nominal_params
from rotorfault.scenarios import make_scenario
from rotorfault.sim import simulate


def test_nominal_hover_is_stable_under_noise_and_turbulence():
    r = simulate(make_scenario("hover", 6.0), "passive", 21, FDIConfig())
    assert not r.crashed
    assert M.max_deviation(r, 0.0) < 0.10
    assert M.settled_error(r) < 0.05


def test_figure_eight_tracking():
    r = simulate(make_scenario("fig8", 12.0), "passive", 22, FDIConfig())
    assert not r.crashed
    assert M.rmse(r, 3.0, 12.0) < 0.08


def test_ftc_recovers_50_percent_loe_and_beats_passive():
    sc = make_scenario("hover", 6.0, rotor=0, magnitude=0.5, onset=2.0)
    passive = simulate(sc, "passive", 23, FDIConfig())
    active = simulate(sc, "estimated", 23, FDIConfig())
    assert not active.crashed
    assert M.settled_error(active) < 0.05
    assert M.max_deviation(active, 2.0) < 0.5 * M.max_deviation(passive, 2.0)


def test_relaxed_hover_equilibrium_balances_forces_and_torques():
    P = nominal_params()
    for rotor in range(4):
        eq = solve_relaxed_hover(P, rotor, 15.0)
        assert eq.residual < 1e-6
        assert eq.f[rotor] == 0.0
        assert np.all(eq.f[eq.active] > 0.1 * P.f_max)
        assert np.all(eq.f[eq.active] < 0.8 * P.f_max)
        assert 15.0 < eq.spin < 0.9 * np.deg2rad(2000)  # inside the gyro range


def test_relaxed_lqr_is_stabilising_on_the_whole_schedule():
    from rotorfault.ftc import c2d

    P = nominal_params()
    d = design_relaxed(P, 0, 0.002)
    for A, Bm, K in zip(d.A_grid, d.B_grid, d.K_grid):
        Ad, Bd = c2d(A, Bm, 0.002)
        eig = np.linalg.eigvals(Ad - Bd @ K)
        assert np.max(np.abs(eig)) < 1.0


def test_yaw_relaxed_control_survives_complete_rotor_loss():
    sc = make_scenario("hover", 7.0, rotor=2, magnitude=1.0, onset=2.0)
    passive = simulate(sc, "passive", 24, FDIConfig())
    active = simulate(sc, "estimated", 24, FDIConfig())
    assert passive.crashed
    assert not active.crashed
    assert M.settled_error(active) < 0.2
    assert 15.0 < M.spin_rate(active) < np.deg2rad(2000)
