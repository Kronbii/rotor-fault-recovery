"""Runs are bit-for-bit reproducible from the seed; different seeds differ."""

import numpy as np

from rotorfault.fdi import FDIConfig
from rotorfault.scenarios import make_scenario
from rotorfault.sim import simulate


def test_same_seed_same_result():
    sc = make_scenario("hover", 3.0, rotor=3, magnitude=0.6, onset=1.0)
    a = simulate(sc, "estimated", 99, FDIConfig())
    b = simulate(sc, "estimated", 99, FDIConfig())
    assert np.array_equal(a.x, b.x)
    assert np.array_equal(a.eta_hat, b.eta_hat)
    assert a.alarm_time == b.alarm_time


def test_different_seed_different_result():
    sc = make_scenario("hover", 2.0)
    a = simulate(sc, "passive", 1, FDIConfig())
    b = simulate(sc, "passive", 2, FDIConfig())
    assert not np.array_equal(a.x, b.x)


def test_baselines_share_the_same_random_draws_before_the_fault():
    sc = make_scenario("hover", 3.0, rotor=0, magnitude=0.5, onset=2.0)
    a = simulate(sc, "passive", 5, FDIConfig())
    b = simulate(sc, "estimated", 5, FDIConfig())
    k = np.searchsorted(a.t, 2.0)
    assert np.array_equal(a.x[:k], b.x[:k])
