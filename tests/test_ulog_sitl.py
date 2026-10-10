"""PX4 ULog reader and replay on a real PX4 SITL log excerpt (tests/data/sitl_excerpt.ulg).

The excerpt (3.2 s, cut by sitl/make_fixture.py) is from a Gazebo x500 flight in which
the simulated ESC of motor 4 was set to deliver 25 % less thrust 2.6 s into the excerpt.
"""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyulog")

from rotorfault.airframe import Airframe  # noqa: E402
from rotorfault.logs.ulog import read_ulog  # noqa: E402
from rotorfault.replay import load_calibration, replay  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
META = json.loads((DATA / "sitl_excerpt.json").read_text())


@pytest.fixture(scope="module")
def log():
    return read_ulog(DATA / "sitl_excerpt.ulg")


def test_reader_frames_rates_and_events(log):
    assert log.rate_hz == pytest.approx(250.0, rel=0.01)
    hover = log.t < META["inject_sent_s"]
    # FRD -> FLU: the accelerometer reads +g on body z in hover
    assert np.median(log.accel[hover, 2]) == pytest.approx(9.8, abs=0.1)
    assert np.all(np.abs(np.median(log.gyro[hover], axis=0)) < 0.05)
    # x500 hover command and the simulated-ESC map 150 + 850 u (actuator_outputs)
    u = np.median(log.motor[hover], axis=0)
    assert np.allclose(u, 0.728, atol=0.01)
    assert np.allclose(np.median(log.extra["esc_output"][hover], axis=0), 150 + 850 * u, atol=2)
    names = [p[1] for p in log.events["param_changes"]]
    assert "SIM_GZ_EC_MAX4" in names and "SIM_GZ_EC_MIN4" in names
    assert log.v_body is not None and np.all(np.abs(log.v_body[hover]) < 0.5)


def test_replay_detects_and_isolates_the_esc_fault(log):
    af = Airframe.from_yaml(ROOT / "sitl/config/x500.yaml")
    cfg, delay = load_calibration(ROOT / "sitl/results/sitl_calibration_proposed.json")
    t0 = log.t[0]
    r = replay(log, af, cfg, t_detect=t0 + 1.5, baseline_s=1.0, warmup_s=0.5, delay_steps=delay)
    onset = META["inject_sent_s"]
    assert r.alarm and onset < r.alarm_time < onset + 0.3
    assert r.isolated + 1 == META["spec"]["motor"] == 4
    k = META["fault_params"]["SIM_GZ_EC_MAX4"] / 1000.0
    assert r.loe_hat(log.t[-1])[3] == pytest.approx(1 - k**2, abs=0.05)
