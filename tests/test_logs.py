"""Log pipeline: simulator -> synthetic Betaflight blackbox CSV -> reader -> replay.

The replay of a synthetic log must reproduce the in-simulation diagnoser, and the CSV
round trip (unit conversion, axis mapping, motor order, DShot and eRPM quantisation)
must not change the diagnosis.
"""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from rotorfault.experiments.common import load_json
from rotorfault.experiments.runner import fdi_config
from rotorfault.logs.betaflight import read_blackbox_csv, write_blackbox_csv
from rotorfault.logs.flightlog import FlightLog, load_any
from rotorfault.logs.synthetic import synthetic_flight
from rotorfault.replay import LogSegment, auto_detect_start, calibrate_on_logs, replay

ROOT = Path(__file__).resolve().parents[1]
CFG = fdi_config(load_json(ROOT / "results/calibration.json")["gyro_x1"])


@pytest.fixture(scope="module")
def faulty():
    # 40 % loss on simulator rotor 2 (rear-left) = Betaflight motor 3, at t = 4 s
    return synthetic_flight("hover", 5.0, rotor=2, loe=0.4, onset=4.0, seed=11, cfg=CFG)


def test_replay_reproduces_in_simulation_diagnoser(faulty):
    log, af, r = faulty
    rr = replay(log, af, CFG, t_detect=3.0, baseline_s=2.5)
    assert rr.alarm and math.isfinite(r.alarm_time)
    assert abs(rr.alarm_time - r.alarm_time) <= 0.004
    assert rr.isolated + 1 == log.events["fault_motor"] == 3
    assert rr.loe_hat(rr.alarm_time + 0.25)[2] == pytest.approx(0.4, abs=0.05)


def test_betaflight_csv_round_trip(faulty, tmp_path):
    log, af, _ = faulty
    af = copy.deepcopy(af)
    af.log = {"gyro_axes": ["+y", "-x", "+z"], "acc_axes": ["+y", "-x", "+z"]}  # permuted
    path = tmp_path / "LOG00001.csv"
    write_blackbox_csv(log, path, af)
    header = path.read_text().splitlines()[0]
    assert "gyroADC[0] (rad/s)" in header and "eRPM[3]" in header
    back = read_blackbox_csv(path, af)
    assert np.allclose(back.gyro, log.gyro, atol=1e-5)
    assert np.allclose(back.accel, log.accel, atol=1e-4)
    assert np.allclose(back.motor, log.motor, atol=1.0 / 1999)
    assert np.allclose(back.rotor_speed, log.rotor_speed, atol=1.2)  # eRPM/100 steps
    a = replay(log, af, CFG, 3.0)
    b = replay(back, af, CFG, 3.0)
    assert b.isolated == a.isolated
    assert abs(b.alarm_time - a.alarm_time) <= 0.006
    # with RPM telemetry instead of the motor model
    c = replay(back, af, CFG, 3.0, measured_speed=True)
    assert c.isolated == a.isolated and c.alarm_time - 4.0 < 0.1


def test_raw_units_and_minimal_format(faulty, tmp_path):
    log, af, _ = faulty
    n = 400
    rows = ["t_s,gyro_x,gyro_y,gyro_z,acc_x,acc_y,acc_z,motor_1,motor_2,motor_3,motor_4"]
    for k in range(n):
        rows.append(
            ",".join(f"{v:.6f}" for v in [log.t[k], *log.gyro[k], *log.accel[k], *log.motor[k]])
        )
    p = tmp_path / "minimal.csv"
    p.write_text("\n".join(rows) + "\n")
    m = load_any(p)
    assert m.meta["source"] == "rotorfault-minimal-csv"
    assert np.allclose(m.gyro, log.gyro[:n], atol=1e-6) and m.rotor_speed is None
    # raw gyro units without a scale factor are refused, not guessed
    q = tmp_path / "raw.csv"
    q.write_text(
        "time (us),gyroADC[0],gyroADC[1],gyroADC[2],accSmooth[0],accSmooth[1],accSmooth[2],"
        "motor[0],motor[1],motor[2],motor[3]\n0,1,2,3,0,0,2048,500,500,500,500\n"
        "125,1,2,3,0,0,2048,500,500,500,500\n"
    )
    with pytest.raises(ValueError, match="raw gyro"):
        read_blackbox_csv(q)


def test_flightlog_storage_and_resampling(faulty, tmp_path):
    log, _, _ = faulty
    p = tmp_path / "log.npz"
    log.save(p)
    back = FlightLog.load(p)
    assert np.array_equal(back.t, log.t)
    assert np.allclose(back.gyro, log.gyro, atol=1e-6)
    assert back.events["fault_motor"] == 3
    half = log.resample(250.0)
    assert half.rate_hz == pytest.approx(250.0)
    assert np.allclose(half.motor[10], log.motor[np.searchsorted(log.t, half.t[10], "right") - 1])


def test_calibration_on_fault_free_logs_and_auto_start(faulty):
    _, af, _ = faulty
    free = [synthetic_flight("hover", 5.0, loe=0.0, seed=s, cfg=CFG)[0] for s in (21, 22)]
    segs = [LogSegment(f, 3.0) for f in free]
    cfg, info = calibrate_on_logs(segs, af, CFG)
    assert np.all(np.isfinite(cfg.thresholds)) and np.all(np.asarray(cfg.sigma) > 0)
    assert info["fault_free_s"] == pytest.approx(4.0, abs=0.1)
    for f in free:
        assert not replay(f, af, cfg, 3.0).alarm
    # the synthetic log starts in hover: take-off at the first sample
    assert auto_detect_start(free[0], af, 0.5, 2.5) == pytest.approx(3.0)
    json.dumps(info)  # serialisable


def test_stored_baseline_detects_a_fault_present_from_take_off(faulty):
    """A weakened propeller is there from take-off: a baseline learned in that flight
    absorbs it, a baseline stored from earlier fault-free flights does not."""
    _, af, _ = faulty
    free = [synthetic_flight("hover", 5.0, loe=0.0, seed=s, cfg=CFG)[0] for s in (31, 32)]
    cfg, info = calibrate_on_logs([LogSegment(f, 3.0) for f in free], af, CFG)
    weak, _, _ = synthetic_flight("hover", 5.0, rotor=1, loe=0.3, onset=0.0, seed=33, cfg=CFG)
    learned = replay(weak, af, cfg, 3.0)
    assert not learned.alarm
    assert learned.loe_hat(4.9)[3] == pytest.approx(0.0, abs=0.05)  # absorbed (FL = motor 4)
    stored = replay(weak, af, cfg, 3.0, eta_ref=info["eta_ref_mean"])
    assert stored.alarm and stored.isolated + 1 == 4
    assert stored.loe_hat(4.9)[3] == pytest.approx(0.3, abs=0.06)
