"""PX4 ULog -> :class:`FlightLog` (needs ``pyulog``).

Diagnoser inputs (all on the ``sensor_combined`` time base, i.e. the IMU samples):

* ``gyro``, ``accel`` from ``sensor_combined`` (FRD -> FLU: y and z negated);
* ``motor`` from ``actuator_motors.control[0:4]`` (normalised thrust command of the
  control allocator, before the output driver; NaN = stopped), held between messages;
* ``v_body`` from ``vehicle_local_position`` (NED) rotated with ``vehicle_attitude``.

Kept for evaluation only (``extra``): the output driver's value ``actuator_outputs``
(``esc_output``; in SITL the rotor-speed command sent to Gazebo, which is where the ESC
faults of ``sitl/`` act), ``esc_status`` RPM, ground-truth position and tilt,
estimated position and tilt, PX4's ``failure_detector_status`` flags, arming and
navigation state. ``events`` holds parameter changes, logged text messages and the
first time each failure-detector flag was raised.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .flightlog import FlightLog

FD_FLAGS = ("fd_roll", "fd_pitch", "fd_alt", "fd_ext", "fd_motor", "fd_imbalanced_prop")


def _held(t_src, values, t_dst):
    """Zero-order hold of ``values`` (len(t_src), ...) onto ``t_dst`` (NaN before the first)."""
    idx = np.searchsorted(t_src, t_dst, side="right") - 1
    out = values[np.clip(idx, 0, len(t_src) - 1)].astype(float)
    out[idx < 0] = np.nan
    return out


def _quat_to_rot(q):
    w, x, y, z = q.T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def _tilt_deg(q):
    """Angle between body z and world z for (N, 4) quaternions (either frame pair)."""
    w, x, y, z = q.T
    return np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))


def read_ulog(path: str | Path, n_motors: int = 4) -> FlightLog:
    from pyulog import ULog

    u = ULog(str(path))
    topics = {}
    for d in u.data_list:
        if d.multi_id == 0:
            topics[d.name] = d.data

    def col(topic, fmt, n):
        return np.column_stack([topics[topic][fmt.format(i)] for i in range(n)]).astype(float)

    sc = topics["sensor_combined"]
    t = sc["timestamp"].astype(np.int64)
    ts = t * 1e-6
    sign = np.array([1.0, -1.0, -1.0])  # FRD -> FLU
    gyro = col("sensor_combined", "gyro_rad[{}]", 3) * sign
    accel = col("sensor_combined", "accelerometer_m_s2[{}]", 3) * sign

    am = topics["actuator_motors"]
    motor = _held(am["timestamp"] * 1e-6, col("actuator_motors", "control[{}]", n_motors), ts)

    att = topics["vehicle_attitude"]
    q = _held(att["timestamp"] * 1e-6, col("vehicle_attitude", "q[{}]", 4), ts)
    q = np.nan_to_num(q, nan=0.0)
    q[np.linalg.norm(q, axis=1) < 0.5] = (1.0, 0.0, 0.0, 0.0)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    extra = {"tilt_deg": _tilt_deg(q)}
    v_body = None
    if "vehicle_local_position" in topics:
        lp = topics["vehicle_local_position"]
        tl = lp["timestamp"] * 1e-6
        v_ned = _held(tl, np.column_stack([lp["vx"], lp["vy"], lp["vz"]]), ts)
        v_ned = np.nan_to_num(v_ned)
        R = _quat_to_rot(q)  # FRD body -> NED
        v_body = np.einsum("nji,nj->ni", R, v_ned) * sign
        extra["pos_ned"] = _held(tl, np.column_stack([lp["x"], lp["y"], lp["z"]]), ts)
    if "actuator_outputs" in topics:
        ao = topics["actuator_outputs"]
        extra["esc_output"] = _held(
            ao["timestamp"] * 1e-6, col("actuator_outputs", "output[{}]", n_motors), ts
        )
    if "esc_status" in topics:
        es = topics["esc_status"]
        extra["esc_rpm"] = _held(
            es["timestamp"] * 1e-6, col("esc_status", "esc[{}].esc_rpm", n_motors), ts
        )
    if "vehicle_local_position_groundtruth" in topics:
        g = topics["vehicle_local_position_groundtruth"]
        extra["gt_pos_ned"] = _held(
            g["timestamp"] * 1e-6, np.column_stack([g["x"], g["y"], g["z"]]), ts
        )
    if "vehicle_attitude_groundtruth" in topics:
        g = topics["vehicle_attitude_groundtruth"]
        qg = _held(g["timestamp"] * 1e-6, col("vehicle_attitude_groundtruth", "q[{}]", 4), ts)
        extra["gt_tilt_deg"] = _tilt_deg(np.nan_to_num(qg))

    events: dict = {}
    if "failure_detector_status" in topics:
        fd = topics["failure_detector_status"]
        tf = fd["timestamp"] * 1e-6
        for flag in FD_FLAGS:
            if flag in fd:
                v = fd[flag].astype(float)
                extra[flag] = np.nan_to_num(_held(tf, v, ts))
                on = np.flatnonzero(v > 0)
                events[f"first_{flag}"] = float(tf[on[0]]) if len(on) else None
    if "vehicle_status" in topics:
        vs = topics["vehicle_status"]
        tv = vs["timestamp"] * 1e-6
        extra["armed"] = np.nan_to_num(_held(tv, (vs["arming_state"] == 2).astype(float), ts))
        extra["nav_state"] = np.nan_to_num(_held(tv, vs["nav_state"].astype(float), ts))
        if "failsafe" in vs:
            on = np.flatnonzero(vs["failsafe"] > 0)
            events["first_failsafe"] = float(tv[on[0]]) if len(on) else None
    events["param_changes"] = [(p[0] * 1e-6, p[1], p[2]) for p in u.changed_parameters]
    events["messages"] = [(m.timestamp * 1e-6, m.message) for m in u.logged_messages]

    meta = {
        "source": "px4-ulog",
        "file": Path(path).name,
        "ver_sw": u.msg_info_dict.get("ver_sw"),
        "sys_name": u.msg_info_dict.get("sys_name"),
        "ver_hw": u.msg_info_dict.get("ver_hw"),
        "frame_note": "sensor_combined FRD converted to FLU; motor = actuator_motors.control",
        "params": {
            k: u.initial_parameters[k]
            for k in sorted(u.initial_parameters)
            if k.startswith(("SIM_GZ_EC", "FD_", "CA_FAILURE", "THR_MDL", "IMU_", "MPC_THR_HOVER"))
        },
    }
    return FlightLog(
        t=ts,
        gyro=gyro,
        accel=accel,
        motor=motor,
        v_body=v_body,
        extra=extra,
        events=events,
        meta=meta,
    )
