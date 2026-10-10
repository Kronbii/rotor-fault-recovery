"""Airframe descriptions for running the diagnoser on logs of other vehicles.

An airframe file (YAML) holds what the diagnoser's regression needs -- mass, inertia,
rotor layout in the flight controller's motor order, spin directions, yaw-torque
coefficient, drag, damping -- plus the map from the logged motor command to rotor
speed, and the axis/unit conventions of the log format. ``sitl/config/x500.yaml``
(generated from PX4's Gazebo model files by :func:`from_gazebo_sdf`) and
``config/5inch.yaml`` (the real 5-inch quadrotor, to be identified) are examples.

Conventions: body frame FLU (x forward, y left, z up). Rotor ``i`` of the file is
motor ``i + 1`` of the flight controller. ``spin`` is the sign ``s_i`` of the yaw
reaction torque ``s_i kappa f_i`` about +z: ``+1`` for a rotor turning clockwise seen
from above, ``-1`` for counter-clockwise.

Motor map: normalised command ``u`` in [0, 1] -> rotor-speed command
``w = speed_at_zero + (speed_at_full - speed_at_zero) u`` -> first-order lag (``tau_up``
when speeding up, ``tau_down`` when slowing down) after ``delay_s`` of transport delay
-> thrust ``k_f w^2``.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .params import G, QuadParams, nominal_params


@dataclass
class MotorMap:
    k_f: float  # thrust coefficient, f = k_f w^2 (N s^2)
    speed_at_zero: float  # rad/s at normalised command 0
    speed_at_full: float  # rad/s at normalised command 1
    tau_up: float  # s
    tau_down: float  # s
    delay_s: float = 0.0  # command -> ESC transport delay

    def speed(self, u: np.ndarray) -> np.ndarray:
        """Rotor-speed command for a normalised command (NaN = motor off -> 0)."""
        u = np.asarray(u, dtype=float)
        w = self.speed_at_zero + (self.speed_at_full - self.speed_at_zero) * np.clip(u, 0.0, 1.0)
        return np.where(np.isfinite(u), w, 0.0)

    def command(self, w: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`speed` (clipped to [0, 1])."""
        u = (np.asarray(w, dtype=float) - self.speed_at_zero) / (
            self.speed_at_full - self.speed_at_zero
        )
        return np.clip(u, 0.0, 1.0)

    def thrust(self, w: np.ndarray) -> np.ndarray:
        return self.k_f * np.asarray(w, dtype=float) ** 2


@dataclass
class Airframe:
    name: str
    params: QuadParams
    motor: MotorMap
    log: dict = field(default_factory=dict)  # log-format conventions (see from_yaml)
    source: str = ""  # provenance of the numbers

    # ------------------------------------------------------------------ YAML
    @classmethod
    def from_dict(cls, d: dict) -> Airframe:
        v, m = d["vehicle"], d["motor"]
        rotors = v["rotors"]
        if len(rotors) != 4:
            raise ValueError("exactly four rotors are supported")
        xy = np.array([[r["x"], r["y"]] for r in rotors], dtype=float)
        spin = np.array([r["spin"] for r in rotors], dtype=float)
        if not np.all(np.isin(spin, (-1.0, 1.0))):
            raise ValueError("rotor spin must be +1 (clockwise) or -1 (counter-clockwise)")
        mm = MotorMap(
            k_f=float(m["k_f"]),
            speed_at_zero=float(m["speed_at_zero"]),
            speed_at_full=float(m["speed_at_full"]),
            tau_up=float(m["tau_up"]),
            tau_down=float(m.get("tau_down", m["tau_up"])),
            delay_s=float(m.get("delay_s", 0.0)),
        )
        J = v["inertia"]
        J = np.diag(J) if np.ndim(J) == 1 else np.array(J, dtype=float)
        p = QuadParams(
            mass=float(v["mass"]),
            inertia=np.array(J, dtype=float),
            arm_length=float(np.mean(np.linalg.norm(xy, axis=1))),
            kappa=float(v["kappa"]),
            k_f=mm.k_f,
            omega_min=mm.speed_at_zero,
            omega_max=mm.speed_at_full,
            tau_motor=0.5 * (mm.tau_up + mm.tau_down),
            rotor_inertia=float(v.get("rotor_inertia", 0.0)),
            drag_lin=np.array(v.get("drag_lin", [0.0, 0.0, 0.0]), dtype=float),
            drag_quad=float(v.get("drag_quad", 0.0)),
            rot_damping=np.array(v.get("rot_damping", [0.0, 0.0, 0.0]), dtype=float),
            g=float(v.get("gravity", G)),
            rotor_xy=xy,
            spin=spin,
        )
        return cls(d.get("name", "airframe"), p, mm, dict(d.get("log", {})), d.get("source", ""))

    @classmethod
    def from_yaml(cls, path: str | Path) -> Airframe:
        import yaml

        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> dict:
        p, m = self.params, self.motor
        rot = [
            {"x": float(x), "y": float(y), "spin": int(s)}
            for (x, y, _), s in zip(p.rotor_positions, p.spin_dir)
        ]
        return {
            "name": self.name,
            "source": self.source,
            "vehicle": {
                "mass": float(p.mass),
                "inertia": [float(j) for j in np.diag(p.inertia)],
                "gravity": float(p.g),
                "rotors": rot,
                "kappa": float(p.kappa),
                "rotor_inertia": float(p.rotor_inertia),
                "drag_lin": [float(d) for d in p.drag_lin],
                "drag_quad": float(p.drag_quad),
                "rot_damping": [float(d) for d in p.rot_damping],
            },
            "motor": {
                "k_f": m.k_f,
                "speed_at_zero": m.speed_at_zero,
                "speed_at_full": m.speed_at_full,
                "tau_up": m.tau_up,
                "tau_down": m.tau_down,
                "delay_s": m.delay_s,
            },
            "log": self.log,
        }

    def to_yaml(self, path: str | Path, header: str = "") -> None:
        import yaml

        txt = yaml.safe_dump(self.to_dict(), sort_keys=False, width=100)
        Path(path).write_text((header.rstrip() + "\n" if header else "") + txt)

    # ------------------------------------------------------------------ helpers
    @property
    def hover_speed(self) -> float:
        return math.sqrt(self.params.hover_thrust / self.motor.k_f)

    @property
    def hover_command(self) -> float:
        return float(self.motor.command(self.hover_speed))


def simulator_airframe(motor_order=(0, 1, 2, 3)) -> Airframe:
    """The simulated 5-inch quadrotor of this study as an :class:`Airframe`.

    ``motor_order[j]`` is the simulator rotor index (0 FR, 1 FL, 2 RL, 3 RR) wired to
    flight-controller motor ``j + 1``. The command map is linear in rotor speed between
    idle and maximum speed, which is what the simulator's ESC model does.
    """
    P = nominal_params()
    order = list(motor_order)
    pos = P.rotor_positions[order, :2]
    spin = P.spin_dir[order]
    p = P.copy(rotor_xy=pos, spin=spin)
    mm = MotorMap(P.k_f, P.omega_min, P.omega_max, P.tau_motor, P.tau_motor, 0.002)
    return Airframe("rotorfault-5inch-sim", p, mm, {}, "rotorfault.params.nominal_params()")


# ======================================================================================
def _vec(text: str | None, n: int) -> np.ndarray:
    vals = [float(v) for v in (text or "").split()]
    return np.array((vals + [0.0] * n)[:n])


def _inertial(link) -> tuple[float, np.ndarray, np.ndarray]:
    """(mass, CoM in the link frame, 3x3 inertia about the CoM in the link frame)."""
    inert = link.find("inertial")
    if inert is None:
        return 0.0, np.zeros(3), np.zeros((3, 3))
    m = float(inert.findtext("mass", "0"))
    com = _vec(inert.findtext("pose"), 6)[:3]
    I = inert.find("inertia")
    g = {k: float(I.findtext(k, "0")) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")}
    J = np.array(
        [
            [g["ixx"], g["ixy"], g["ixz"]],
            [g["ixy"], g["iyy"], g["iyz"]],
            [g["ixz"], g["iyz"], g["izz"]],
        ]
    )
    return m, com, J


def from_gazebo_sdf(
    base_sdf: str | Path, motors_sdf: str | Path, gravity: float = 9.8, name: str = "x500"
) -> Airframe:
    """Airframe of a PX4 Gazebo multicopter from its ``model.sdf`` files.

    ``base_sdf`` has the links (``base_link`` and ``rotor_<i>``); ``motors_sdf`` the
    ``MulticopterMotorModel`` plugins (PX4's ``x500_base`` / ``x500`` split). Mass and
    inertia are composed over all links about the common centre of mass (rotor links at
    their poses, rotations of the link frames assumed zero, as in the x500 model). Rotor
    ``i`` is the plugin with ``motorNumber`` ``i`` (PX4 motor ``i + 1``). The plugin
    spins the rotor joint at ``1 / rotorVelocitySlowdownSim`` of the simulated rotor
    speed, so the effective rotor inertia is divided by that factor.
    """
    base = ET.parse(base_sdf).getroot().find("model")
    links = {lk.get("name"): lk for lk in base.findall("link")}
    masses, coms, Js = [], [], []
    for lk in links.values():
        m, com, J = _inertial(lk)
        pose = _vec(lk.findtext("pose"), 6)
        if np.any(np.abs(pose[3:]) > 1e-9):
            raise ValueError(f"rotated link frames are not supported ({lk.get('name')})")
        masses.append(m)
        coms.append(pose[:3] + com)
        Js.append(J)
    M = float(sum(masses))
    c = sum(m * r for m, r in zip(masses, coms)) / M
    Jt = np.zeros((3, 3))
    for m, r, J in zip(masses, coms, Js):
        d = r - c
        Jt += J + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

    plugins = [
        pl
        for pl in ET.parse(motors_sdf).getroot().iter("plugin")
        if pl.get("name", "").endswith("MulticopterMotorModel")
    ]
    plugins.sort(key=lambda pl: int(pl.findtext("motorNumber")))
    rotors, kf, km, tup, tdn, wmax, slow, drag, jr = [], [], [], [], [], [], [], [], []
    for pl in plugins:
        link = links[pl.findtext("linkName")]
        pos = _vec(link.findtext("pose"), 6)[:3] - c
        direction = pl.findtext("turningDirection").strip().lower()
        # MulticopterMotorModel: drag torque -turningDirection * k_m * f about the rotor
        # axis, turningDirection = +1 for ccw -> yaw sign s = -1 for ccw, +1 for cw
        rotors.append(
            {"x": float(pos[0]), "y": float(pos[1]), "spin": -1 if direction == "ccw" else 1}
        )
        kf.append(float(pl.findtext("motorConstant")))
        km.append(float(pl.findtext("momentConstant")))
        tup.append(float(pl.findtext("timeConstantUp")))
        tdn.append(float(pl.findtext("timeConstantDown")))
        wmax.append(float(pl.findtext("maxRotVelocity")))
        slow.append(float(pl.findtext("rotorVelocitySlowdownSim")))
        drag.append(float(pl.findtext("rotorDragCoefficient")))
        jr.append(_inertial(link)[2][2, 2])
    if len(rotors) != 4:
        raise ValueError("expected four MulticopterMotorModel plugins")
    for vals in (kf, km, tup, tdn, wmax, slow, drag):
        if max(vals) - min(vals) > 1e-12 * max(1.0, abs(max(vals))):
            raise ValueError("rotors with different motor parameters are not supported")

    k_f, kappa = kf[0], km[0]
    w_hover = math.sqrt(M * gravity / 4.0 / k_f)
    # in-plane rotor drag -lambda * w * v_perp of every rotor -> body-frame drag in x/y
    lam = drag[0] * w_hover
    arm = float(np.mean([math.hypot(r["x"], r["y"]) for r in rotors]))
    vehicle = {
        "mass": M,
        "inertia": [float(Jt[0, 0]), float(Jt[1, 1]), float(Jt[2, 2])],
        "gravity": gravity,
        "rotors": rotors,
        "kappa": kappa,
        "rotor_inertia": float(np.mean(jr) / slow[0]),
        "drag_lin": [4 * lam, 4 * lam, 0.0],
        "drag_quad": 0.0,
        # rotor drag acting at the arm when the body yaws; roll/pitch: none modelled
        "rot_damping": [0.0, 0.0, 4 * lam * arm**2],
    }
    motor = {
        "k_f": k_f,
        "speed_at_zero": 0.0,
        "speed_at_full": wmax[0],
        "tau_up": tup[0],
        "tau_down": tdn[0],
        "delay_s": 0.0,
    }
    src = (
        "generated by rotorfault.airframe.from_gazebo_sdf from "
        f"{Path(base_sdf).parent.name}/{Path(base_sdf).name} and "
        f"{Path(motors_sdf).parent.name}/{Path(motors_sdf).name}; hover rotor speed "
        f"{w_hover:.0f} rad/s"
    )
    return Airframe.from_dict(
        {"name": name, "source": src, "vehicle": vehicle, "motor": motor, "log": {}}
    )
