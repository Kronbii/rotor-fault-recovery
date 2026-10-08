"""Nominal flight controller: geometric tracking controller on SE(3).

Follows Lee, Leok & McClamroch (CDC 2010), written for a z-up world frame, with two
practical additions that every real flight stack has: integral action on position and
on attitude (with clamping anti-windup), and a tilt / thrust limit on the commanded
acceleration.

Why SE(3) rather than a cascaded Euler-angle PID: the attitude error is defined on
SO(3), so it has no singularities and it behaves sensibly for the large attitude
errors that follow an actuator failure; it is also a well-documented baseline that a
reviewer can check against the original paper. The desired angular velocity and
acceleration feedforward terms of the original are omitted (we use ``Omega_d = 0``),
which is common practice for the moderate trajectories flown here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .params import QuadParams
from .rotations import cross3, norm3, np_clip, vee
from .trajectory import Reference


@dataclass
class SE3Gains:
    # position loop, acceleration units (1/s^2, 1/s, 1/s^3); closed-loop poles
    # (-2, -2, -1) rad/s horizontally and (-3.6, -1.4, -1) rad/s vertically
    kp: np.ndarray = field(default_factory=lambda: np.array([8.0, 8.0, 10.0]))
    kv: np.ndarray = field(default_factory=lambda: np.array([5.0, 5.0, 6.0]))
    ki: np.ndarray = field(default_factory=lambda: np.array([4.0, 4.0, 5.0]))
    i_limit: float = 1.0  # m s
    # attitude loop: natural frequency / damping per axis
    att_wn: np.ndarray = field(default_factory=lambda: np.array([18.0, 18.0, 7.0]))
    att_zeta: np.ndarray = field(default_factory=lambda: np.array([0.75, 0.75, 0.9]))
    att_ki: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.0, 1.0]))  # 1/s
    att_i_limit: float = 0.3  # rad s
    max_tilt_deg: float = 45.0


@dataclass
class ControlOutput:
    thrust: float
    torque: np.ndarray
    R_des: np.ndarray
    a_cmd: np.ndarray


class SE3Controller:
    def __init__(self, params: QuadParams, gains: SE3Gains | None = None, dt: float = 0.002):
        self.P = params
        self.g = gains or SE3Gains()
        self.dt = dt
        Jd = np.diag(params.inertia)
        self.KR = Jd * self.g.att_wn**2
        self.KW = Jd * 2 * self.g.att_zeta * self.g.att_wn
        self.KIR = self.KR * self.g.att_ki
        self.int_p = np.zeros(3)
        self.int_R = np.zeros(3)
        self.J = params.inertia.copy()

    def reset(self):
        self.int_p[:] = 0
        self.int_R[:] = 0

    def position_command(self, p, v, ref: Reference) -> np.ndarray:
        """Desired acceleration including gravity compensation (world frame)."""
        g = self.g
        ep = p - ref.p
        ev = v - ref.v
        self.int_p = np_clip(self.int_p + ep * self.dt, -g.i_limit, g.i_limit)
        a = -g.kp * ep - g.kv * ev - g.ki * self.int_p + ref.a
        a[2] += self.P.g
        # nominal drag feed-forward along the reference velocity
        vr = ref.v
        P = self.P
        a += (P.drag_lin * vr + P.drag_quad * norm3(vr) * vr) / P.mass
        # thrust / tilt limits
        a[2] = max(a[2], 0.3 * self.P.g)
        hmax = a[2] * np.tan(np.deg2rad(g.max_tilt_deg))
        hn = np.hypot(a[0], a[1])
        if hn > hmax:
            a[0:2] *= hmax / hn
        return a

    def attitude_torque(self, R, omega, R_des) -> np.ndarray:
        eR = 0.5 * vee(R_des.T @ R - R.T @ R_des)
        self.int_R = np_clip(self.int_R + eR * self.dt, -self.g.att_i_limit, self.g.att_i_limit)
        tau = (
            -self.KR * eR
            - self.KW * omega
            - self.KIR * self.int_R
            + cross3(omega, self.J @ omega)
            + self.P.rot_damping * omega
        )
        return tau

    def __call__(self, p, v, R, omega, ref: Reference) -> ControlOutput:
        a = self.position_command(p, v, ref)
        F = self.P.mass * a
        b3d = F / norm3(F)
        b1c = np.array([np.cos(ref.yaw), np.sin(ref.yaw), 0.0])
        b2d = cross3(b3d, b1c)
        b2d /= norm3(b2d)
        b1d = cross3(b2d, b3d)
        R_des = np.column_stack([b1d, b2d, b3d])
        T = max(float(F @ R[:, 2]), 0.0)
        tau = self.attitude_torque(R, omega, R_des)
        return ControlOutput(thrust=T, torque=tau, R_des=R_des, a_cmd=a)
