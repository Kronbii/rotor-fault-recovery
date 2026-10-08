"""6-DOF rigid-body quadrotor dynamics with first-order rotor-speed lag.

State vector (17):
    x[0:3]   position p in world (m)
    x[3:6]   velocity v in world (m/s)
    x[6:10]  unit quaternion q = (w, x, y, z), body to world
    x[10:13] body angular velocity omega (rad/s)
    x[13:17] rotor speeds Omega_i (rad/s)

Equations (body frame for rotation, world for translation):
    m dv/dt   = -m g e3 + R [0, 0, T] - R D R^T v - c |v| v + F_d
    dq/dt     = 0.5 q (x) [0, omega]
    J domega/dt = tau_thrust + tau_yaw + tau_rotor_accel
                  - omega x (J omega + h_r) - K_w omega + tau_d
    dOmega_i/dt = (Omega_cmd_i - Omega_i) / tau_m,  Omega_cmd in [Omega_min, Omega_max]

with rotor thrust ``f_i = eta_i k_f s_i^kf Omega_i^2`` (``eta_i`` = loss-of-effectiveness
factor, ``s_i^kf`` = per-rotor manufacturing spread), yaw torque ``s_i kappa f_i``,
rotor angular momentum ``h_r = -J_r sum_i s_i Omega_i e3`` and the reaction torque of
accelerating rotors ``s_i J_r dOmega_i/dt`` about z. ``F_d`` / ``tau_d`` are external
disturbances (turbulence, see :mod:`rotorfault.disturbance`).

The derivative is written with scalar Python arithmetic; for 17 states this is several
times faster than small-array NumPy operations and dominates simulation speed.
"""

from __future__ import annotations

import math

import numpy as np

from .params import QuadParams
from .rotations import np_clip, quat_to_rot

N_STATE = 17
_NO_DIST = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class QuadDynamics:
    """Integrates the true vehicle with fixed-step RK4."""

    def __init__(self, params: QuadParams):
        self.p = params
        J = np.diag(params.inertia)
        self._J = tuple(float(j) for j in J)
        r = params.rotor_positions
        self._rx = tuple(float(v) for v in r[:, 0])
        self._ry = tuple(float(v) for v in r[:, 1])
        self._s = tuple(float(v) for v in params.spin_dir)
        self._kf = tuple(float(params.k_f * s) for s in params.k_f_scale)
        self._D = tuple(float(d) for d in params.drag_lin)
        self._Kw = tuple(float(d) for d in params.rot_damping)

    # ------------------------------------------------------------------
    def thrusts(self, omega_rotor, eta) -> np.ndarray:
        kf = self._kf
        return np.array([eta[i] * kf[i] * omega_rotor[i] ** 2 for i in range(4)])

    def deriv(self, x: list, wcmd: list, eta: list, dist=_NO_DIST) -> list:
        """Time derivative of the state (lists of floats in and out).

        ``dist`` = (Fx, Fy, Fz, tx, ty, tz): external disturbance force (world frame,
        N) and torque (body frame, N m), e.g. turbulence.
        """
        P = self.p
        m = P.mass
        g = P.g
        Jx, Jy, Jz = self._J
        rx, ry, s, kf = self._rx, self._ry, self._s, self._kf
        Dx, Dy, Dz = self._D
        Kx, Ky, Kz = self._Kw
        kappa = P.kappa
        Jr = P.rotor_inertia
        itau = 1.0 / P.tau_motor

        vx, vy, vz = x[3], x[4], x[5]
        qw, qx, qy, qz = x[6], x[7], x[8], x[9]
        wx, wy, wz = x[10], x[11], x[12]
        W = x[13:17]

        # rotor dynamics and forces
        T = 0.0
        tx = 0.0
        ty = 0.0
        tz = 0.0
        hz = 0.0
        dW = [0.0, 0.0, 0.0, 0.0]
        for i in range(4):
            Wi = W[i]
            dWi = (wcmd[i] - Wi) * itau
            dW[i] = dWi
            fi = eta[i] * kf[i] * Wi * Wi
            T += fi
            tx += ry[i] * fi
            ty -= rx[i] * fi
            tz += s[i] * (kappa * fi + Jr * dWi)
            hz -= Jr * s[i] * Wi

        # rotation matrix
        r00 = 1 - 2 * (qy * qy + qz * qz)
        r01 = 2 * (qx * qy - qw * qz)
        r02 = 2 * (qx * qz + qw * qy)
        r10 = 2 * (qx * qy + qw * qz)
        r11 = 1 - 2 * (qx * qx + qz * qz)
        r12 = 2 * (qy * qz - qw * qx)
        r20 = 2 * (qx * qz - qw * qy)
        r21 = 2 * (qy * qz + qw * qx)
        r22 = 1 - 2 * (qx * qx + qy * qy)

        # body-frame velocity and drag
        bvx = r00 * vx + r10 * vy + r20 * vz
        bvy = r01 * vx + r11 * vy + r21 * vz
        bvz = r02 * vx + r12 * vy + r22 * vz
        fbx = -Dx * bvx
        fby = -Dy * bvy
        fbz = T - Dz * bvz
        sp = math.sqrt(vx * vx + vy * vy + vz * vz)
        cq = P.drag_quad * sp
        im = 1.0 / m
        ax = (r00 * fbx + r01 * fby + r02 * fbz - cq * vx + dist[0]) * im
        ay = (r10 * fbx + r11 * fby + r12 * fbz - cq * vy + dist[1]) * im
        az = (r20 * fbx + r21 * fby + r22 * fbz - cq * vz + dist[2]) * im - g

        # quaternion kinematics
        dqw = 0.5 * (-qx * wx - qy * wy - qz * wz)
        dqx = 0.5 * (qw * wx + qy * wz - qz * wy)
        dqy = 0.5 * (qw * wy + qz * wx - qx * wz)
        dqz = 0.5 * (qw * wz + qx * wy - qy * wx)

        # Euler equation with rotor gyroscopics
        Hx = Jx * wx
        Hy = Jy * wy
        Hz = Jz * wz + hz
        cx = wy * Hz - wz * Hy
        cy = wz * Hx - wx * Hz
        cz = wx * Hy - wy * Hx
        dwx = (tx - cx - Kx * wx + dist[3]) / Jx
        dwy = (ty - cy - Ky * wy + dist[4]) / Jy
        dwz = (tz - cz - Kz * wz + dist[5]) / Jz

        return [vx, vy, vz, ax, ay, az, dqw, dqx, dqy, dqz, dwx, dwy, dwz, *dW]

    def specific_force(self, x, wcmd, eta, dist=_NO_DIST) -> np.ndarray:
        """Specific force in body frame (what an ideal accelerometer measures)."""
        d = self.deriv(x.tolist(), list(wcmd), list(eta), dist)
        R = _rot(x[6:10])
        a_world = np.array(d[3:6]) + np.array([0.0, 0.0, self.p.g])
        return R.T @ a_world

    def step(self, x: np.ndarray, wcmd, eta, dt: float, dist=_NO_DIST) -> np.ndarray:
        """One RK4 step; rotor commands and eta are held constant over the step."""
        xl = x.tolist()
        wc = [float(w) for w in wcmd]
        et = [float(e) for e in eta]
        h2 = 0.5 * dt
        k1 = self.deriv(xl, wc, et, dist)
        k2 = self.deriv([a + h2 * b for a, b in zip(xl, k1)], wc, et, dist)
        k3 = self.deriv([a + h2 * b for a, b in zip(xl, k2)], wc, et, dist)
        k4 = self.deriv([a + dt * b for a, b in zip(xl, k3)], wc, et, dist)
        h6 = dt / 6.0
        xn = [a + h6 * (b + 2 * c + 2 * d + e) for a, b, c, d, e in zip(xl, k1, k2, k3, k4)]
        # renormalise the quaternion
        qn = math.sqrt(xn[6] ** 2 + xn[7] ** 2 + xn[8] ** 2 + xn[9] ** 2)
        for i in range(6, 10):
            xn[i] /= qn
        return np.array(xn)


def _rot(q):
    return quat_to_rot(q)


def hover_state(params: QuadParams, position=(0.0, 0.0, 2.0)) -> np.ndarray:
    """State at rest with rotors spinning at the nominal hover speed."""
    x = np.zeros(N_STATE)
    x[0:3] = position
    x[6] = 1.0
    w_h = math.sqrt(params.hover_thrust / params.k_f)
    x[13:17] = w_h
    return x


def thrust_to_speed(f_cmd, params: QuadParams) -> np.ndarray:
    """ESC/firmware map from commanded (nominal) thrust to rotor-speed command."""
    f = np_clip(np.asarray(f_cmd, dtype=float), params.f_min, params.f_max)
    return np.sqrt(f / params.k_f)
