"""Fault-tolerant control.

(a) Partial loss of effectiveness
    Handled in :mod:`rotorfault.stack` by passing the effectiveness vector (estimated or,
    for the oracle baseline, true) to :class:`~rotorfault.allocation.PriorityAllocator`,
    i.e. allocating with ``B diag(eta_hat)``; yaw is the first axis given up under
    saturation.

(b) Complete loss of one rotor -- yaw-relaxed control
    In the spirit of Mueller & D'Andrea (ICRA 2014): give up yaw, let the vehicle spin
    about a body-fixed *primary axis* ``n_bar`` and control only the direction of that
    axis and the thrust along it.

    1. :func:`solve_relaxed_hover` computes the relaxed-hover equilibrium numerically:
       three rotor thrusts and a constant body rate ``omega_bar`` (parallel to
       ``n_bar``) such that all body torques balance (thrust moments, rotor drag
       torques, gyroscopic terms of body and rotors, rotational damping) and the
       spin-averaged thrust ``f_T n_bar_z`` carries the weight. These solutions form
       a two-parameter family; we fix the tilt of ``n_bar`` from the thrust axis and
       pick its azimuth so that the spin rate is minimal (gyro range).
    2. The reduced attitude dynamics -- body rates ``(p, q)``, the two components of
       the desired-axis error in the plane orthogonal to ``n_bar`` and the three motor
       states -- are linearised *numerically* about quasi-equilibria with the yaw rate
       ``r`` frozen at a grid of values between 0 and 1.3 ``r_bar``. A discrete LQR
       (``scipy.linalg.solve_discrete_are``) is computed at every grid point and the
       gain is scheduled on the measured ``r`` at run time. The torque feed-forward
       that balances the gyroscopic moment is also evaluated at the measured ``r``;
       with ``r = 0`` it is zero, which makes the controller start from the
       "two opposing rotors" configuration that spins the vehicle up fastest.
    3. An outer position loop commands an acceleration ``a_cmd``; its direction is the
       desired primary-axis direction and ``f_T = m |a_cmd| / n_bar_z`` the thrust.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm, solve_discrete_are
from scipy.optimize import brentq, minimize_scalar

from .allocation import priority_solve
from .params import QuadParams
from .rotations import cross3, norm3, np_clip
from .trajectory import Reference

DIAGONAL = {0: 2, 1: 3, 2: 0, 3: 1}


# ======================================================================================
# Model pieces shared by the equilibrium solver and the linearisation
# ======================================================================================
def rotor_speeds(f: np.ndarray, P: QuadParams) -> np.ndarray:
    return np.sqrt(np_clip(f, P.f_min, P.f_max) / P.k_f)


def body_torque_balance(omega: np.ndarray, f: np.ndarray, P: QuadParams) -> np.ndarray:
    """Net torque (body frame) for constant rotor speeds: J dw/dt."""
    B = P.mixer()
    tau = B[1:4] @ f
    W = rotor_speeds(f, P)
    h = np.array([0.0, 0.0, -P.rotor_inertia * float(P.spin_dir @ W)])
    J = P.inertia
    return tau - cross3(omega, J @ omega + h) - P.rot_damping * omega


@dataclass
class RelaxedEquilibrium:
    failed: int
    active: np.ndarray  # indices of the three working rotors (adjacent, adjacent, diagonal)
    f: np.ndarray  # 4 rotor thrusts (failed rotor = 0)
    omega: np.ndarray  # body angular velocity at equilibrium
    n: np.ndarray  # primary axis (unit, body frame)
    tilt_deg: float
    residual: float

    @property
    def spin(self) -> float:
        return float(np.linalg.norm(self.omega))

    @property
    def total_thrust(self) -> float:
        return float(self.f.sum())

    @property
    def axis_tilt_deg(self) -> float:
        return math.degrees(math.acos(min(1.0, abs(self.n[2]))))


def _equilibrium_for_axis(
    P: QuadParams, failed: int, n: np.ndarray, spin_bounds=(1.0, 120.0)
) -> tuple[float, np.ndarray] | None:
    """For a given body-fixed primary axis ``n`` (unit, n_z > 0) find the spin rate
    and the three rotor thrusts that balance all torques and carry the weight."""
    diag = DIAGONAL[failed]
    act = [i for i in range(4) if i not in (failed, diag)] + [diag]
    Pi = np.linalg.inv(P.mixer()[1:4][:, act])
    sigma = -P.spin_dir[failed]  # sign of the yaw rate

    def thrusts(W):
        w = sigma * W * n
        f = np.full(4, P.hover_thrust)
        f[failed] = 0.0
        for _ in range(6):  # fixed point on the (small) rotor angular momentum
            Wr = rotor_speeds(f, P)
            Wr[failed] = P.omega_min
            h = np.array([0.0, 0.0, -P.rotor_inertia * float(P.spin_dir @ Wr)])
            treq = cross3(w, P.inertia @ w + h) + P.rot_damping * w
            f = np.zeros(4)
            f[act] = Pi @ treq
        return f

    def g(W):
        return thrusts(W).sum() * n[2] - P.weight

    lo, hi = spin_bounds
    if g(lo) * g(hi) > 0:
        return None
    W = brentq(g, lo, hi, xtol=1e-12)
    return W, thrusts(W)


def solve_relaxed_hover(P: QuadParams, failed: int, tilt_deg: float = 15.0) -> RelaxedEquilibrium:
    """Relaxed-hover equilibrium with the primary axis tilted ``tilt_deg`` from body z.

    The relaxed-hover solutions form a two-parameter family (Mueller & D'Andrea). We
    fix the tilt of the primary axis away from the thrust axis (a design parameter
    trading efficiency, ``n_z``, against spin rate) and choose its azimuth to
    *minimise the spin rate* subject to all rotor thrusts being within limits, because
    the spin must stay well inside the +-2000 deg/s gyro range of typical FPV IMUs.
    """
    diag = DIAGONAL[failed]
    active = np.array([i for i in range(4) if i not in (failed, diag)] + [diag])
    th = math.radians(tilt_deg)

    def axis(phi):
        return np.array([math.sin(th) * math.cos(phi), math.sin(th) * math.sin(phi), math.cos(th)])

    def cost(phi):
        sol = _equilibrium_for_axis(P, failed, axis(phi))
        if sol is None:
            return 1e3
        W, f = sol
        fa = f[active]
        pen = 0.0
        if fa.min() < 0.1 * P.f_max:
            pen += 100 * (0.1 * P.f_max - fa.min())
        if fa.max() > 0.8 * P.f_max:
            pen += 100 * (fa.max() - 0.8 * P.f_max)
        return W + pen

    phis = np.linspace(0, 2 * math.pi, 73)[:-1]
    costs = [cost(p) for p in phis]
    k = int(np.argmin(costs))
    if tilt_deg == 0.0:
        phi = 0.0
    else:
        res = minimize_scalar(
            cost, bounds=(phis[k] - 0.1, phis[k] + 0.1), method="bounded", options={"xatol": 1e-6}
        )
        phi = float(res.x)
    n = axis(phi)
    sol = _equilibrium_for_axis(P, failed, n)
    if sol is None:
        raise RuntimeError("no relaxed-hover equilibrium for this tilt")
    W, f = sol
    sigma = -P.spin_dir[failed]
    w = sigma * W * n
    resid = body_torque_balance(w, f, P)
    return RelaxedEquilibrium(
        failed=failed,
        active=active,
        f=f,
        omega=w,
        n=n,
        tilt_deg=tilt_deg,
        residual=float(np.linalg.norm(resid)) + abs(f.sum() * n[2] - P.weight),
    )


def _basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal basis (t1, t2) of the plane orthogonal to n."""
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = a - (a @ n) * n
    t1 /= np.linalg.norm(t1)
    t2 = cross3(n, t1)
    return t1, t2


# ======================================================================================
# Gain-scheduled LQR design
# ======================================================================================
@dataclass
class RelaxedDesign:
    eq: RelaxedEquilibrium
    Psi_inv: np.ndarray  # 3x3: [T, tau_x, tau_y] -> active rotor thrusts
    t1: np.ndarray
    t2: np.ndarray
    r_grid: np.ndarray
    K_grid: np.ndarray  # (len(r_grid), 2, 7)
    A_grid: np.ndarray
    B_grid: np.ndarray


@dataclass
class RelaxedWeights:
    q_rate: float = 1.0 / 3.0**2
    q_axis: float = 1.0 / 0.05**2
    q_motor: float = 1.0 / 1.0**2
    r_torque: float = 1.0 / 0.05**2


def tau_feedforward(r: float, eq: RelaxedEquilibrium, f: np.ndarray, P: QuadParams) -> np.ndarray:
    """Roll/pitch torque that keeps omega = (r / n_z) n_bar steady (z not balanced)."""
    w = (r / eq.n[2]) * eq.n
    W = rotor_speeds(f, P)
    h = np.array([0.0, 0.0, -P.rotor_inertia * float(P.spin_dir @ W)])
    tb = cross3(w, P.inertia @ w + h) + P.rot_damping * w
    return tb[0:2]


def linearise(
    P: QuadParams, eq: RelaxedEquilibrium, Psi_inv: np.ndarray, t1, t2, r: float
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous-time (A, B) of x = [dp, dq, dn1, dn2, df_a, df_b, df_c], u = [dtx, dty]."""
    act = eq.active
    n = eq.n
    w0 = (r / n[2]) * n
    # quasi-equilibrium thrusts: the nominal total thrust and the feed-forward torques
    fT = eq.total_thrust
    f0 = eq.f.copy()
    for _ in range(3):
        tff = tau_feedforward(r, eq, f0, P)
        f0 = np.zeros(4)
        f0[act] = Psi_inv @ np.array([fT, tff[0], tff[1]])
    Jd = np.diag(P.inertia)

    def wdot(pq, fa):
        f = f0.copy()
        f[act] = fa
        w = np.array([pq[0], pq[1], w0[2]])
        return body_torque_balance(w, f, P)[0:2] / Jd[0:2]

    eps_w = 1e-4
    eps_f = 1e-4
    A = np.zeros((7, 7))
    Bm = np.zeros((7, 2))
    base_pq = w0[0:2]
    base_f = f0[act]
    for k in range(2):
        d = np.zeros(2)
        d[k] = eps_w
        A[0:2, k] = (wdot(base_pq + d, base_f) - wdot(base_pq - d, base_f)) / (2 * eps_w)
    for k in range(3):
        d = np.zeros(3)
        d[k] = eps_f
        A[0:2, 4 + k] = (wdot(base_pq, base_f + d) - wdot(base_pq, base_f - d)) / (2 * eps_f)
    # axis error kinematics: d(dn)/dt = -dw x n - w0 x dn, projected on (t1, t2)
    T = np.column_stack([t1, t2])
    for k in range(2):
        dw = np.zeros(3)
        dw[k] = 1.0
        A[2:4, k] = T.T @ (-cross3(dw, n))
    for k in range(2):
        A[2:4, 2 + k] = T.T @ (-cross3(w0, T[:, k]))
    # first-order motors in thrust space
    A[4:7, 4:7] = -np.eye(3) / P.tau_motor
    Bm[4:7, :] = Psi_inv[:, 1:3] / P.tau_motor
    return A, Bm


def c2d(A, B, dt):
    n, m = B.shape
    M = expm(np.block([[A, B], [np.zeros((m, n + m))]]) * dt)
    return M[:n, :n], M[:n, n:]


def design_relaxed(
    P: QuadParams,
    failed: int,
    dt: float,
    tilt_deg: float = 15.0,
    weights: RelaxedWeights | None = None,
    n_grid: int = 14,
) -> RelaxedDesign:
    w = weights or RelaxedWeights()
    eq = solve_relaxed_hover(P, failed, tilt_deg)
    act = eq.active
    B = P.mixer()
    Psi = B[0:3][:, act]
    Psi_inv = np.linalg.inv(Psi)
    t1, t2 = _basis(eq.n)
    r_bar = eq.omega[2]
    r_grid = np.linspace(0.0, 1.3, n_grid) * r_bar
    Q = np.diag([w.q_rate, w.q_rate, w.q_axis, w.q_axis, w.q_motor, w.q_motor, w.q_motor])
    R = np.diag([w.r_torque, w.r_torque])
    Ks, As, Bs = [], [], []
    for r in r_grid:
        A, Bm = linearise(P, eq, Psi_inv, t1, t2, r)
        Ad, Bd = c2d(A, Bm, dt)
        X = solve_discrete_are(Ad, Bd, Q, R)
        K = np.linalg.solve(R + Bd.T @ X @ Bd, Bd.T @ X @ Ad)
        Ks.append(K)
        As.append(A)
        Bs.append(Bm)
    return RelaxedDesign(
        eq=eq,
        Psi_inv=Psi_inv,
        t1=t1,
        t2=t2,
        r_grid=r_grid,
        K_grid=np.array(Ks),
        A_grid=np.array(As),
        B_grid=np.array(Bs),
    )


# ======================================================================================
# Run-time controller
# ======================================================================================
@dataclass
class RelaxedGains:
    kp: np.ndarray = field(default_factory=lambda: np.array([3.0, 3.0, 5.0]))
    kv: np.ndarray = field(default_factory=lambda: np.array([3.0, 3.0, 4.0]))
    ki: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 1.5]))
    i_limit: float = 1.0
    max_tilt_deg: float = 25.0
    axis_err_limit: float = 0.6


@dataclass
class RelaxedOutput:
    f: np.ndarray
    n_des_body: np.ndarray
    axis_error: float  # angle between primary axis and desired direction (rad)


class RelaxedHoverController:
    def __init__(
        self,
        P: QuadParams,
        failed: int,
        dt: float,
        tilt_deg: float = 15.0,
        gains: RelaxedGains | None = None,
        weights: RelaxedWeights | None = None,
        design: RelaxedDesign | None = None,
    ):
        self.P = P
        self.dt = dt
        self.d = design or design_relaxed(P, failed, dt, tilt_deg, weights)
        self.g = gains or RelaxedGains()
        self.failed = failed
        self.int_p = np.zeros(3)
        self.lo = np.full(3, P.f_min)
        self.hi = np.full(3, P.f_max)

    def reset(self):
        self.int_p[:] = 0

    def gain(self, r: float) -> np.ndarray:
        rg = self.d.r_grid
        s = r / rg[-1] * (len(rg) - 1)
        s = min(max(s, 0.0), len(rg) - 1.0)
        i = min(int(s), len(rg) - 2)
        a = s - i
        return (1 - a) * self.d.K_grid[i] + a * self.d.K_grid[i + 1]

    def __call__(self, p, v, R, omega, ref: Reference, f_hat: np.ndarray) -> RelaxedOutput:
        P, g, d = self.P, self.g, self.d
        eq = d.eq
        n = eq.n
        # ---- outer position loop
        ep = p - ref.p
        ev = v - ref.v
        self.int_p = np_clip(self.int_p + ep * self.dt, -g.i_limit, g.i_limit)
        a = -g.kp * ep - g.kv * ev - g.ki * self.int_p + ref.a
        a[2] += P.g
        a[2] = max(a[2], 0.4 * P.g)
        hmax = a[2] * math.tan(math.radians(g.max_tilt_deg))
        hn = math.hypot(a[0], a[1])
        if hn > hmax:
            a[0:2] *= hmax / hn
        an = norm3(a)
        n_des_w = a / an
        fT = P.mass * an / n[2]
        n_des_b = R.T @ n_des_w
        # ---- reduced attitude error
        r = float(omega[2])
        act = eq.active
        f_act_hat = f_hat[act]
        tff = tau_feedforward(r, eq, f_hat, P)
        f_bar = d.Psi_inv @ np.array([fT, tff[0], tff[1]])
        w_bar = (r / n[2]) * n
        en = n_des_b - n
        e1 = float(d.t1 @ en)
        e2 = float(d.t2 @ en)
        lim = g.axis_err_limit
        e1 = min(max(e1, -lim), lim)
        e2 = min(max(e2, -lim), lim)
        x = np.array(
            [
                omega[0] - w_bar[0],
                omega[1] - w_bar[1],
                e1,
                e2,
                *(f_act_hat - f_bar),
            ]
        )
        u = -self.gain(r) @ x
        tau = tff + u
        Mi = d.Psi_inv
        h = Mi[:, 1] * tau[0] + Mi[:, 2] * tau[1]
        res = priority_solve(Mi[:, 0], fT, h, None, self.lo, self.hi)
        f = np.full(4, P.f_min)
        f[act] = res.f
        ang = math.acos(max(-1.0, min(1.0, float(n @ n_des_b))))
        return RelaxedOutput(f=f, n_des_body=n_des_b, axis_error=ang)
