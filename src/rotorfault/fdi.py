"""Fault detection, isolation and effectiveness estimation.

Model
-----
Rigid-body rotational dynamics plus the collective (body-z) force balance:

    m a_z + D_z v_z + c|v| v_z                              = sum_i eta_i f_i
    J dw/dt + w x (J w + h_r) + K_w w - s J_r dOmega/dt e3  = B_tau diag(eta) f

i.e. ``y = B diag(f) eta = Phi eta`` with ``y`` (4) built from the gyro, the
accelerometer and the velocity estimate, and ``f`` the rotor thrusts *predicted* from
the commands through the nominal first-order motor model and the known 2 ms transport
delay (no RPM telemetry is assumed).

Signals
-------
The gyro is differentiated with a 2nd-order state-variable filter (Butterworth,
``fc`` Hz) which outputs the filtered rate and its derivative. *Every* other signal in
the regression (predicted thrusts, nonlinear terms, accelerometer) goes through the
same filter, so the regression stays exact for constant ``eta`` while the noise of the
differentiated gyro is attenuated above ``fc``.

Residual scale
--------------
Each row is normalised by ``sigma_k = sqrt(sigma_0^2 + u_k^2)``: a calibrated floor
``sigma_0`` (sensor noise, turbulence, static mismatch) plus the residual that the
assumed parametric uncertainty can produce at this instant -- inertia error times the
filtered angular acceleration and motor-lag error times the predicted thrust rate (an
adaptive threshold in the sense of robust residual evaluation).

Estimation
----------
``eta`` is tracked by :class:`EffectivenessKF`, a Kalman filter with a random-walk
model (equivalent to RLS with forgetting), measurement noise ``(k sigma_k)^2`` and
projection onto ``[eta_min, eta_max]``. Rotors commanded near idle are frozen.

Healthy baseline
----------------
Static mismatch (thrust-coefficient spread between rotors, mass error) looks exactly
like a small loss of effectiveness. During the initial hover (assumed fault-free) the
estimate is averaged into ``eta_ref`` and detection is then performed against it.

Detection and isolation
-----------------------
The residual ``r = (y - Phi eta_ref) / sigma_k`` of a loss ``delta`` on rotor ``j`` has
mean ``delta d_j`` with the known signature ``d_j = -Phi[:, j] eta_ref_j / sigma_k``.
Projecting on the unit signature gives ``t_j = d_j^T r / |d_j|`` (N(0, 1) for white
noise) and :class:`CusumBank` runs, for each rotor and each design magnitude
``delta_k``, the one-sided CUSUM of the log-likelihood ratio (Page 1954; Basseville &
Nikiforov 1993)

    g_kj <- max(0, g_kj + mu_kj (t_j - mu_kj / 2)),   mu_kj = delta_k |d_j|,

and raises an alarm when any ``g_kj > h_k``; the bank approximates the GLR test for an
unknown fault size. The isolated rotor is the ``argmax_j`` of the test that fired.
Because the residual is coloured and the model is not exact, average-run-length
formulas do not apply: ``sigma_0`` and ``h_k`` are calibrated on fault-free flights
with seeds disjoint from all evaluation seeds (``rotorfault.experiments.calibration``).
The defaults below are the values that calibration produced.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm

from .params import QuadParams
from .rotations import cross3, np_clip


@dataclass
class FDIConfig:
    filter_hz: float = 15.0
    filter_zeta: float = 0.7071
    # residual floor per row (N, N m, N m, N m); calibrated, see calibration.py
    sigma: np.ndarray = field(default_factory=lambda: np.array([0.170, 4.49e-3, 4.49e-3, 2.43e-3]))
    # assumed relative uncertainty of inertia and motor time constant (used to widen
    # the residual scale during manoeuvres; matches the parameter spread simulated)
    unc_inertia: float = 0.05
    unc_tau: float = 0.10
    kf_r_inflation: float = 4.0  # coloured-noise inflation of R in the KF
    kf_q: float = 0.6  # random-walk intensity of eta (1/sqrt(s))
    excitation_min: float = 0.2  # N; below this predicted thrust eta_i is frozen
    actuator_delay_steps: int = 1  # known command transport delay (control periods)
    eta_min: float = 0.0
    eta_max: float = 1.1
    # healthy-baseline learning: eta_hat is averaged over [baseline_start,
    # detect_after) (initial hover, assumed fault-free); detection starts afterwards
    use_baseline: bool = True
    baseline_start: float = -2.5
    detect_after: float = 0.0
    # bank of CUSUM tests, one per design LOE magnitude, each with its own threshold
    deltas: tuple = (0.15, 0.3, 0.6)
    thresholds: np.ndarray = field(default_factory=lambda: np.array([1519.0, 741.0, 2964.0]))
    cusum_max: float = 1e5


@dataclass
class FDIOutput:
    eta_hat: np.ndarray
    alarm: bool
    isolated: int  # -1 if no alarm yet
    alarm_time: float  # nan if no alarm yet
    cusum: np.ndarray  # normalised decision statistic per rotor (alarm at 1)
    residual: np.ndarray  # normalised residual
    eta_ref: np.ndarray  # healthy baseline learned before detection is enabled
    sigma: np.ndarray  # residual scale used at this step
    cusum_bank: np.ndarray  # max over rotors of each CUSUM in the bank (raw units)

    @property
    def loe_hat(self) -> np.ndarray:
        """Estimated loss of effectiveness relative to the healthy baseline."""
        return 1.0 - self.eta_hat / self.eta_ref


def hover_signature_norm(params: QuadParams, sigma: np.ndarray) -> float:
    """|d_j| at nominal hover thrust (identical for the four rotors)."""
    col = params.mixer()[:, 0]
    return float(params.hover_thrust * np.linalg.norm(col / sigma))


# ======================================================================================
class EffectivenessKF:
    """Kalman filter for ``y = Phi eta + v`` with a random-walk model on ``eta``."""

    def __init__(self, n: int = 4, q: float = 0.6, p0: float = 1e-3, lo=0.0, hi=1.1):
        self.n = n
        self.q = q
        self.lo, self.hi = lo, hi
        self.eta = np.ones(n)
        self.P = np.eye(n) * p0
        self._I = np.eye(n)

    def update(self, Phi: np.ndarray, y: np.ndarray, r_std: np.ndarray, dt: float, excited=None):
        if excited is None:
            excited = np.ones(self.n, dtype=bool)
        self.P.flat[:: self.n + 1] += np.where(excited, self.q**2 * dt, 0.0)
        S = Phi @ self.P @ Phi.T
        S.flat[:: len(y) + 1] += r_std**2
        K = np.linalg.solve(S, Phi @ self.P).T
        K[~excited, :] = 0.0
        innov = y - Phi @ self.eta
        self.eta = np_clip(self.eta + K @ innov, self.lo, self.hi)
        self.P = (self._I - K @ Phi) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        return self.eta


class CusumBank:
    """One-sided CUSUM tests for a mean shift along known signatures."""

    def __init__(self, deltas, thresholds, n: int = 4, g_max: float = 1e5):
        self.deltas = np.asarray(deltas, dtype=float)[:, None]
        self.h = np.asarray(thresholds, dtype=float)
        self.g = np.zeros((len(self.deltas), n))
        self.g_max = g_max
        self.alarm = False
        self.isolated = -1
        self.bank = -1

    def statistics(self, r: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Projected statistic ``t_j`` and signature norms for residual ``r`` (m) and
        signatures ``D`` (m x n, one column per hypothesis)."""
        dn = np.sqrt((D * D).sum(axis=0)) + 1e-12
        return (D.T @ r) / dn, dn

    def update(self, tj: np.ndarray, dn: np.ndarray, active: bool = True) -> bool:
        mu = self.deltas * dn[None, :]
        if active:
            self.g = np_clip(self.g + mu * (tj[None, :] - 0.5 * mu), 0.0, self.g_max)
        ratio = self.g / self.h[:, None]
        if not self.alarm and ratio.max() > 1.0:
            k = int(np.argmax(ratio.max(axis=1)))
            self.alarm = True
            self.bank = k
            self.isolated = int(np.argmax(self.g[k]))
            return True
        return False

    @property
    def normalised(self) -> np.ndarray:
        return (self.g / self.h[:, None]).max(axis=0)


# ======================================================================================
class RotorFDI:
    def __init__(self, params: QuadParams, dt: float, cfg: FDIConfig | None = None):
        self.P = params
        self.cfg = cfg or FDIConfig()
        self.dt = dt
        c = self.cfg
        wc = 2 * math.pi * c.filter_hz
        A = np.array([[0.0, 1.0], [-(wc**2), -2 * c.filter_zeta * wc]])
        Bc = np.array([[0.0], [wc**2]])
        # zero-order-hold discretisation of the state-variable filter
        M = expm(np.block([[A, Bc], [np.zeros((1, 3))]]) * dt)
        self._Ad = M[:2, :2]
        self._Bd = M[:2, 2]
        self.B = params.mixer()
        self.J = np.diag(params.inertia).copy()
        self.s = params.spin_dir
        self.n_ch = 3 + 3 + 4 + 1  # gyro, nonlinear terms, thrusts, collective
        self._absB = np.abs(self.B[1:4])
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        P, c = self.P, self.cfg
        self._x1 = np.zeros(self.n_ch)
        self._x2 = np.zeros(self.n_ch)
        self._primed = False
        self.omega_hat = np.full(4, math.sqrt(P.hover_thrust / P.k_f))  # predicted speeds
        self._cmd_fifo = deque(maxlen=c.actuator_delay_steps + 1)
        self.kf = EffectivenessKF(4, c.kf_q, lo=c.eta_min, hi=c.eta_max)
        self.cusum = CusumBank(c.deltas, c.thresholds, 4, c.cusum_max)
        self.alarm_time = float("nan")
        self.t = 0.0
        self.eta_ref = np.ones(4)
        self._ref_sum = np.zeros(4)
        self._ref_n = 0
        self._ref_done = False

    @property
    def eta_hat(self) -> np.ndarray:
        return self.kf.eta

    @property
    def alarm(self) -> bool:
        return self.cusum.alarm

    @property
    def isolated(self) -> int:
        return self.cusum.isolated

    def _motor_model(self, f_cmd: np.ndarray):
        """Propagate the nominal motor model over one step; returns (f_pred, dOmega)."""
        P = self.P
        w_cmd = np.sqrt(np_clip(f_cmd, P.f_min, P.f_max) / P.k_f)
        a = math.exp(-self.dt / P.tau_motor)
        w_new = w_cmd + (self.omega_hat - w_cmd) * a
        dw = (w_new - self.omega_hat) / self.dt
        self.omega_hat = w_new
        return P.k_f * w_new**2, dw

    def update(
        self,
        t: float,
        gyro: np.ndarray,
        accel: np.ndarray,
        f_cmd_applied: np.ndarray,
        v_body: np.ndarray,
    ) -> FDIOutput:
        """One FDI step.

        ``gyro`` must be bias-corrected; ``f_cmd_applied`` is the latest thrust command
        sent (the known transport delay is applied internally); ``v_body`` is the
        estimated velocity in body axes (nominal drag term of the collective row).
        """
        P = self.P
        c = self.cfg
        self.t = t
        self._cmd_fifo.append(np.asarray(f_cmd_applied, dtype=float))
        f_pred, dW = self._motor_model(self._cmd_fifo[0])
        w = gyro
        J = self.J
        h_z = -P.rotor_inertia * float(self.s @ self.omega_hat)
        nonlin = cross3(w, J * w + np.array([0.0, 0.0, h_z])) + P.rot_damping * w
        nonlin[2] -= P.rotor_inertia * float(self.s @ dW)
        sp = math.sqrt(float(v_body @ v_body))
        coll = P.mass * accel[2] + P.drag_lin[2] * v_body[2] + P.drag_quad * sp * v_body[2]

        # ---------------- common state-variable filter ----------------
        u = np.concatenate([w, nonlin, f_pred, [coll]])
        if not self._primed:
            self._x1 = u.copy()
            self._x2 = np.zeros_like(u)
            self._primed = True
        Ad, Bd = self._Ad, self._Bd
        x1 = Ad[0, 0] * self._x1 + Ad[0, 1] * self._x2 + Bd[0] * u
        x2 = Ad[1, 0] * self._x1 + Ad[1, 1] * self._x2 + Bd[1] * u
        self._x1, self._x2 = x1, x2
        dw_f = x2[0:3]
        f_f = x1[6:10]
        y = np.empty(4)
        y[0] = x1[10]
        y[1:4] = J * dw_f + x1[3:6]
        Phi = self.B * f_f[None, :]

        # ---------------- adaptive residual scale ----------------
        fdot = np.abs(x2[6:10])
        lag = c.unc_tau * P.tau_motor
        unc = np.empty(4)
        unc[0] = lag * fdot.sum()
        unc[1:4] = c.unc_inertia * np.abs(J * dw_f) + lag * (self._absB @ fdot)
        sig = np.sqrt(c.sigma**2 + unc**2)

        # ---------------- effectiveness estimate ----------------
        excited = f_f > c.excitation_min
        self.kf.update(Phi, y, sig * c.kf_r_inflation, self.dt, excited)

        # ---------------- healthy baseline ----------------
        if t < c.detect_after:
            if c.use_baseline and t >= c.baseline_start:
                self._ref_sum += self.kf.eta
                self._ref_n += 1
        elif not self._ref_done:
            if self._ref_n > 0:
                self.eta_ref = self._ref_sum / self._ref_n
            self._ref_done = True

        # ---------------- detection / isolation ----------------
        Phi_ref = Phi * self.eta_ref[None, :]
        r = (y - Phi_ref.sum(axis=1)) / sig
        tj, dn = self.cusum.statistics(r, -Phi_ref / sig[:, None])
        if self.cusum.update(tj, dn, active=self._ref_done):
            self.alarm_time = t
        return FDIOutput(
            eta_hat=self.kf.eta.copy(),
            alarm=self.cusum.alarm,
            isolated=self.cusum.isolated,
            alarm_time=self.alarm_time,
            cusum=self.cusum.normalised,
            cusum_bank=self.cusum.g.max(axis=1),
            residual=r,
            eta_ref=self.eta_ref,
            sigma=sig,
        )
