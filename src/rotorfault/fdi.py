"""Fault detection, isolation and effectiveness estimation.

Model
-----
Rigid-body rotational dynamics plus the collective (body-z) force balance:

    m a_z + D_z v_z + c|v| v_z        = sum_i eta_i f_i
    J dw/dt + w x (J w + h_r) + K_w w - s J_r dOmega/dt e3  = B_tau diag(eta) f

i.e. ``y = B diag(f) eta = Phi eta`` with ``y`` (4) built from the gyro, the
accelerometer and the velocity estimate, and ``f`` the rotor thrusts *predicted* from
the commands through the nominal first-order motor model (no RPM telemetry is assumed).

Signals
-------
The gyro is differentiated with a 2nd-order state-variable filter (Butterworth,
``fc`` Hz) which outputs the filtered rate and its derivative. *Every* other signal in
the regression (predicted thrusts, nonlinear terms, accelerometer) goes through the
same filter, so the regression stays exact for constant ``eta`` while the noise of the
differentiated gyro is attenuated above ``fc``.

Estimation
----------
``eta`` is tracked by a Kalman filter with a random-walk model (equivalent to RLS with
forgetting), measurement noise from the per-row residual variance, and projection onto
``[eta_min, eta_max]``.

Detection and isolation
-----------------------
The residual against the healthy model ``r = y - Phi 1`` is normalised by the per-row
noise level. A loss of effectiveness ``delta`` on rotor ``j`` shifts its mean along the
known signature ``d_j = -Phi[:, j] / sigma``. For each rotor we project onto the unit
signature, ``t_j = d_j^T r / |d_j|`` (N(0, 1) per sample for white noise), and run a
one-sided CUSUM on the log-likelihood ratio of a design shift ``mu_j = delta_min |d_j|``
(Page 1954; Basseville & Nikiforov 1993):

    g_j <- max(0, g_j + mu_j (t_j - mu_j / 2)),   alarm when max_j g_j > h.

The isolated rotor is ``argmax_j g_j`` at the alarm. Because the filtered residual is
coloured and the model is not exact (parameter spread), the nominal average-run-length
formulas do not apply; ``sigma`` and ``h`` are calibrated on fault-free runs with seeds
disjoint from all evaluation seeds (see ``rotorfault.experiments.calibration``).
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
    # per-row residual noise (N, N m, N m, N m); calibrated, see calibration.py
    sigma: np.ndarray = field(default_factory=lambda: np.array([0.08, 2.0e-3, 2.0e-3, 1.0e-3]))
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
    # bank of CUSUM tests, one per design LOE magnitude (approximates the GLR test
    # for an unknown fault size); each has its own threshold (log-likelihood units)
    deltas: tuple = (0.15, 0.3, 0.6)
    thresholds: np.ndarray = field(default_factory=lambda: np.array([2000.0, 2000.0, 2000.0]))
    cusum_max: float = 1e5


@dataclass
class FDIOutput:
    eta_hat: np.ndarray
    alarm: bool
    isolated: int  # -1 if no alarm yet
    alarm_time: float  # nan if no alarm yet
    cusum: np.ndarray
    residual: np.ndarray
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
        self._I4 = np.eye(4)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        P = self.P
        self._x1 = np.zeros(self.n_ch)
        self._x2 = np.zeros(self.n_ch)
        self._primed = False
        w_h = math.sqrt(P.hover_thrust / P.k_f)
        self.omega_hat = np.full(4, w_h)  # predicted rotor speeds
        self._cmd_fifo = deque(maxlen=self.cfg.actuator_delay_steps + 1)
        self.eta_hat = np.ones(4)
        self.Pk = np.eye(4) * 1e-3
        self.g = np.zeros((len(self.cfg.deltas), 4))
        self._deltas = np.asarray(self.cfg.deltas, dtype=float)[:, None]
        self.alarm = False
        self.isolated = -1
        self.alarm_time = float("nan")
        self.alarm_bank = -1
        self.t = 0.0
        self.last_residual = np.zeros(4)
        self.last_sigma = self.cfg.sigma.copy()
        self.eta_ref = np.ones(4)
        self._ref_sum = np.zeros(4)
        self._ref_n = 0
        self._ref_done = False

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
        sent (the known transport delay is applied internally); ``v_body`` is the estimated velocity
        in body axes (for the nominal drag term in the collective row).
        """
        P = self.P
        c = self.cfg
        self.t = t
        self._cmd_fifo.append(np.asarray(f_cmd_applied, dtype=float))
        f_pred, dW = self._motor_model(self._cmd_fifo[0])
        w = gyro
        J = self.J
        h_z = -P.rotor_inertia * float(self.s @ self.omega_hat)
        Hw = J * w + np.array([0.0, 0.0, h_z])
        nonlin = cross3(w, Hw) + P.rot_damping * w
        nonlin[2] -= P.rotor_inertia * float(self.s @ dW)
        sp = math.sqrt(float(v_body @ v_body))
        coll = P.mass * accel[2] + P.drag_lin[2] * v_body[2] + P.drag_quad * sp * v_body[2]

        u = np.concatenate([w, nonlin, f_pred, [coll]])
        if not self._primed:
            # initialise the filter at steady state for this input
            self._x1 = u.copy()
            self._x2 = np.zeros_like(u)
            self._primed = True
        Ad, Bd = self._Ad, self._Bd
        x1 = Ad[0, 0] * self._x1 + Ad[0, 1] * self._x2 + Bd[0] * u
        x2 = Ad[1, 0] * self._x1 + Ad[1, 1] * self._x2 + Bd[1] * u
        self._x1, self._x2 = x1, x2

        dw_f = x2[0:3]
        nonlin_f = x1[3:6]
        f_f = x1[6:10]
        coll_f = x1[10]
        y = np.empty(4)
        y[0] = coll_f
        y[1:4] = J * dw_f + nonlin_f
        Phi = self.B * f_f[None, :]

        # Adaptive residual scale: calibrated floor plus the residual that the assumed
        # parametric uncertainty can produce *at this instant* (inertia error times
        # angular acceleration, motor-lag error times thrust rate).
        fdot = np.abs(x2[6:10])
        unc = np.empty(4)
        lag = c.unc_tau * P.tau_motor
        unc[0] = lag * fdot.sum()
        unc[1:4] = c.unc_inertia * np.abs(J * dw_f) + lag * (self._absB @ fdot)
        sig = np.sqrt(c.sigma**2 + unc**2)
        self.last_sigma = sig

        # ---------------- Kalman filter on eta ----------------
        # A rotor commanded (near) idle carries no information about its own
        # effectiveness: its estimate and covariance are frozen (no random walk,
        # no update) instead of being allowed to drift.
        excited = f_f > c.excitation_min
        self.Pk = self.Pk + np.diag(np.where(excited, c.kf_q**2 * self.dt, 0.0))
        S = Phi @ self.Pk @ Phi.T + np.diag((sig * c.kf_r_inflation) ** 2)
        K = np.linalg.solve(S, Phi @ self.Pk).T
        K[~excited, :] = 0.0
        innov = y - Phi @ self.eta_hat
        self.eta_hat = np_clip(self.eta_hat + K @ innov, c.eta_min, c.eta_max)
        self.Pk = (self._I4 - K @ Phi) @ self.Pk
        self.Pk = 0.5 * (self.Pk + self.Pk.T)

        # ---------------- healthy baseline ----------------
        if t < c.detect_after:
            if c.use_baseline and t >= c.baseline_start:
                self._ref_sum += self.eta_hat
                self._ref_n += 1
        elif not self._ref_done:
            if self._ref_n > 0:
                self.eta_ref = self._ref_sum / self._ref_n
            self._ref_done = True

        # ---------------- CUSUM detection / isolation ----------------
        Phi_ref = Phi * self.eta_ref[None, :]
        r = (y - Phi_ref.sum(axis=1)) / sig
        self.last_residual = r
        D = -Phi_ref / sig[:, None]  # columns = fault signatures
        dn = np.sqrt((D * D).sum(axis=0)) + 1e-12
        tj = (D.T @ r) / dn
        mu = self._deltas * dn[None, :]
        if self._ref_done:
            self.g = np_clip(self.g + mu * (tj[None, :] - 0.5 * mu), 0.0, c.cusum_max)
        h = np.asarray(c.thresholds, dtype=float)
        ratio = self.g / h[:, None]  # normalised decision statistics, alarm at 1
        if not self.alarm and ratio.max() > 1.0:
            k = int(np.argmax(ratio.max(axis=1)))
            self.alarm = True
            self.isolated = int(np.argmax(self.g[k]))
            self.alarm_time = t
            self.alarm_bank = k
        return FDIOutput(
            eta_hat=self.eta_hat.copy(),
            alarm=self.alarm,
            isolated=self.isolated,
            alarm_time=self.alarm_time,
            cusum=ratio.max(axis=0),
            cusum_bank=self.g.max(axis=1),
            residual=r,
            eta_ref=self.eta_ref,
            sigma=sig,
        )
