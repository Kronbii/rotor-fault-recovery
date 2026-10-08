"""The simulated on-board software ("flight stack"): state estimation, FDI, supervisor,
nominal / reconfigured / yaw-relaxed control and allocation.

Three baselines share everything except how fault information reaches the controller:

* ``passive``   -- nominal controller and allocator, faults are never acted upon
                   (the FDI still runs, for logging only).
* ``oracle``    -- the true effectiveness vector and onset time are handed to the
                   supervisor (upper bound on what the active scheme can achieve).
* ``estimated`` -- the proposed pipeline: CUSUM alarm + isolation, then reconfiguration
                   with the Kalman-filter estimate of ``eta``.

Supervisor (modes): ``NOMINAL`` -> ``PARTIAL`` (allocation with ``B diag(eta)``) on an
alarm, -> ``RELAXED`` (yaw-relaxed controller, failed rotor commanded to idle) when the
effectiveness of the isolated rotor is below ``eta_relax`` for ``relax_persist`` s.
``eta_relax`` defaults to 0.35: below ~0.29 the rotor cannot produce its quarter of the
hover thrust, so yaw-holding hover becomes infeasible; the margin leaves room for
attitude control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .allocation import PriorityAllocator
from .control import SE3Controller, SE3Gains
from .estimation import StateEstimator
from .fdi import FDIConfig, RotorFDI
from .ftc import RelaxedDesign, RelaxedHoverController, design_relaxed
from .params import QuadParams, nominal_params
from .sensors import ImuSample, MocapSample
from .trajectory import Reference

NOMINAL, PARTIAL, RELAXED = 0, 1, 2
MODE_NAMES = {NOMINAL: "nominal", PARTIAL: "partial", RELAXED: "relaxed"}
BASELINES = ("passive", "oracle", "estimated")


@dataclass
class StackOptions:
    eta_relax: float = 0.35
    relax_persist: float = 0.03
    eta_floor_partial: float = 0.25
    relaxed_tilt_deg: float = 15.0
    enable_relaxed: bool = True


@lru_cache(maxsize=16)
def _cached_design(failed: int, dt: float, tilt: float) -> RelaxedDesign:
    return design_relaxed(nominal_params(), failed, dt, tilt)


class FlightStack:
    def __init__(
        self,
        baseline: str,
        dt: float,
        params: QuadParams | None = None,
        fdi_cfg: FDIConfig | None = None,
        options: StackOptions | None = None,
        gains: SE3Gains | None = None,
        mocap_latency_steps: int = 0,
    ):
        if baseline not in BASELINES:
            raise ValueError(f"unknown baseline {baseline!r}")
        self.baseline = baseline
        self.P = params or nominal_params()
        self.dt = dt
        self.opt = options or StackOptions()
        self.est = StateEstimator(dt, latency_steps=mocap_latency_steps)
        self.ctrl = SE3Controller(self.P, gains, dt)
        self.alloc = PriorityAllocator(self.P.mixer(), self.P.f_min, self.P.f_max)
        self.fdi = RotorFDI(self.P, dt, fdi_cfg)
        self.mode = NOMINAL
        self.rotor = -1
        self.eta_used = np.ones(4)
        self.relaxed: RelaxedHoverController | None = None
        self.f_cmd = np.full(4, self.P.hover_thrust)
        self._below_since = None
        self.switch_time = {PARTIAL: math.nan, RELAXED: math.nan}
        self.last_fdi = None
        self.axis_error = math.nan

    def initialise(self, p, q):
        self.est.reset(p, q)

    # ------------------------------------------------------------------
    def _enter_relaxed(self, t, rotor):
        self.mode = RELAXED
        self.rotor = rotor
        self.switch_time[RELAXED] = t
        design = _cached_design(rotor, self.dt, self.opt.relaxed_tilt_deg)
        self.relaxed = RelaxedHoverController(self.P, rotor, self.dt, design=design)
        # hand over the position integrator so the altitude bias is not re-learned
        self.relaxed.int_p = self.ctrl.int_p.copy()

    def _supervise(self, t, fdi, eta_true, onset_known):
        opt = self.opt
        if self.baseline == "passive" or self.mode == RELAXED:
            return
        if self.baseline == "oracle":
            if onset_known and np.min(eta_true) < 1.0 - 1e-9:
                j = int(np.argmin(eta_true))
                if eta_true[j] < opt.eta_relax and opt.enable_relaxed:
                    self._enter_relaxed(t, j)
                else:
                    if self.mode == NOMINAL:
                        self.switch_time[PARTIAL] = t
                    self.mode = PARTIAL
                    self.rotor = j
                    self.eta_used = np.maximum(eta_true, opt.eta_floor_partial)
            return
        # estimated
        if self.mode == NOMINAL and fdi.alarm:
            self.mode = PARTIAL
            self.rotor = fdi.isolated
            self.switch_time[PARTIAL] = t
        if self.mode == PARTIAL:
            j = self.rotor
            e = float(fdi.eta_hat[j])
            self.eta_used = np.ones(4)
            self.eta_used[j] = min(max(e, opt.eta_floor_partial), 1.0)
            if opt.enable_relaxed and e < opt.eta_relax:
                if self._below_since is None:
                    self._below_since = t
                if t - self._below_since >= opt.relax_persist - 1e-9:
                    self._enter_relaxed(t, j)
            else:
                self._below_since = None

    def update(
        self,
        t: float,
        imu: ImuSample,
        mocap: MocapSample | None,
        ref: Reference,
        eta_true: np.ndarray | None = None,
        onset_known: bool = False,
    ) -> np.ndarray:
        est = self.est
        est.predict(imu.gyro, imu.accel)
        if mocap is not None:
            est.correct(mocap.position, mocap.quat)
        R = est.R
        omega = imu.gyro - est.bias
        v_body = R.T @ est.v
        fdi = self.fdi.update(t, omega, imu.accel, self.f_cmd, v_body)
        self.last_fdi = fdi
        self._supervise(t, fdi, eta_true, onset_known)

        if self.mode == RELAXED:
            f_hat = self.P.k_f * self.fdi.omega_hat**2
            out = self.relaxed(est.p, est.v, R, omega, ref, f_hat)
            f = out.f
            self.axis_error = out.axis_error
        else:
            c = self.ctrl(est.p, est.v, R, omega, ref)
            eta = self.eta_used if self.mode == PARTIAL else None
            f = self.alloc.allocate(c.thrust, c.torque, eta).f
        self.f_cmd = f
        return f
