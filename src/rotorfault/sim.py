"""Closed-loop simulation: true vehicle + sensors + flight stack."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .disturbance import Disturbance, DisturbanceParams
from .dynamics import QuadDynamics, hover_state, thrust_to_speed
from .faults import FaultSchedule
from .fdi import FDIConfig
from .params import SensorParams, SimConfig, nominal_params, perturbed_params
from .rotations import norm3, tilt_angle
from .sensors import Sensors
from .stack import FlightStack, StackOptions
from .trajectory import Hover, Reference

# A run is declared lost (and stopped) when the vehicle hits the ground, flips past
# 90 deg of tilt (inverted), or strays more than 3 m from the
# reference. See metrics.py for the definition used in the tables.
LOST_POSITION_ERROR = 3.0


@dataclass
class Scenario:
    name: str
    duration: float
    reference: Callable[[float], Reference] = field(default_factory=Hover)
    faults: FaultSchedule = field(default_factory=FaultSchedule)
    sensors: SensorParams = field(default_factory=SensorParams)
    disturbance: DisturbanceParams = field(default_factory=DisturbanceParams)
    uncertainty: bool = True
    init_position: tuple = (0.0, 0.0, 2.0)
    # closed-loop settling before t = 0 so integrators have absorbed the parameter
    # mismatch when the experiment starts (identical for all baselines)
    settle_time: float = 3.0
    # optional multiplicative changes of *true* parameters (robustness studies), e.g.
    # {"tau_motor": 1.6}; the flight stack keeps using the nominal model
    true_scale: dict = field(default_factory=dict)


@dataclass
class SimResult:
    scenario: str
    baseline: str
    seed: int
    t: np.ndarray
    x: np.ndarray  # (N, 17) true state
    p_ref: np.ndarray
    f_cmd: np.ndarray
    thrust: np.ndarray  # actual rotor thrusts (N)
    eta_true: np.ndarray
    eta_hat: np.ndarray
    eta_used: np.ndarray
    mode: np.ndarray
    cusum: np.ndarray  # (N, 4) normalised decision statistic per rotor (alarm at 1)
    residual: np.ndarray  # raw FDI residual y - Phi diag(eta_ref) 1 (N, N m, N m, N m)
    p_est: np.ndarray
    cusum_bank: np.ndarray  # (N, n_bank) raw CUSUM maxima per design magnitude
    alarm_time: float
    isolated: int
    eta_ref: np.ndarray
    switch_partial: float
    switch_relaxed: float
    crashed: bool
    crash_time: float
    crash_reason: str
    fault_onset: float
    fault_rotor: int
    fault_magnitude: float
    true_params: dict = field(default_factory=dict)

    @property
    def position(self):
        return self.x[:, 0:3]

    @property
    def quat(self):
        return self.x[:, 6:10]

    @property
    def omega(self):
        return self.x[:, 10:13]


def simulate(
    scenario: Scenario,
    baseline: str,
    seed: int,
    fdi_cfg: FDIConfig | None = None,
    options: StackOptions | None = None,
    cfg: SimConfig | None = None,
) -> SimResult:
    cfg = cfg or SimConfig()
    ss = np.random.SeedSequence(seed)
    rng_par, rng_sens, rng_dist = (np.random.default_rng(s) for s in ss.spawn(3))
    nominal = nominal_params()
    true_p = perturbed_params(rng_par, nominal) if scenario.uncertainty else nominal
    if scenario.true_scale:
        true_p = true_p.copy()
        for name, factor in scenario.true_scale.items():
            setattr(true_p, name, getattr(true_p, name) * factor)
    dyn = QuadDynamics(true_p)
    sens = Sensors(scenario.sensors, rng_sens, cfg.dt_ctrl)
    dist_proc = Disturbance(scenario.disturbance, rng_dist, cfg.dt_ctrl)
    lat = int(round(scenario.sensors.mocap_latency / cfg.dt_ctrl))
    stack = FlightStack(baseline, cfg.dt_ctrl, nominal, fdi_cfg, options, mocap_latency_steps=lat)
    pose_hist = deque(maxlen=lat + 1)
    cmd_fifo = deque(maxlen=cfg.actuator_delay_steps + 1)

    x = hover_state(true_p, scenario.init_position)
    # rotor speeds that hover the *true* vehicle with the nominal command map
    x[13:17] = thrust_to_speed(np.full(4, nominal.hover_thrust), nominal)
    stack.initialise(x[0:3], x[6:10])

    n = int(round(scenario.duration / cfg.dt_ctrl))
    n_settle = int(round(scenario.settle_time / cfg.dt_ctrl))
    log_t = np.zeros(n)
    log_x = np.zeros((n, 17))
    log_ref = np.zeros((n, 3))
    log_f = np.zeros((n, 4))
    log_thr = np.zeros((n, 4))
    log_eta = np.zeros((n, 4))
    log_eta_hat = np.zeros((n, 4))
    log_eta_used = np.zeros((n, 4))
    log_mode = np.zeros(n, dtype=np.int8)
    log_cusum = np.zeros((n, 4))
    log_resid = np.zeros((n, 4))
    log_bank = np.zeros((n, len(stack.fdi.cfg.deltas)))
    log_est = np.zeros((n, 3))

    faults = scenario.faults
    onset = faults.first_onset
    w_cmd = x[13:17].copy()
    crashed, crash_time, reason = False, math.nan, ""
    k_end = n
    for k in range(-n_settle, n):
        t = k * cfg.dt_ctrl
        eta = faults.eta(t)
        dist = dist_proc.step()
        sf = dyn.specific_force(x, w_cmd, eta, dist)
        imu = sens.imu(x[10:13], sf)
        pose_hist.append((x[0:3].copy(), x[6:10].copy()))
        mocap = sens.mocap(t, *pose_hist[0])
        ref = scenario.reference(t)
        onset_known = onset is not None and t >= onset
        f = stack.update(t, imu, mocap, ref, eta, onset_known)
        cmd_fifo.append(thrust_to_speed(f, nominal))
        w_cmd = cmd_fifo[0]
        if k < 0:  # settling phase before t = 0 (not logged, never faulted)
            for _ in range(cfg.physics_substeps):
                x = dyn.step(x, w_cmd, eta, cfg.dt_phys, dist)
            continue

        log_t[k] = t
        log_x[k] = x
        log_ref[k] = ref.p
        log_f[k] = f
        log_thr[k] = dyn.thrusts(x[13:17], eta)
        log_eta[k] = eta
        fd = stack.last_fdi
        log_eta_hat[k] = fd.eta_hat
        log_eta_used[k] = stack.eta_used if stack.mode != 2 else 0.0
        log_mode[k] = stack.mode
        log_cusum[k] = fd.cusum
        log_resid[k] = fd.residual * fd.sigma
        log_bank[k] = fd.cusum_bank
        log_est[k] = stack.est.p

        for _ in range(cfg.physics_substeps):
            x = dyn.step(x, w_cmd, eta, cfg.dt_phys, dist)

        err = norm3(x[0:3] - ref.p)
        if x[2] <= 0.0:
            crashed, reason = True, "ground"
        elif err > LOST_POSITION_ERROR:
            crashed, reason = True, "position"
        elif tilt_angle(x[6:10]) > math.pi / 2:
            crashed, reason = True, "flip"
        elif not np.all(np.isfinite(x)):
            crashed, reason = True, "diverged"
        if crashed:
            crash_time = t + cfg.dt_ctrl
            k_end = k + 1
            break

    sl = slice(0, k_end)
    first = faults.faults[0] if faults.faults else None
    fd = stack.last_fdi
    return SimResult(
        scenario=scenario.name,
        baseline=baseline,
        seed=seed,
        t=log_t[sl],
        x=log_x[sl],
        p_ref=log_ref[sl],
        f_cmd=log_f[sl],
        thrust=log_thr[sl],
        eta_true=log_eta[sl],
        eta_hat=log_eta_hat[sl],
        eta_used=log_eta_used[sl],
        mode=log_mode[sl],
        cusum=log_cusum[sl],
        residual=log_resid[sl],
        cusum_bank=log_bank[sl],
        p_est=log_est[sl],
        alarm_time=fd.alarm_time,
        isolated=fd.isolated,
        eta_ref=fd.eta_ref.copy(),
        switch_partial=stack.switch_time[1],
        switch_relaxed=stack.switch_time[2],
        crashed=crashed,
        crash_time=crash_time,
        crash_reason=reason,
        fault_onset=math.nan if first is None else first.onset,
        fault_rotor=-1 if first is None else first.rotor,
        fault_magnitude=0.0 if first is None else first.magnitude,
        true_params={
            "mass": true_p.mass,
            "inertia": np.diag(true_p.inertia).tolist(),
            "tau_motor": true_p.tau_motor,
            "rot_damping": true_p.rot_damping.tolist(),
            "k_f_scale": true_p.k_f_scale.tolist(),
        },
    )
