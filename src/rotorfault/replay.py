"""Offline replay of the diagnoser on a recorded flight log.

The estimator, residual scale and CUSUM bank are exactly those of :mod:`rotorfault.fdi`
(the class below only replaces the motor model): rotor thrusts are predicted from the
logged motor commands through the airframe's command-to-speed map, an asymmetric
first-order lag and the transport delay -- or, with ``measured_speed=True``, taken from
RPM telemetry. Detection is enabled at ``t_detect`` (log time); the healthy baseline is
learned over the ``baseline_s`` seconds before it, which must be fault-free flight.

:func:`calibrate_on_logs` is the calibration of ``experiments/calibration.py`` applied
to logs: residual floor = RMS of the fault-free residual (roll and pitch pooled),
threshold of each CUSUM = ``SAFETY`` x the largest statistic seen on fault-free logs,
floored at the evidence a design-size fault gives in 20 ms.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from .airframe import Airframe
from .experiments.calibration import SAFETY
from .fdi import FDIConfig, RotorFDI, hover_signature_norm
from .logs.flightlog import FlightLog

FLOOR_EVIDENCE_S = 0.02  # threshold floor: evidence of a design-size fault in 20 ms


class ReplayFDI(RotorFDI):
    """:class:`RotorFDI` with the airframe's motor map instead of the simulator's."""

    def __init__(
        self, airframe: Airframe, dt: float, cfg: FDIConfig, measured_speed=False, eta_ref=None
    ):
        self.airframe = airframe
        self.measured = measured_speed
        self.fixed_ref = None if eta_ref is None else np.asarray(eta_ref, dtype=float)
        m = airframe.motor
        self._a_up = math.exp(-dt / m.tau_up) if m.tau_up > 0 else 0.0
        self._a_dn = math.exp(-dt / m.tau_down) if m.tau_down > 0 else 0.0
        super().__init__(airframe.params, dt, cfg)

    def reset(self):
        super().reset()
        if getattr(self, "fixed_ref", None) is not None:
            # baseline from earlier fault-free flights: detection from the first sample
            # (a fault present from take-off, e.g. a weakened propeller, is then visible)
            self.eta_ref = self.fixed_ref.copy()
            self._ref_done = True

    def _motor_model(self, cmd: np.ndarray):
        m = self.airframe.motor
        if self.measured:
            w_new = np.nan_to_num(np.asarray(cmd, dtype=float))
        else:
            w_cmd = m.speed(cmd)
            a = np.where(w_cmd > self.omega_hat, self._a_up, self._a_dn)
            w_new = w_cmd + (self.omega_hat - w_cmd) * a
        dw = (w_new - self.omega_hat) / self.dt
        self.omega_hat = w_new
        return m.k_f * w_new**2, dw


@dataclass
class ReplayResult:
    t: np.ndarray
    eta_hat: np.ndarray
    cusum: np.ndarray  # (N, 4) normalised statistic per rotor (alarm at 1)
    cusum_bank: np.ndarray  # (N, n_bank) raw maxima per design magnitude
    residual: np.ndarray  # (N, 4) raw residual y - Phi eta_ref
    sigma: np.ndarray  # (N, 4) residual scale used (floor + model-uncertainty term)
    alarm_time: float  # log time, NaN if none
    isolated: int  # 0-based rotor (FC motor - 1), -1 if none
    eta_ref: np.ndarray
    t_detect: float
    cfg: FDIConfig

    @property
    def alarm(self) -> bool:
        return math.isfinite(self.alarm_time)

    def loe_hat(self, t: float) -> np.ndarray:
        k = min(int(np.searchsorted(self.t, t)), len(self.t) - 1)
        return 1.0 - self.eta_hat[k] / self.eta_ref


def replay_config(
    cfg: FDIConfig | None, dt: float, airframe: Airframe, baseline_s: float, delay_steps=None
) -> FDIConfig:
    cfg = cfg or FDIConfig()
    if delay_steps is None:
        delay_steps = int(round(airframe.motor.delay_s / dt))
    return dataclasses.replace(
        cfg, baseline_start=-baseline_s, detect_after=0.0, actuator_delay_steps=int(delay_steps)
    )


def replay(
    log: FlightLog,
    airframe: Airframe,
    cfg: FDIConfig | None = None,
    t_detect: float | None = None,
    baseline_s: float = 2.5,
    warmup_s: float = 1.0,
    t_end: float | None = None,
    measured_speed: bool = False,
    delay_steps: int | None = None,
    eta_ref: np.ndarray | None = None,
) -> ReplayResult:
    """Run the diagnoser over ``log`` from ``t_detect - baseline_s - warmup_s`` to ``t_end``.

    ``eta_ref``: use this healthy baseline (e.g. ``eta_ref_mean`` of a calibration)
    instead of learning it at the start of the log."""
    dt = log.dt
    if t_detect is None:
        t_detect = float(log.t[0]) + warmup_s + baseline_s
    c = replay_config(cfg, dt, airframe, baseline_s, 0 if measured_speed else delay_steps)
    fdi = ReplayFDI(airframe, dt, c, measured_speed, eta_ref)
    k0 = int(np.searchsorted(log.t, t_detect - baseline_s - warmup_s))
    k1 = len(log.t) if t_end is None else int(np.searchsorted(log.t, t_end))
    if measured_speed and log.rotor_speed is None:
        raise ValueError("log has no rotor-speed telemetry")
    cmd = log.rotor_speed if measured_speed else log.motor
    v = log.v_body if log.v_body is not None else np.zeros((len(log.t), 3))
    n = max(0, k1 - k0)
    nb = len(c.deltas)
    eta = np.zeros((n, 4))
    cus = np.zeros((n, 4))
    bank = np.zeros((n, nb))
    res = np.zeros((n, 4))
    sig = np.zeros((n, 4))
    for i, k in enumerate(range(k0, k1)):
        o = fdi.update(float(log.t[k] - t_detect), log.gyro[k], log.accel[k], cmd[k], v[k])
        eta[i] = o.eta_hat
        cus[i] = o.cusum
        bank[i] = o.cusum_bank
        res[i] = o.residual * o.sigma
        sig[i] = o.sigma
    at = fdi.alarm_time + t_detect if math.isfinite(fdi.alarm_time) else math.nan
    return ReplayResult(
        t=log.t[k0:k1],
        eta_hat=eta,
        cusum=cus,
        cusum_bank=bank,
        residual=res,
        sigma=sig,
        alarm_time=at,
        isolated=fdi.isolated,
        eta_ref=fdi.eta_ref.copy(),
        t_detect=t_detect,
        cfg=c,
    )


@dataclass
class LogSegment:
    """A fault-free stretch of a log used for calibration."""

    log: FlightLog
    t_detect: float
    t_end: float | None = None


def calibrate_on_logs(
    segments: list[LogSegment],
    airframe: Airframe,
    base: FDIConfig | None = None,
    baseline_s: float = 2.5,
    safety: float = SAFETY,
    measured_speed: bool = False,
    delay_steps: int | None = None,
    iterations: int = 2,
    mapper=map,
) -> tuple[FDIConfig, dict]:
    """Residual floor and CUSUM thresholds from fault-free log segments.

    ``mapper(fn, tasks)`` runs the replays (e.g. a process-pool map)."""
    # no clipping of the statistic while it is being measured
    base = dataclasses.replace(base or FDIConfig(), cusum_max=1e15)
    sigma = np.asarray(base.sigma, dtype=float)
    kw = dict(baseline_s=baseline_s, measured_speed=measured_speed, delay_steps=delay_steps)

    def run(cfg):
        return list(mapper(_replay_job, [(s, airframe, cfg, kw) for s in segments]))

    rms = None
    for _ in range(iterations):  # the KF gain depends on sigma: iterate the floor
        cfg = dataclasses.replace(base, sigma=sigma, thresholds=np.full(len(base.deltas), np.inf))
        outs = run(cfg)
        ms = np.mean([np.mean(o.residual[o.t >= o.t_detect] ** 2, axis=0) for o in outs], axis=0)
        rms = np.sqrt(ms)
        rp = math.sqrt(0.5 * (ms[1] + ms[2]))
        sigma = np.array([rms[0], rp, rp, rms[3]])
    cfg = dataclasses.replace(base, sigma=sigma, thresholds=np.full(len(base.deltas), np.inf))
    outs = run(cfg)
    gmax = np.array([o.cusum_bank[o.t >= o.t_detect].max(axis=0) for o in outs])
    dt = segments[0].log.dt
    n_min = max(1, int(round(FLOOR_EVIDENCE_S / dt)))
    mu = np.asarray(base.deltas) * hover_signature_norm(airframe.params, sigma)
    floor = n_min * mu**2 / 2
    thresholds = np.maximum(floor, safety * gmax.max(axis=0))
    # the statistic is clipped well above the largest threshold (FDIConfig's default
    # clip is sized for the simulator's thresholds)
    cfg = dataclasses.replace(
        base, sigma=sigma, thresholds=thresholds, cusum_max=100.0 * float(thresholds.max())
    )
    info = {
        "n_segments": len(segments),
        "eta_ref_mean": np.mean([o.eta_ref for o in outs], axis=0).tolist(),
        "eta_ref_spread": np.std([o.eta_ref for o in outs], axis=0).tolist(),
        "fault_free_s": float(sum(len(o.t[o.t >= o.t_detect]) * dt for o in outs)),
        "sigma": sigma.tolist(),
        "residual_rms_per_row": rms.tolist(),
        "h0_cusum_max_per_segment": gmax.tolist(),
        "threshold_floor": floor.tolist(),
        "thresholds": thresholds.tolist(),
        "safety_factor": safety,
        "deltas": list(base.deltas),
        "dt": dt,
        "measured_speed": measured_speed,
    }
    return cfg, info


def _replay_job(job):
    seg, airframe, cfg, kw = job
    return replay(seg.log, airframe, cfg, seg.t_detect, t_end=seg.t_end, **kw)


def best_delay(
    segments: list[LogSegment],
    airframe: Airframe,
    max_steps: int = 4,
    baseline_s: float = 2.5,
    mapper=map,
) -> tuple[int, list[float]]:
    """Transport delay (in samples) that minimises the fault-free residual (identification
    on calibration logs only)."""
    ms = []
    cfg = FDIConfig(thresholds=np.full(3, np.inf))
    for d in range(max_steps + 1):
        kw = dict(baseline_s=baseline_s, delay_steps=d)
        outs = list(mapper(_replay_job, [(s, airframe, cfg, kw) for s in segments]))
        r = np.concatenate([o.residual[o.t >= o.t_detect] for o in outs])
        ms.append(np.mean(r**2, axis=0))
    ms = np.array(ms)
    scores = (ms / ms[0]).sum(axis=1).tolist()  # each row relative to zero delay
    return int(np.argmin(scores)), scores


def auto_detect_start(
    log: FlightLog, airframe: Airframe, settle_s: float = 1.0, baseline_s: float = 2.5
) -> float:
    """Detection start for a log without event markers: take-off (mean command above 70 %
    of the airframe's hover command) + ``settle_s`` + the baseline window. The baseline
    must be fault-free flight; give ``t_detect`` explicitly when that is not the case."""
    u = np.nan_to_num(log.motor).mean(axis=1)
    k = np.flatnonzero(u > 0.7 * airframe.hover_command)
    if len(k) == 0:
        raise ValueError("no take-off found in the log (motor commands never reach hover)")
    return float(log.t[k[0]]) + settle_s + baseline_s


def save_calibration(path, cfg: FDIConfig, info: dict | None = None, delay_steps=None) -> None:
    from .experiments.common import save_json

    d = {
        "sigma": np.asarray(cfg.sigma).tolist(),
        "thresholds": np.asarray(cfg.thresholds).tolist(),
        "cusum_max": float(cfg.cusum_max),
        "unc_inertia": cfg.unc_inertia,
        "unc_tau": cfg.unc_tau,
        "delay_steps": delay_steps,
        "eta_ref_mean": (info or {}).get("eta_ref_mean"),
    }
    if info:
        d["info"] = info
    save_json(path, d)


def load_calibration(path) -> tuple[FDIConfig, int | None]:
    """(config, transport delay in samples) of a calibration file; the stored baseline is
    available with :func:`load_baseline`."""
    import json

    with open(path) as f:
        d = json.load(f)
    cfg = FDIConfig(
        sigma=np.array(d["sigma"]),
        thresholds=np.array(d["thresholds"]),
        cusum_max=float(d.get("cusum_max", 1e5)),
        unc_inertia=float(d.get("unc_inertia", FDIConfig.unc_inertia)),
        unc_tau=float(d.get("unc_tau", FDIConfig.unc_tau)),
    )
    return cfg, d.get("delay_steps")


def load_baseline(path) -> np.ndarray | None:
    import json

    with open(path) as f:
        v = json.load(f).get("eta_ref_mean")
    return None if v is None else np.asarray(v, dtype=float)
