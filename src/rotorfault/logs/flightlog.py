"""Common in-memory format of a flight log, whatever produced it.

Every reader (PX4 ULog, Betaflight blackbox CSV, the simulator) produces a
:class:`FlightLog` on the IMU time base, in the body FLU frame used everywhere in this
package. The diagnoser only ever sees ``gyro``, ``accel``, ``motor`` and, when present,
``v_body`` and ``rotor_speed``; everything else (ground truth, events) is kept for
evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class FlightLog:
    t: np.ndarray  # (N,) s, increasing
    gyro: np.ndarray  # (N, 3) rad/s, body FLU
    accel: np.ndarray  # (N, 3) m/s^2 specific force, body FLU (+g on z at rest)
    motor: np.ndarray  # (N, 4) normalised command in [0, 1] (NaN = motor stopped)
    v_body: np.ndarray | None = None  # (N, 3) m/s, velocity estimate in body FLU
    rotor_speed: np.ndarray | None = None  # (N, 4) rad/s, measured (RPM telemetry)
    extra: dict = field(default_factory=dict)  # name -> (N, ...) arrays on the same base
    events: dict = field(default_factory=dict)  # name -> time (s) or list of (t, text)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.t)
        for name in ("gyro", "accel", "motor"):
            a = getattr(self, name)
            if a.shape[0] != n:
                raise ValueError(f"{name} has {a.shape[0]} rows, expected {n}")

    @property
    def dt(self) -> float:
        return float(np.median(np.diff(self.t)))

    @property
    def rate_hz(self) -> float:
        return 1.0 / self.dt

    def window(self, t0: float, t1: float) -> FlightLog:
        m = (self.t >= t0) & (self.t < t1)
        return self._take(m)

    def _take(self, idx) -> FlightLog:
        def tk(a):
            return None if a is None else a[idx]

        return FlightLog(
            t=self.t[idx],
            gyro=self.gyro[idx],
            accel=self.accel[idx],
            motor=self.motor[idx],
            v_body=tk(self.v_body),
            rotor_speed=tk(self.rotor_speed),
            extra={k: v[idx] for k, v in self.extra.items()},
            events=dict(self.events),
            meta=dict(self.meta),
        )

    def resample(self, rate_hz: float) -> FlightLog:
        """Uniform time base at ``rate_hz``: block-averaged when down-sampling (simple
        anti-aliasing), linear interpolation otherwise. Motor commands are held."""
        dt = 1.0 / rate_hz
        tg = np.arange(self.t[0], self.t[-1], dt)
        ratio = max(1, int(round(dt / self.dt)))

        def smooth(a):
            if ratio <= 1:
                return a
            k = np.ones(ratio) / ratio
            return np.column_stack(
                [np.convolve(a[:, j], k, mode="same") for j in range(a.shape[1])]
            )

        def interp(a, hold=False):
            if a is None:
                return None
            a2 = a.reshape(len(self.t), -1).astype(float)
            if hold:
                idx = np.clip(np.searchsorted(self.t, tg, side="right") - 1, 0, len(self.t) - 1)
                out = a2[idx]
            else:
                a2 = smooth(a2)
                out = np.column_stack([np.interp(tg, self.t, a2[:, j]) for j in range(a2.shape[1])])
            return out.reshape((len(tg),) + a.shape[1:])

        return FlightLog(
            t=tg,
            gyro=interp(self.gyro),
            accel=interp(self.accel),
            motor=interp(self.motor, hold=True),
            v_body=interp(self.v_body),
            rotor_speed=interp(self.rotor_speed),
            extra={k: interp(v, hold=True) for k, v in self.extra.items()},
            events=dict(self.events),
            meta=dict(self.meta, resampled_hz=rate_hz),
        )

    # ------------------------------------------------------------------ storage
    def save(self, path: str | Path) -> None:
        arrays = {"t": self.t, "gyro": self.gyro, "accel": self.accel, "motor": self.motor}
        if self.v_body is not None:
            arrays["v_body"] = self.v_body
        if self.rotor_speed is not None:
            arrays["rotor_speed"] = self.rotor_speed
        for k, v in self.extra.items():
            arrays[f"extra__{k}"] = v
        np.savez_compressed(
            path,
            **{
                k: np.asarray(v, dtype=np.float32 if k != "t" else np.float64)
                for k, v in arrays.items()
            },
            meta=json.dumps({"events": self.events, "meta": self.meta}, default=_json_default),
        )

    @classmethod
    def load(cls, path: str | Path) -> FlightLog:
        d = np.load(path, allow_pickle=False)
        info = json.loads(str(d["meta"]))
        extra = {k[7:]: d[k].astype(float) for k in d.files if k.startswith("extra__")}

        def opt(k):
            return d[k].astype(float) if k in d.files else None

        return cls(
            t=d["t"].astype(float),
            gyro=d["gyro"].astype(float),
            accel=d["accel"].astype(float),
            motor=d["motor"].astype(float),
            v_body=opt("v_body"),
            rotor_speed=opt("rotor_speed"),
            extra=extra,
            events=info.get("events", {}),
            meta=info.get("meta", {}),
        )


def load_any(path: str | Path, airframe=None) -> FlightLog:
    """Read a ``.npz`` (this format), ``.ulg`` (PX4) or ``.csv`` (Betaflight) log."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".npz":
        return FlightLog.load(path)
    if suf == ".ulg":
        from .ulog import read_ulog

        return read_ulog(path)
    if suf == ".csv":
        from .betaflight import read_blackbox_csv

        return read_blackbox_csv(path, airframe)
    raise ValueError(f"unknown log format: {path.name}")


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
