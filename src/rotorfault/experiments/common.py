"""Shared infrastructure for the experiments: paths, seeds, parallel execution, I/O."""

from __future__ import annotations

import csv
import json
import math
import multiprocessing as mp
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Disjoint seed ranges: no evaluation run ever reuses a calibration seed.
SEED_CALIBRATION = 900_000
SEED_E1 = 100_000
SEED_E2_FAULT = 200_000
SEED_E2_FREE = 210_000
SEED_E2_SWEEP = 220_000
SEED_E3 = 300_000
SEED_E4 = 400_000
SEED_E5 = 500_000


@dataclass
class Paths:
    root: Path

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def replays(self) -> Path:
        return self.results / "replays"

    @property
    def cache(self) -> Path:
        return self.results / "cache"

    @property
    def paper(self) -> Path:
        return self.root / "paper"

    def ensure(self) -> Paths:
        for p in (self.results, self.figures, self.replays, self.cache):
            p.mkdir(parents=True, exist_ok=True)
        return self


def default_paths() -> Paths:
    return Paths(Path(os.environ.get("ROTORFAULT_ROOT", os.getcwd()))).ensure()


def n_workers() -> int:
    env = os.environ.get("ROTORFAULT_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, min(os.cpu_count() or 1, 16))


def pmap(fn: Callable, tasks: Iterable, desc: str = "", workers: int | None = None) -> list:
    """Parallel map preserving order (fork start method; falls back to serial)."""
    tasks = list(tasks)
    workers = workers or n_workers()
    t0 = time.time()
    if workers <= 1 or len(tasks) <= 1:
        out = [fn(t) for t in tasks]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers) as pool:
            out = pool.map(fn, tasks, chunksize=1)
    if desc:
        print(f"  [{desc}] {len(tasks)} runs in {time.time() - t0:.1f} s ({workers} workers)")
    return out


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if not math.isfinite(f) else round(f, 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_clean(obj), f, indent=1)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: _clean(r.get(k)) for k in keys})


def nan_stats(x) -> dict:
    a = np.array([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if len(a) == 0:
        return {"n": 0, "mean": None, "median": None, "p25": None, "p75": None, "max": None}
    return {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "median": float(np.median(a)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "p95": float(np.percentile(a, 95)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for a binomial proportion."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def save_run_npz(path: Path, r) -> None:
    """Cache a representative run (float32, every sample)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        t=r.t.astype(np.float32),
        x=r.x.astype(np.float32),
        p_ref=r.p_ref.astype(np.float32),
        f_cmd=r.f_cmd.astype(np.float32),
        thrust=r.thrust.astype(np.float32),
        eta_true=r.eta_true.astype(np.float32),
        eta_hat=r.eta_hat.astype(np.float32),
        mode=r.mode,
        cusum=r.cusum.astype(np.float32),
        residual=r.residual.astype(np.float32),
        eta_ref=r.eta_ref,
        meta=json.dumps(
            _clean(
                {
                    "baseline": r.baseline,
                    "seed": r.seed,
                    "alarm_time": r.alarm_time,
                    "isolated": r.isolated,
                    "switch_partial": r.switch_partial,
                    "switch_relaxed": r.switch_relaxed,
                    "crashed": r.crashed,
                    "crash_time": r.crash_time,
                    "crash_reason": r.crash_reason,
                    "fault_onset": r.fault_onset,
                    "fault_rotor": r.fault_rotor,
                    "fault_magnitude": r.fault_magnitude,
                }
            )
        ),
    )


class CachedRun:
    """Light-weight stand-in for SimResult loaded from an npz cache."""

    def __init__(self, path: Path):
        d = np.load(path, allow_pickle=False)
        for k in d.files:
            if k != "meta":
                setattr(self, k, d[k].astype(float) if d[k].dtype.kind == "f" else d[k])
        meta = json.loads(str(d["meta"]))
        for k, v in meta.items():
            setattr(self, k, math.nan if v is None else v)

    @property
    def position(self):
        return self.x[:, 0:3]

    @property
    def quat(self):
        return self.x[:, 6:10]

    @property
    def omega(self):
        return self.x[:, 10:13]


def export_replay(path: Path, r, rate_hz: float = 50.0, extra: dict | None = None) -> int:
    """Down-sampled replay for a web viewer: time, position, quaternion, rotor thrusts,
    true / estimated effectiveness and supervisor mode. Returns the file size."""
    dt = float(r.t[1] - r.t[0]) if len(r.t) > 1 else 0.002
    step = max(1, int(round(1.0 / (rate_hz * dt))))
    idx = np.arange(0, len(r.t), step)

    def rnd(a, n):
        return np.round(np.asarray(a, dtype=float), n).tolist()

    data = {
        "format": "rotorfault-replay-v1",
        "frame": "world ENU-like (x fwd, y left, z up); quaternion (w, x, y, z) body->world",
        "rotor_layout": {
            "positions_body_m": [
                [0.0778, -0.0778, 0.0],
                [0.0778, 0.0778, 0.0],
                [-0.0778, 0.0778, 0.0],
                [-0.0778, -0.0778, 0.0],
            ],
            "names": ["front-right", "front-left", "rear-left", "rear-right"],
        },
        "baseline": r.baseline,
        "seed": int(r.seed),
        "rate_hz": 1.0 / (step * dt),
        "fault": {
            "rotor": int(r.fault_rotor),
            "magnitude": float(r.fault_magnitude),
            "onset_s": None if not np.isfinite(r.fault_onset) else float(r.fault_onset),
        },
        "events": {
            "alarm_s": None if not np.isfinite(r.alarm_time) else float(r.alarm_time),
            "isolated_rotor": int(r.isolated),
            "partial_ftc_s": None if not np.isfinite(r.switch_partial) else float(r.switch_partial),
            "relaxed_ftc_s": None if not np.isfinite(r.switch_relaxed) else float(r.switch_relaxed),
            "lost": bool(r.crashed),
            "lost_s": None if not np.isfinite(r.crash_time) else float(r.crash_time),
        },
        "t": rnd(r.t[idx], 3),
        "position": rnd(r.position[idx], 4),
        "reference": rnd(r.p_ref[idx], 4),
        "quaternion": rnd(r.quat[idx], 5),
        "rotor_thrust_N": rnd(r.thrust[idx], 3),
        "eta_true": rnd(r.eta_true[idx], 3),
        "eta_hat": rnd(r.eta_hat[idx], 3),
        "mode": [int(m) for m in r.mode[idx]],
        "mode_names": {"0": "nominal", "1": "partial-loss FTC", "2": "yaw-relaxed FTC"},
    }
    if extra:
        data.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    return path.stat().st_size
