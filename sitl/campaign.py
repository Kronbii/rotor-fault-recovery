"""Run the PX4 SITL campaign of ``sitl/config/campaign.yaml``.

    .venv/bin/python sitl/campaign.py                 # everything (resumes finished runs)
    .venv/bin/python sitl/campaign.py --only partial  # one group
    .venv/bin/python sitl/campaign.py --list          # print the run list

ULogs and per-run manifests go to ``build/sitl/logs`` (not in git: about 5-10 MB per
run). Each finished run is converted right away to a compact ``.npz`` flight log in
``build/sitl/npz``, which is what ``sitl/analyze.py`` reads. Needs PX4-Autopilot built
for SITL (``PX4_DIR``, default ``~/repos/PX4-Autopilot``) and Gazebo Harmonic.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from harness import RunSpec, fly  # noqa: E402

LOGS = HERE.parent / "build" / "sitl" / "logs"
NPZ = HERE.parent / "build" / "sitl" / "npz"


def run_list(cfg: dict) -> list[RunSpec]:
    runs = []
    for gname, g in cfg["groups"].items():
        conds = g.get("conditions", list(cfg["conditions"]))
        for ci, cname in enumerate(conds):
            c = cfg["conditions"][cname]
            for i in range(g["per_condition"]):
                seed = g["seed0"] + 100 * ci + i
                rng = np.random.default_rng(seed)
                spec = RunSpec(
                    name=f"{gname}-{cname}-{i:02d}",
                    seed=seed,
                    traj=c["traj"],
                    world=c["world"],
                    fault=g["fault"],
                    altitude=cfg["altitude_m"],
                    orbit_radius=cfg["orbit"]["radius_m"],
                    orbit_speed=cfg["orbit"]["speed_m_s"],
                    speed_factor=cfg["speed_factor"],
                    extra_params=dict(g.get("extra_params", {})),
                )
                if g["fault"] == "none":
                    spec.pre_s, spec.post_s = float(g["pre_s"]), 0.0
                else:
                    loe = g["loe"]
                    spec.loe = float(rng.uniform(*loe)) if isinstance(loe, list) else float(loe)
                    spec.loe = round(spec.loe, 3)
                    # complete losses visit the four motors in turn, partial ones are random
                    spec.motor = (i % 4) + 1 if spec.loe >= 1.0 else int(rng.integers(1, 5))
                    spec.pre_s = round(float(rng.uniform(*g["onset_s"])), 2)
                    spec.post_s = float(g["post_s"])
                runs.append(spec)
    return runs


def convert(name: str) -> None:
    from rotorfault.logs.ulog import read_ulog

    log = read_ulog(LOGS / f"{name}.ulg")
    log.meta["manifest"] = json.loads((LOGS / f"{name}.json").read_text())
    NPZ.mkdir(parents=True, exist_ok=True)
    log.save(NPZ / f"{name}.npz")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(HERE / "config" / "campaign.yaml"))
    ap.add_argument("--only", help="comma-separated groups")
    ap.add_argument("--workers", type=int, default=2, help="parallel SITL instances")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--redo-failed", action="store_true")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text())
    runs = run_list(cfg)
    if a.only:
        keep = set(a.only.split(","))
        runs = [r for r in runs if r.name.split("-", 1)[0] in keep]
    if a.list:
        for r in runs:
            print(r.name, r.seed, r.traj, r.world, r.fault, r.motor, r.loe, r.pre_s, r.post_s)
        print(len(runs), "runs")
        return

    def todo(r):
        m = LOGS / f"{r.name}.json"
        if not m.exists():
            return True
        ok = json.loads(m.read_text()).get("ok")
        return (not ok) and a.redo_failed

    pending = [r for r in runs if todo(r)]
    print(f"{len(pending)} of {len(runs)} runs to do, {a.workers} workers", flush=True)
    t0 = time.time()
    slots = list(range(a.workers))

    def work(job):
        idx, r = job
        inst = slots[idx % a.workers]
        m = fly(r, LOGS, instance=inst)
        if m.get("ulog"):
            try:
                convert(r.name)
            except Exception as e:  # noqa: BLE001
                m["convert_error"] = str(e)
        print(
            f"[{time.time() - t0:6.0f} s] {r.name}: ok={m['ok']} wall={m['wall_s']} s"
            + (f" error={m['error']}" if m["error"] else ""),
            flush=True,
        )
        return m

    # one thread per instance slot; jobs are assigned round-robin to slots
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futures = []
        for w in range(a.workers):
            jobs = [(i, r) for i, r in enumerate(pending) if i % a.workers == w]
            futures.append(ex.submit(lambda js: [work(j) for j in js], jobs))
        results = [m for f in futures for m in f.result()]
    n_ok = sum(1 for m in results if m["ok"])
    print(f"done: {n_ok}/{len(results)} ok in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
