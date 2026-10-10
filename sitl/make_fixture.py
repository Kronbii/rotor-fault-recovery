"""Cut a small ULog excerpt of one SITL run for the CI tests (``tests/data/sitl_excerpt.ulg``).

    .venv/bin/python sitl/make_fixture.py [run-name]

Keeps only the topics the log reader needs, from 2.6 s before the fault onset to 0.6 s
after it, and writes them with pyulog's ULog writer (about 0.35 MB).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent / "build" / "sitl" / "logs"
OUT = HERE.parent / "tests" / "data" / "sitl_excerpt.ulg"
TOPICS = [
    "sensor_combined",
    "actuator_motors",
    "actuator_outputs",
    "vehicle_attitude",
    "vehicle_local_position",
    "failure_detector_status",
    "vehicle_status",
]


def main(name: str = "partial-hover-calm-03"):
    from pyulog import ULog

    man = json.loads((LOGS / f"{name}.json").read_text())
    onset = man["events"]["inject_sent_s"]
    t0, t1 = int((onset - 2.6) * 1e6), int((onset + 0.6) * 1e6)
    u = ULog(str(LOGS / f"{name}.ulg"), TOPICS)
    for d in u.data_list:
        t = d.data["timestamp"]
        m = (t >= t0) & (t <= t1)
        if d.name in ("failure_detector_status", "vehicle_status"):
            # keep the last message before the window so that the state is defined
            k = np.flatnonzero(t < t0)
            if len(k):
                m[k[-1]] = True
        d.data = {k: v[m] for k, v in d.data.items()}
    u._logged_messages = [m for m in u.logged_messages if t0 <= m.timestamp <= t1]
    u._changed_parameters = [p for p in u.changed_parameters if t0 <= p[0] <= t1]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    u.write_ulog(str(OUT))
    meta = {
        "run": name,
        "spec": man["spec"],
        "inject_sent_s": onset,
        "manoeuvre_s": man["events"]["manoeuvre_s"],
        "fault_params": man.get("fault_params"),
        "window_s": [t0 * 1e-6, t1 * 1e-6],
    }
    OUT.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB) from {name}")


if __name__ == "__main__":
    main(*sys.argv[1:])
