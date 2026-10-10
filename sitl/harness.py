"""Headless PX4 SITL runs (Gazebo Harmonic, x500) with rotor-fault injection.

One run = a fresh Gazebo server (fixed ``--seed``) + a fresh PX4 instance in its own
working directory, controlled over MAVLink with pymavlink:

    boot -> arm -> take off -> hover or orbit -> inject fault -> fly on -> disarm -> stop

The ULog written by PX4 is the only output that matters; everything the diagnoser uses
is read back from it offline (``rotorfault.logs.ulog``).

Fault injection (see ``sitl/README.md`` for why there are two mechanisms):

* ``esc``: the simulated ESC of one motor delivers only ``sqrt(1 - loe)`` of the commanded
  rotor speed, i.e. ``1 - loe`` of the thrust, by scaling that output's range
  (``SIM_GZ_EC_MIN<n>`` / ``SIM_GZ_EC_MAX<n>``) at runtime. PX4's controller and allocator
  never learn about it; the logged motor command stays the intended one. ``loe = 1``
  sets the range to zero (rotor stops).
* ``px4_off``: PX4's own failure injection, ``failure motor off -i <n>`` (MAV_CMD_INJECT_FAILURE
  with SYS_FAILURE_EN = 1). In this PX4 version only ``off`` exists for motors, and it takes
  effect only with ``CA_FAILURE_MODE`` >= 1 (the allocator applies the stop mask); the
  stopped motor's command is then logged as NaN, which reveals the fault to any log-based
  diagnoser, so these runs are used for PX4's behaviour only.

Nothing in the PX4 source tree is modified; parameters are set at runtime over MAVLink.
This module is not imported by the package or by CI.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
PX4_DIR = Path(os.environ.get("PX4_DIR", "~/repos/PX4-Autopilot")).expanduser()
PX4_BUILD = PX4_DIR / "build" / "px4_sitl_default"
GZ_DIR = PX4_DIR / "Tools" / "simulation" / "gz"


@dataclass
class RunSpec:
    name: str
    seed: int  # Gazebo sensor-noise seed (and wind seed); PX4 itself has no seed
    traj: str = "hover"  # hover | orbit
    fault: str = "none"  # none | esc | px4_off
    motor: int = 1  # PX4 numbering 1..4 (1 front-right, 2 rear-left, 3 front-left, 4 rear-right)
    loe: float = 0.0  # loss of thrust effectiveness of that motor, 0..1
    pre_s: float = 20.0  # flight time in the manoeuvre before the onset (or total if no fault)
    post_s: float = 8.0  # flight time after the onset
    world: str = "default"  # Gazebo world: "default" (calm) or "windy" (steady 5.4 m/s)
    altitude: float = 6.0
    orbit_radius: float = 5.0
    orbit_speed: float = 3.0
    speed_factor: float = 1.0
    extra_params: dict = field(default_factory=dict)


# ======================================================================================
def gz_env(instance: int) -> dict:
    """Environment of PX4's generated gz_env.sh, plus a private gz-transport partition."""
    env = dict(os.environ)
    plugins = PX4_BUILD / "src" / "modules" / "simulation" / "gz_plugins"
    env.update(
        PX4_GZ_MODELS=str(GZ_DIR / "models"),
        PX4_GZ_WORLDS=str(GZ_DIR / "worlds"),
        PX4_GZ_PLUGINS=str(plugins),
        PX4_GZ_SERVER_CONFIG=str(PX4_DIR / "src/modules/simulation/gz_bridge/server.config"),
        GZ_SIM_RESOURCE_PATH=f"{GZ_DIR / 'models'}:{GZ_DIR / 'worlds'}",
        GZ_SIM_SYSTEM_PLUGIN_PATH=str(plugins),
        GZ_SIM_SERVER_CONFIG_PATH=str(PX4_DIR / "src/modules/simulation/gz_bridge/server.config"),
        GZ_PARTITION=f"rotorfault{instance}",
        GZ_IP="127.0.0.1",
    )
    return env


def prepare_workdir(work: Path) -> None:
    """PX4 rootfs: real ``etc/`` with links to the build's startup files plus our logger list."""
    if work.exists():
        shutil.rmtree(work)
    (work / "etc" / "logging").mkdir(parents=True)
    for sub in ("init.d", "init.d-posix", "extras"):
        src = PX4_BUILD / "etc" / sub
        if src.exists():
            (work / "etc" / sub).symlink_to(src)
    shutil.copy(HERE / "config" / "logger_topics.txt", work / "etc" / "logging")


class Sitl:
    """Gazebo server + PX4 daemon for one run (process groups, killed on exit)."""

    def __init__(self, spec: RunSpec, work: Path, instance: int = 0):
        self.spec = spec
        self.work = work
        self.instance = instance
        self.env = gz_env(instance)
        self.gz = None
        self.px4 = None
        self._logs = []

    def start(self, timeout: float = 60.0) -> None:
        prepare_workdir(self.work)
        world = GZ_DIR / "worlds" / f"{self.spec.world}.sdf"
        gz_log = open(self.work / "gz.log", "w")
        px4_log = open(self.work / "px4.log", "w")
        self._logs += [gz_log, px4_log]
        self.gz = subprocess.Popen(
            ["gz", "sim", "-s", "-r", "-v", "2", "--seed", str(self.spec.seed), str(world)],
            env=self.env,
            cwd=self.work,
            stdout=gz_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        env = dict(self.env)
        env.update(
            PX4_SYS_AUTOSTART="4001",
            PX4_SIM_MODEL="gz_x500",
            PX4_GZ_STANDALONE="1",
            PX4_GZ_WORLD=self.spec.world,
            PX4_GZ_NO_FOLLOW="1",
            HEADLESS="1",
        )
        if self.spec.speed_factor != 1.0:
            env["PX4_SIM_SPEED_FACTOR"] = f"{self.spec.speed_factor:g}"
        self.px4 = subprocess.Popen(
            [
                str(PX4_BUILD / "bin" / "px4"),
                "-d",
                "-i",
                str(self.instance),
                "-w",
                str(self.work),
                str(PX4_BUILD / "etc"),
            ],
            env=env,
            cwd=self.work,
            stdout=px4_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def alive(self) -> bool:
        return (
            self.gz is not None
            and self.px4 is not None
            and self.gz.poll() is None
            and self.px4.poll() is None
        )

    def stop(self) -> None:
        for p in (self.px4, self.gz):
            if p is not None and p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
        t0 = time.time()
        for p in (self.px4, self.gz):
            if p is None:
                continue
            while p.poll() is None and time.time() - t0 < 10:
                time.sleep(0.1)
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()
        for f in self._logs:
            f.close()

    def ulogs(self) -> list[Path]:
        return sorted((self.work / "log").rglob("*.ulg"))


# ======================================================================================
class Vehicle:
    """Minimal MAVLink client for PX4 (pymavlink)."""

    def __init__(self, instance: int = 0, timeout: float = 90.0):
        from pymavlink import mavutil

        self.mavutil = mavutil
        self.m = mavutil.mavlink_connection(
            f"udpin:127.0.0.1:{14540 + instance}", source_system=245, dialect="common"
        )
        hb = self.m.wait_heartbeat(timeout=timeout)
        if hb is None:
            raise TimeoutError("no heartbeat from PX4")
        self.sysid = hb.get_srcSystem()  # MAV_SYS_ID = instance + 1
        self.boot_ms = 0
        self.pos = None
        self.att = None
        self.statustexts = []

    # ---------------------------------------------------------------- messages
    def pump(self, duration: float = 0.0, until=None):
        """Read messages for ``duration`` wall seconds (or until ``until()`` is true)."""
        t_end = time.time() + duration
        while True:
            msg = self.m.recv_match(blocking=True, timeout=0.05)
            if msg is not None:
                self._handle(msg)
            if until is not None and until():
                return True
            if time.time() >= t_end and (until is None or duration > 0):
                return False

    def _handle(self, msg):
        t = msg.get_type()
        if hasattr(msg, "time_boot_ms"):
            self.boot_ms = max(self.boot_ms, int(msg.time_boot_ms))
        if t == "LOCAL_POSITION_NED":
            self.pos = msg
        elif t == "ATTITUDE":
            self.att = msg
        elif t == "STATUSTEXT":
            self.statustexts.append((self.boot_ms, msg.text))

    def sim_time(self) -> float:
        return self.boot_ms / 1000.0

    def wait_sim(self, dt: float, abort=None):
        t_end = self.sim_time() + dt
        return self.pump(
            3600, until=lambda: self.sim_time() >= t_end or (abort is not None and abort())
        )

    # ---------------------------------------------------------------- commands
    def command(self, cmd, *params, timeout=3.0, retries=3):
        p = list(params) + [0.0] * (7 - len(params))
        for _ in range(retries):
            self.m.mav.command_long_send(self.sysid, 1, cmd, 0, *p)
            t_end = time.time() + timeout
            while time.time() < t_end:
                msg = self.m.recv_match(blocking=True, timeout=0.1)
                if msg is None:
                    continue
                self._handle(msg)
                if msg.get_type() == "COMMAND_ACK" and msg.command == cmd:
                    return msg.result
        return None

    def set_param(self, name: str, value, timeout=3.0):
        mav = self.mavutil.mavlink
        is_int = isinstance(value, int)
        ptype = mav.MAV_PARAM_TYPE_INT32 if is_int else mav.MAV_PARAM_TYPE_REAL32
        if is_int:  # PX4 sends/accepts integers bytewise-encoded in the float field
            import struct

            fval = struct.unpack("<f", struct.pack("<i", value))[0]
        else:
            fval = float(value)
        for _ in range(3):
            self.m.mav.param_set_send(self.sysid, 1, name.encode(), fval, ptype)
            t_end = time.time() + timeout
            while time.time() < t_end:
                msg = self.m.recv_match(blocking=True, timeout=0.1)
                if msg is None:
                    continue
                self._handle(msg)
                if msg.get_type() == "PARAM_VALUE" and msg.param_id == name:
                    return True
        return False

    def send_params_burst(self, values: dict):
        """Send several PARAM_SETs back to back (applied in the same receive cycle)."""
        mav = self.mavutil.mavlink
        import struct

        for name, value in values.items():
            if isinstance(value, int):
                fval = struct.unpack("<f", struct.pack("<i", value))[0]
                ptype = mav.MAV_PARAM_TYPE_INT32
            else:
                fval, ptype = float(value), mav.MAV_PARAM_TYPE_REAL32
            self.m.mav.param_set_send(self.sysid, 1, name.encode(), fval, ptype)

    def set_mode(self, main: int, sub: int = 0):
        return self.command(176, 1, main, sub)  # MAV_CMD_DO_SET_MODE, custom mode

    def arm(self, timeout: float = 60.0) -> bool:
        t_end = time.time() + timeout
        while time.time() < t_end:
            if self.command(400, 1) == 0:
                return True
            self.pump(1.0)
        return False

    def disarm(self, force: bool = True):
        return self.command(400, 0, 21196 if force else 0)

    def altitude(self) -> float:
        return -self.pos.z if self.pos is not None else float("nan")


PX4_MAIN_AUTO, PX4_AUTO_LOITER, PX4_AUTO_TAKEOFF, PX4_AUTO_LAND = 4, 3, 2, 6
X500_MIN, X500_MAX = 150, 1000  # SIM_GZ_EC_MIN/MAX defaults of PX4 airframe 4001_gz_x500


# ======================================================================================
def fly(spec: RunSpec, out_dir: Path, instance: int = 0, keep_work: bool = False) -> dict:
    """Execute one run and copy its ULog to ``out_dir/<name>.ulg``; returns the manifest."""
    work = HERE.parent / "build" / "sitl" / f"work{instance}"
    sitl = Sitl(spec, work, instance)
    manifest = {"spec": asdict(spec), "events": {}, "ok": False, "error": None}
    t_wall = time.time()
    try:
        sitl.start()
        v = Vehicle(instance)
        manifest["events"]["heartbeat_s"] = v.sim_time()
        v.pump(2.0)
        params = {
            "SYS_FAILURE_EN": 1,
            "COM_RC_IN_MODE": 4,  # no RC in this setup
            "NAV_DLL_ACT": 0,
            "NAV_RCL_ACT": 0,
            "COM_DISARM_LAND": -1.0,
            "MIS_TAKEOFF_ALT": float(spec.altitude),
            "MPC_TKO_SPEED": 2.0,
            # PX4's ESC-current motor-failure check, enabled so that it gets its chance
            "FD_ACT_EN": 1,
        }
        if spec.fault == "px4_off":
            params["CA_FAILURE_MODE"] = 1
        params.update(spec.extra_params)
        for k, val in params.items():
            if not v.set_param(k, val):
                raise RuntimeError(f"could not set {k}")
        if not v.arm():
            raise RuntimeError("arming refused")
        manifest["events"]["armed_s"] = v.sim_time()
        if v.command(22, 0, 0, 0, float("nan"), float("nan"), float("nan"), float("nan")) != 0:
            raise RuntimeError("takeoff refused")
        ok = v.pump(180, until=lambda: v.pos is not None and v.altitude() > spec.altitude - 0.3)
        if not ok:
            raise RuntimeError("takeoff did not reach altitude")
        v.wait_sim(3.0)
        v.set_mode(PX4_MAIN_AUTO, PX4_AUTO_LOITER)
        v.wait_sim(2.0)
        if spec.traj == "orbit":
            # MAV_CMD_DO_ORBIT: radius, velocity, yaw behaviour (0 = front to centre),
            # centre = current position, current altitude
            r = v.command(
                34,
                spec.orbit_radius,
                spec.orbit_speed,
                0,
                0,
                float("nan"),
                float("nan"),
                float("nan"),
            )
            if r != 0:
                raise RuntimeError(f"orbit refused ({r})")
        manifest["events"]["manoeuvre_s"] = v.sim_time()
        v.wait_sim(spec.pre_s, abort=lambda: not sitl.alive())
        if spec.fault != "none" and spec.loe > 0:
            n = spec.motor
            manifest["events"]["inject_sent_s"] = v.sim_time()
            if spec.fault == "esc":
                # the x500 airframe maps the command u to 150 + 850 u rad/s; scaling both
                # ends by k scales the rotor speed by k at every u, the thrust by k^2
                k = math.sqrt(max(0.0, 1.0 - spec.loe))
                vals = {f"SIM_GZ_EC_MAX{n}": int(round(X500_MAX * k))}
                vals[f"SIM_GZ_EC_MIN{n}"] = int(round(X500_MIN * k))
                v.send_params_burst(vals)
                manifest["fault_params"] = vals
            elif spec.fault == "px4_off":
                # MAV_CMD_INJECT_FAILURE: unit 101 (motor), type 1 (off), instance n
                manifest["events"]["inject_ack"] = v.command(420, 101, 1, n)
            v.wait_sim(spec.post_s, abort=lambda: not sitl.alive())
        manifest["events"]["end_s"] = v.sim_time()
        manifest["final_altitude_m"] = v.altitude()
        manifest["statustext"] = v.statustexts[-40:]
        v.disarm(force=True)
        v.pump(2.0)
        manifest["ok"] = True
    except Exception as e:  # noqa: BLE001 - every failure is recorded in the manifest
        manifest["error"] = f"{type(e).__name__}: {e}"
    finally:
        sitl.stop()
        manifest["wall_s"] = round(time.time() - t_wall, 1)
    logs = sitl.ulogs()
    out_dir.mkdir(parents=True, exist_ok=True)
    if logs:
        dst = out_dir / f"{spec.name}.ulg"
        shutil.copy(logs[-1], dst)
        manifest["ulog"] = dst.name
        manifest["ulog_bytes"] = dst.stat().st_size
    else:
        manifest["ok"] = False
        manifest["error"] = manifest["error"] or "no ULog written"
    for name in ("px4.log", "gz.log"):
        if (work / name).exists():
            shutil.copy(work / name, out_dir / f"{spec.name}.{name}")
    (out_dir / f"{spec.name}.json").write_text(json.dumps(manifest, indent=1))
    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)
    return manifest


if __name__ == "__main__":  # pragma: no cover - manual single run
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default="manual")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--traj", default="hover", choices=["hover", "orbit"])
    ap.add_argument("--fault", default="none", choices=["none", "esc", "px4_off"])
    ap.add_argument("--motor", type=int, default=1)
    ap.add_argument("--loe", type=float, default=0.0)
    ap.add_argument("--pre", type=float, default=20.0)
    ap.add_argument("--post", type=float, default=8.0)
    ap.add_argument("--world", default="default", choices=["default", "windy"])
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--out", default=str(HERE.parent / "build" / "sitl" / "manual"))
    ap.add_argument("--keep-work", action="store_true")
    a = ap.parse_args()
    spec = RunSpec(
        a.name,
        a.seed,
        a.traj,
        a.fault,
        a.motor,
        a.loe,
        a.pre,
        a.post,
        a.world,
        speed_factor=a.speed,
    )
    print(json.dumps(fly(spec, Path(a.out), keep_work=a.keep_work), indent=1))
