# PX4 SITL validation

The diagnoser of this repository (`rotorfault.fdi`, unchanged) replayed offline on ULogs
of PX4 software-in-the-loop flights: PX4's x500 quadrotor in Gazebo Harmonic, headless,
with motor faults injected during hover and during an orbit, with and without wind.
Everything here is simulation: PX4's real flight stack, Gazebo's multicopter model. The
results are in the main README (section "PX4 SITL validation") and the paper; this file
says how to reproduce them.

## What is run

| step | script | output |
|---|---|---|
| airframe file from PX4's model files | `make_airframe.py` | `config/x500.yaml` |
| one SITL flight | `harness.py` | ULog + manifest |
| the campaign (110 flights) | `campaign.py` with `config/campaign.yaml` | `build/sitl/logs/*.ulg`, `build/sitl/npz/*.npz` |
| replay + comparison with PX4 | `analyze.py` (+ `figures_sitl.py`) | `results/*.json`, `results/sitl_runs.csv`, `figures/*.png`, README/paper blocks |
| CI test fixture | `make_fixture.py` | `tests/data/sitl_excerpt.ulg` |

One flight: a fresh `gz sim -s` server (`--seed` = run seed, sensor noise) and a fresh
PX4 daemon in its own working directory and gz-transport partition; pymavlink over UDP
sets parameters, arms, takes off to 6 m, holds (hover) or flies `MAV_CMD_DO_ORBIT`
(5 m radius, 3 m/s, nose to the centre), injects the fault, flies on, disarms. Gazebo
runs at 4x real time (`PX4_SIM_SPEED_FACTOR`, PX4 follows the simulation clock); two
instances run in parallel. A flight takes about 20 s of wall time.

Conditions: `default` world (calm) and PX4's `windy` world (steady 5.4 m/s wind; it acts
on the x500 through the rotor-drag terms of Gazebo's motor model, which tilts the hover
by about 3.5 deg and makes the four motor commands unequal).

## Fault injection

PX4 at this version (v1.18.0-beta1, main of 2026-08-18) has two relevant mechanisms, and
both were tried:

1. **PX4 failure injection**: `failure motor off -i N` / `MAV_CMD_INJECT_FAILURE`
   (unit 101, type 1) with `SYS_FAILURE_EN = 1`. For motors only `off` (and `ok`) are
   implemented; there is no `slow` or partial mode. The injection is acknowledged but has
   **no effect with the default `CA_FAILURE_MODE = 0`**: the control allocator applies the
   motor stop mask only when `CA_FAILURE_MODE` is 1 or 2 (checked in a SITL flight: motor
   kept spinning). With `CA_FAILURE_MODE = 1` the motor stops, but the allocator then
   writes NaN as that motor's command, so the fault is visible in the logged command
   itself. These runs (`px4_off`) are used for PX4's behaviour only.
2. **Simulated-ESC loss** (used for the diagnoser): the gz bridge maps PX4's normalised
   motor command `u` to a rotor-speed command `SIM_GZ_EC_MIN + (SIM_GZ_EC_MAX -
   SIM_GZ_EC_MIN) u` (150 + 850 u rad/s on the x500). Setting both limits of motor N to
   `k` times their default at runtime (two PARAM_SET messages in one burst) makes that
   ESC deliver `k` times the commanded speed, i.e. `k^2` of the thrust, at every command;
   `k = 0` stops the rotor. PX4's controller and allocator are not told: the logged
   `actuator_motors` command stays the intended one, as it would on a real vehicle with a
   damaged motor or a throttled ESC. The true onset is taken from `actuator_outputs`
   (logged for evaluation only), 12-40 ms (median 16 ms) after the parameter message was
   sent; detection delays are measured from that onset.

Nothing in the PX4 tree is modified; all parameters are set at runtime. PX4's ESC-current
motor-failure check is switched on in every run (`FD_ACT_EN = 1`) so that it gets its
chance; Gazebo's ESC model reports no current, so it cannot fire in SITL.

## What the diagnoser sees

`rotorfault.logs.ulog` reads `sensor_combined` (gyro, accelerometer; FRD converted to
FLU), `actuator_motors` (normalised command) and `vehicle_local_position` +
`vehicle_attitude` (body velocity) on the IMU time base (250 Hz). The airframe file
(`config/x500.yaml`) is generated from `x500_base/model.sdf`, `x500/model.sdf`, the
`default` world and the SITL airframe `4001_gz_x500`: mass and inertia composed over
the links, rotor positions, spin directions from `turningDirection`, `motorConstant`,
`momentConstant`, the asymmetric motor time constants, the command map. The only
quantity fitted to data is the command-to-thrust transport delay (one sample, 4 ms),
chosen on the calibration flights. The healthy baseline is learned over the 2.5 s of
hover before the manoeuvre command; detection runs from the manoeuvre command on.

Calibration: 24 fault-free flights (6 per condition, seeds 9000+), the same procedure as
in the simulation study (residual floor = RMS of the fault-free residual; threshold =
2x the largest CUSUM statistic seen, floored at the evidence of a design-size fault in
20 ms). Evaluation: 24 other fault-free flights (seeds 1000+), 40 partial losses (random
motor, loss 15-50 %, onset 8-14 s into the manoeuvre, seeds 2000+), 16 complete losses
(each motor in each condition, seeds 3000+), plus 6 flights with PX4's own failure
injection (2 with the default `CA_FAILURE_MODE = 0`, 4 with mode 1). Two diagnoser
variants: as designed, and without the adaptive residual scale.

## Reproduce

Requirements: PX4-Autopilot built for SITL (`make px4_sitl` in `$PX4_DIR`, default
`~/repos/PX4-Autopilot`), Gazebo Harmonic (`gz sim` 8.x), Python with
`pip install -e ".[dev,sitl]"`. About 2 GB of disk for the logs.

```bash
make sitl-setup       # installs pyulog/pymavlink, regenerates config/x500.yaml
make sitl-campaign    # 110 SITL flights, ~20 min with 2 parallel instances
make sitl-analyze     # offline: replay, comparison, figures, README/paper blocks
# a single flight, e.g. a 30 % loss of motor 2 during an orbit in wind:
.venv/bin/python sitl/harness.py --traj orbit --world windy --fault esc --motor 2 --loe 0.3 --speed 4
```

`campaign.py` resumes: finished runs are skipped (`--redo-failed` retries failed ones).
SITL runs are not bit-reproducible (thread scheduling, two instances sharing the CPU):
the seeds fix the Gazebo noise and the drawn fault, not the exact trajectory. The
ULogs are not in git; `results/sitl_runs.csv` has one row per run and variant.

## Not done / limits

* PX4's x500 is a 2 kg, 1.7 thrust-to-weight quadrotor, not the 5-inch quad of the study,
  and Gazebo's sensors are much cleaner than a real IMU on a vibrating frame (gyro noise
  0.05 deg/s, no vibration): the residual floor here is far below what a real vehicle
  will show.
* The diagnoser runs offline on the logs; PX4 does not act on it. Nothing is said about
  fault-tolerant control in PX4.
* Wind is Gazebo's steady wind through the rotor-drag model; there are no gusts.
* The two QGroundControl telemetry logs in `~/Documents/QGroundControl/Telemetry`
  (2026-08-19, 66 s and 837 s, PX4 v1.18 development build, quadrotor, home position at
  PX4's default SITL location) are earlier SITL sessions. They hold MAVLink telemetry
  at 50 Hz (attitude, rates, `SERVO_OUTPUT_RAW`), no accelerometer and no high-rate IMU,
  so they cannot drive the diagnoser and were not used.
