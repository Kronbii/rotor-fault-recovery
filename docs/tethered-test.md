# Tethered test on the real 5-inch quadrotor (plan)

**Status: planned, not done.** Nothing in this document has been flown or measured yet.
It is the protocol for the first measured flights; results will go into the README and
the paper only once they exist.

## Goal

Measure, on the real 5-inch quadrotor, with the diagnoser of this repository replayed
offline on Betaflight blackbox logs (`rotorfault replay`):

1. the residual floor and the false-alarm rate on fault-free flights (the number the
   simulation and SITL cannot give: real vibration, real aerodynamics);
2. detection, isolation and the loss estimate for a **throttled ESC channel** switched
   on in flight (onset known from the log);
3. the loss estimate for a **weakened propeller** fitted before take-off;
4. the vehicle parameters that the study only assumed (mass, inertia, thrust coefficient,
   motor time constant, command-to-speed map), with `rotorfault identify`.

Out of scope for this campaign: complete rotor loss and yaw-relaxed control. That needs
the controller running on the vehicle, not an offline replay, and comes later.

## Rig

The quad hovers at 1.0 m inside a netted volume, held by two slack lines that only become
taut when it leaves a small band around the hover point. In the measurement window the
lines hang slack, so they disturb the dynamics as little as possible; outside it they
arrest the vehicle.

```
          beam / ceiling hook (2.4 m)
                   |
                   |  top line: 1.6 m Dyneema + 0.3 m bungee, swivel at the quad
                   |  (taut below ~0.8 m: catches a fall)
                   |
        ___________o___________   <- quad, attachment at the top-plate centre
       (  prop  ) [==] (  prop  )     hover point 1.0 m above the mat
                   o
                   |  bottom line: 1.2 m Dyneema, swivel at the quad
                   |  (taut above ~1.2 m: stops a climb or fly-away)
                   |
   ================#================  foam mat, floor anchor (10 kg sandbag or bolt)
   |<------- net enclosure ~2 x 2 m, 2.5 m high, operator outside ------->|
```

Geometry (anchors vertically above and below the hover point):

| line | anchor height | length | taut when the quad is |
|---|---|---|---|
| top | 2.4 m | 1.6 m (+ bungee) | below 0.8 m on the axis, or more than ~0.75 m off-axis at 1.0 m |
| bottom | 0.0 m | 1.2 m | above 1.2 m on the axis, or more than ~0.65 m off-axis at 1.0 m |

So the free volume is a vertical band of 0.8-1.2 m and a radius of about 0.65 m at the
hover height. Both lines attach at the frame centre through ball-bearing swivels (yaw is
free) and leave the frame vertically, between the arms, so that they cannot reach the
propeller disks; prop guards are fitted for all flights. 1.5 mm Dyneema weighs about
1.5 g/m: a slack line hangs a few grams on the frame, which shows up as a constant
offset (absorbed by the healthy baseline) rather than as a dynamic disturbance.

### Bill of materials

| item | qty | purpose |
|---|---|---|
| braided Dyneema line, 1.5 mm, 10 m | 1 | tether (low stretch) |
| ball-bearing fishing swivels with snaps, rated > 20 kg | 4 | yaw freedom at the quad and the anchors |
| bungee cord 4 mm, 0.5 m | 1 | shock absorber in the top line |
| ceiling hook / eye bolt rated > 50 kg, floor anchor (10 kg sandbag or bolted plate) | 2 | anchors |
| sports netting 3 x 3 m | 3 | enclosure |
| foam mat about 2 x 2 m | 1 | landing area |
| 5-inch prop guards (printed TPU) | 1 set | lines and net away from the props |
| propellers: matched new set, plus spares of the same type | 8+ | reference props |
| propellers with deliberately trimmed tips (see step 4) | 2-3 | weakened prop |
| kitchen scale (0.1 g) and scale for the quad (1 g), caliper | 1 each | mass, prop mass, geometry |
| thrust stand: 2-5 kg load cell + HX711, motor mount (optional) | 1 | bench k_f, lag, weakened-prop ground truth |
| optical tachometer (optional) | 1 | eRPM unit check |
| smoke stopper for the battery lead | 1 | first power-up after every change |
| LiPo bag, sand bucket, CO2 extinguisher | 1 each | battery safety |
| safety glasses | 2 | operator and observer |

## Firmware and logging

* Betaflight 4.3 or later (eRPM fields in the blackbox), bidirectional DShot on
  (`dshot_bidir = ON`), `motor_poles` set for the motors (14 for 12N14P).
* Blackbox on flash or SD at 1 kHz or more (loop 8 kHz with `blackbox_sample_rate`
  1/4 or 1/8; check that the flash holds a full flight), accelerometer logged
  (`blackbox_disable_acc = OFF`, accelerometer enabled and calibrated), RPM logged
  (`blackbox_disable_rpm = OFF`). Note the `motorOutput` range from the log header.
* Throttled-ESC test only: a test build with a small patch that scales the value sent
  to one ESC by a factor `k` while an AUX switch is on, and writes the switch state to a
  `debug` field. The scaling must be applied in the motor output driver, **after** the
  value is copied to the blackbox `motor[]` field: the log has to show the intended
  command, as a damaged motor would, otherwise the fault is visible in the command
  itself (the same problem as PX4's failure injection in SITL, see `sitl/README.md`).
  Default `k = 1`; the switch is off at arming.
* Decode every log with
  `blackbox_decode --unit-rotation rad/s --unit-acceleration m/s2 --unit-frame-time s --save-headers LOG.BBL`
  (betaflight/blackbox-tools) and keep the `.BBL`, the decoded CSV, the header CSV and
  `diff all`.

## Protocol

### 1. Bench checks (props off, every new build or firmware)

1. **Axes.** Record a short blackbox log while rotating the quad by hand: roll right
   (right side down), pitch nose-down, yaw nose-left, one at a time, then level and
   still. The replay assumes x forward / y left / z up: each of these rotations must give
   a positive rate on the matching axis, and level must read about +9.8 m/s^2 on z. If
   not, fix `gyro_axes` / `acc_axes` in `config/5inch.yaml`.
2. **Motor order and spin.** Betaflight motors tab: motor 1 rear-right, 2 front-right,
   3 rear-left, 4 front-left; spin directions as in `config/5inch.yaml` (props-in by
   default; with `yaw_motors_reversed = ON` all four `spin` signs flip).
3. **eRPM units.** Spin one motor at a fixed command and compare the `eRPM[i]` column
   with the tachometer (or the motors-tab RPM): sets `erpm_units`.
4. **Mass and geometry.** Weigh the flight-ready quad (battery, camera, guards, tether
   attachments); measure motor-to-motor distances.
5. **Thrust stand (optional).** Thrust versus command and eRPM for one motor with the
   reference prop and with each weakened prop; command steps for the motor time
   constant. These give the ground truth of step 4 below.

### 2. Safety checklist (before every session)

* Props off for any work on the bench; battery connected only in the cage.
* Smoke stopper on the first power-up after any change.
* Arm switch on the radio, failsafe set to drop (`failsafe_procedure = DROP`); check the
  receiver failsafe by switching the radio off with props off.
* Tether: inspect knots, swivels and anchors; pull-test each line at about 5x the
  vehicle weight (3 kg); check by hand (props off) that no line can reach a prop disk at
  any position inside the band.
* Net closed; operator and observer outside it, with glasses; nobody in the prop plane.
* Batteries: cell check before flight, land at 3.5 V/cell under load, charge in a LiPo
  bag; sand bucket and extinguisher at hand.
* Abort and disarm (the quad drops onto the mat) if a line stays taut for more than about
  1 s, on an oscillation that grows, or on an unexpected yaw spin.
* Weakened props: check the balance first and look at the gyro spectrum of a short hover
  before longer flights.

### 3. Fault-free flights (residual floor, false alarms, identification)

* At least 10 flights of 60 s: arm, climb into the band, hold still for 10 s, then hover
  with gentle stick inputs on each axis (small steps), land. Fresh matched props, charged
  batteries; note battery and prop IDs for each flight.
* Split by flight number: odd flights for calibration and identification, even flights
  held out for the false-alarm rate (the same separation as the disjoint seeds of the
  simulation and SITL).
* `rotorfault identify` on the calibration flights; replace the assumed values in
  `config/5inch.yaml` with the identified ones (record both).
* `rotorfault calibrate-logs <odd flights> --airframe config/5inch.yaml --out cal.json`
  (residual floor, thresholds, transport delay, stored baseline), then
  `rotorfault replay <even flights> --airframe config/5inch.yaml --calibration cal.json`
  for the false alarms per fault-free minute, with and without `--rpm`.
* Watch for: battery sag late in a flight (a slow common thrust loss on all rotors, which
  has a component along every rotor's fault signature), tether contact, the vibration
  level in the gyro. If the false-alarm rate is not acceptable, the residual floor is
  re-measured with more flights before any fault is flown; the thresholds are never tuned
  on the faulty flights.

### 4. Weakened propeller (loss present from take-off)

* One prop replaced by a prop of the same type with trimmed blade tips, cut symmetrically
  on all blades (start with about 3 mm, then 6 mm) so that it stays balanced. Its thrust
  loss at the hover command, measured on the thrust stand against the reference prop, is
  the ground truth.
* Same flight pattern as step 3, 3 flights per prop and position (move the weak prop
  between at least two motors).
* A baseline learned at the start of the flight would absorb a loss that is there from
  take-off, so these flights are replayed with the baseline stored from the fault-free
  calibration flights: `rotorfault replay ... --calibration cal.json --stored-baseline`.
  Measured: isolation and loss estimate versus the thrust-stand value. This depends on the
  baseline being stable from flight to flight (battery, props), which step 3 measures
  (`eta_ref_spread` in the calibration file).

### 5. Throttled ESC (onset in flight)

* Test firmware of the logging section. Bench check first (props off): with the switch on,
  the eRPM of the chosen motor drops by the expected factor and the blackbox `motor[]`
  value of that motor does not change.
* In the cage: hover for at least 10 s, switch the fault on, hold 5 s, switch off, land.
  Thrust ratio `k^2` = 0.85, then 0.75, then 0.65 (15, 25, 35 % loss), each on at least two
  motors, 3 flights per setting. Do not go beyond 35 % in this campaign; the simulation and
  SITL say nothing about how this frame handles more.
* Ground truth: onset from the `debug` field; loss from the eRPM ratio of that motor
  against the speed map identified in step 3, and from the thrust stand.
* Replay with the learned baseline (the onset is in flight), with and without `--rpm`.
  Measured: detection delay, isolation, loss estimate, alarms before the onset.

## What is logged for each flight

The `.BBL` and its decoded CSV and header CSV; the Betaflight `diff all`; flight number,
date, battery ID and its voltage before and after, prop set and position of a weakened
prop, fault setting and switch times; the vehicle mass; a phone video of the flight for
sanity checks; any tether contact or abort.

## How the results feed the paper

* A section "Measured on the real 5-inch quadrotor (tethered)": the residual floor next to
  the simulated and the SITL floors; false alarms per fault-free minute on the held-out
  flights; detection delay, isolation and loss error for the throttled ESC; loss error for
  the weakened props; each with and without RPM telemetry, and with the number of
  flights.
* The identified parameters next to the assumed ones, and E1-E6 re-run with them, to see
  which simulated conclusions survive (in particular the complete-loss limits of E6,
  which depend on the motor time constant and the actuator delay).
* Tether effects, the small number of flights and the single airframe stated as
  limitations. Simulation plus this one measured campaign is the scope of a workshop
  paper; nothing in it will be called a flight test of fault-tolerant control.
