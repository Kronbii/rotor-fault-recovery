# Portfolio summary

Material for a portfolio page and a CV. Every number below is from the generated results
of this repository (`results/headline.json`, `sitl/results/sitl_summary.json`); all of it
is simulation or PX4 software-in-the-loop. Nothing has flown yet. If the results are
regenerated, re-check the numbers against those files.

## One paragraph

Detect, isolate, recover is a reproducible study of rotor-fault tolerance for a 5-inch FPV
quadrotor. I wrote a 6-DOF simulator of the vehicle with sensor noise, motion-capture
latency, parameter spread and turbulence, and on it a pipeline that estimates each rotor's
effectiveness with a Kalman filter on a filtered rigid-body regression, detects and
isolates a loss with a bank of CUSUM tests, re-allocates thrust after a partial loss and
switches to a gain-scheduled yaw-relaxed LQR controller (after Mueller and D'Andrea) after
a complete loss. In Monte-Carlo simulation it detected and isolated every fault from 20 to
90 % loss and recovered every complete rotor loss in hover and during a 3.3 m/s
figure-eight, and robustness sweeps located where recovery breaks (actuator delay, slow
motors, strong turbulence). I then replayed the same diagnoser offline on 110 PX4
software-in-the-loop flights of PX4's x500 in Gazebo, with parameters read from the model
files: it detected and isolated all injected partial and complete losses, while PX4's own
failure detector flagged none of the partial losses. The log pipeline for Betaflight
blackbox logs and the protocol for a tethered test on the real quad are written; the test
is the next step.

## Headline results (simulation and SITL)

1. 200/200 simulated faults of 20-90 % loss detected and isolated to the right rotor;
   median detection delay 30 ms (95th percentile 58 ms), median loss-estimate error 1.5
   points; no false alarm in 32 min of fault-free simulated flight.
2. Complete loss of one rotor in simulated hover: the nominal controller lost the vehicle
   in 32/32 runs (median 0.30 s); the pipeline recovered 32/32 (median peak deviation
   38 cm, spin 23 rad/s), and 24/24 during a 3.3 m/s figure-eight.
3. 50 % loss in simulated hover: median peak deviation 28 cm with the passive controller,
   4.3 cm with the pipeline.
4. PX4 SITL (x500, Gazebo Harmonic, 110 flights): 40/40 partial losses (19-48 %) detected
   and isolated, median 120 ms; 16/16 complete losses, median 132 ms, all before the
   vehicle tilted past 90 deg; no false alarm in 22 min of fault-free SITL flight. PX4's
   failure detector: no flag for any partial loss, attitude-limit trigger after a median
   664 ms for the complete losses.
5. Limits found: complete-loss recovery started to fail at 24 ms actuator delay (2 ms
   assumed), an 80 ms motor time constant (25 ms assumed) and 3x the default turbulence.

## CV bullets

* Built a reproducible simulation study of rotor-fault detection and recovery for a 5-inch
  quadrotor (Python; Kalman-filter effectiveness estimation, CUSUM bank, yaw-relaxed LQR):
  200/200 simulated faults detected and isolated (median 30 ms), 56/56 simulated complete
  rotor losses recovered.
* Validated the diagnoser on 110 headless PX4 SITL flights (Gazebo, pymavlink automation,
  ULog replay): all 56 injected motor faults detected and isolated, median 120-132 ms,
  where PX4's failure detector flagged no partial loss.
* Wrote the log pipeline (PX4 ULog, Betaflight blackbox) with parameter identification and
  a tethered-test protocol for the first measured flights on the real quad.

## Portfolio fields

**What.** Detect, isolate, recover: rotor-fault tolerance for a 5-inch quadrotor, from a
simulation study to PX4 software-in-the-loop, with the real-vehicle test prepared.

**Problem.** A quadrotor has no actuator redundancy: a chipped prop or a failing motor
has to be noticed and compensated within tens of milliseconds, and a lost rotor can only
be survived by giving up yaw. Flight software for small quads mostly just disarms.

**How.** Rigid-body regression of the rotor effectiveness with a common state-variable
filter, Kalman filter and healthy baseline; CUSUM bank calibrated on fault-free flights
with disjoint seeds; re-allocation for partial losses; numerically solved relaxed-hover
equilibrium and gain-scheduled LQR for a complete loss; Monte-Carlo evaluation with
passive and oracle baselines, ablation and robustness sweeps; offline replay on PX4 SITL
ULogs with faults injected in the simulated ESC.

**Role.** Sole author: models, algorithms, simulator, experiments, SITL automation and
analysis, paper.

**Stages.**

1. Vehicle, sensor and turbulence model of a 5-inch quad; SE(3) control and allocation.
2. Diagnoser (regression, Kalman filter, CUSUM bank) and its calibration.
3. Fault-tolerant control: re-allocation and the yaw-relaxed controller.
4. Experiments E1-E7: partial and complete losses, trajectories, robustness, ablation.
5. PX4 SITL validation and the log pipeline for the real quad (Betaflight, tethered test).

**Measurements** (label, value, context).

| label | value | context |
|---|---|---|
| detection delay, simulation | 30 ms median (58 ms p95) | 200 faults, 20-90 % loss, hover and figure-eight |
| isolation, simulation | 200/200 | same runs |
| complete-loss recovery, simulation | 32/32 hover, 24/24 figure-eight | passive controller lost 32/32 and 24/24 |
| detection delay, PX4 SITL | 120 ms (partial), 132 ms (complete), median | x500 in Gazebo, 56 faults, offline replay |
| PX4's own detector, SITL | no flag for 40 partial losses; 664 ms for complete losses | attitude-limit check, after the vehicle had flipped |
| false alarms | 0 in 32 min (simulation), 0 in 22 min (SITL) | fault-free flights not used for calibration |
| robustness limit | 24 ms actuator delay, 80 ms motor lag, 3x turbulence | complete-loss recovery starts failing |

**Limits.** All results are simulation or SITL; vehicle parameters are assumed, not
identified; the simulated and Gazebo sensors are cleaner than a real vibrating frame, so
false-alarm rates are optimistic; in SITL the diagnoser ran offline and no fault-tolerant
control was run in PX4; the tethered test is planned, not done.

**Hero figure.** `figures/fig_summary.png` (peak deviation and loss rate versus fault
size, and a complete-loss recovery against the passive controller's flip). Second figure
for the SITL part: `sitl/figures/fig_sitl_delay.png`, or panel (b) of
`sitl/figures/fig_sitl_timeline.png` (diagnoser alarm before the flip, PX4's detector
after it).
