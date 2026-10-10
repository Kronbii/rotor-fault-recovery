# Results summary (generated)

## E1: hover with a step loss of effectiveness on rotor 1.

| LOE rotor 1 | Passive | Oracle | Estimated (proposed) |
|---|---|---|---|
| 20 % | 7.6 / 2.7 (0/8) | 4.3 / 2.5 (0/8) | 4.3 / 2.2 (0/8) |
| 40 % | 18.1 / 3.3 (0/8) | 4.3 / 2.5 (0/8) | 4.3 / 2.2 (0/8) |
| 60 % | 46.1 / 5.5 (0/8) | 4.3 / 2.5 (0/8) | 4.4 / 2.2 (0/8) |
| 70 % | 73.5 / 16.1 (3/8) | 19.5 / 5.6 (0/8) | 12.9 / 5.5 (0/8) |
| 80 % | 97.7 / 31.0 (0/8) | 19.8 / 5.7 (0/8) | 18.7 / 5.5 (0/8) |
| 90 % | -- / -- (8/8) | 20.1 / 5.8 (0/8) | 27.2 / 6.2 (0/8) |

*Hover, fault at t = 2 s. Each cell: median max deviation / median settled error (cm) over non-lost runs, (lost/total).*

## E2: fault detection and isolation Monte Carlo.

| LOE | detected | delay ms: median (p95) | isolation | abs. LOE error, pts: median (p95) |
|---|---|---|---|---|
| 20-35 % | 33/33 | 51 (80) | 100.0 % | 1.7 (4.6) |
| 35-50 % | 44/44 | 37 (43) | 100.0 % | 1.0 (4.4) |
| 50-65 % | 44/44 | 30 (35) | 100.0 % | 1.1 (2.7) |
| 65-80 % | 46/46 | 27 (32) | 100.0 % | 1.3 (3.7) |
| 80-90 % | 33/33 | 24 (27) | 100.0 % | 3.8 (5.4) |
| all 20-90 % | 200/200 | 30 (58) | 100.0 % | 1.5 (4.8) |

*200 faulty runs (hover and figure-eight). False alarms: 0 in 32 min of fault-free flight (100 fault-free runs incl. unseen aggressive steps, plus pre-onset time of the faulty runs).*

## E2: sensitivity to gyro noise.

| gyro noise (deg/s) | detected | delay ms: median (p95) | isolation | abs. LOE error, pts | false alarms | false alarms, 1x thresholds | lost |
|---|---|---|---|---|---|---|---|
| 0.5x (0.9) | 40/40 | 18 (30) | 100.0 % | 1.1 | 0 | 0/20 runs | 0 |
| 1x (1.7) | 200/200 | 30 (58) | 100.0 % | 1.5 | 0 | -- | 0 |
| 2x (3.4) | 40/40 | 26 (34) | 100.0 % | 1.3 | 0 | 0/20 runs | 0 |
| 4x (6.9) | 40/40 | 29 (53) | 100.0 % | 1.7 | 0 | 0/20 runs | 0 |
| 8x (13.8) | 40/40 | 25 (48) | 100.0 % | 2.0 | 0 | 0/20 runs | 0 |

*1x = main experiment (200 + 100 runs); other levels 40 faulty + 20 fault-free runs with residual floor and thresholds re-calibrated for that level; the second false-alarm column re-flies the 20 fault-free runs with the thresholds calibrated at 1x.*

## E3: figure-eight tracking with a 50 % loss.

| Baseline | RMSE before | RMSE 0-3 s after | RMSE 3-10 s after | max dev. | lost |
|---|---|---|---|---|---|
| Passive | 3.5 | 17.6 | 8.5 | 24.7 | 0/10 |
| Oracle | 3.5 | 3.7 | 4.0 | 7.0 | 0/10 |
| Estimated (proposed) | 3.5 | 3.5 | 3.9 | 6.8 | 0/10 |

*Figure-eight (peak 3.3 m/s), 50 % LOE on rotor 1 at t = 10 s; medians over 10 seeds, cm.*

## E4/E5: complete loss of one rotor.

| case | baseline | lost | time to loss (s) | max dev. (cm) | settled (cm) | spin (rad/s) | switch delay (ms) |
|---|---|---|---|---|---|---|---|
| hover | Passive | 32/32 | 0.30 | -- | -- | -- | -- |
| hover | Oracle | 0/32 | -- | 21.5 | 6.4 | 23.2 | 0 |
| hover | Estimated (proposed) | 0/32 | -- | 37.9 | 7.6 | 23.2 | 62 |
| figure-eight | Passive | 24/24 | 0.29 | -- | -- | -- | -- |
| figure-eight | Oracle | 0/24 | -- | 25.4 | 16.7 | 23.6 | 0 |
| figure-eight | Estimated (proposed) | 0/24 | -- | 39.2 | 18.3 | 23.6 | 62 |

*Complete loss of one rotor (all four rotors in turn, several seeds each). Spin = mean |omega| over the last 2 s; switch delay = onset to yaw-relaxed control.*

## E6: robustness of complete-loss recovery.

| true-system change | runs lost per value |
|---|---|
| actuator delay (ms) | 0: 0/8, 2: 0/8, 4: 0/8, 8: 0/8, 12: 0/8, 16: 0/8, 20: 0/8, 24: 5/8 |
| motor time constant (ms) | 15: 0/8, 25: 0/8, 40: 0/8, 60: 0/8, 80: 8/8, 100: 8/8 |
| rotational damping (x) | 0.25: 0/8, 0.5: 0/8, 1: 0/8, 1.5: 0/8, 2: 0/8 |
| inertia (x) | 0.7: 0/8, 0.85: 0/8, 1: 0/8, 1.2: 0/8, 1.4: 0/8 |
| turbulence (x) | 0: 0/8, 1: 0/8, 2: 0/8, 3: 5/8, 4: 6/8 |

*Complete loss in hover, estimated pipeline; the controller always assumes the nominal model.*

## E7: FDI ablation.

| FDI variant | detected | delay ms: median (p95) | isolation | abs. LOE error, pts | false-alarm runs (+ before onset) | lost |
|---|---|---|---|---|---|---|
| proposed (all elements) | 60/60 | 30 (64) | 100.0 % | 1.6 | 0/30 (+0) | 0 |
| no healthy baseline | 59/60 | 28 (63) | 100.0 % | 1.8 | 1/30 (+1) | 0 |
| no adaptive residual scale | 60/60 | 25 (56) | 100.0 % | 1.2 | 0/30 (+0) | 0 |
| single CUSUM (15 %) | 60/60 | 61 (99) | 100.0 % | 1.2 | 0/30 (+0) | 0 |

*60 faulty + 30 fault-free runs per variant (same seeds), each variant calibrated with the same procedure; detected counts alarms after onset.*

## PX4 SITL validation.

| fault | condition | runs | detected | isolated | delay ms: median (p95) | no adaptive scale: delay (p95), isolated | abs. loss error, pts | PX4 failure detector | PX4 after the fault |
|---|---|---|---|---|---|---|---|---|---|
| partial | hover, calm | 10 | 10/10 | 10/10 | 120 (126) | 40 (50), 10/10 | 0.1 | none | held 4, descended 6 |
| partial | orbit, calm | 10 | 10/10 | 10/10 | 120 (128) | 42 (56), 10/10 | 0.9 | none | held 8, descended 2 |
| partial | hover, wind | 10 | 10/10 | 10/10 | 124 (124) | 38 (48), 10/10 | 0.4 | none | held 6, descended 4 |
| partial | orbit, wind | 10 | 10/10 | 10/10 | 122 (137) | 36 (55), 8/10 | 2.3 | none | held 3, descended 6, flipped 1 |
| partial | all | 40 | 40/40 | 40/40 | 120 (128) | 40 (56), 38/40 | 0.5 | none | held 21, descended 18, flipped 1 |
| complete | all four | 16 | 16/16 | 16/16 | 132 (141) | 20 (24), 14/16 | 0.0 | fd_roll 16/16, 664 ms | flipped 16 |

*PX4 v1.18.0-beta1-322-g18098d3bbe SITL, x500 in Gazebo Harmonic; the diagnoser is replayed offline on the ULogs. Partial: thrust loss of 19-48 % on a random motor, onset 8-14 s into the manoeuvre; complete: each motor in each condition (the loss estimate is clipped at 100 %). False alarms: 0 (as designed) and 0 (no adaptive scale) in 22 min of fault-free flight (24 held-out flights plus the time before each onset); calibration on 24 other fault-free flights. PX4 after the fault, within 8 s (6 s for complete losses): held altitude (less than 1 m lost), descended (more than 1 m lost, tilt below 90 deg) or flipped.*
