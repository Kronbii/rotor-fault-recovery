# Short summary for emails

Paste-ready text, written to be accurate to the results in this repository. Replace the
link once the repository is public.

---

**Project: rotor-fault detection and recovery for a 5-inch quadrotor (simulation study)**

I built an open, reproducible simulation of a 5-inch FPV quadrotor (my own airframe class,
0.64 kg, thrust-to-weight 3.6) with sensor noise, motion-capture latency, parameter spread
and turbulence, and on it a full detect-isolate-recover pipeline: a Kalman-filter estimate
of each rotor's effectiveness from a filtered rigid-body regression, a bank of CUSUM tests
for detection and isolation, re-allocation for partial losses, and a gain-scheduled
yaw-relaxed LQR controller (following Mueller and D'Andrea) for the complete loss of a rotor.
In 200 Monte-Carlo runs with 20-90 % loss of effectiveness every fault was detected (median
30 ms) and isolated to the right rotor with no false alarms in 32 minutes of fault-free
flight, and the controller recovered all 56 complete rotor losses it was tested on (hover
and a 3.3 m/s figure-eight) where the nominal controller flipped within about 0.3 s. I also
mapped where it breaks: about 24 ms of actuator delay, motors three times slower than
modelled, or three times the default turbulence.

To check that the diagnoser is not tied to my own simulator, I then replayed it, with the
same settings, on 110 PX4 software-in-the-loop flights of PX4's x500 in Gazebo, with its
parameters read from the simulator's model files and thresholds calibrated on separate
fault-free flights. It detected and isolated all 40 injected partial losses (19-48 %,
median delay 120 ms) and all 16 complete losses (median 132 ms), with no false alarm in
22 minutes of fault-free flight; PX4's own failure detector flagged none of the partial
losses and the complete losses only after about 0.66 s, once the vehicle had flipped.

Attach `paper/main.pdf` (6 pages, IEEE format). Key figure: `figures/fig_summary.png` (loss
rate and peak deviation versus fault size, and a complete-loss recovery); for the SITL part,
`sitl/figures/fig_sitl_delay.png`.

These are simulation results with assumed parameters (SITL included: Gazebo's sensors are
much cleaner than a real IMU). The next step, for which the blackbox-log pipeline and the
test protocol are ready, is a tethered test of my real 5-inch quad with the same estimator
replayed on its logs, which is where I think your group's work on UAV fault diagnosis and
fault-tolerant control would teach me the most.

---

Notes for adapting the last sentence: name the specific paper or method of the professor
(for example INDI-based FTC, observer-based FDI, or learning-based diagnosis) and say what
in this project it would change or test. Do not describe the work as published or
peer-reviewed; it is a simulation study with code and data, and the PX4 part is SITL,
not a flight. Do not describe the tethered test as done.
