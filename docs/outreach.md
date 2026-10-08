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

Key figure: `figures/fig_summary.png` (loss rate and peak deviation versus fault size, and a
complete-loss recovery), with the detail in `figures/fig_e4_complete_loss.png`.

These are simulation results with assumed parameters; the next step I want to take is to
test the diagnoser on real flight logs and in SITL, which is where I think your group's work
on UAV fault diagnosis and fault-tolerant control would teach me the most.

---

Notes for adapting the last sentence: name the specific paper or method of the professor
(for example INDI-based FTC, observer-based FDI, or learning-based diagnosis) and say what
in this project it would change or test. Do not describe the work as published or
peer-reviewed; it is a simulation study with code and data.
