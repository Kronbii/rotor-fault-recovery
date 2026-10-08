"""Vehicle, sensor and simulation parameters for a 5-inch X-configuration quadrotor.

Every number in this file is an *assumption* chosen to be representative of a
5-inch FPV-class quadrotor carrying a 4S battery and a small HD camera. None of
them has been identified on a real airframe; see ``docs`` and the README for the
justification of each choice and for the limitations this implies.

Conventions
-----------
* World frame: z up (ENU-like), gravity ``-g e3``.
* Body frame: x forward, y left, z up (FLU). Collective thrust acts along +z_b.
* Rotor numbering (seen from above):

        front
     2 (FL)   1 (FR)
        \\     /
         \\   /
         /   \\
        /     \\
     3 (RL)   4 (RR)
        rear

  Rotor ``i`` sits at azimuth ``theta_i`` = -45, 45, 135, 225 deg.
  Diagonal pairs (1,3) and (2,4) spin in the same direction.
* Yaw reaction torque of rotor ``i`` on the body is ``s_i * kappa * f_i`` with
  ``s = (+1, -1, +1, -1)``; the rotor's own angular velocity is ``-s_i * Omega_i * e3``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

G = 9.81  # m/s^2


# --------------------------------------------------------------------------------------
# Mass-distribution model used to derive the inertia tensor
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MassModel:
    """Simple lumped-mass model of a 5-inch quad (all masses in kg, lengths in m).

    The central body (battery, stack, cameras, frame plates) is a solid box centred
    on the rotor plane; each arm is a thin rod from the centre to the motor; each
    motor+propeller is a point mass at the arm tip.
    """

    battery: float = 0.200  # 4S 1500 mAh LiPo
    frame_body: float = 0.060  # carbon top/bottom plates, standoffs
    electronics: float = 0.030  # FC + 4-in-1 ESC stack
    fpv_system: float = 0.050  # FPV camera, VTX, receiver, antennas
    hd_camera: float = 0.070  # small action camera
    hardware: float = 0.030  # screws, straps, wiring
    arm: float = 0.012  # each of 4 carbon arms
    motor: float = 0.033  # each 2306-class motor
    prop: float = 0.0045  # each 5-inch tri-blade propeller
    body_box: tuple[float, float, float] = (0.16, 0.07, 0.07)  # central box (x, y, z)
    arm_length: float = 0.11  # motor axis to centre of mass

    @property
    def body_mass(self) -> float:
        return (
            self.battery
            + self.frame_body
            + self.electronics
            + self.fpv_system
            + self.hd_camera
            + self.hardware
        )

    @property
    def rotor_mass(self) -> float:
        return self.motor + self.prop

    @property
    def total_mass(self) -> float:
        return self.body_mass + 4 * self.arm + 4 * self.rotor_mass

    def inertia(self) -> np.ndarray:
        """Diagonal inertia tensor about the centre of mass (kg m^2)."""
        lx, ly, lz = self.body_box
        mb = self.body_mass
        l = self.arm_length
        a = l / np.sqrt(2.0)
        # Solid box
        jb = mb / 12.0 * np.array([ly**2 + lz**2, lx**2 + lz**2, lx**2 + ly**2])
        # Point-mass rotors at (+-a, +-a, 0)
        mr = self.rotor_mass
        jr = 4 * mr * np.array([a**2, a**2, 2 * a**2])
        # Thin rods from the origin along the diagonals: about own end J = m l^2 / 3,
        # projected onto x/y with sin^2(45 deg) = 1/2.
        ma = self.arm
        ja = 4 * ma * l**2 / 3.0 * np.array([0.5, 0.5, 1.0])
        return np.diag(jb + jr + ja)


# --------------------------------------------------------------------------------------
# Quadrotor parameters
# --------------------------------------------------------------------------------------
@dataclass
class QuadParams:
    """Physical parameters of the simulated vehicle.

    Attributes are deliberately explicit; :func:`nominal_params` documents how each
    value was chosen.
    """

    mass: float
    inertia: np.ndarray  # 3x3
    arm_length: float
    kappa: float  # yaw reaction torque per unit thrust (m)
    k_f: float  # thrust coefficient, f = k_f * Omega^2 (N s^2)
    omega_min: float  # idle rotor speed (rad/s)
    omega_max: float  # max rotor speed (rad/s)
    tau_motor: float  # first-order rotor-speed time constant (s)
    rotor_inertia: float  # propeller + bell inertia about the spin axis (kg m^2)
    drag_lin: np.ndarray  # body-frame linear drag diag (N s/m)
    drag_quad: float  # world-frame isotropic quadratic drag (N s^2/m^2)
    rot_damping: np.ndarray  # body-frame rotational damping diag (N m s/rad)
    k_f_scale: np.ndarray = field(default_factory=lambda: np.ones(4))  # per-rotor k_f spread
    g: float = G

    # ---------------- derived quantities ----------------
    @property
    def J(self) -> np.ndarray:
        return self.inertia

    @property
    def weight(self) -> float:
        return self.mass * self.g

    @property
    def f_max(self) -> float:
        """Nominal maximum thrust of one rotor (N)."""
        return self.k_f * self.omega_max**2

    @property
    def f_min(self) -> float:
        """Nominal idle thrust of one rotor (N)."""
        return self.k_f * self.omega_min**2

    @property
    def thrust_to_weight(self) -> float:
        return 4 * self.f_max / self.weight

    @property
    def hover_thrust(self) -> float:
        return self.weight / 4.0

    @property
    def spin_dir(self) -> np.ndarray:
        return np.array([1.0, -1.0, 1.0, -1.0])

    @property
    def rotor_positions(self) -> np.ndarray:
        """4x3 array of rotor positions in the body frame."""
        th = np.deg2rad([-45.0, 45.0, 135.0, 225.0])
        l = self.arm_length
        return np.stack([l * np.cos(th), l * np.sin(th), np.zeros(4)], axis=1)

    def mixer(self) -> np.ndarray:
        """Effectiveness matrix B (4x4): [T, tau_x, tau_y, tau_z] = B @ f."""
        r = self.rotor_positions
        return np.vstack(
            [
                np.ones(4),
                r[:, 1],  # tau_x = y_i f_i
                -r[:, 0],  # tau_y = -x_i f_i
                self.kappa * self.spin_dir,
            ]
        )

    def copy(self, **changes) -> QuadParams:
        p = dataclasses.replace(self, **changes)
        p.inertia = np.array(p.inertia, dtype=float).copy()
        p.drag_lin = np.array(p.drag_lin, dtype=float).copy()
        p.rot_damping = np.array(p.rot_damping, dtype=float).copy()
        p.k_f_scale = np.array(p.k_f_scale, dtype=float).copy()
        return p


def nominal_params() -> QuadParams:
    """Nominal 5-inch quadrotor.

    Choices (all assumed, not identified):

    * **Mass / inertia** from :class:`MassModel`: 0.64 kg total, inertia
      ``diag(1.47, 2.30, 3.37) e-3 kg m^2``.
    * **Arm length** 0.11 m (220 mm motor-to-motor diagonal).
    * **Thrust**: maximum rotor speed 2500 rad/s (~23.9 k rpm) and
      ``k_f = 8.96e-7 N s^2`` give 5.6 N per rotor, i.e. a total thrust-to-weight
      ratio of about 3.6. Racing builds exceed this; the lower value is chosen on
      purpose (task requirement 3-4x) and makes fault tolerance *harder*.
    * **Idle** 250 rad/s (1 % of max thrust), mimicking a motor idle setting.
    * **Yaw torque / thrust** ``kappa = 0.015 m``.
    * **Motor lag** ``tau = 25 ms`` first order on rotor speed.
    * **Rotor inertia** 6e-6 kg m^2 (5-inch prop + 2306 bell).
    * **Drag**: per-rotor induced (H-force) drag of 0.04 N s/m gives body-frame
      linear drag 0.16 N s/m in x/y; 0.20 N s/m in z (inflow loss); quadratic
      ``0.5 rho C_d A = 0.006 N s^2/m^2``.
    * **Rotational damping**: rotor H-force drag acting at the arm gives
      ``4 d l^2 ~ 2e-3 N m s/rad`` about z; inflow-induced thrust change of a
      rotor moving vertically gives ``~2.5e-3 N m s/rad`` about x/y. This term
      sets the equilibrium spin rate after a complete rotor loss and is the
      least certain parameter of the model.
    """
    mm = MassModel()
    omega_max = 2500.0
    f_max = 5.6
    return QuadParams(
        mass=mm.total_mass,
        inertia=mm.inertia(),
        arm_length=mm.arm_length,
        kappa=0.015,
        k_f=f_max / omega_max**2,
        omega_min=250.0,
        omega_max=omega_max,
        tau_motor=0.025,
        rotor_inertia=6.0e-6,
        drag_lin=np.array([0.16, 0.16, 0.20]),
        drag_quad=0.006,
        rot_damping=np.array([2.5e-3, 2.5e-3, 2.0e-3]),
    )


def perturbed_params(
    rng: np.random.Generator,
    base: QuadParams | None = None,
    mass_pct: float = 0.03,
    inertia_pct: float = 0.05,
    kf_pct: float = 0.03,
    tau_pct: float = 0.10,
    damping_pct: float = 0.20,
) -> QuadParams:
    """Draw a 'true' vehicle around the nominal model (uniform relative errors).

    Controllers and the estimator always use the *nominal* model, so this
    represents unmodelled parametric uncertainty (manufacturing spread, battery
    swaps, prop wear).
    """
    base = base or nominal_params()

    def u(pct, size=None):
        return 1.0 + rng.uniform(-pct, pct, size=size)

    J = np.diag(np.diag(base.inertia) * u(inertia_pct, 3))
    return base.copy(
        mass=base.mass * u(mass_pct),
        inertia=J,
        tau_motor=base.tau_motor * u(tau_pct),
        rot_damping=base.rot_damping * u(damping_pct, 3),
        k_f_scale=u(kf_pct, 4),
    )


# --------------------------------------------------------------------------------------
# Sensor parameters
# --------------------------------------------------------------------------------------
@dataclass
class SensorParams:
    """Noise levels of the simulated sensors (all 1-sigma, per sample).

    * Gyro sampled at the control rate (500 Hz). 0.03 rad/s (1.7 deg/s) per sample
      is meant to represent the *effective* noise after the usual flight-controller
      low-pass filtering on a vibrating airframe, not the datasheet noise density.
      Bias follows a random walk; range +-2000 deg/s like common FPV IMUs.
    * Accelerometer 0.2 m/s^2 per sample (vibration dominated).
    * Motion capture at 100 Hz: 1 mm position, 0.3 deg attitude.
    """

    gyro_noise: float = 0.03  # rad/s
    gyro_bias_init: float = 0.01  # rad/s, std of initial bias per axis
    gyro_bias_walk: float = 0.002  # rad/s/sqrt(s)
    gyro_range: float = np.deg2rad(2000.0)  # rad/s saturation
    accel_noise: float = 0.2  # m/s^2
    accel_bias: float = 0.05  # m/s^2, std of constant bias per axis
    mocap_rate: float = 100.0  # Hz
    mocap_pos_noise: float = 0.001  # m
    mocap_att_noise: float = np.deg2rad(0.3)  # rad

    def scaled(self, gyro_scale: float = 1.0) -> SensorParams:
        return dataclasses.replace(self, gyro_noise=self.gyro_noise * gyro_scale)


# --------------------------------------------------------------------------------------
# Simulation timing
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SimConfig:
    """Timing. Physics: RK4 at 1 kHz. Control, estimation and FDI: 500 Hz."""

    dt_ctrl: float = 0.002
    physics_substeps: int = 2

    @property
    def dt_phys(self) -> float:
        return self.dt_ctrl / self.physics_substeps
