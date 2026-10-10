"""Airframe files: YAML round trip, default geometry unchanged, Gazebo SDF parsing."""

from pathlib import Path

import numpy as np
import pytest

from rotorfault.airframe import Airframe, from_gazebo_sdf, simulator_airframe
from rotorfault.params import nominal_params


def test_default_geometry_is_unchanged():
    P = nominal_params()
    a = 0.11 / np.sqrt(2)
    expected = np.array(
        [
            [1, 1, 1, 1],
            [-a, a, a, -a],
            [-a, -a, a, a],
            [0.015, -0.015, 0.015, -0.015],
        ]
    )
    assert np.allclose(P.mixer(), expected)
    assert P.rotor_xy is None and P.spin is None


def test_reordered_simulator_airframe_permutes_mixer_columns():
    order = (3, 0, 2, 1)
    af = simulator_airframe(order)
    assert np.allclose(af.params.mixer(), nominal_params().mixer()[:, list(order)])


def test_yaml_round_trip(tmp_path):
    af = simulator_airframe((3, 0, 2, 1))
    af.log = {"motor_range": [48, 2047]}
    path = tmp_path / "a.yaml"
    af.to_yaml(path, header="# test")
    b = Airframe.from_yaml(path)
    assert np.allclose(b.params.mixer(), af.params.mixer())
    assert b.params.mass == pytest.approx(af.params.mass)
    assert b.motor.k_f == pytest.approx(af.motor.k_f)
    assert b.log["motor_range"] == [48, 2047]
    u = np.array([0.0, 0.5, 1.0, np.nan])
    assert np.allclose(b.motor.speed(u), [250.0, 1375.0, 2500.0, 0.0])
    assert np.allclose(b.motor.command(b.motor.speed(u[:3])), u[:3])


def test_shipped_airframe_files_load():
    root = Path(__file__).resolve().parents[1]
    for path in ("config/5inch.yaml", "sitl/config/x500.yaml"):
        af = Airframe.from_yaml(root / path)
        B = af.params.mixer()
        assert np.linalg.matrix_rank(B) == 4
        assert 0.0 < af.hover_command < 1.0


BASE = """<?xml version="1.0"?>
<sdf version="1.9"><model name="q">
  <link name="base_link"><inertial><mass>2.0</mass>
    <inertia><ixx>0.02</ixx><iyy>0.02</iyy><izz>0.04</izz></inertia></inertial></link>
  {rotors}
</model></sdf>"""
ROTOR = """<link name="rotor_{i}"><pose>{x} {y} 0.06 0 0 0</pose><inertial><mass>0.016</mass>
  <inertia><ixx>4e-7</ixx><iyy>2.6e-5</iyy><izz>2.65e-5</izz></inertia></inertial></link>"""
PLUGIN = """<plugin filename="gz-sim-multicopter-motor-model-system"
  name="gz::sim::systems::MulticopterMotorModel"><linkName>rotor_{i}</linkName>
  <turningDirection>{d}</turningDirection><timeConstantUp>0.0125</timeConstantUp>
  <timeConstantDown>0.025</timeConstantDown><maxRotVelocity>1000</maxRotVelocity>
  <motorConstant>8.54858e-06</motorConstant><momentConstant>0.016</momentConstant>
  <motorNumber>{i}</motorNumber><rotorDragCoefficient>8.06428e-05</rotorDragCoefficient>
  <rotorVelocitySlowdownSim>10</rotorVelocitySlowdownSim></plugin>"""


def test_gazebo_sdf_parser(tmp_path):
    pos = [(0.174, -0.174), (-0.174, 0.174), (0.174, 0.174), (-0.174, -0.174)]
    dirs = ["ccw", "ccw", "cw", "cw"]
    rotors = "\n".join(ROTOR.format(i=i, x=x, y=y) for i, (x, y) in enumerate(pos))
    plugins = "\n".join(PLUGIN.format(i=i, d=d) for i, d in enumerate(dirs))
    (tmp_path / "base.sdf").write_text(BASE.format(rotors=rotors))
    (tmp_path / "motors.sdf").write_text(
        f'<?xml version="1.0"?><sdf version="1.9"><model name="m">{plugins}</model></sdf>'
    )
    af = from_gazebo_sdf(tmp_path / "base.sdf", tmp_path / "motors.sdf", 9.8)
    p = af.params
    assert p.mass == pytest.approx(2.064)
    assert np.allclose(p.spin_dir, [-1, -1, 1, 1])
    assert np.allclose(p.rotor_positions[:, :2], pos)
    # parallel-axis terms about the composite centre of mass (slightly above the base)
    cz = 4 * 0.016 * 0.06 / 2.064
    jx = 0.02 + 2.0 * cz**2 + 4 * (4e-7 + 0.016 * (0.174**2 + (0.06 - cz) ** 2))
    assert p.inertia[0, 0] == pytest.approx(jx, rel=1e-9)
    assert p.rotor_inertia == pytest.approx(2.65e-6)
    assert af.motor.tau_up == 0.0125 and af.motor.tau_down == 0.025
    assert af.hover_speed == pytest.approx(np.sqrt(2.064 * 9.8 / 4 / 8.54858e-06))
