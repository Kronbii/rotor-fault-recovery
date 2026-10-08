"""Scenario factory shared by the experiments and the CLI."""

from __future__ import annotations

from .disturbance import DisturbanceParams
from .faults import FaultSchedule
from .params import SensorParams
from .sim import Scenario
from .trajectory import FigureEight, Hover, WaypointSteps

TRAJECTORIES = ("hover", "fig8", "steps")


def reference(name: str):
    if name == "hover":
        return Hover()
    if name == "fig8":
        return FigureEight()
    if name == "steps":
        return WaypointSteps()
    raise ValueError(f"unknown trajectory {name!r}")


def make_scenario(
    traj: str,
    duration: float,
    rotor: int = 0,
    magnitude: float = 0.0,
    onset: float = 2.0,
    gyro_scale: float = 1.0,
    turbulence: float = 1.0,
    uncertainty: bool = True,
) -> Scenario:
    return Scenario(
        name=traj,
        duration=duration,
        reference=reference(traj),
        faults=FaultSchedule.single(rotor, magnitude, onset),
        sensors=SensorParams().scaled(gyro_scale),
        disturbance=DisturbanceParams().scaled(turbulence),
        uncertainty=uncertainty,
    )
