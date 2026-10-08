"""Actuator fault model: multiplicative loss of effectiveness (LOE) per rotor.

Rotor ``i`` produces ``eta_i * f_i`` where ``f_i`` is the healthy thrust and
``eta_i in [0, 1]`` (``eta_i = 0`` is a complete loss, e.g. a broken propeller or a
dead ESC channel). The same factor scales the rotor's drag torque. Faults are steps
with a configurable onset time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RotorFault:
    rotor: int  # 0-based index
    magnitude: float  # loss of effectiveness, 0 = healthy, 1 = complete loss
    onset: float  # s

    def __post_init__(self):
        if not 0 <= self.rotor < 4:
            raise ValueError("rotor index must be 0..3")
        if not 0.0 <= self.magnitude <= 1.0:
            raise ValueError("magnitude must be in [0, 1]")

    @property
    def eta(self) -> float:
        return 1.0 - self.magnitude


@dataclass
class FaultSchedule:
    faults: list[RotorFault] = field(default_factory=list)

    def eta(self, t: float) -> np.ndarray:
        e = np.ones(4)
        for f in self.faults:
            if t >= f.onset:
                e[f.rotor] = min(e[f.rotor], f.eta)
        return e

    @property
    def first_onset(self) -> float | None:
        return min((f.onset for f in self.faults), default=None)

    @classmethod
    def none(cls) -> FaultSchedule:
        return cls([])

    @classmethod
    def single(cls, rotor: int, magnitude: float, onset: float) -> FaultSchedule:
        if magnitude <= 0.0:
            return cls([])
        return cls([RotorFault(rotor, magnitude, onset)])
