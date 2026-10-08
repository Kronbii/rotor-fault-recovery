"""Control allocation with saturation priorities.

Given a desired collective thrust ``T`` and body torques ``tau`` the allocator returns
rotor thrust commands ``f`` (nominal thrust units, i.e. before the loss-of-effectiveness
factor) that satisfy ``f_min <= f <= f_max`` and, in order of priority,

1. reproduce the roll/pitch torques exactly if at all possible (scaled down by the
   largest feasible factor ``beta`` otherwise),
2. reproduce the collective thrust, shifting it by the smallest feasible amount when
   rotors would saturate (an "air-mode"-like shift that protects attitude control),
3. reproduce the yaw torque with the largest remaining feasible factor ``alpha``.

The effectiveness model is ``[T, tau] = B diag(eta) f``; passing ``eta`` (estimated
or known) gives the reconfigured allocation used by the fault-tolerant controller.
All three steps are solved in closed form (each is a 1-D linear feasibility problem),
so the allocator is exact and cheap enough for 500 Hz Monte-Carlo simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rotations import np_clip


@dataclass
class AllocationResult:
    f: np.ndarray  # rotor thrust commands (N)
    beta: float  # roll/pitch scaling actually applied
    alpha: float  # yaw scaling actually applied
    thrust_shift: float  # change in collective thrust (N)


def priority_solve(
    mT: np.ndarray,
    T: float,
    h: np.ndarray,
    d: np.ndarray | None,
    lo: np.ndarray,
    hi: np.ndarray,
) -> AllocationResult:
    """Solve ``f = mT (T + dT) + beta h + alpha d`` within ``[lo, hi]``.

    ``mT`` (entries >= 0) is the thrust direction, ``h`` the roll/pitch part and
    ``d`` the (lowest-priority) yaw part. Maximises ``beta`` first, then minimises
    ``|dT|``, then maximises ``alpha``. Rotors with ``mT_i = 0`` (possible with three
    rotors, where the rotor diagonal to the failed one carries no collective thrust)
    only constrain ``beta`` through their upper bound and are clipped at the lower one.
    """
    g = mT * T
    # fast path: everything feasible
    full = g + h if d is None else g + h + d
    if np.all(full >= lo) and np.all(full <= hi):
        return AllocationResult(
            f=full, beta=1.0, alpha=1.0 if d is not None else 0.0, thrust_shift=0.0
        )
    P = mT > 1e-9
    beta = 1.0
    # rotors that the thrust shift cannot move: upper bound only
    Z = ~P
    if np.any(Z):
        hz = h[Z]
        up = hz > 1e-12
        if np.any(up):
            beta = min(beta, float(np.min((hi[Z][up] - g[Z][up]) / hz[up])))
    mP = mT[P]
    a_lo = (lo[P] - g[P]) / mP
    a_hi = (hi[P] - g[P]) / mP
    s = h[P] / mP
    # feasible iff for all i, k: a_lo_i - b s_i <= a_hi_k - b s_k
    coef = s[None, :] - s[:, None]
    rhs = a_hi[None, :] - a_lo[:, None]
    pos = coef > 1e-12
    if np.any(pos):
        beta = min(beta, float(np.min(rhs[pos] / coef[pos])))
    beta = max(beta, 0.0)
    lo_dT = float(np.max(a_lo - beta * s))
    hi_dT = float(np.min(a_hi - beta * s))
    if lo_dT > hi_dT:
        dT = 0.5 * (lo_dT + hi_dT)
    else:
        dT = min(max(0.0, lo_dT), hi_dT)
    f0 = g + beta * h + mT * dT
    alpha = 0.0
    if d is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            up = np.where(d > 1e-12, (hi - f0) / d, np.inf)
            dn = np.where(d < -1e-12, (lo - f0) / d, np.inf)
        alpha = min(max(min(1.0, float(up.min()), float(dn.min())), 0.0), 1.0)
        f0 = f0 + alpha * d
    return AllocationResult(f=np_clip(f0, lo, hi), beta=beta, alpha=alpha, thrust_shift=dT)


class PriorityAllocator:
    """Four-rotor allocator for ``[T, tau] = B diag(eta) f``."""

    def __init__(self, B: np.ndarray, f_min: float, f_max: float, eta_floor: float = 0.05):
        self.B = np.asarray(B, dtype=float)
        self.lo = np.full(4, float(f_min))
        self.hi = np.full(4, float(f_max))
        self.eta_floor = eta_floor
        self._cache_key = None
        self._M = None

    def inverse(self, eta: np.ndarray) -> np.ndarray:
        key = tuple(np.round(eta, 6))
        if key != self._cache_key:
            e = np.maximum(eta, self.eta_floor)
            self._M = np.linalg.inv(self.B * e[None, :])
            self._cache_key = key
        return self._M

    def allocate(
        self, T: float, tau: np.ndarray, eta: np.ndarray | None = None
    ) -> AllocationResult:
        if eta is None:
            eta = np.ones(4)
        M = self.inverse(np.asarray(eta, dtype=float))
        h = M[:, 1] * tau[0] + M[:, 2] * tau[1]
        d = M[:, 3] * tau[2]
        return priority_solve(M[:, 0], T, h, d, self.lo, self.hi)
