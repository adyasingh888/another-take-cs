"""
Stage 2: multi-organ PBPK where mass conservation is exact BY CONSTRUCTION.

THE MATHEMATICAL CLAIM
----------------------
The brief states the conservation law

    Σ_m A_m(t) + A_cleared(t) = D_0    for all t

and says it is "computed deterministically by the mathematical engine" rather than
enforced by a penalty. This module is that engine, and the construction is what
makes the claim true rather than approximately true.

HOW
---
Write the compartment amounts as a linear system

    dA/dt = K A,    A(0) = D_0 e_plasma

and build K as a METZLER matrix with ZERO COLUMN SUMS:

    K_ij >= 0  for i != j        (transfers are non-negative)
    Σ_i K_ij = 0  for every j     (what leaves j arrives somewhere)

Two consequences follow for free, and they are the whole point:

  1. Zero column sums make 1^T K = 0, so d(1^T A)/dt = 0 and total amount is
     conserved EXACTLY -- as an algebraic identity, not a fitted property. The
     solution A(t) = exp(Kt) A(0) inherits it because exp(Kt) has unit column sums
     whenever K has zero column sums.

  2. Metzler structure (non-negative off-diagonals) makes exp(Kt) non-negative, so
     no compartment can go negative. A negative organ amount is impossible rather
     than clipped.

Elimination is NOT a leak out of the system. It is a transfer into an ABSORBING
`cleared` compartment, which is the only way Σ A_m + A_cleared = D_0 can hold with
a drug that is actually eliminated. An audit of this programme's earlier work
raised exactly this: a generator with zero column sums conserves total mass, so
"nothing is ever eliminated" unless one compartment is a sink. The sink is named
here, and `cleared_fraction()` reports it alongside the organ trajectories.

WHAT THIS DOES NOT CLAIM
------------------------
Conservation is a structural guarantee, not a biological result. It says the model
cannot invent or destroy material; it says nothing about whether the rate
constants are right. Reporting "0.0% mass-balance violation" as an empirical
finding would be reporting that arithmetic works -- the brief's own audit makes
this point and it is repeated here because the temptation is strong.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy.linalg import expm


class ConstructionError(ValueError):
    """The generator does not satisfy the structure the guarantees depend on."""


#: The compartments of the brief's 6-organ benchmark, plus plasma and the sink.
ORGANS = ("liver", "spleen", "kidney", "lung", "heart", "tumour")
COMPARTMENTS = ("plasma",) + ORGANS + ("cleared",)


@dataclass(frozen=True)
class Generator:
    """A validated Metzler, zero-column-sum generator.

    Built from uptake and return rates rather than written out, because writing a
    matrix by hand and then asserting its column sums is how a sign error survives
    to the results chapter.
    """

    compartments: tuple[str, ...]
    K: np.ndarray

    def __post_init__(self) -> None:
        K = self.K
        n = len(self.compartments)
        if K.shape != (n, n):
            raise ConstructionError(f"K is {K.shape}, expected {(n, n)}")
        off = K - np.diag(np.diag(K))
        if (off < -1e-12).any():
            i, j = np.unravel_index(np.argmin(off), off.shape)
            raise ConstructionError(
                f"K[{self.compartments[i]},{self.compartments[j]}] = {off[i, j]:.3g} < 0. "
                f"A negative off-diagonal is a negative transfer rate; it breaks "
                f"non-negativity of exp(Kt) and lets a compartment go negative.")
        colsum = K.sum(axis=0)
        bad = np.abs(colsum) > 1e-10
        if bad.any():
            j = int(np.argmax(np.abs(colsum)))
            raise ConstructionError(
                f"column {self.compartments[j]!r} sums to {colsum[j]:.3g}, not 0. "
                f"A non-zero column sum means material leaves that compartment "
                f"without arriving anywhere -- mass conservation is then a fitted "
                f"property rather than an identity.")

    def solve(self, dose: float, t_h: Sequence[float],
              start: str = "plasma") -> dict[str, np.ndarray]:
        """A(t) = exp(Kt) A(0) by matrix exponential.

        The exponential rather than a stepper: conservation then holds to machine
        precision at every requested time, with no dependence on step size or
        tolerance. A Runge-Kutta solution conserves mass only to its own error
        tolerance, which is the difference between a guarantee and a setting.
        """
        if start not in self.compartments:
            raise ConstructionError(f"unknown start compartment {start!r}")
        A0 = np.zeros(len(self.compartments))
        A0[self.compartments.index(start)] = float(dose)
        out = {c: np.empty(len(t_h)) for c in self.compartments}
        for k, t in enumerate(t_h):
            if t < 0:
                raise ConstructionError("negative time")
            A = expm(self.K * float(t)) @ A0
            for i, c in enumerate(self.compartments):
                out[c][k] = A[i]
        return out

    def residual(self, dose: float, t_h: Sequence[float]) -> float:
        """Worst absolute conservation residual over the requested times."""
        sol = self.solve(dose, t_h)
        tot = sum(sol[c] for c in self.compartments)
        return float(np.max(np.abs(tot - dose)))


def build_generator(uptake: Mapping[str, float],
                    release: Mapping[str, float],
                    clearance_per_h: float,
                    organs: Sequence[str] = ORGANS) -> Generator:
    """Assemble a valid generator from physiological rates.

    `uptake[o]`  plasma -> organ o, per hour
    `release[o]` organ o -> plasma, per hour
    `clearance_per_h` plasma -> cleared (the absorbing sink), per hour

    Diagonals are SET to minus the column's outflow rather than supplied, so the
    zero-column-sum property holds by construction instead of by assertion. There
    is no way to call this function and get a leaky generator.
    """
    comps = ("plasma",) + tuple(organs) + ("cleared",)
    idx = {c: i for i, c in enumerate(comps)}
    n = len(comps)
    K = np.zeros((n, n))
    if clearance_per_h < 0:
        raise ConstructionError("clearance rate cannot be negative")
    for o in organs:
        u, r = float(uptake.get(o, 0.0)), float(release.get(o, 0.0))
        if u < 0 or r < 0:
            raise ConstructionError(f"negative rate for {o!r}: uptake {u}, release {r}")
        K[idx[o], idx["plasma"]] += u            # into organ
        K[idx["plasma"], idx[o]] += r            # back to plasma
    K[idx["cleared"], idx["plasma"]] += float(clearance_per_h)
    # diagonals absorb each column's total outflow: this is the construction
    for j in range(n):
        K[j, j] = 0.0
        K[j, j] = -(K[:, j].sum())
    return Generator(comps, K)


def organ_shares(sol: Mapping[str, np.ndarray], dose: float,
                 organs: Sequence[str] = ORGANS) -> dict[str, np.ndarray]:
    """Percent of injected dose per organ -- the quantity the literature reports.

    Returned as %ID, explicitly NOT %ID/g: converting between them needs an organ
    mass, and conflating the two is a documented error in this literature.
    """
    return {o: 100.0 * sol[o] / dose for o in organs}


def cleared_fraction(sol: Mapping[str, np.ndarray], dose: float) -> np.ndarray:
    """The sink's trajectory, reported alongside the organs.

    Exposed as a first-class output because conservation without a named sink is
    a model in which nothing is ever eliminated -- true of the mathematics and
    false of the biology.
    """
    return sol["cleared"] / dose
