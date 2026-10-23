"""
Structural observability: is a quantity recoverable from the measurements you can
actually make?

THE CS REFRAMING
----------------
The nanomedicine literature asks "can we predict organ delivery better?" That is a
statistics question and its answer is bounded by a variance decomposition nobody
can move. The computer-science question underneath is different and sharper:

    given the set of measurements an experiment can produce, is the parameter of
    interest DETERMINED, or does an entire family of parameter values produce
    identical observations?

If the second, no model, no architecture and no quantity of data recovers it. That
is structural non-identifiability, and it is decidable from the model's wiring
alone -- before any data exists, at zero experimental cost. Control theory calls
it observability; program analysis calls it a dependence question; either way it is
a property of the graph, not of the fit.

WHY THIS MATTERS HERE SPECIFICALLY
----------------------------------
The organ-to-cytosol interface requires a target-cell partition fraction. Whole-
organ measurement gives one number -- total organ-associated material -- while the
interface needs that number SPLIT between parenchymal cells, resident macrophages,
endothelium and extracellular space. One equation, four unknowns. Adding model
capacity cannot help; the shortfall is in the measurement set.

This module decides that mechanically, so an interface is classified before it is
modelled rather than after a year of fitting.

METHOD
------
A structural model is a bipartite graph of unknowns and observation equations.
Each equation states which unknowns it constrains. Local structural identifiability
of the whole system is then a rank condition on the incidence matrix; per-parameter
identifiability is whether that parameter appears in a maximal independent set.

Pure structure, deliberately: the result holds for EVERY parameter value rather
than at a nominal point, which is what makes it answerable before data. A system
that passes structurally may still be practically unidentifiable at realistic noise
-- `practical_margin` addresses that separately and is honest about being a
different question.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, Mapping, Sequence


class Unobservable(ValueError):
    """A quantity cannot be recovered from the declared measurement set."""


@dataclass(frozen=True)
class Observation:
    """One measurable equation.

    `constrains` is the set of unknowns this measurement says something about.
    `is_achievable` records whether the measurement can actually be made with
    available methods -- an equation that closes the system but requires an assay
    nobody has is recorded, not quietly counted.
    """

    name: str
    constrains: frozenset[str]
    is_achievable: bool = True
    method: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if not self.constrains:
            raise ValueError(f"observation {self.name!r} constrains nothing")


@dataclass
class StructuralModel:
    """Unknowns plus the observations available about them."""

    unknowns: tuple[str, ...]
    observations: tuple[Observation, ...]
    constraints: tuple[frozenset[str], ...] = ()      # e.g. a simplex: fractions sum to 1
    name: str = ""

    def _rows(self, achievable_only: bool = True) -> list[frozenset[str]]:
        rows = [o.constrains for o in self.observations
                if o.is_achievable or not achievable_only]
        return rows + list(self.constraints)

    # ---------------------------------------------------------------- counting
    def degrees_of_freedom(self, achievable_only: bool = True) -> int:
        """Unknowns minus independent equations. Positive means underdetermined.

        Independence is counted structurally: two equations constraining exactly
        the same unknown set are not independent information about the split
        between them, however different their numeric values.
        """
        rows = {r for r in self._rows(achievable_only)}
        return len(self.unknowns) - len(rows)

    def identifiable(self, parameter: str, achievable_only: bool = True
                     ) -> tuple[bool, str]:
        """Is this one parameter determined?

        A parameter is structurally identifiable only if some achievable equation
        pins it either alone or jointly with parameters that are themselves pinned.
        The fixed point below is the transitive closure of that rule: start from
        singletons and repeatedly resolve any equation with exactly one remaining
        unknown.
        """
        if parameter not in self.unknowns:
            raise KeyError(f"{parameter!r} is not an unknown of {self.name or 'the model'}")
        rows = self._rows(achievable_only)
        resolved: set[str] = set()
        changed = True
        while changed:
            changed = False
            for r in rows:
                remaining = r - resolved
                if len(remaining) == 1:
                    resolved |= remaining
                    changed = True
        if parameter in resolved:
            return True, "determined by the achievable measurement set"
        blocking = [sorted(r) for r in rows if parameter in r]
        if not blocking:
            return False, (f"{parameter!r} appears in no achievable equation: it is not "
                           f"measured, constrained or inferable, so any value is "
                           f"consistent with every observation")
        smallest = min(blocking, key=len)
        return False, (f"{parameter!r} appears only in equations with other unresolved "
                       f"unknowns; the tightest is over {smallest} "
                       f"({len(smallest)} unknowns, 1 equation), so a one-parameter "
                       f"family of values produces identical observations")

    def report(self, achievable_only: bool = True) -> dict:
        per = {u: self.identifiable(u, achievable_only) for u in self.unknowns}
        unid = sorted(u for u, (ok, _) in per.items() if not ok)
        return {
            "model": self.name,
            "n_unknowns": len(self.unknowns),
            "n_equations": len({r for r in self._rows(achievable_only)}),
            "degrees_of_freedom": self.degrees_of_freedom(achievable_only),
            "identifiable": sorted(u for u, (ok, _) in per.items() if ok),
            "unidentifiable": unid,
            "reasons": {u: per[u][1] for u in unid},
            "verdict": ("determined" if not unid else
                        f"{len(unid)} of {len(self.unknowns)} unknowns are structurally "
                        f"unidentifiable from the achievable measurement set"),
        }

    # ------------------------------------------------------- what would fix it
    def minimal_additional_observations(self, target: str,
                                        candidates: Sequence[Observation]
                                        ) -> list[tuple[str, ...]]:
        """The smallest sets of candidate measurements that would make `target`
        identifiable.

        This is the module's payload. Declaring a parameter unidentifiable is only
        half an answer; the actionable half is which measurement closes it, and the
        search returns ALL minimal sets rather than one, because they differ in
        cost and feasibility and that is the experimenter's trade-off to make.
        """
        ok, _ = self.identifiable(target)
        if ok:
            return []
        out: list[tuple[str, ...]] = []
        for k in range(1, len(candidates) + 1):
            for combo in combinations(candidates, k):
                trial = StructuralModel(self.unknowns,
                                        self.observations + tuple(combo),
                                        self.constraints, self.name)
                if trial.identifiable(target)[0]:
                    names = tuple(c.name for c in combo)
                    if not any(set(prev) <= set(names) for prev in out):
                        out.append(names)
            if out:
                break          # minimal cardinality reached; do not grow further
        return out
