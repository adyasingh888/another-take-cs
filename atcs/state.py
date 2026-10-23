"""
Typed state groups and the Process-and-Store registry.

THE ARCHITECTURAL CLAIM, MADE CHECKABLE
---------------------------------------
The brief's §4 states that MesoReach is "a Modular Process-and-Store Registry
operating over extensible typed state groups rather than a rigid fixed-dimension
vector", so that "new biological assays, organs, or intracellular pathways can be
plugged in without rewriting the core architecture".

That is a falsifiable engineering claim, and this module is where it is tested
rather than asserted. The state is

    x = { x_id      corona identity        (simplex over adsorbed protein species)
          x_organ   multi-organ PBPK       (amounts, mass-conserving)
          x_barrier sub-organ / endosomal  (fractions through a barrier)
          x_cell    intracellular network  (concentrations and species)
          x_c       context                (study, patient, protocol) }

Each group declares an INVARIANT its values must satisfy. A process declares which
groups it reads and which it writes, and the registry refuses a process whose
output violates the target group's invariant. The extensibility claim then has a
test: adding an organ, a protein species or a pathway node must not require
editing this file, and `test_state.py` adds all three at runtime to prove it.

WHY INVARIANTS LIVE ON THE GROUP RATHER THAN THE PROCESS
-------------------------------------------------------
Because the failure mode is a process that is individually correct and jointly
wrong. A corona model that returns probabilities and a PBPK model that returns
amounts are each fine; a pipeline that feeds the first into the second as though
probabilities were amounts is the fraction-as-concentration defect again. Putting
the invariant on the store means the violation is caught at the write, not at the
eventual implausible result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np


class InvariantViolation(RuntimeError):
    """A store write violated the invariant its state group declares."""


class WiringError(RuntimeError):
    """A process was registered or run against groups it did not declare."""


@dataclass(frozen=True)
class StateGroup:
    """One typed group of the state vector.

    `invariant` returns (ok, reason). It is checked on EVERY write, which is the
    point: a group is a contract, not a namespace.
    """

    name: str
    description: str
    invariant: Callable[[Mapping[str, float]], tuple[bool, str]]
    unit: str = ""

    def check(self, values: Mapping[str, float]) -> None:
        ok, why = self.invariant(values)
        if not ok:
            raise InvariantViolation(f"{self.name}: {why}")


# --------------------------------------------------------------- invariants
def simplex(tol: float = 1e-8) -> Callable[[Mapping[str, float]], tuple[bool, str]]:
    """Non-negative and summing to one. Corona composition is a composition."""
    def _inv(v: Mapping[str, float]) -> tuple[bool, str]:
        if not v:
            return True, ""
        a = np.asarray(list(v.values()), float)
        if (a < -tol).any():
            return False, f"negative share: {min(v, key=v.get)!r} = {min(a):.3g}"
        s = float(a.sum())
        if abs(s - 1.0) > max(tol, 1e-6 * len(a)):
            return False, (f"shares sum to {s:.6f}, not 1. A composition whose parts "
                           f"do not sum to the whole is not a composition, and a "
                           f"downstream ratio computed from it is meaningless")
        return True, ""
    return _inv


def nonnegative_amounts(dose: float | None = None, tol: float = 1e-7
                        ) -> Callable[[Mapping[str, float]], tuple[bool, str]]:
    """Amounts are non-negative and, if a dose is declared, never exceed it.

    The dose ceiling is the cheap half of mass conservation: a compartment holding
    more than was injected is a bug no fit can excuse.
    """
    def _inv(v: Mapping[str, float]) -> tuple[bool, str]:
        for k, x in v.items():
            if x < -tol:
                return False, f"negative amount in {k!r}: {x:.3g}"
            if dose is not None and x > dose * (1 + 1e-6) + tol:
                return False, (f"{k!r} holds {x:.4g} of an injected {dose:.4g}: a "
                               f"compartment cannot contain more than was administered")
        return True, ""
    return _inv


def unit_interval(tol: float = 1e-9) -> Callable[[Mapping[str, float]], tuple[bool, str]]:
    """Barrier passage fractions live in [0, 1]."""
    def _inv(v: Mapping[str, float]) -> tuple[bool, str]:
        for k, x in v.items():
            if not (-tol <= x <= 1 + tol):
                return False, (f"{k!r} = {x:.4g} is outside [0, 1]; a passage fraction "
                               f"above one creates material at a barrier")
        return True, ""
    return _inv


def nonnegative_concentrations(tol: float = 1e-12
                               ) -> Callable[[Mapping[str, float]], tuple[bool, str]]:
    def _inv(v: Mapping[str, float]) -> tuple[bool, str]:
        for k, x in v.items():
            if x < -tol:
                return False, f"negative concentration in {k!r}: {x:.3g}"
        return True, ""
    return _inv


def unconstrained() -> Callable[[Mapping[str, float]], tuple[bool, str]]:
    """Context carries protocol and patient descriptors; no numeric invariant."""
    return lambda v: (True, "")


#: The five groups of the brief's state vector. Extending the state means adding a
#: KEY to a group or a new group here -- never editing a process.
GROUPS: dict[str, StateGroup] = {
    "x_id": StateGroup("x_id", "corona identity: share of adsorbed protein by species",
                       simplex(), unit="fraction of adsorbed protein"),
    "x_organ": StateGroup("x_organ", "multi-organ PBPK amounts, including a cleared sink",
                          nonnegative_amounts(), unit="nmol"),
    "x_barrier": StateGroup("x_barrier", "sub-organ and endosomal passage fractions",
                            unit_interval(), unit="fraction"),
    "x_cell": StateGroup("x_cell", "intracellular species concentrations",
                         nonnegative_concentrations(), unit="nmol/L"),
    "x_c": StateGroup("x_c", "context: study, protocol and patient descriptors",
                      unconstrained()),
}


# ------------------------------------------------------------------- store
class Store:
    """The typed state. Writes are checked against the group's invariant."""

    def __init__(self, groups: Mapping[str, StateGroup] | None = None) -> None:
        self.groups = dict(groups or GROUPS)
        self._data: dict[str, dict[str, float]] = {g: {} for g in self.groups}

    def write(self, group: str, values: Mapping[str, float]) -> None:
        if group not in self.groups:
            raise WiringError(f"no such state group {group!r}; have {sorted(self.groups)}")
        merged = {**self._data[group], **dict(values)}
        self.groups[group].check(merged)
        self._data[group] = merged

    def read(self, group: str) -> dict[str, float]:
        if group not in self.groups:
            raise WiringError(f"no such state group {group!r}")
        return dict(self._data[group])

    def add_group(self, g: StateGroup) -> None:
        """Extensibility, exercised. A new scale is a new group, not an edit here."""
        if g.name in self.groups:
            raise WiringError(f"group {g.name!r} already exists")
        self.groups[g.name] = g
        self._data[g.name] = {}

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {k: dict(v) for k, v in self._data.items()}


# ----------------------------------------------------------------- process
@dataclass(frozen=True)
class Process:
    """A transformation that declares what it reads and what it writes.

    The declaration is enforced: a process that reads a group it did not declare
    gets a WiringError, so an undeclared dependence cannot hide in the code.
    """

    name: str
    reads: tuple[str, ...]
    writes: str
    fn: Callable[..., Mapping[str, float]]
    note: str = ""

    def run(self, store: Store, **params) -> dict[str, float]:
        inputs = {}
        for g in self.reads:
            if g not in store.groups:
                raise WiringError(f"{self.name} reads undeclared group {g!r}")
            inputs[g] = store.read(g)
        out = self.fn(**inputs, **params)
        store.write(self.writes, out)
        return dict(out)


class Registry:
    """Processes plus the order they compose in."""

    def __init__(self) -> None:
        self._p: dict[str, Process] = {}

    def register(self, p: Process) -> Process:
        if p.name in self._p:
            raise WiringError(f"{p.name!r} already registered")
        if p.writes not in GROUPS and p.writes not in self._p:
            pass            # a process may write a group added at runtime
        self._p[p.name] = p
        return p

    def get(self, name: str) -> Process:
        if name not in self._p:
            raise WiringError(f"{name!r} not registered; have {sorted(self._p)}")
        return self._p[name]

    def run_chain(self, store: Store, order: Sequence[str], **params) -> Store:
        """Run processes in order, checking every intermediate write.

        Deliberately not a dataflow solver. The order is explicit because an
        implicit topological sort hides which stage produced which quantity, and
        that attribution is exactly what a multiscale result needs.
        """
        for n in order:
            p = self.get(n)
            kw = {k: v for k, v in params.items() if k in _fn_params(p.fn)}
            p.run(store, **kw)
        return store

    def wiring(self) -> list[dict]:
        return [{"process": p.name, "reads": "|".join(p.reads), "writes": p.writes,
                 "note": p.note} for p in self._p.values()]

    def __len__(self) -> int:
        return len(self._p)


def _fn_params(fn: Callable) -> set[str]:
    import inspect
    return set(inspect.signature(fn).parameters)
