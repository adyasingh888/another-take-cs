"""
Provenance as taint: measuredness does not launder.

THE CS REFRAMING
----------------
Scientific provenance is usually a metadata field -- a string saying "simulated"
or "measured" that travels beside a value and is read by whoever remembers to
look. Treat it instead as a TAINT LATTICE with a propagation rule, and the
property becomes structural: a result computed from any simulated input is itself
simulated, and no amount of downstream measured data cleans it.

This is the same discipline as information-flow control. The failure it prevents
is specific and has occurred repeatedly in this programme's own history: a figure
captioned as a statistical validation whose inputs were typed literals; a
conservation rate assigned a constant for one arm and measured for others; a
trajectory assembled from unpaired corpora and presented as an experimental
series. In each case a metadata field would have said the right thing if anyone
had propagated it. A lattice propagates it whether or not anyone looks.

THE LATTICE
-----------
    MEASURED  <  DERIVED  <  SIMULATED  <  ASSUMED  <  ILLUSTRATIVE

Join is max: combining a measured value with an assumed one yields assumed. There
is no operation that lowers a taint, because lowering it is exactly the error.
A measurement that CALIBRATES a simulation does not clean the simulation -- it
produces a new, separately-tainted calibrated object, and `calibrate()` is written
to make that explicit rather than to launder.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Any, Iterable, Sequence


class Taint(IntEnum):
    """Ordered so that `max` is the join. Higher is weaker evidence."""

    MEASURED = 0        # an instrument produced this number
    DERIVED = 1         # computed from measured inputs by a stated mechanism
    SIMULATED = 2       # produced by a model, however well-validated
    ASSUMED = 3         # a value chosen because one was needed
    ILLUSTRATIVE = 4    # a placeholder, explicitly not a quantity


#: What each level may be used for, and what it may not.
USE = {
    Taint.MEASURED: "may appear in a results table as an observation",
    Taint.DERIVED: ("may appear in a results table WITH its derivation chain; the "
                    "chain is part of the claim, not a footnote"),
    Taint.SIMULATED: ("may be shown as a model output and compared against "
                      "observations; may NOT be reported as an observation"),
    Taint.ASSUMED: ("may appear only in a methods section with the assumption "
                    "stated and a sensitivity analysis over it"),
    Taint.ILLUSTRATIVE: "may not appear in any results artefact in any form",
}


class TaintViolation(RuntimeError):
    """A value was used at an evidential level above its taint."""


@dataclass(frozen=True)
class Tainted:
    """A value carrying its provenance and the chain that produced it."""

    value: Any
    taint: Taint
    origin: str
    parents: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.origin.strip():
            raise TaintViolation(
                "a tainted value must name its origin; an unnamed origin is how a "
                "typed literal becomes indistinguishable from a measurement")

    @property
    def chain(self) -> str:
        return self.origin if not self.parents else \
            f"{self.origin} <- [{', '.join(self.parents)}]"


def combine(name: str, *vals: Tainted, op: str = "combined") -> Tainted:
    """Join. The result is as weak as the weakest input.

    `op` is recorded rather than inferred, because 'what was done to these numbers'
    is the part a reader needs and the part a metadata field never carries.
    """
    if not vals:
        raise TaintViolation("combining nothing produces nothing to attribute")
    t = max(v.taint for v in vals)
    worst = [v.origin for v in vals if v.taint == t]
    return Tainted(value=None, taint=t,
                   origin=f"{name} ({op}; taint from {', '.join(worst)})",
                   parents=tuple(v.chain for v in vals))


def calibrate(sim: Tainted, against: Tainted, name: str) -> Tainted:
    """Calibrating a simulation against a measurement does NOT make it measured.

    Written as its own function because this is the single most tempting
    laundering operation in the field: a simulator tuned to data feels like data.
    It is not. The calibrated object is a new simulated object whose parameters
    were fitted -- and `fitted to` is recorded in the origin so a reader can see
    the measurement was consumed as a target rather than reported as a result.
    """
    if sim.taint < Taint.SIMULATED:
        raise TaintViolation(
            f"calibrate() expects a simulated object; {sim.origin!r} is "
            f"{sim.taint.name}. Calibrating a measurement against a measurement is a "
            f"comparison, not a calibration -- use combine().")
    if against.taint != Taint.MEASURED:
        raise TaintViolation(
            f"calibration target {against.origin!r} is {against.taint.name}, not "
            f"MEASURED. Fitting a simulation to another simulation's output "
            f"propagates its assumptions and reports them as agreement.")
    return Tainted(value=None, taint=Taint.SIMULATED,
                   origin=f"{name} (simulated, fitted to {against.origin})",
                   parents=(sim.chain, against.chain))


def require(v: Tainted, maximum: Taint, claim: str) -> None:
    """Gate a claim on a value's taint."""
    if v.taint > maximum:
        raise TaintViolation(
            f"claim {claim!r} requires taint <= {maximum.name}, but this value is "
            f"{v.taint.name}. {USE[v.taint]} Chain: {v.chain}")


def audit(vals: Iterable[Tainted]) -> dict:
    """Taint census over a set of values, for a methods section.

    Reports the WORST taint and what carries it, because an artefact's evidential
    standing is set by its weakest input and an average would hide that.
    """
    vs = list(vals)
    if not vs:
        return {"n": 0}
    worst = max(v.taint for v in vs)
    counts = {t.name: sum(1 for v in vs if v.taint == t) for t in Taint}
    return {
        "n": len(vs),
        "counts": {k: n for k, n in counts.items() if n},
        "worst": worst.name,
        "worst_carriers": [v.origin for v in vs if v.taint == worst],
        "artefact_standing": USE[worst],
        "note": ("An artefact's evidential standing is its WEAKEST input, not its "
                 "average. A single assumed parameter in a chain of measurements "
                 "makes the output assumed."),
    }
