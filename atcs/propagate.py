"""
Uncertainty that cannot shrink.

THE CS REFRAMING
----------------
Treat the chain as a program whose values carry intervals, and impose a
MONOTONICITY INVARIANT: no composition may be more certain than its least certain
input. Stated that way it is obvious; in practice it is violated constantly,
because every stage of a multiscale pipeline is usually fitted independently and
each fit reports the uncertainty of ITS OWN residuals, discarding the uncertainty
of the input it was given.

The result is a four-stage chain whose final interval is narrower than its first
stage's -- a pipeline that becomes more confident the further it gets from the
data. The 20 October briefing names the requirement ("uncertainty propagates
rather than being re-estimated at each stage"); this module makes re-estimation
impossible rather than discouraged.

WHAT IS AND IS NOT CLAIMED
--------------------------
The arithmetic here is first-order: relative uncertainties combine in quadrature
for products of independent terms, and additively in the worst case for dependent
ones. That is a BOUND, not a distribution, and the module says so. The point is
not precision in the propagated width -- it is that the width cannot decrease,
which is a structural property no amount of distributional sophistication changes.

Dependence is required rather than defaulted: assuming independence is the
commonest way a propagated interval comes out too narrow, and in this chain the
stages are emphatically not independent (the same study offset enters every one).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Sequence


class Dependence(Enum):
    """How two uncertain quantities are related. There is no default."""

    INDEPENDENT = "independent"      # quadrature
    DEPENDENT = "dependent"          # linear worst case
    UNKNOWN = "unknown"              # linear worst case, and flagged


class MonotonicityViolation(RuntimeError):
    """A composition came out more certain than its least certain input."""


@dataclass(frozen=True)
class Uncertain:
    """A value with a relative uncertainty and the stages that produced it."""

    value: float
    rel_sd: float                     # relative standard deviation, dimensionless
    name: str
    stages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.rel_sd < 0:
            raise ValueError("relative uncertainty cannot be negative")
        if self.value == 0 and self.rel_sd > 0:
            raise ValueError(
                f"{self.name}: a relative uncertainty on a zero value is undefined; "
                f"use an absolute interval or rescale")

    @property
    def interval(self) -> tuple[float, float]:
        d = abs(self.value) * self.rel_sd
        return self.value - 1.96 * d, self.value + 1.96 * d

    @property
    def fold_width(self) -> float:
        """Width as a multiplicative factor, which is how delivery data is read.

        Returns inf when the interval crosses zero. That is not a failure mode to
        be papered over: a relative SD above roughly 0.51 puts the 95% lower bound
        below zero for a positive quantity, at which point a MULTIPLICATIVE width
        is undefined rather than large. The honest reading is that a quantity this
        uncertain should be carried on the log scale, where the interval is
        symmetric and the fold-width is always defined -- which is also how this
        literature reports delivery. `width_note` says which case you are in.
        """
        lo, hi = self.interval
        return float("inf") if lo <= 0 else hi / lo

    @property
    def fold_width_defined(self) -> bool:
        return self.interval[0] > 0

    @property
    def width_note(self) -> str:
        if self.fold_width_defined:
            return f"{self.fold_width:.1f}x multiplicative width"
        return (f"relative SD {self.rel_sd:.2f} puts the 95% lower bound at or below "
                f"zero, so a multiplicative fold-width is undefined, not infinite; "
                f"carry this quantity on the log scale instead")


def product(name: str, a: Uncertain, b: Uncertain, dependence: Dependence) -> Uncertain:
    """Multiply two uncertain quantities, propagating rather than re-estimating.

    `dependence` is positional and has no default. In this chain the stages share a
    study offset, so INDEPENDENT is usually the wrong assumption and defaulting to
    it would silently narrow every interval in the pipeline.
    """
    if dependence is Dependence.INDEPENDENT:
        rel = math.hypot(a.rel_sd, b.rel_sd)
    else:
        rel = a.rel_sd + b.rel_sd
    out = Uncertain(a.value * b.value, rel, name,
                    stages=a.stages + b.stages + (name,))
    _check_monotone(out, (a, b))
    return out


def chain(name: str, stages: Sequence[Uncertain], dependence: Dependence) -> Uncertain:
    """Compose a whole pipeline. The invariant is checked at every step, not only
    at the end, so the stage that broke it can be named."""
    if not stages:
        raise ValueError("an empty chain has no uncertainty to propagate")
    # Seed the accumulator's stage list with the FIRST stage's own name. An
    # earlier version started from `stages[0]` unchanged, whose `.stages` is empty
    # for a leaf, so the first stage vanished from the attribution -- defeating the
    # only thing the field exists for. Caught by a test asserting the first stage
    # was present.
    acc = replace(stages[0], stages=stages[0].stages + (stages[0].name,))
    for s in stages[1:]:
        acc = product(f"{name}:{s.name}", acc, s, dependence)
    return Uncertain(acc.value, acc.rel_sd, name, stages=acc.stages)


def _check_monotone(out: Uncertain, inputs: Sequence[Uncertain]) -> None:
    worst = max(i.rel_sd for i in inputs)
    if out.rel_sd < worst - 1e-12:
        culprit = max(inputs, key=lambda i: i.rel_sd)
        raise MonotonicityViolation(
            f"{out.name!r} has relative uncertainty {out.rel_sd:.4f}, below its least "
            f"certain input {culprit.name!r} at {culprit.rel_sd:.4f}. A composition "
            f"cannot be more certain than what it is composed of; this is the "
            f"signature of a stage re-estimating uncertainty from its own residuals "
            f"instead of propagating its input's.")


def reestimate_check(declared: float, propagated: Uncertain) -> dict:
    """Compare a stage's SELF-REPORTED uncertainty against the propagated bound.

    This is the audit to run on an existing multiscale pipeline. A stage reporting
    a tighter interval than its inputs permit is not better calibrated -- it has
    dropped its input's uncertainty on the floor, and the ratio quantifies by how
    much.
    """
    ratio = propagated.rel_sd / declared if declared > 0 else float("inf")
    return {
        "declared_rel_sd": declared,
        "propagated_rel_sd": round(propagated.rel_sd, 5),
        "understatement_factor": round(ratio, 2),
        "declared_fold": round(Uncertain(propagated.value, declared, "declared").fold_width, 2)
                         if propagated.value > 0 else None,
        "propagated_fold": round(propagated.fold_width, 2) if propagated.value > 0 else None,
        "verdict": ("declared interval is consistent with propagation"
                    if ratio <= 1.0 + 1e-9 else
                    f"declared interval is {ratio:.1f}x too narrow: the stage reports "
                    f"the uncertainty of its own residuals, not of its inputs"),
    }
