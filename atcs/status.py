"""
Evidential status: the brief's own four-label vocabulary, made comparable.

WHY THIS EXISTS HERE
--------------------
The 20 October brief's evidence section separates "independently executed audit
findings from project-reported benchmarks and proposed biological capabilities",
and uses a four-label vocabulary throughout. This module implements that vocabulary
from the brief.

It is deliberately duplicated rather than imported from the sibling `mesoloop`
repository. The shared source is the brief, not either package, and a cross-repo
import would mean neither installs standalone -- which for a prototype someone is
meant to clone and run is a worse problem than two short enum definitions.

WHY IT IS AN ORDERED ENUM
-------------------------
Because the useful operations are comparisons. "At least REPORTED" is a real
requirement a registry needs to express, and a composition's status is the MINIMUM
over its parts -- a well-evidenced front end cannot raise a proposed handoff, which
is the formal content of the brief's warning that a connected diagram is a
hypothesis until each material handoff is validated.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class StatusViolation(RuntimeError):
    """A status was declared without the evidence a status requires."""


class Status(IntEnum):
    """Ordered weakest to strongest, so `>=` and `min` mean what they look like."""

    PROPOSED = 1       # named and scoped; no supporting measurement
    ILLUSTRATIVE = 2   # code runs; its numbers are not evidence
    REPORTED = 3       # the project measured it; not independently reproduced
    INDEPENDENT = 4    # reproduced by someone who did not build it


#: What each status licenses. Quoted into error messages so a refusal explains
#: itself at the point of refusal rather than in documentation.
LICENCE: dict[Status, str] = {
    Status.PROPOSED: ("may appear in a research plan and in a figure labelled as a "
                      "proposal; may not appear in a results section"),
    Status.ILLUSTRATIVE: ("may demonstrate that a computation is possible; its "
                          "numbers may not be quoted as findings, even with a "
                          "caveat attached"),
    Status.REPORTED: ("may be quoted with its source named; may not be presented as "
                      "established, because nobody outside the project has run it"),
    Status.INDEPENDENT: ("may be quoted as a finding; still carries the limits of "
                         "the data it was measured on"),
}


@dataclass(frozen=True)
class Evidenced:
    """A status with the basis for it, and what would raise it."""

    status: Status
    basis: str
    what_would_raise_it: str = ""

    def __post_init__(self) -> None:
        if len(self.basis) < 15:
            raise StatusViolation(
                f"status {self.status.name} does not name a source. A label without "
                f"a basis is a claim about evidence that cannot itself be checked.")
        if self.status is not Status.INDEPENDENT and not self.what_would_raise_it:
            raise StatusViolation(
                f"{self.status.name} must state what would raise it. A research "
                f"programme is distinguished from a prospectus by naming the "
                f"measurement that would change each claim's standing.")

    @property
    def licence(self) -> str:
        return LICENCE[self.status]


def composition_status(*parts: Evidenced | Status) -> Status:
    """The status of a chain is the status of its WEAKEST handoff.

    Not the average and not the headline. A chain with four independent stages and
    one proposed handoff is proposed, because the proposed handoff is where the
    chain breaks and the other four cannot carry it.
    """
    if not parts:
        raise StatusViolation("a composition of nothing has no status")
    return min(p.status if isinstance(p, Evidenced) else p for p in parts)


def require(evidence: Evidenced, minimum: Status, claim: str) -> None:
    """Refuse a claim its evidence does not license."""
    if evidence.status < minimum:
        raise StatusViolation(
            f"claim {claim!r} requires {minimum.name} evidence; this is "
            f"{evidence.status.name}, which {evidence.licence}. "
            f"Raise it by: {evidence.what_would_raise_it or 'unspecified'}.")
