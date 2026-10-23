"""
Loop A: the five-phase scientific bench chain.

WHY FIVE PHASES AND NOT A PIPELINE
----------------------------------
The brief's §04 is explicit that this is "not just a linear target -> drug chain:
evidence can force revision of the target, delivery system, or entire mechanistic
hypothesis". A pipeline cannot express that. A phase graph with REVISION EDGES
can, and the distinction is load-bearing: the thing that makes a research loop a
loop is that a late phase can invalidate an early one.

So each phase declares, besides what it produces, which earlier phases its
failure can send you back to. `revision_target()` then answers the question a
research plan actually needs: given this failure, what is the cheapest phase to
re-enter?

SIMULATION HAPPENS TWICE, AND THE TWO ARE DIFFERENT
---------------------------------------------------
The brief's §3.1 notes simulation occurs at Phase 2 AND Phase 4. These are not
the same operation and conflating them is a real error:

  Phase 2  target validation INSIDE the virtual cell, BEFORE any chemistry
           exists. Asks: if we could deliver this perfectly, would suppressing
           this target do anything? Delivery is assumed perfect by construction,
           so a Phase 2 result says nothing about reachability.
  Phase 4  full physical-to-cellular simulation of an actual candidate. Asks:
           does THIS formulation reach the target cell? Delivery is the question.

A Phase 2 success followed by a Phase 4 failure is the transport-limited case and
is informative. A Phase 2 result REPORTED as though it were Phase 4 is the
"assumes the therapeutic is already inside the cell" error the brief names in §1.
`PhaseResult.may_claim_reachability()` refuses it.

PHASE 5 IS INFERENCE, NOT A VICTORY LAP
---------------------------------------
Phase 5 is in-silico stress-testing plus Bayesian competing-mechanism inference.
Its output is a POSTERIOR OVER EXPLANATIONS, not a confirmation, and it is the
phase that decides which revision edge to take. A Phase 5 that returns a single
mechanism without comparing rivals has not done its job -- see `atcs.critic`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Mapping, Sequence

from .taint import Taint


class PhaseError(RuntimeError):
    """A phase was entered without what it requires, or its output misused."""


class Phase(IntEnum):
    """Ordered so that `<` means 'earlier', which revision edges depend on."""

    DISEASE_HYPOTHESIS = 1      # disease biology -> a causal target hypothesis
    TARGET_VALIDATION = 2       # simulation #1: virtual cell, delivery assumed
    CANDIDATE_DESIGN = 3        # carrier/cargo chemistry and formulation
    DELIVERY_SIMULATION = 4     # simulation #2: MesoReach, delivery is the question
    STRESS_AND_INFERENCE = 5    # ablation, stress tests, competing-mechanism posterior


#: What each phase consumes, produces, and what its failure can revise.
#: `revises` is the set of EARLIER phases a failure here can send you back to --
#: the edges that make this a loop rather than a chain.
SPEC: dict[Phase, dict] = {
    Phase.DISEASE_HYPOTHESIS: {
        "consumes": ("disease biology", "genetic and epidemiological evidence"),
        "produces": ("a causal target hypothesis", "a measurable disease endpoint"),
        "revises": (),
        "simulation": None,
        "gate": ("the endpoint must be measurable in the species the loop will use; "
                 "a target whose effect is only definable in humans cannot be "
                 "validated in Phase 2"),
    },
    Phase.TARGET_VALIDATION: {
        "consumes": ("a causal target hypothesis", "a cell model"),
        "produces": ("predicted response to perfect suppression",
                     "an effect-size floor worth pursuing"),
        "revises": (Phase.DISEASE_HYPOTHESIS,),
        "simulation": ("virtual cell, DELIVERY ASSUMED PERFECT -- this phase cannot "
                       "speak to reachability"),
        "gate": ("the predicted effect must exceed the measurement noise of the "
                 "disease endpoint, or no delivery system can rescue it"),
    },
    Phase.CANDIDATE_DESIGN: {
        "consumes": ("a validated target", "a payload modality"),
        "produces": ("a carrier/cargo formulation", "physicochemical descriptors"),
        "revises": (Phase.TARGET_VALIDATION,),
        "simulation": None,
        "gate": ("the formulation's descriptors must be measurable, or the "
                 "downstream model has no inputs"),
    },
    Phase.DELIVERY_SIMULATION: {
        "consumes": ("a formulation", "a biological context"),
        "produces": ("organ disposition", "target-cell exposure", "predicted response"),
        "revises": (Phase.CANDIDATE_DESIGN, Phase.TARGET_VALIDATION),
        "simulation": ("MesoReach four-stage chain, DELIVERY IS THE QUESTION"),
        "gate": ("the chain's interfaces must be identifiable from achievable "
                 "measurements, or the prediction is unfalsifiable"),
    },
    Phase.STRESS_AND_INFERENCE: {
        "consumes": ("predictions", "observations", "candidate mechanisms"),
        "produces": ("a posterior over competing mechanisms",
                     "an ablation table", "the next experiment"),
        "revises": (Phase.DISEASE_HYPOTHESIS, Phase.TARGET_VALIDATION,
                    Phase.CANDIDATE_DESIGN, Phase.DELIVERY_SIMULATION),
        "simulation": ("stress tests and perturbation of model assumptions"),
        "gate": ("a single mechanism without a comparison against rivals is not an "
                 "inference; the posterior must be over at least two explanations"),
    },
}


@dataclass(frozen=True)
class PhaseResult:
    """What a phase produced, and what it is therefore allowed to claim."""

    phase: Phase
    passed: bool
    quantities: Mapping[str, float]
    taint: Taint
    note: str = ""
    mechanisms_compared: int = 0

    def may_claim_reachability(self) -> tuple[bool, str]:
        """May this result be read as evidence the therapeutic reaches the target?

        Only Phase 4 can. A Phase 2 result assumes perfect delivery BY
        CONSTRUCTION, so reading it as reachability evidence is the structural
        error the brief names in its executive summary -- virtual cell models that
        'assume the therapeutic is already inside the target cell'.
        """
        if self.phase is Phase.TARGET_VALIDATION:
            return False, (
                "Phase 2 assumes perfect delivery by construction, so it establishes "
                "that suppressing the target WOULD matter -- not that any formulation "
                "reaches it. Reading it as reachability evidence is the error that "
                "motivates this whole programme.")
        if self.phase is Phase.DELIVERY_SIMULATION:
            return True, "Phase 4 simulates delivery, so reachability is its question"
        return False, f"Phase {int(self.phase)} does not model delivery at all"

    def revision_target(self) -> tuple[Phase | None, str]:
        """On failure, the cheapest earlier phase worth re-entering.

        'Cheapest' means latest, because re-entering an early phase discards
        everything after it. A Phase 4 failure goes to Phase 3 (reformulate) before
        Phase 2 (requestion the target), which is both cheaper and the right
        scientific order: exhaust delivery explanations before abandoning a target.
        """
        if self.passed:
            return None, "no failure to revise"
        edges = SPEC[self.phase]["revises"]
        if not edges:
            return None, ("Phase 1 has no earlier phase: a failed disease hypothesis "
                          "ends this line of work rather than revising it")
        target = max(edges)
        return target, (
            f"re-enter Phase {int(target)} first -- it is the latest revisable phase, "
            f"so it discards the least work, and exhausting delivery explanations "
            f"before abandoning a target is also the right scientific order")


def gate(phase: Phase, available: Sequence[str]) -> tuple[bool, list[str]]:
    """May this phase be entered with what is available?"""
    missing = [c for c in SPEC[phase]["consumes"] if c not in available]
    return not missing, missing


def validate_inference(result: PhaseResult, minimum_mechanisms: int = 2
                       ) -> PhaseResult:
    """Phase 5 must compare rivals. Enforced rather than requested."""
    if result.phase is not Phase.STRESS_AND_INFERENCE:
        return result
    if result.mechanisms_compared < minimum_mechanisms:
        raise PhaseError(
            f"Phase 5 compared {result.mechanisms_compared} mechanism(s); an "
            f"inference requires at least {minimum_mechanisms}. A single mechanism "
            f"reported without rivals is a hypothesis restated, not a posterior. "
            f"{SPEC[Phase.STRESS_AND_INFERENCE]['gate']}")
    return result


def loop_a_table() -> list[dict]:
    """The loop as a table, including its revision edges."""
    return [{"phase": int(p), "name": p.name,
             "simulation": SPEC[p]["simulation"] or "none",
             "produces": "; ".join(SPEC[p]["produces"]),
             "failure_revises": ", ".join(f"P{int(x)}" for x in SPEC[p]["revises"]) or "-",
             "gate": SPEC[p]["gate"]}
            for p in Phase]
