"""
The planner: search over admissible model compositions, scored by correct refusal.

WHY A PLANNER AT ALL
--------------------
This is the module that answers the brief's originality boundary. Five prompt
personas do not make a new scientist; a search over compositions that is CONSTRAINED
BY TYPED INTERFACES does something a fixed workflow cannot, and the difference is
measurable.

The search is deliberately exhaustive and small. The point is not scale -- the
admissible space is tiny because the constraints are tight, which is itself the
finding. A fixed workflow explores one path through that space; a general agent
explores many and accepts invalid ones; this explores all of them and REJECTS the
invalid ones with a reason attached. The rejection log is the output that matters.

WHAT MAKES A COMPOSITION ADMISSIBLE
-----------------------------------
Four conditions, each checked by a different module, and a plan failing any one is
rejected rather than warned about:

  1. the output of each stage is an acceptable input to the next  (atcs.quantities)
  2. every model is used inside its validated context            (atcs.registry)
  3. the chain's provenance does not launder a simulation         (atcs.taint)
  4. the target quantity is identifiable from the measurements    (atcs.observability)

Condition 4 is the one that eliminates most candidates, and it is the one a
workflow built from "which models do we have" never checks at all.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .observability import StructuralModel
from .registry import ContextOfUse, ContextViolation, RegisteredModel, Registry
from .status import Status
from .taint import Taint


class PlanningError(RuntimeError):
    """A plan could not be assembled or scored."""


@dataclass(frozen=True)
class Candidate:
    """One ordered composition of registered models."""

    model_ids: tuple[str, ...]

    def __str__(self) -> str:             # pragma: no cover
        return " -> ".join(self.model_ids)


@dataclass(frozen=True)
class Rejection:
    """Why a candidate was eliminated. The planner's most useful output."""

    candidate: Candidate
    condition: str
    reason: str


def _chain_taint(ids: Sequence[str], stage_taint: Mapping[str, Taint]) -> Taint:
    """The weakest taint in the chain, which is the chain's taint.

    `max` directly: Taint is an ordered IntEnum whose maximum IS the lattice join.
    An earlier version passed `key=lambda t: list(Taint).index(t)`, which agreed only
    because the declaration order happens to match the values -- a redundant
    indirection that would disagree silently if either were ever reordered.
    """
    return max((stage_taint.get(i, Taint.ASSUMED) for i in ids), default=Taint.ASSUMED)


def enumerate_plans(registry: Registry, stages: Sequence[Sequence[str]]) -> list[Candidate]:
    """Every ordered composition picking one model per stage."""
    for i, opts in enumerate(stages):
        if not opts:
            raise PlanningError(
                f"stage {i} has no candidate models, so no plan exists. This is a "
                f"finding about model availability rather than a planner failure.")
    return [Candidate(tuple(c)) for c in itertools.product(*stages)]


def admissible_plans(registry: Registry, stages: Sequence[Sequence[str]],
                     *, stage_contexts: Sequence[Mapping[str, str]], decision: str,
                     stage_taint: Mapping[str, Taint],
                     max_taint: Taint,
                     structural: StructuralModel | None = None,
                     target: str | None = None,
                     minimum_status: Status = Status.ILLUSTRATIVE,
                     ) -> dict:
    """Search the space, returning admissible plans AND every rejection with a reason.

    `stage_contexts` is PER STAGE and must align with `stages`. An earlier version
    took one species and one compartment for the whole composition, which made every
    multi-stage plan inadmissible on context -- a chain crosses compartments by
    construction, that is what makes it a chain. The error was not harmless: it
    produced an empty admissible space attributed to context of use when the real
    binding constraint was elsewhere, which is exactly the wrong thing for a
    rejection log to say.

    The rejection log is returned first-class rather than discarded, because "the
    admissible space is empty and here is why, condition by condition" is a more
    useful answer than "no plan found" -- and it is the answer a research plan can
    act on.
    """
    if len(stage_contexts) != len(stages):
        raise PlanningError(
            f"{len(stage_contexts)} contexts for {len(stages)} stages. Each stage "
            f"operates in its own compartment and must be checked against its own "
            f"context; a single context for the whole chain cannot be right for a "
            f"chain that crosses compartments.")
    cands = enumerate_plans(registry, stages)
    ok: list[Candidate] = []
    rejected: list[Rejection] = []

    # EVERY condition is evaluated for every candidate, not just the first failing
    # one. An earlier version returned on the first failure, which made the
    # "dominant rejection condition" an artefact of check ORDER rather than of
    # which constraint actually binds -- a candidate failing provenance was never
    # tested for identifiability, so a rejection log whose whole purpose is to say
    # what is blocking the work reported the wrong blocker. Found by a figure whose
    # title asserted identifiability emptied the space while the data said
    # provenance, because provenance happened to be checked first.
    for c in cands:
        fails: list[tuple[str, str]] = []

        for mid, ctx in zip(c.model_ids, stage_contexts):
            try:
                registry.get(mid)
            except KeyError as e:
                fails.append(("context_of_use", str(e)))
                continue
            adm = registry.admissible(species=ctx["species"],
                                      compartment=ctx["compartment"],
                                      minimum_status=minimum_status)
            if mid not in adm:
                fails.append(("context_of_use",
                              f"{mid} is not admissible for species "
                              f"{ctx['species']!r} / compartment "
                              f"{ctx['compartment']!r} at status "
                              f"{minimum_status.name} or better"))

        t = _chain_taint(c.model_ids, stage_taint)
        if t > max_taint:
            fails.append(("provenance",
                          f"chain taint is {t.name}; this decision permits at most "
                          f"{max_taint.name}. Provenance does not launder: the chain "
                          f"is as weak as its weakest input, however many measured "
                          f"stages surround it."))

        if structural is not None and target is not None:
            idf, why = structural.identifiable(target)
            if not idf:
                fails.append(("identifiability",
                              f"{target!r} is not identifiable from the achievable "
                              f"measurements: {why}. No choice of model at any stage "
                              f"changes this, which is why it eliminates the whole "
                              f"space at once."))

        if fails:
            for cond, why in fails:
                rejected.append(Rejection(c, cond, why))
        else:
            ok.append(c)

    by_condition: dict[str, int] = {}
    for r in rejected:
        by_condition[r.condition] = by_condition.get(r.condition, 0) + 1
    dominant = max(by_condition, key=by_condition.get) if by_condition else None
    n_blocked = len({r.candidate.model_ids for r in rejected})
    # A condition that blocks EVERY candidate is binding on the space as a whole,
    # which is a stronger and more useful statement than "most frequent".
    binding = sorted(c for c, n in by_condition.items()
                     if len({r.candidate.model_ids for r in rejected
                             if r.condition == c}) == len(cands))
    return {"n_enumerated": len(cands), "n_admissible": len(ok),
            "admissible": [c.model_ids for c in ok],
            "n_rejected": n_blocked,
            "n_rejection_reasons": len(rejected),
            "conditions_binding_on_every_candidate": binding,
            "rejections_by_condition": by_condition,
            "dominant_rejection_condition": dominant,
            "rejections": [{"plan": " -> ".join(r.candidate.model_ids),
                            "condition": r.condition, "reason": r.reason}
                           for r in rejected],
            "verdict": (f"{len(ok)} of {len(cands)} compositions are admissible"
                        if ok else
                        f"the admissible space is EMPTY. All {len(cands)} compositions "
                        f"were rejected; {binding or [dominant]} block(s) every "
                        f"candidate, so lifting anything else changes nothing. That is "
                        f"a finding about the evidence base, not a planner failure -- "
                        f"and it is the answer a research plan can act on.")}


def plan_advantage(ours: Mapping[str, object],
                   fixed_workflow_plan: Sequence[str],
                   stage_taint: Mapping[str, Taint],
                   max_taint: Taint) -> dict:
    """Would a fixed workflow have run a plan this search rejects?

    The concrete form of the agent-advantage question. A fixed workflow picks one
    composition -- typically the best-performing model at each stage independently --
    and runs it. If that composition is in the rejection log, the planner's value is
    demonstrated on this instance rather than argued in the abstract.
    """
    fixed = " -> ".join(fixed_workflow_plan)
    rejected = {r["plan"]: r for r in ours.get("rejections", [])}  # type: ignore[union-attr]
    admissible = {" -> ".join(p) for p in ours.get("admissible", [])}  # type: ignore[union-attr]
    if fixed in rejected:
        r = rejected[fixed]
        return {"fixed_workflow_plan": fixed, "would_have_been_run": True,
                "is_admissible": False, "rejected_on": r["condition"],
                "reason": r["reason"],
                "verdict": (f"the fixed workflow would have run a composition this "
                            f"search rejects on {r['condition']}. The advantage is "
                            f"demonstrated on this instance, not argued.")}
    if fixed in admissible:
        return {"fixed_workflow_plan": fixed, "would_have_been_run": True,
                "is_admissible": True, "rejected_on": None,
                "verdict": ("the fixed workflow's plan is admissible, so the planner "
                            "adds nothing here. Reporting that is the honest outcome "
                            "and it should not be buried.")}
    return {"fixed_workflow_plan": fixed, "would_have_been_run": True,
            "is_admissible": None, "rejected_on": None,
            "verdict": ("the fixed workflow's plan was not in the enumerated space, "
                        "so no comparison can be made; enumerate over the same stages "
                        "before claiming an advantage")}
