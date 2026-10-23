"""
ICH M15 credibility assessment: risk-based, as the guidance actually frames it.

WHAT THE BRIEF ASKS FOR
-----------------------
§16 proposes using ICH M15 "now to define and report the research", with a bounded
context of use, and is careful that doing so "does not make the existing MesoReach
model regulatory-qualified". That distinction is the module's design: it produces a
CREDIBILITY PLAN, which is a statement of what evidence would be needed, not a
qualification.

THE RISK FRAMING
----------------
M15's organising idea is that the credibility evidence required scales with MODEL
RISK, and model risk has two components that must be assessed separately:

  MODEL INFLUENCE       how much of the decision rests on the model, as against
                        other evidence? A model that ranks candidates for a
                        confirmatory experiment has low influence; a model that
                        replaces an experiment has high influence.
  DECISION CONSEQUENCE  what happens if the model is wrong? A wasted experiment is
                        a different order of consequence from a dosing error.

Risk is their combination, and the common error is to assess only the second. A
low-consequence decision taken ENTIRELY on a model still carries real risk, because
nothing else will catch the error -- and that is exactly the research-prioritisation
case this programme is proposing.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping, Sequence


class CredibilityError(ValueError):
    """A credibility plan was requested for an unassessable context."""


class Influence(IntEnum):
    SUPPORTING = 1      # the model informs; other evidence decides
    CONTRIBUTING = 2    # the model is one of several comparable inputs
    PRIMARY = 3         # the decision rests mainly on the model
    SUBSTITUTING = 4    # the model replaces an experiment that would otherwise be run


class Consequence(IntEnum):
    WASTED_EFFORT = 1       # a misdirected experiment
    WRONG_PRIORITY = 2      # a programme pursues the wrong candidate
    PRECLINICAL_HARM = 3    # unnecessary animal use or a missed safety signal
    PATIENT_HARM = 4        # a clinical decision is affected


#: Evidence activities, ordered roughly by cost. Which are REQUIRED depends on risk.
ACTIVITIES = {
    "code_verification": "the implementation computes what the equations say",
    "numerical_verification": "solver error is bounded and reported",
    "sensitivity_analysis": "which parameters the output actually depends on",
    "applicability_bounds": "the input region where the model was evaluated",
    "internal_validation": "held-out performance within the development data",
    "external_validation": "performance on data withheld before development",
    "uncertainty_quantification": "calibrated intervals, not point predictions",
    "independent_reproduction": "someone else runs it and gets the same answer",
    "prospective_validation": "predictions made before the outcome was observed",
}


def model_risk(influence: Influence, consequence: Consequence) -> dict:
    """Risk as the combination, with the common error named.

    The product is used rather than the maximum because the two compound: a
    substituting model in a patient-harm decision is not merely 'high', it is
    categorically different from either factor alone.
    """
    score = int(influence) * int(consequence)
    band = ("low" if score <= 3 else "moderate" if score <= 6 else
            "high" if score <= 11 else "very high")
    return {"influence": influence.name, "consequence": consequence.name,
            "risk_score": score, "risk_band": band,
            "note": ("risk is the COMBINATION. A low-consequence decision taken "
                     "entirely on a model still carries real risk, because nothing "
                     "else will catch the error -- which is exactly the research-"
                     "prioritisation case, and the component usually left unassessed")}


#: What each risk band requires. Cumulative: a higher band requires everything below.
REQUIRED: dict[str, tuple[str, ...]] = {
    "low": ("code_verification", "applicability_bounds"),
    "moderate": ("numerical_verification", "sensitivity_analysis",
                 "internal_validation"),
    "high": ("external_validation", "uncertainty_quantification"),
    "very high": ("independent_reproduction", "prospective_validation"),
}


def required_activities(band: str) -> list[str]:
    order = ["low", "moderate", "high", "very high"]
    if band not in order:
        raise CredibilityError(f"unknown risk band {band!r}")
    out: list[str] = []
    for b in order[:order.index(band) + 1]:
        out.extend(REQUIRED[b])
    return out


@dataclass(frozen=True)
class ContextOfUse:
    """A bounded question, decision and scope. All three are required."""

    question: str
    decision: str
    scope: str
    influence: Influence
    consequence: Consequence

    def __post_init__(self) -> None:
        for fld in ("question", "decision", "scope"):
            if len(getattr(self, fld)) < 20:
                raise CredibilityError(
                    f"a context of use must state its {fld} specifically. "
                    f"An unbounded context cannot have its evidence requirements "
                    f"assessed, which is the entire function of the exercise.")


def credibility_plan(cou: ContextOfUse, evidence_available: Sequence[str]) -> dict:
    """What this context requires, what exists, and the gap.

    Returns the GAP as the primary output, because that is the only part a research
    plan can act on. A plan that reports a percentage complete invites the reading
    that 80% is nearly enough, which for a gated requirement it is not.
    """
    risk = model_risk(cou.influence, cou.consequence)
    need = required_activities(risk["risk_band"])
    have = set(evidence_available)
    unknown = [e for e in have if e not in ACTIVITIES]
    if unknown:
        raise CredibilityError(
            f"unrecognised evidence activities {unknown}; declared activities are "
            f"{sorted(ACTIVITIES)}")
    missing = [e for e in need if e not in have]
    return {"question": cou.question, "decision": cou.decision, "scope": cou.scope,
            **risk,
            "required_activities": need,
            "available": sorted(have & set(need)),
            "missing": missing,
            "credible_for_this_context": not missing,
            "surplus_not_required_here": sorted(have - set(need)),
            "verdict": ("the available evidence meets this context's requirements; "
                        "this is a credibility statement for THIS context of use and "
                        "is not a qualification of the model"
                        if not missing else
                        f"{len(missing)} required activities are absent: "
                        f"{', '.join(missing)}. The model is not credible for this "
                        f"context until they exist -- a gated requirement, not a "
                        f"percentage to be nearly met.")}


#: The brief's own proposed first context, §16, written out so it can be assessed.
PROPOSED_FIRST_CONTEXT = ContextOfUse(
    question=("Does one reference measurement plus a predictive model improve "
              "withheld preclinical biodistribution predictions in new studies?"),
    decision=("Research-stage interpretation and prioritisation of experiments; "
              "not human dosing and not substitution of clinical evidence"),
    scope=("Preclinical rodent biodistribution of lipid and polymeric carriers, "
           "within the descriptor ranges of the development corpus"),
    influence=Influence.PRIMARY,
    consequence=Consequence.WRONG_PRIORITY,
)
