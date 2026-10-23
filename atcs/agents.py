"""
The five agent roles, and the only scoring that can show they matter.

THE ORIGINALITY BOUNDARY, QUOTED
--------------------------------
The brief's §14 is blunt: "Building six prompt personas does not make a new
scientist. We must demonstrate that evidence-aware model construction yields
better biological predictions or research decisions."

So this module is not an agent framework. It is the SCORING PROTOCOL that would
settle whether an agent layer earns its place, plus the typed roles the brief
specifies, written so each role's output is checkable rather than persuasive.

THE DECISIVE METRIC
-------------------
Of the six criteria the brief lists -- valid model links, correct refusal,
reproducibility, answer quality, decision value, cost -- only CORRECT REFUSAL can
distinguish an evidence-aware agent from a fixed workflow with the same tools. A
fixed workflow with good tools gets the valid links right too. What it cannot do
is decline a link that looks computable and is not supported.

And refusal is asymmetric, which is why `score_refusals` reports a confusion table
and refuses to average it: a false permit propagates an unsupported number into a
conclusion, while a false refusal costs an analysis that could have been run.
Those are not interchangeable, and an F1 that treats them as such is the wrong
summary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping, Sequence


class AgentError(RuntimeError):
    """An agent produced output its role does not permit."""


class Role(Enum):
    """The brief's §14 roles. Each has an output type and a thing it may not do."""

    EVIDENCE_RETRIEVAL = "evidence retrieval and provenance"
    COMPATIBILITY_CHECKER = "model registry and compatibility checking"
    MODEL_EXECUTOR = "scientific model execution"
    MECHANISTIC_CRITIC = "competing-mechanism comparison"
    EXPERIMENT_DECISION = "experiment selection"


#: What each role returns, and -- more usefully -- what it is forbidden to do.
#: The forbidden column is where an agent architecture actually fails: roles that
#: quietly acquire each other's powers.
ROLE_CONTRACT: dict[Role, dict] = {
    Role.EVIDENCE_RETRIEVAL: {
        "returns": "studies with provenance, and whether assays are PAIRED",
        "forbidden": ("asserting that two corpora are paired without a shared "
                      "identity key; this is the error that creates invented "
                      "trajectories"),
        "checkable_by": "atcs.multimodal.linkage",
    },
    Role.COMPATIBILITY_CHECKER: {
        "returns": "admissible models for a context of use",
        "forbidden": ("admitting a model outside its validated context because no "
                      "better one exists; scarcity is not a credential"),
        "checkable_by": "atcs.registry.Registry.admissible",
    },
    Role.MODEL_EXECUTOR: {
        "returns": "predictions with propagated uncertainty",
        "forbidden": ("re-estimating uncertainty from its own residuals, which "
                      "discards the input's"),
        "checkable_by": "atcs.propagate.reestimate_check",
    },
    Role.MECHANISTIC_CRITIC: {
        "returns": "a posterior over at least two rival mechanisms",
        "forbidden": ("returning a single mechanism, which is a hypothesis restated "
                      "rather than an inference"),
        "checkable_by": "atcs.critic.compare",
    },
    Role.EXPERIMENT_DECISION: {
        "returns": "the measurement with the greatest expected decision value",
        "forbidden": ("ranking by where the favoured mechanism looks best rather "
                      "than by where rivals disagree"),
        "checkable_by": "atcs.critic.discriminating_design",
    },
}


@dataclass(frozen=True)
class RefusalCase:
    """One link an agent was asked to make, with ground truth."""

    case_id: str
    should_refuse: bool        # ground truth: is this link unsupported?
    did_refuse: bool
    reason: str = ""
    consequence_if_wrong: str = ""


def score_refusals(cases: Sequence[RefusalCase]) -> dict:
    """Confusion table over refusal decisions. Deliberately NOT summarised to one number.

    A false permit lets an unsupported number into a conclusion; a false refusal
    costs an analysis. They are not interchangeable, so no F1 is returned -- the
    caller must decide the exchange rate for their own setting, and making that
    explicit is the point.
    """
    if not cases:
        raise AgentError("no cases to score")
    tp = sum(1 for c in cases if c.should_refuse and c.did_refuse)
    fn = sum(1 for c in cases if c.should_refuse and not c.did_refuse)
    fp = sum(1 for c in cases if not c.should_refuse and c.did_refuse)
    tn = sum(1 for c in cases if not c.should_refuse and not c.did_refuse)
    n_should = tp + fn
    n_shouldnt = fp + tn
    return {
        "n": len(cases),
        "correct_refusals": tp, "missed_refusals": fn,
        "false_refusals": fp, "correct_permits": tn,
        "refusal_recall": round(tp / n_should, 4) if n_should else None,
        "permit_precision": round(tn / n_shouldnt, 4) if n_shouldnt else None,
        "missed_refusal_cases": [c.case_id for c in cases
                                 if c.should_refuse and not c.did_refuse],
        "no_single_score": ("a false permit propagates an unsupported number into a "
                            "conclusion; a false refusal costs an analysis. These are "
                            "not interchangeable, so no F1 is reported -- the exchange "
                            "rate is the caller's to set"),
    }


def versus_baselines(ours: Sequence[RefusalCase],
                     fixed_workflow: Sequence[RefusalCase],
                     general_agent: Sequence[RefusalCase] | None = None) -> dict:
    """The comparison the brief specifies: same tools, same evidence, same budget.

    Returns the advantage on the ONE axis where an evidence-aware agent can differ,
    and says plainly when there is none. A result of 'no measured advantage' is a
    publishable finding about agent architectures and should not be buried.
    """
    arms = {"ours": score_refusals(ours),
            "fixed_workflow": score_refusals(fixed_workflow)}
    if general_agent is not None:
        arms["general_biomedical_agent"] = score_refusals(general_agent)
    base = arms["fixed_workflow"]["refusal_recall"] or 0.0
    mine = arms["ours"]["refusal_recall"] or 0.0
    adv = round(mine - base, 4)
    cost = round((arms["ours"]["false_refusals"] -
                  arms["fixed_workflow"]["false_refusals"]) / max(len(ours), 1), 4)
    return {"arms": arms, "refusal_recall_advantage": adv,
            "additional_false_refusal_rate": cost,
            "verdict": (
                "no measured advantage over a fixed workflow on the one axis where "
                "an evidence-aware agent could differ -- which is a finding about "
                "agent architectures, not a null to bury" if adv <= 0 else
                f"refuses {adv:.0%} more of the genuinely unsupported links, at "
                f"{cost:+.0%} additional false-refusal rate")}


def contract_table() -> list[dict]:
    return [{"role": r.value, "returns": c["returns"],
             "forbidden": c["forbidden"], "checkable_by": c["checkable_by"]}
            for r, c in ROLE_CONTRACT.items()]
