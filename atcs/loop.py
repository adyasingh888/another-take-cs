"""
Closing the cycle: a clinical observation becomes a bench experiment.

WHAT WAS MISSING
----------------
`patient.reverse_translate` produces a research hypothesis. `phases` describes a
bench loop whose Phase 5 consumes hypotheses. `experiment` can specify and assess an
experiment. Nothing connected them, so the architecture was two loops drawn beside
each other rather than a cycle.

This module is the connection, and it is deliberately narrow: ONE function, with
every gate from both loops applied in order. A wide API here would invite bypassing
a gate; one function with no alternative path cannot be routed around.

THE ORDER OF THE GATES IS THE ARGUMENT
--------------------------------------
  1. the observation cleared provenance and confounding      (patient)
  2. the effective sample size is reported, not the row count (patient)
  3. the hypothesis names a parameter of a structural model   (observability)
  4. the experiment that would test it is identifiable        (experiment)
  5. ... and adequately powered at an achievable size         (experiment)
  6. ... and the parameter is one the output depends on       (sensitivity)
  7. re-entry is at Phase 5, never at Phase 1                 (phases)

Gate 7 is the one worth stating. A clinical surprise does not restart the
programme; it enters at the inference phase, where it is compared against rival
explanations before anything upstream is revised. Allowing re-entry at Phase 1
would let one unexpected trajectory in 29 patients redirect a research programme,
which is how a laboratory artefact becomes a research direction.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from .experiment import ExperimentSpec, SpecificationError
from .observability import StructuralModel
from .patient import GovernanceError, effective_n, reverse_translate
from .phases import Phase


class CycleError(RuntimeError):
    """The cycle could not be closed, with the gate that stopped it."""


def close_cycle(*, observation: str, n_patients: int, n_observations: int,
                confounders_assessed: Sequence[str], provenance_cleared: bool,
                hypothesis_parameter: str, model: StructuralModel,
                proposed: ExperimentSpec,
                sensitivity: Mapping[str, object] | None = None,
                total_effect_floor: float = 0.05) -> dict:
    """A clinical observation -> a bench experiment, or a named refusal.

    Returns the re-entry point and the full gate record. Every gate that passed is
    recorded alongside the ones that did not, because "it failed gate 4" is only
    actionable if you can see that gates 1-3 passed.
    """
    gates: list[dict] = []

    # 1-2. the clinical side
    try:
        rt = reverse_translate(observation, n_patients, n_observations,
                               confounders_assessed, provenance_cleared)
    except GovernanceError as e:
        raise CycleError(f"gate 1-2 (clinical governance): {e}") from e
    gates.append({"gate": "clinical governance", "passed": True,
                  "detail": f"effective n {rt['effective_n']} from "
                            f"{n_observations} observations"})

    # 3. the hypothesis must name something the model contains
    if hypothesis_parameter not in model.unknowns:
        raise CycleError(
            f"gate 3 (hypothesis is about a modelled quantity): "
            f"{hypothesis_parameter!r} is not an unknown of {model.name!r}. A "
            f"hypothesis that names no parameter of any model cannot be tested by "
            f"an experiment on that model -- it needs a different model first.")
    gates.append({"gate": "hypothesis names a modelled quantity", "passed": True,
                  "detail": f"{hypothesis_parameter} is an unknown of {model.name}"})

    # 4-5. the proposed experiment
    if proposed.targets != hypothesis_parameter:
        raise CycleError(
            f"gate 4 (the experiment tests the hypothesis): the proposal targets "
            f"{proposed.targets!r} but the hypothesis is about "
            f"{hypothesis_parameter!r}. An experiment that measures something else "
            f"is not a test of this hypothesis, however informative it is.")
    assessment = proposed.assess(model)
    if not assessment["worth_running"]:
        raise CycleError(
            f"gate {'4' if assessment['binding_constraint'] == 'identifiability' else '5'} "
            f"({assessment['binding_constraint']}): "
            f"{assessment['blockers'][0][1]}")
    gates.append({"gate": "experiment is identifiable and powered", "passed": True,
                  "detail": f"power {assessment['power']['power']:.2f} at n_eff "
                            f"{assessment['power']['n_effective']:.0f}"})

    # 6. does the output depend on it at all?
    if sensitivity is not None:
        ent = next((r for r in sensitivity["indices"]              # type: ignore[index]
                    if r["parameter"] == hypothesis_parameter), None)
        if ent is None:
            gates.append({"gate": "output depends on the parameter", "passed": None,
                          "detail": f"{hypothesis_parameter} absent from the "
                                    f"sensitivity analysis; not assessed"})
        elif ent["total_effect"] < total_effect_floor:
            raise CycleError(
                f"gate 6 (sensitivity): total effect {ent['total_effect']} is below "
                f"{total_effect_floor}, so the output barely depends on "
                f"{hypothesis_parameter!r}. Measuring it would close an unknown that "
                f"does not matter -- which is a cheaper mistake than the alternatives "
                f"but still a mistake.")
        else:
            gates.append({"gate": "output depends on the parameter", "passed": True,
                          "detail": f"total effect {ent['total_effect']}"})

    return {"observation": observation,
            "hypothesis_parameter": hypothesis_parameter,
            "experiment": proposed.name,
            "reenters_at": int(Phase.STRESS_AND_INFERENCE),
            "reentry_rationale": (
                "Phase 5, never Phase 1. A clinical surprise enters at the inference "
                "phase, where it is compared against rival explanations before "
                "anything upstream is revised. Re-entry at Phase 1 would let one "
                "unexpected trajectory in a small cohort redirect a research "
                "programme, which is how a laboratory artefact becomes a research "
                "direction."),
            "gates": gates,
            "clinical_record": rt,
            "experiment_assessment": assessment,
            "cycle_closed": True}
