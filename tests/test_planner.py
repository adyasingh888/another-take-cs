"""The planner: the rejection log is the output that matters."""
import pytest

from atcs.models import ORGAN_TO_CYTOSOL
from atcs.planner import (Candidate, PlanningError, admissible_plans,
                          enumerate_plans, plan_advantage)
from atcs.registry import ContextOfUse, RegisteredModel, Registry
from atcs.status import Evidenced, Status
from atcs.taint import Taint

IND = Evidenced(Status.INDEPENDENT, "executed study-grouped diagnostic, Oct 2026")
REP = Evidenced(Status.REPORTED, "project-reported benchmark, 20 Oct brief",
                what_would_raise_it="independent reproduction on withheld data")


def _reg():
    r = Registry()
    for name, status, comp in [("organ_rf", IND, "whole_liver_homogenate"),
                               ("organ_gbm", REP, "whole_liver_homogenate"),
                               ("cyto_a", REP, "hepatocyte_cytosol_risc"),
                               ("cyto_b", REP, "hepatocyte_cytosol_risc")]:
        r.register(RegisteredModel(
            name, status,
            ContextOfUse(species="mouse", compartment=comp,
                         question="organ-level disposition of lipid carriers",
                         descriptor_ranges={"size_nm": (20.0, 200.0)},
                         decision="research prioritisation of formulations"),
            required_inputs=("size_nm",), output="log10 delivery fraction"))
    return r


ORGAN_CTX = {"species": "mouse", "compartment": "whole_liver_homogenate"}
CYTO_CTX = {"species": "mouse", "compartment": "hepatocyte_cytosol_risc"}
HUMAN_CTX = {"species": "human", "compartment": "whole_liver_homogenate"}

TAINT_OK = {"organ_rf": Taint.MEASURED, "organ_gbm": Taint.MEASURED,
            "cyto_a": Taint.DERIVED, "cyto_b": Taint.DERIVED}
TAINT_SIM = {**TAINT_OK, "cyto_a": Taint.SIMULATED, "cyto_b": Taint.SIMULATED}


def test_enumeration_covers_the_product_of_the_stages():
    plans = enumerate_plans(_reg(), [["organ_rf", "organ_gbm"], ["cyto_a", "cyto_b"]])
    assert len(plans) == 4


def test_a_stage_with_no_candidates_is_a_finding_about_model_availability():
    with pytest.raises(PlanningError, match="finding about model availability"):
        enumerate_plans(_reg(), [["organ_rf"], []])


def test_a_context_per_stage_is_required():
    """A chain crosses compartments by construction. An earlier version took one
    context for the whole composition, which made every multi-stage plan inadmissible
    on context -- producing an empty space attributed to the wrong constraint."""
    with pytest.raises(PlanningError, match="crosses compartments"):
        admissible_plans(_reg(), [["organ_rf"], ["cyto_a"]],
                         stage_contexts=[ORGAN_CTX], decision="x",
                         stage_taint=TAINT_OK, max_taint=Taint.DERIVED)


def test_a_two_stage_plan_crossing_compartments_is_admissible():
    out = admissible_plans(
        _reg(), [["organ_rf"], ["cyto_a"]], stage_contexts=[ORGAN_CTX, CYTO_CTX],
        decision="research prioritisation", stage_taint=TAINT_OK,
        max_taint=Taint.DERIVED)
    assert out["n_admissible"] == 1, out["rejections"]


def test_context_of_use_eliminates_models_outside_their_compartment():
    out = admissible_plans(
        _reg(), [["organ_rf", "cyto_a"]], stage_contexts=[ORGAN_CTX],
        decision="research prioritisation",
        stage_taint=TAINT_OK, max_taint=Taint.DERIVED)
    assert out["n_admissible"] == 1
    assert out["rejections_by_condition"].get("context_of_use") == 1


def test_a_species_the_models_were_not_validated_for_empties_the_space():
    out = admissible_plans(
        _reg(), [["organ_rf", "organ_gbm"]], stage_contexts=[HUMAN_CTX], decision="anything",
        stage_taint=TAINT_OK, max_taint=Taint.DERIVED)
    assert out["n_admissible"] == 0
    assert out["dominant_rejection_condition"] == "context_of_use"
    assert "finding about the evidence base" in out["verdict"]


def test_provenance_does_not_launder_across_a_composition():
    """The chain is as weak as its weakest input, however many measured stages
    surround it."""
    out = admissible_plans(
        _reg(), [["organ_rf"], ["cyto_a"]], stage_contexts=[ORGAN_CTX, CYTO_CTX],
        decision="research prioritisation",
        stage_taint=TAINT_SIM, max_taint=Taint.DERIVED)
    assert out["n_admissible"] == 0
    assert out["rejections_by_condition"]["provenance"] == 1
    assert "does not launder" in out["rejections"][0]["reason"]


def test_identifiability_eliminates_the_whole_space_at_once():
    """The condition a workflow built from 'which models do we have' never checks --
    and the one no choice of model can fix."""
    out = admissible_plans(
        _reg(), [["organ_rf", "organ_gbm"]], stage_contexts=[ORGAN_CTX],
        decision="target-cell exposure",
        stage_taint=TAINT_OK, max_taint=Taint.DERIVED,
        structural=ORGAN_TO_CYTOSOL, target="f_parenchymal")
    assert out["n_admissible"] == 0
    assert out["rejections_by_condition"]["identifiability"] == 2
    assert "eliminates the whole space at once" in out["rejections"][0]["reason"]


def test_all_conditions_are_evaluated_not_just_the_first_failing_one():
    """Found by a figure whose title asserted identifiability emptied the space
    while the data said provenance -- because provenance happened to be checked
    first. A rejection log reporting the wrong blocker defeats its own purpose."""
    out = admissible_plans(
        _reg(), [["organ_rf", "organ_gbm"], ["cyto_a", "cyto_b"]],
        stage_contexts=[ORGAN_CTX, CYTO_CTX], decision="target-cell exposure",
        stage_taint=TAINT_SIM, max_taint=Taint.DERIVED,
        structural=ORGAN_TO_CYTOSOL, target="f_parenchymal")
    assert out["n_admissible"] == 0
    conds = out["rejections_by_condition"]
    assert conds["provenance"] == 4 and conds["identifiability"] == 4, (
        "both conditions must be reported, not whichever is checked first")
    assert out["n_rejected"] == 4, "four candidates, not eight reasons"
    assert out["n_rejection_reasons"] == 8


def test_a_condition_blocking_every_candidate_is_named_as_binding():
    """Stronger and more useful than 'most frequent': lifting anything else changes
    nothing."""
    out = admissible_plans(
        _reg(), [["organ_rf", "organ_gbm"]], stage_contexts=[ORGAN_CTX],
        decision="target-cell exposure", stage_taint=TAINT_OK,
        max_taint=Taint.DERIVED, structural=ORGAN_TO_CYTOSOL, target="f_parenchymal")
    assert out["conditions_binding_on_every_candidate"] == ["identifiability"]
    assert "block(s) every candidate" in out["verdict"]


def test_every_rejection_carries_a_condition_and_a_reason():
    out = admissible_plans(
        _reg(), [["organ_rf", "cyto_a"]], stage_contexts=[HUMAN_CTX], decision="x",
        stage_taint=TAINT_OK, max_taint=Taint.DERIVED)
    for r in out["rejections"]:
        assert r["condition"] and len(r["reason"]) > 30


# ---- the concrete agent-advantage question --------------------------------
def test_the_advantage_is_demonstrated_on_an_instance_not_argued():
    """A fixed workflow picks the best model per stage and runs it. If that
    composition is in the rejection log, the advantage is shown on this instance."""
    ours = admissible_plans(
        _reg(), [["organ_rf"], ["cyto_a"]], stage_contexts=[ORGAN_CTX, CYTO_CTX],
        decision="research prioritisation",
        stage_taint=TAINT_SIM, max_taint=Taint.DERIVED)
    adv = plan_advantage(ours, ["organ_rf", "cyto_a"], TAINT_SIM, Taint.DERIVED)
    assert adv["would_have_been_run"] and adv["is_admissible"] is False
    assert adv["rejected_on"] == "provenance"
    assert "not argued" in adv["verdict"]


def test_no_advantage_is_reported_honestly():
    ours = admissible_plans(
        _reg(), [["organ_rf"]], stage_contexts=[ORGAN_CTX],
        decision="research prioritisation",
        stage_taint=TAINT_OK, max_taint=Taint.DERIVED)
    adv = plan_advantage(ours, ["organ_rf"], TAINT_OK, Taint.DERIVED)
    assert adv["is_admissible"] is True
    assert "adds nothing here" in adv["verdict"]
    assert "should not be buried" in adv["verdict"]


def test_a_plan_outside_the_enumerated_space_refuses_comparison():
    ours = admissible_plans(
        _reg(), [["organ_rf"]], stage_contexts=[ORGAN_CTX],
        decision="x",
        stage_taint=TAINT_OK, max_taint=Taint.DERIVED)
    adv = plan_advantage(ours, ["something_else"], TAINT_OK, Taint.DERIVED)
    assert adv["is_admissible"] is None
    assert "before claiming an advantage" in adv["verdict"]
