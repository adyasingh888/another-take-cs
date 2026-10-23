"""Loop A's five phases and Loop B's six steps, including what each may not do."""
import numpy as np
import pytest

from atcs.phases import (Phase, PhaseError, PhaseResult, SPEC, gate, loop_a_table,
                         validate_inference)
from atcs.patient import (C_PATIENT, ContextField, GovernanceError, Step,
                          companion_diagnostic, context_coverage, effective_n,
                          loop_b_table, require_independent_patients,
                          reverse_translate, simulate_regimens, state_estimate)
from atcs.taint import Taint


# ---- Loop A ---------------------------------------------------------------
def test_simulation_happens_twice_and_the_two_are_different():
    """Phase 2 assumes perfect delivery; Phase 4 asks whether delivery happens."""
    assert SPEC[Phase.TARGET_VALIDATION]["simulation"] is not None
    assert SPEC[Phase.DELIVERY_SIMULATION]["simulation"] is not None
    assert "ASSUMED PERFECT" in SPEC[Phase.TARGET_VALIDATION]["simulation"]
    assert "DELIVERY IS THE QUESTION" in SPEC[Phase.DELIVERY_SIMULATION]["simulation"]


def test_a_phase_2_result_cannot_claim_reachability():
    """The structural error the whole programme exists to address: a virtual-cell
    result read as though it proved the therapeutic gets there."""
    r = PhaseResult(Phase.TARGET_VALIDATION, True, {"response": 0.8}, Taint.SIMULATED)
    ok, why = r.may_claim_reachability()
    assert not ok
    assert "assumes perfect delivery by construction" in why


def test_a_phase_4_result_can():
    r = PhaseResult(Phase.DELIVERY_SIMULATION, True, {"response": 0.4}, Taint.SIMULATED)
    assert r.may_claim_reachability()[0]


def test_the_loop_has_revision_edges_so_it_is_not_a_chain():
    assert SPEC[Phase.DELIVERY_SIMULATION]["revises"]
    assert len(SPEC[Phase.STRESS_AND_INFERENCE]["revises"]) == 4
    assert SPEC[Phase.DISEASE_HYPOTHESIS]["revises"] == ()


def test_a_phase_4_failure_revises_chemistry_before_the_target():
    """Latest revisable phase first: it discards the least work, and exhausting
    delivery explanations before abandoning a target is the right order."""
    r = PhaseResult(Phase.DELIVERY_SIMULATION, False, {}, Taint.SIMULATED)
    target, why = r.revision_target()
    assert target is Phase.CANDIDATE_DESIGN
    assert "discards the least work" in why


def test_a_phase_1_failure_ends_the_line_rather_than_revising():
    r = PhaseResult(Phase.DISEASE_HYPOTHESIS, False, {}, Taint.DERIVED)
    target, why = r.revision_target()
    assert target is None and "ends this line of work" in why


def test_phase_5_must_compare_rivals():
    one = PhaseResult(Phase.STRESS_AND_INFERENCE, True, {}, Taint.SIMULATED,
                      mechanisms_compared=1)
    with pytest.raises(PhaseError, match="hypothesis restated"):
        validate_inference(one)
    two = PhaseResult(Phase.STRESS_AND_INFERENCE, True, {}, Taint.SIMULATED,
                      mechanisms_compared=2)
    assert validate_inference(two) is two


def test_a_phase_cannot_be_entered_without_its_inputs():
    ok, missing = gate(Phase.DELIVERY_SIMULATION, ["a formulation"])
    assert not ok and "a biological context" in missing


def test_every_phase_declares_a_gate():
    for row in loop_a_table():
        assert len(row["gate"]) > 40


# ---- Loop B ---------------------------------------------------------------
def test_the_context_vector_is_mechanism_typed_not_a_feature_vector():
    """Each field declares which stage it modulates and in which direction, so a
    fitted effect contradicting the declared direction is a finding."""
    stages = {f.modulates_stage for f in C_PATIENT.values()}
    assert stages >= {"x_id", "x_organ", "x_barrier", "x_cell"}
    assert C_PATIENT["apoe_plasma"].direction == +1
    assert C_PATIENT["complement_c3"].direction == -1


def test_an_unestablished_direction_is_declared_zero_rather_than_guessed():
    assert C_PATIENT["albumin"].direction == 0
    assert "not established" in C_PATIENT["albumin"].rationale


def test_a_context_field_without_a_mechanism_is_refused():
    with pytest.raises(GovernanceError, match="must state WHY"):
        ContextField("x", "x_id", 1, "unit", "too short", True)


def test_coverage_reports_stages_that_cannot_be_personalised():
    cov = context_coverage(["complement_c3", "crp"])
    assert cov["by_stage"]["x_id"]["available"] == 2
    assert "x_barrier" in cov["stages_without_any_available_modulator"]
    assert "on population parameters" in cov["note"]


# ---- the patient-count problem, enforced ----------------------------------
def test_215_observations_from_29_patients_is_not_215_samples():
    """Stated twice in the brief, and the kind of error that survives into print."""
    e = effective_n(29, 215, icc=0.5)
    assert e["obs_per_patient"] == pytest.approx(7.41, abs=0.01)
    assert e["effective_n"] < 60
    assert e["design_effect"] > 3


def test_higher_icc_shrinks_the_effective_sample_further():
    e = [effective_n(29, 215, icc=i)["effective_n"] for i in (0.0, 0.3, 0.7, 1.0)]
    assert e == sorted(e, reverse=True)
    assert e[0] == 215.0 and e[-1] == pytest.approx(29.0, abs=0.1)


def test_a_claim_using_observations_as_patients_is_refused():
    with pytest.raises(GovernanceError, match="not independent samples"):
        require_independent_patients(215, 29, "cross-sectional regression")


def test_fewer_observations_than_patients_is_refused():
    with pytest.raises(GovernanceError, match="fewer observations"):
        effective_n(29, 10)


# ---- the governance separation --------------------------------------------
def test_updating_a_patient_posterior_does_not_touch_the_population_model():
    s = state_estimate(0.5, 0.2, [0.7, 0.72, 0.68], 0.05)
    assert s["population_model_untouched"] is True
    assert s["posterior_sd"] < 0.2
    assert "not retraining" in s["note"]


def test_with_no_observations_the_posterior_is_the_prior():
    s = state_estimate(0.5, 0.2, [], 0.05)
    assert s["posterior_mean"] == 0.5 and s["shrinkage"] == 0.0


def test_regimens_are_ranked_with_overlap_flagged_rather_than_a_winner_declared():
    out = simulate_regimens(
        {"q3w_0.3": {"dose_mg_per_kg": 0.3, "interval_weeks": 3},
         "q3w_0.4": {"dose_mg_per_kg": 0.4, "interval_weeks": 3},
         "q6w_0.3": {"dose_mg_per_kg": 0.3, "interval_weeks": 6}},
        posterior_mean=0.8, posterior_sd=0.25)
    assert [r["regimen"] for r in out][0] == "q3w_0.4"
    assert any(r["distinguishable_from_next"] is False for r in out)


def test_a_nonpositive_dose_is_refused():
    with pytest.raises(GovernanceError, match="non-positive"):
        simulate_regimens({"bad": {"dose_mg_per_kg": 0.0, "interval_weeks": 3}}, 0.8, 0.1)


def test_the_best_measurement_and_the_best_deployable_one_can_differ():
    out = companion_diagnostic(
        ["apoe_plasma", "complement_c3", "hepatic_fibrosis_stage"],
        posterior_sd=0.3,
        variance_explained={"apoe_plasma": 0.55, "complement_c3": 0.20,
                            "hepatic_fibrosis_stage": 0.30})
    assert out["best_overall"] == "apoe_plasma"
    assert out["best_deployable"] == "hepatic_fibrosis_stage"
    assert not C_PATIENT["apoe_plasma"].routinely_measured


def test_reverse_translation_yields_a_hypothesis_and_cannot_update_a_model():
    r = reverse_translate("biomarker trajectory outside the predicted range",
                          n_patients=29, n_observations=215,
                          confounders_assessed=["differential exposure",
                                                "assay variation", "adherence"],
                          provenance_cleared=True)
    assert r["outcome"] == "research hypothesis"
    assert "does not" in str(r).lower() or "does_not" in r
    assert "update a deployed model" in r["does_not"]
    assert r["effective_n"] < 215


def test_reverse_translation_refuses_before_provenance_clearance():
    with pytest.raises(GovernanceError, match="provenance, privacy"):
        reverse_translate("x", 29, 215, ["a", "b", "c"], provenance_cleared=False)


def test_reverse_translation_refuses_a_single_explanation():
    """Promoting one explanation before the others are assessed is how a laboratory
    artefact becomes a mechanism."""
    with pytest.raises(GovernanceError, match="how an artefact"):
        reverse_translate("x", 29, 215, ["differential exposure"], True)


def test_both_loops_tabulate():
    assert len(loop_a_table()) == 5
    assert len(loop_b_table()) == 6
    assert loop_b_table()[-1]["name"] == "REVERSE_TRANSLATION"
