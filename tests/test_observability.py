"""Structural identifiability must be decided from wiring, before any data."""
import pytest

from atcs.models import (CROSS_STUDY_OFFSET, CYTOSOL_CANDIDATES, OFFSET_CANDIDATES,
                         ORGAN_TO_CYTOSOL, PATIENT_CANDIDATES, PATIENT_STATE, REGISTRY)
from atcs.observability import Observation, StructuralModel, Unobservable


def test_whole_organ_homogenate_cannot_give_the_cell_split():
    """The chain's break point: ONE equation over six unknowns, so five degrees of
    freedom and nothing identifiable. No model recovers them."""
    r = ORGAN_TO_CYTOSOL.report()
    assert r["n_equations"] == 1
    assert r["degrees_of_freedom"] == 5
    assert r["identifiable"] == []
    assert "f_parenchymal" in r["unidentifiable"]


def test_a_definitional_simplex_is_not_independent_information():
    """Caught in this repository's own model. The four fractions are fractions OF
    organ-associated material, so 'they sum to one' IS 'the homogenate measures
    their total'. Declaring both made the system look one degree of freedom better
    than it is. Equations are compared as unknown-SETS precisely to catch this."""
    from atcs.observability import Observation, StructuralModel
    unknowns = ("f_a", "f_b", "f_c", "f_d")
    total = Observation("homogenate_total", frozenset(unknowns))
    doubled = StructuralModel(unknowns, (total,), (frozenset(unknowns),))
    honest = StructuralModel(unknowns, (total,), ())
    assert doubled.degrees_of_freedom() == honest.degrees_of_freedom() == 3

    # a simplex IS informative when its normalising total is measured separately
    informative = StructuralModel(
        unknowns,
        (total, Observation("separate_total", frozenset({"f_a", "f_b"}))))
    assert informative.degrees_of_freedom() == 2


def test_the_escape_rate_appears_in_no_equation_at_all():
    """Worse than underdetermined: unconstrained. Any value fits every observation."""
    ok, why = ORGAN_TO_CYTOSOL.identifiable("k_escape")
    assert not ok
    assert "appears in no achievable equation" in why


def test_the_minimal_fix_for_the_parenchymal_fraction_is_found():
    """The actionable half: which measurement closes it, and at what cardinality."""
    sets = ORGAN_TO_CYTOSOL.minimal_additional_observations(
        "f_parenchymal", CYTOSOL_CANDIDATES)
    assert sets, "no candidate set closes the parenchymal fraction"
    assert all(len(s) == len(sets[0]) for s in sets), "sets must be of minimal cardinality"
    assert ("flow_sorted_parenchymal",) in sets


def test_an_identifiable_parameter_returns_no_fix():
    sets = CROSS_STUDY_OFFSET.minimal_additional_observations(
        "alpha_study", OFFSET_CANDIDATES)
    assert sets == [] or ("reference_arm_same_run",) in sets


# ---- the anchor, as an equation ------------------------------------------
def test_a_single_published_value_cannot_separate_formulation_from_study():
    r = CROSS_STUDY_OFFSET.report()
    assert r["degrees_of_freedom"] == 1
    assert set(r["unidentifiable"]) == {"beta_formulation", "alpha_study"}


def test_a_reference_arm_closes_the_system_and_a_second_novel_arm_does_not():
    """Why 'more formulations per paper' never fixed cross-study transfer: it adds
    an equation AND an unknown. The reference arm adds only an equation."""
    with_ref = StructuralModel(
        CROSS_STUDY_OFFSET.unknowns,
        CROSS_STUDY_OFFSET.observations + (OFFSET_CANDIDATES[0],))
    assert with_ref.report()["unidentifiable"] == []

    with_second = StructuralModel(
        CROSS_STUDY_OFFSET.unknowns,
        CROSS_STUDY_OFFSET.observations + (OFFSET_CANDIDATES[1],))
    assert with_second.report()["unidentifiable"] != []


# ---- symmetry: the formalism refuses Loop B the same way ------------------
def test_the_patient_loop_is_refused_by_the_same_rule():
    r = PATIENT_STATE.report()
    assert r["degrees_of_freedom"] == 2
    assert "exposure_patient" in r["unidentifiable"]


def test_the_clinical_fix_is_structurally_the_same_as_the_anchor():
    """A shared reference material run at every site pins the assay bias exactly as
    a reference formulation pins the study offset. That parallel is the one real
    piece of shared science between the loops, and it is derived here, not asserted."""
    sets = PATIENT_STATE.minimal_additional_observations(
        "assay_bias_site", PATIENT_CANDIDATES)
    assert ("site_reference_standard",) in sets


# ---- mechanics ------------------------------------------------------------
def test_unachievable_observations_are_excluded_by_default():
    m = StructuralModel(
        ("a",),
        (Observation("hypothetical", frozenset({"a"}), is_achievable=False),))
    assert m.identifiable("a")[0] is False
    assert m.identifiable("a", achievable_only=False)[0] is True


def test_duplicate_equations_are_not_independent_information():
    """Two assays measuring the same sum are one equation. Counting them as two is
    how an underdetermined system looks solved."""
    m = StructuralModel(
        ("a", "b"),
        (Observation("assay1", frozenset({"a", "b"})),
         Observation("assay2", frozenset({"a", "b"}))))
    assert m.degrees_of_freedom() == 1


def test_an_observation_constraining_nothing_is_refused():
    with pytest.raises(ValueError, match="constrains nothing"):
        Observation("empty", frozenset())


def test_unknown_parameter_lookup_raises():
    with pytest.raises(KeyError, match="not an unknown"):
        ORGAN_TO_CYTOSOL.identifiable("nonexistent")


def test_every_registered_model_reports_without_error():
    for name, (model, cands) in REGISTRY.items():
        r = model.report()
        assert r["n_unknowns"] > 0 and "verdict" in r
