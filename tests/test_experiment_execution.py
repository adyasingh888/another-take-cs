"""Experiment specification, the execution engine, M15 and the planner."""
import numpy as np
import pytest

from atcs.execution import Engine, ExecutionError, digest, environment_fingerprint
from atcs.experiment import ExperimentSpec, Resource, SpecificationError, rank
from atcs.icm15 import (ACTIVITIES, Consequence, ContextOfUse, CredibilityError,
                        Influence, PROPOSED_FIRST_CONTEXT, credibility_plan,
                        model_risk, required_activities)
from atcs.models import CYTOSOL_CANDIDATES, ORGAN_TO_CYTOSOL
from atcs.observability import Observation
from atcs.propagate import Dependence, Uncertain
from atcs.taint import Taint


CAND = {c.name: c for c in CYTOSOL_CANDIDATES}


def spec(**kw):
    base = dict(
        name="flow-sorted parenchymal fraction",
        measures=(CAND["flow_sorted_parenchymal"],),
        targets="f_parenchymal", effect_size=0.30, n_groups=12, m_per_group=4,
        rho=0.75, residual_sd=0.35,
        resource=Resource(animals=48, assay_runs=12, weeks=6.0),
        rationale="closes the parenchymal fraction, which no whole-organ assay can")
    base.update(kw)
    return ExperimentSpec(**base)


# ---- power at the effective sample size -----------------------------------
def test_the_design_effect_is_applied_and_the_gap_is_reported():
    """At the reported clustering the naive row count overstates power by a factor
    larger than most effects anyone is looking for."""
    p = spec().power()
    assert p["design_effect"] == pytest.approx(1 + 3 * 0.75)
    assert p["n_effective"] < p["n_raw"]
    assert p["power_if_rows_were_independent"] > p["power"]
    assert "effective sample size" in p["note"]


def test_groups_not_rows_are_what_buy_power():
    """Adding repeated measures within a group buys almost nothing once rho is high,
    which is the practical lesson a row-count calculation hides."""
    more_groups = spec(n_groups=40).power()["power"]
    more_reps = spec(m_per_group=40).power()["power"]
    assert more_groups > more_reps


def test_groups_needed_solves_on_groups():
    s = spec(n_groups=3)
    assert not s.power()["adequate"]
    need = s.groups_needed(0.80)
    assert need > 3
    assert spec(n_groups=need).power()["power"] >= 0.78


def test_a_single_group_design_is_refused():
    with pytest.raises(SpecificationError, match="at least two independent groups"):
        spec(n_groups=1)


def test_an_unargued_proposal_is_refused():
    with pytest.raises(SpecificationError, match="unargued proposal"):
        spec(rationale="do it")


# ---- identifiability ------------------------------------------------------
def test_nothing_is_identifiable_before_a_new_measurement():
    assert spec().closes(ORGAN_TO_CYTOSOL)["identifiable_before"] is False


def test_a_measurement_that_closes_the_target_actually_closes_it():
    out = spec().closes(ORGAN_TO_CYTOSOL)
    assert out["closes_it"] and not out["insufficient"]


def test_an_insufficient_measurement_says_so_rather_than_passing():
    weak = spec(measures=(Observation("total_only", frozenset({
        "f_parenchymal", "f_macrophage", "f_endothelial", "f_extracellular"})),))
    out = weak.closes(ORGAN_TO_CYTOSOL)
    assert out["insufficient"]
    assert "remains unidentifiable" in out["verdict"]


def test_an_experiment_may_yield_several_equations_from_one_sample():
    """A flow-sorting run measures every sorted fraction in the same sample. An
    earlier version took a single Observation and could not express that, so every
    multi-readout assay looked insufficient."""
    multi = spec(name="sort + release", targets="phi_intact",
                 measures=(CAND["flow_sorted_parenchymal"], CAND["cargo_release_assay"]))
    assert multi.closes(ORGAN_TO_CYTOSOL)["closes_it"]
    with pytest.raises(SpecificationError, match="is a TUPLE"):
        spec(measures=CAND["flow_sorted_parenchymal"])


def test_an_experiment_cannot_target_a_parameter_the_model_lacks():
    with pytest.raises(SpecificationError, match="not an unknown"):
        spec(targets="nonexistent").closes(ORGAN_TO_CYTOSOL)


def test_assess_names_the_binding_constraint():
    """Three failures, three different fixes -- so which one failed is the output."""
    a = spec(n_groups=3).assess(ORGAN_TO_CYTOSOL)
    assert not a["worth_running"]
    assert a["binding_constraint"] in {"identifiability", "power", "feasibility"}


def test_a_capability_the_lab_lacks_is_a_blocker():
    a = spec(resource=Resource(animals=48, requires_new_capability="a cell sorter")
             ).assess(ORGAN_TO_CYTOSOL)
    assert any(c == "feasibility" for c, _ in a["blockers"])


def test_ranking_puts_non_closing_experiments_last_regardless_of_cost():
    """A cheap measurement that resolves nothing is not cheap -- it is wasted."""
    cheap_useless = spec(name="cheap", measures=(Observation(
        "homogenate_again", frozenset({"f_parenchymal", "f_macrophage",
                                       "f_endothelial", "f_extracellular"})),),
        resource=Resource(animals=4, assay_runs=1, weeks=0.5))
    out = rank([cheap_useless, spec()], ORGAN_TO_CYTOSOL)
    assert out["n_closing"] == 1
    assert out["ranked"][-1]["experiment"] == "cheap"
    assert out["ranked"][0]["closes"] is True


def test_a_field_where_nothing_closes_the_target_returns_no_ordering():
    """Caught by a test that expected an order and got an arbitrary one: with every
    value_per_cost at zero the sort was stable on input order, which is not a
    ranking -- and a ranked list of useless experiments invites running the
    cheapest."""
    useless = Observation("homogenate_again", frozenset({
        "f_parenchymal", "f_macrophage", "f_endothelial", "f_extracellular"}))
    out = rank([spec(name="a", measures=(useless,),
                     resource=Resource(animals=4)),
                spec(name="b", measures=(useless,),
                     resource=Resource(animals=400))], ORGAN_TO_CYTOSOL)
    assert out["ranked"] is None and out["n_closing"] == 0
    assert "No ordering is returned" in out["headline"]
    assert len(out["candidates"]) == 2


# ---- the execution engine -------------------------------------------------
def test_an_unversioned_model_cannot_be_registered():
    e = Engine()
    with pytest.raises(ExecutionError, match="unattributable"):
        e.register("m", "  ")


def test_running_an_unregistered_model_is_refused():
    e = Engine()
    with pytest.raises(ExecutionError, match="not registered"):
        e.run("m", lambda x: x, {"x": 1}, taint=Taint.MEASURED,
              dependence=Dependence.UNKNOWN)


def test_identical_runs_get_identical_run_ids():
    e = Engine(); e.register("m", "1.0")
    a = e.run("m", lambda x: x * 2, {"x": 3}, taint=Taint.MEASURED,
              dependence=Dependence.DEPENDENT)
    b = e.run("m", lambda x: x * 2, {"x": 3}, taint=Taint.MEASURED,
              dependence=Dependence.DEPENDENT)
    assert a["run_id"] == b["run_id"]


def test_a_different_version_gives_a_different_run_id():
    e = Engine(); e.register("m", "1.0")
    a = e.run("m", lambda x: x * 2, {"x": 3}, taint=Taint.MEASURED,
              dependence=Dependence.DEPENDENT)
    e.register("m", "2.0")
    b = e.run("m", lambda x: x * 2, {"x": 3}, taint=Taint.MEASURED,
              dependence=Dependence.DEPENDENT)
    assert a["run_id"] != b["run_id"]


def test_the_digest_is_stable_under_floating_point_noise():
    """Without rounding, a different BLAS gives different digests for runs that are
    scientifically identical -- so the check would report irreproducibility where
    there is none, and be ignored."""
    assert digest({"x": 0.1 + 0.2}) == digest({"x": 0.30000000000000004})
    assert digest({"x": 0.3}) == digest({"x": 0.30000000000001})
    assert digest({"x": 0.3}) != digest({"x": 0.31})


def test_an_unseeded_stochastic_model_fails_the_reproducibility_check_loudly():
    e = Engine(); e.register("rand", "1.0")
    out = e.reproduce_check("rand", lambda n: float(np.random.rand()), {"n": 1},
                            taint=Taint.SIMULATED, dependence=Dependence.UNKNOWN)
    assert not out["deterministic"]
    assert "not attributable to its inputs" in out["verdict"]


def test_a_deterministic_model_passes():
    e = Engine(); e.register("det", "1.0")
    out = e.reproduce_check("det", lambda n: n ** 2, {"n": 7},
                            taint=Taint.MEASURED, dependence=Dependence.DEPENDENT)
    assert out["deterministic"] and out["distinct_outputs"] == 1


def test_undeclared_dependence_yields_a_bound_not_a_joint_uncertainty():
    """Multiplying marginals and calling the product joint is the default behaviour
    of most pipelines, and is wrong whenever the stages share a parameter."""
    e = Engine()
    stages = [Uncertain(0.3, 0.2, "organ"), Uncertain(0.4, 0.3, "partition")]
    unk = e.compose("chain", stages, Dependence.UNKNOWN)
    dep = e.compose("chain", stages, Dependence.DEPENDENT)
    assert unk["is_joint"] is False and "a BOUND" in unk["interpretation"]
    assert dep["is_joint"] is True


def test_the_ledger_attributes_a_prediction_to_its_run():
    e = Engine(); e.register("m", "3.1")
    r = e.run("m", lambda x: x + 1, {"x": 1}, taint=Taint.DERIVED,
              dependence=Dependence.DEPENDENT)
    att = e.attribution(r["run_id"])
    assert att["version"] == "3.1" and att["taint"] == "DERIVED"
    assert "numpy" in att["environment"]
    assert len(e.ledger()) == 1


def test_an_unlocatable_run_is_refused():
    with pytest.raises(ExecutionError, match="not attributable"):
        Engine().attribution("deadbeef")


# ---- ICH M15 --------------------------------------------------------------
def test_risk_is_the_combination_not_the_consequence_alone():
    """The component usually left unassessed: a low-consequence decision taken
    entirely on a model still carries real risk."""
    low_both = model_risk(Influence.SUPPORTING, Consequence.WASTED_EFFORT)
    high_infl = model_risk(Influence.SUBSTITUTING, Consequence.WASTED_EFFORT)
    assert high_infl["risk_score"] > low_both["risk_score"]
    assert "nothing else will catch the error" in high_infl["note"]


def test_requirements_are_cumulative_across_bands():
    low = set(required_activities("low"))
    high = set(required_activities("high"))
    assert low < high
    assert "prospective_validation" in required_activities("very high")
    assert "prospective_validation" not in high


def test_an_unbounded_context_of_use_is_refused():
    with pytest.raises(CredibilityError, match="cannot have its evidence"):
        ContextOfUse("too short", "a real decision statement here", "a real scope here",
                     Influence.PRIMARY, Consequence.WRONG_PRIORITY)


def test_the_plan_reports_the_gap_as_a_gate_not_a_percentage():
    plan = credibility_plan(PROPOSED_FIRST_CONTEXT,
                            ["code_verification", "applicability_bounds",
                             "sensitivity_analysis"])
    assert not plan["credible_for_this_context"]
    assert plan["missing"]
    assert "not a percentage to be nearly met" in plan["verdict"]


def test_meeting_the_requirements_is_a_context_statement_not_a_qualification():
    plan = credibility_plan(PROPOSED_FIRST_CONTEXT,
                            required_activities(model_risk(
                                PROPOSED_FIRST_CONTEXT.influence,
                                PROPOSED_FIRST_CONTEXT.consequence)["risk_band"]))
    assert plan["credible_for_this_context"]
    assert "is not a qualification of the model" in plan["verdict"]


def test_unrecognised_evidence_is_refused_rather_than_counted():
    with pytest.raises(CredibilityError, match="unrecognised evidence"):
        credibility_plan(PROPOSED_FIRST_CONTEXT, ["vibes"])


def test_the_briefs_own_proposed_context_is_assessable():
    r = model_risk(PROPOSED_FIRST_CONTEXT.influence, PROPOSED_FIRST_CONTEXT.consequence)
    assert r["risk_band"] in {"moderate", "high"}
    assert "not human dosing" in PROPOSED_FIRST_CONTEXT.decision
