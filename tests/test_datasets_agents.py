"""The data wall, and the only agent metric that can show an advantage."""
import numpy as np
import pandas as pd
import pytest

from atcs.agents import (AgentError, ROLE_CONTRACT, RefusalCase, Role,
                         contract_table, score_refusals, versus_baselines)
from atcs.corona import CompositionError, round_composition, softmax_composition
from atcs.datasets import (DataProvenanceError, as_measurement, corpus_registry,
                           generate_biodistribution, generate_corona,
                           generate_escape_panel, generate_patient_series,
                           reported_ablation, reported_variance)


# ---- reported figures, with the one thing recomputed ----------------------
def test_the_reported_variance_shares_sum_to_one():
    v = reported_variance()
    assert v.sums_to_one.all()
    assert v.between_study_share.min() >= 0.748
    assert v.between_study_share.max() <= 0.882


def test_the_formulation_ceiling_follows_from_the_reported_shares():
    v = reported_variance()
    assert v.max_r2_from_formulation_alone.max() <= 0.252
    assert v.max_r2_from_formulation_alone.min() >= 0.118


def test_the_sign_test_recomputes_to_the_reported_p_value():
    """Transcribed shares, but THIS p-value is arithmetic over them -- so it can be
    checked, and it matches the brief's reported 0.0312 exactly."""
    a = reported_ablation()
    st = a.attrs["sign_test"]
    assert st["n_improved"] == 6 and st["n_organs"] == 6
    assert st["exact_binomial_p"] == pytest.approx(0.03125)
    assert st["matches"] is True


def test_pruning_improved_every_organ_as_reported():
    assert reported_ablation().pruning_improved.all()


def test_reported_rows_carry_the_brief_as_their_origin():
    for df in (reported_variance(), reported_ablation(), corpus_registry()):
        assert (~df.is_synthetic).all()
        assert df.origin.str.contains("transcribed, not reproduced").all()


# ---- generated tables reproduce STRUCTURE, not values ---------------------
def test_generated_biodistribution_has_the_reported_clustering_structure():
    df = generate_biodistribution(n_studies=86)
    assert df.study_id.nunique() == 86
    assert df.is_synthetic.all()
    assert (df.groupby("study_id").size() > 1).any(), "multi-arm studies must exist"


def test_the_generated_between_study_variance_share_matches_the_reported_one():
    """The shape worth reproducing, measured with the RIGHT estimator.

    The first version of this test used var(study means) / var(all) and failed:
    0.861 against a target of 0.748. The generator was correct and the test was
    wrong -- that ratio estimates (tau2 + sigma2/m) / (tau2 + sigma2) and so
    exceeds the ICC for every m > 1. With four arms per study the overstatement is
    large. This is the same upward bias that group-count effect sizes carry, and
    it is how a reported variance decomposition can be too high without anyone
    making an arithmetic error.
    """
    from atcs.multimodal import icc_oneway
    df = generate_biodistribution(n_studies=300, seed=7)
    for organ, target in (("liver", 0.748), ("heart", 0.882), ("lung", 0.841)):
        r = icc_oneway(df[f"log10_pid_{organ}"], df.study_id)
        assert abs(r["icc"] - target) < 0.08, (
            f"{organ}: ICC {r['icc']} against target {target}")
        assert r["naive_overstatement"] > 0.02, (
            "the naive ratio must be demonstrably higher, or this test is not "
            "exercising the distinction it exists for")


def test_generated_corona_satisfies_the_simplex_at_written_precision():
    """Caught a real defect: naive rounding of a composition breaks sum-to-one, and
    the artefact would have been written to disk violating the invariant the module
    enforces."""
    df = generate_corona(n_studies=40)
    cols = [c for c in df.columns if c.startswith("share_")]
    assert np.allclose(df[cols].sum(axis=1), 1.0, atol=1e-9)


def test_largest_remainder_rounding_preserves_the_simplex():
    s = softmax_composition({k: v for k, v in zip("abcdef", [0.3, -1.2, 0.7, 2.1, -0.4, 0.0])})
    naive = {k: round(v, 5) for k, v in s.items()}
    fixed = round_composition(s, dp=5)
    assert abs(sum(fixed.values()) - 1.0) < 1e-12
    assert all(abs(fixed[k] - s[k]) <= 1e-5 + 1e-12 for k in s)
    # the naive version is what would have shipped
    assert abs(sum(naive.values()) - 1.0) <= 3e-5


def test_rounding_an_empty_composition_is_refused():
    with pytest.raises(CompositionError, match="no composition"):
        round_composition({})


def test_the_escape_panel_spans_the_optimum_on_both_sides():
    df = generate_escape_panel(n=300)
    assert set(df.side_of_optimum) >= {"below", "above"}
    best = df.loc[df.escape_fraction.idxmax()]
    assert 5.8 <= best.pka <= 7.0


def test_the_patient_series_has_the_reported_patient_and_visit_structure():
    """Generated so the observations-are-not-patients error can be DEMONSTRATED on
    a table rather than only described."""
    df = generate_patient_series(n_patients=29)
    assert df.patient_id.nunique() == 29
    assert 150 < len(df) < 320
    assert (df.groupby("patient_id").size() > 1).all()


# ---- the wall -------------------------------------------------------------
def test_generated_data_cannot_support_a_measurement_claim():
    with pytest.raises(DataProvenanceError, match="the 20 Oct brief forbids"):
        as_measurement(generate_biodistribution(n_studies=5), "organ delivery R2")


def test_reported_data_can():
    df = as_measurement(reported_variance(), "the variance decomposition")
    assert len(df) == 6


def test_a_frame_with_unknown_provenance_is_refused():
    with pytest.raises(DataProvenanceError, match="provenance is unknown"):
        as_measurement(pd.DataFrame({"x": [1, 2]}), "anything")


def test_there_is_no_override_flag_on_the_wall():
    """The entire value of the rule is that a caller in a hurry cannot bypass it."""
    import inspect
    sig = inspect.signature(as_measurement)
    assert set(sig.parameters) == {"df", "claim"}


def test_every_generated_frame_records_what_it_does_not_reproduce():
    for gen in (generate_biodistribution(n_studies=5), generate_corona(n_studies=5),
                generate_escape_panel(n=20), generate_patient_series(n_patients=5)):
        assert gen.origin.str.contains("does NOT reproduce").all()


# ---- agents: the one metric that can distinguish them ---------------------
def _cases(recall, n=20, false_refusals=0):
    out = []
    n_should = n // 2
    for i in range(n_should):
        out.append(RefusalCase(f"u{i}", True, i < int(recall * n_should)))
    for i in range(n - n_should):
        out.append(RefusalCase(f"s{i}", False, i < false_refusals))
    return out


def test_refusal_scoring_returns_a_confusion_table_and_no_single_score():
    s = score_refusals(_cases(0.8, 20, 2))
    assert s["correct_refusals"] == 8 and s["missed_refusals"] == 2
    assert s["false_refusals"] == 2
    assert "f1" not in {k.lower() for k in s}
    assert "not interchangeable" in s["no_single_score"]


def test_missed_refusals_are_named_so_they_can_be_inspected():
    s = score_refusals(_cases(0.5, 20))
    assert len(s["missed_refusal_cases"]) == 5


def test_no_advantage_is_reported_as_a_finding_not_buried():
    same = _cases(0.7, 20)
    out = versus_baselines(same, same)
    assert out["refusal_recall_advantage"] == 0.0
    assert "not a null to bury" in out["verdict"]


def test_a_real_advantage_is_reported_with_its_false_refusal_cost():
    out = versus_baselines(_cases(1.0, 20, 3), _cases(0.4, 20, 0))
    assert out["refusal_recall_advantage"] > 0.5
    assert out["additional_false_refusal_rate"] > 0
    assert "additional false-refusal rate" in out["verdict"]


def test_the_general_agent_arm_is_optional_but_scored_when_given():
    out = versus_baselines(_cases(0.9, 20), _cases(0.5, 20), _cases(0.6, 20))
    assert "general_biomedical_agent" in out["arms"]


def test_scoring_nothing_is_refused():
    with pytest.raises(AgentError, match="no cases"):
        score_refusals([])


def test_every_role_declares_what_it_may_not_do():
    """Where an agent architecture actually fails: roles that quietly acquire each
    other's powers."""
    assert len(ROLE_CONTRACT) == len(Role) == 5
    for row in contract_table():
        assert len(row["forbidden"]) > 40
        assert row["checkable_by"].startswith("atcs.")
