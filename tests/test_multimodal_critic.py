"""Data mixing: what each technique assumes, and the one that is refused."""
import numpy as np
import pytest

from atcs.critic import (CriticError, DEFAULT_MECHANISMS, Mechanism, compare,
                         degradation_limited, discriminating_design,
                         release_limited, target_response_limited, uptake_limited)
from atcs.multimodal import (MODALITIES, MixingError, Unit, covariate_shift_weights,
                             early_fusion, inventory, late_fusion, linkage,
                             naive_total, partial_pooling, shared_parameter,
                             technique_table)


# ---- the unit-of-observation problem --------------------------------------
def test_the_corpora_span_several_units_of_observation():
    units = {m.unit for m in MODALITIES.values()}
    assert len(units) >= 5
    assert Unit.PATIENT_VISIT in units and Unit.CELL_LINE_WELL in units


def test_the_naive_row_total_is_computed_in_order_to_be_refused():
    t = naive_total()
    assert t["naive_row_total"] == sum(m.n_rows for m in MODALITIES.values())
    assert t["distinct_units_of_observation"] >= 5
    assert "not a cohort size" in t["verdict"]
    assert t["largest_genuinely_pooled"] < t["naive_row_total"]


def test_the_clinical_corpus_records_patients_not_observations():
    m = MODALITIES["patisiran_clinical"]
    assert m.n_rows == 215 and m.n_groups == 29
    assert "not 215 independent samples" in m.caution


def test_every_modality_declares_a_caution():
    for row in inventory():
        assert len(row["caution"]) > 30


# ---- 1. linkage -----------------------------------------------------------
def test_linkage_requires_formulation_and_batch_not_formulation_alone():
    """Matching on formulation alone pairs a particle with a differently
    manufactured namesake, and batch variation here is not small."""
    a = [{"formulation_id": "LNP-A", "batch_id": "B1"}]
    b = [{"formulation_id": "LNP-A", "batch_id": "B2"}]
    strict = linkage("corona_proteomics", "ntdb_biodistribution", a, b)
    loose = linkage("corona_proteomics", "ntdb_biodistribution", a, b,
                    require=("formulation_id",))
    assert strict["n_linked"] == 0 and loose["n_linked"] == 1


def test_linkage_reports_when_the_units_differ():
    out = linkage("hepg2_response", "ntdb_biodistribution", [], [])
    assert out["units_differ"] is True
    assert "cannot be joined at the row level" in out["verdict"]


# ---- 2. late fusion -------------------------------------------------------
def test_late_fusion_combines_predictions_and_flags_correlated_errors():
    out = late_fusion({"a": [1.0, 2.0], "b": [3.0, 4.0]},
                      variances={"a": 0.01, "b": 0.09})
    assert out["weights"]["a"] > out["weights"]["b"], "lower variance gets more weight"
    assert np.allclose(out["fused"], [1.2, 2.2], atol=0.01)
    assert "overconfident" in out["caveat"]


def test_late_fusion_refuses_misaligned_predictions():
    with_msg = "combines predictions FOR THE SAME ITEMS"
    with pytest.raises(MixingError, match=with_msg):
        late_fusion({"a": [1.0, 2.0], "b": [3.0]})


# ---- 3. partial pooling ---------------------------------------------------
def test_partial_pooling_shrinks_small_groups_more_than_large_ones():
    out = partial_pooling({"big": 1.0, "small": 3.0, "mid": 2.0},
                          {"big": 100, "small": 2, "mid": 20}, within_var=1.0)
    g = out["groups"]
    assert g["small"]["pooling_factor"] > g["big"]["pooling_factor"]
    assert abs(g["small"]["shrunk"] - out["grand_mean"]) < abs(
        g["small"]["raw"] - out["grand_mean"])


def test_the_pooling_factor_shows_when_a_group_was_ignored():
    out = partial_pooling({"a": 1.0, "b": 1.0001}, {"a": 5, "b": 5}, within_var=10.0)
    assert out["groups"]["a"]["pooling_factor"] > 0.9
    assert "that group's data were ignored" in out["note"]


def test_partial_pooling_needs_at_least_two_groups():
    with pytest.raises(MixingError, match="at least two groups"):
        partial_pooling({"a": 1.0}, {"a": 5}, 1.0)


# ---- 4. shared parameter: the test that can refuse ------------------------
def test_consistent_estimates_support_a_shared_parameter():
    out = shared_parameter({"corona": (0.50, 0.05), "pbpk": (0.52, 0.06)},
                           parameter="apoe_affinity")
    assert out["shared_parameter_supported"]
    assert out["pooled_estimate"] is not None
    assert 0.50 <= out["pooled_estimate"] <= 0.52


def test_heterogeneous_estimates_refuse_to_pool_rather_than_widening():
    """The substantive behaviour: if two modalities disagree beyond their stated
    precision they are not measuring the same quantity, and pooling hides that."""
    out = shared_parameter({"corona": (0.20, 0.02), "pbpk": (0.90, 0.02)},
                           parameter="apoe_affinity")
    assert not out["shared_parameter_supported"]
    assert out["pooled_estimate"] is None
    assert "REFUTED" in out["verdict"]
    assert out["i_squared"] > 0.9


def test_a_shared_parameter_needs_two_modalities():
    with pytest.raises(MixingError, match="at least two modalities"):
        shared_parameter({"only": (1.0, 0.1)}, "p")


# ---- 5. reweighting: the support check ------------------------------------
def test_reweighting_refuses_when_the_target_leaves_the_source_support():
    """A clinical cohort contains patients no mouse study covers. A large clipped
    weight is extrapolation wearing a weight's clothing."""
    rng = np.random.default_rng(0)
    src = {"age": rng.normal(8, 1, 200)}           # mouse weeks
    tgt = {"age": rng.normal(60, 10, 200)}         # human years
    out = covariate_shift_weights(src, tgt)
    assert not out["support_adequate"]
    assert "never covered" in out["verdict"]


def test_reweighting_is_usable_under_mild_shift():
    rng = np.random.default_rng(1)
    src = {"size_nm": rng.normal(80, 25, 500)}
    tgt = {"size_nm": rng.normal(85, 20, 200)}
    out = covariate_shift_weights(src, tgt)
    assert out["support_adequate"]
    assert 0 < out["ess_fraction"] <= 1


def test_reweighting_needs_shared_features():
    with pytest.raises(MixingError, match="share no features"):
        covariate_shift_weights({"a": [1.0]}, {"b": [1.0]})


# ---- the refusal ----------------------------------------------------------
def test_early_fusion_always_refuses_and_names_the_alternatives():
    with pytest.raises(MixingError, match="refused"):
        early_fusion("corona_proteomics", "hepg2_response", "patisiran_clinical")
    try:
        early_fusion("corona_proteomics", "patisiran_clinical")
    except MixingError as e:
        s = str(e)
        assert "late_fusion" in s and "partial_pooling" in s and "shared_parameter" in s
        assert "the 20 Oct brief forbids" in s


def test_every_technique_states_what_it_assumes():
    for row in technique_table():
        assert row["assumes"] and len(row["assumes"]) > 15


# ---- the critic -----------------------------------------------------------
def _obs(organ, escape, delay):
    return {"organ_fraction": organ, "escape_fraction": escape, "delay_h": delay}


def test_the_critic_identifies_release_limitation_from_a_pka_series():
    conds = [_obs(0.6, e, 0.0) for e in (0.005, 0.015, 0.030, 0.015, 0.005)]
    truth = [0.6 * c["escape_fraction"] for c in conds]
    out = compare(conds, truth)
    assert out["top"] == "release-limited"
    assert out["decisive"]


def test_a_flat_response_favours_target_limitation():
    conds = [_obs(o, 0.03, 0.0) for o in (0.1, 0.3, 0.6, 0.9)]
    out = compare(conds, [0.3] * 4)
    assert out["top"] == "target-response-limited"


def test_an_indistinguishable_comparison_says_so_and_names_the_experiment():
    """The most useful output is often that the observations do not separate the
    rivals -- a statement about the DATA, not about the models."""
    conds = [_obs(0.5, 0.02, 0.0), _obs(0.5, 0.02, 0.0)]
    out = compare(conds, [0.01, 0.0100001])
    if not out["decisive"]:
        assert "NOT DISTINGUISHABLE" in out["verdict"]
        assert out["next_measurement"]["measurement"]
        assert "signature_a" in out["next_measurement"]


def test_a_single_mechanism_is_refused():
    """Checked BEFORE the observation count: an earlier ordering made this refusal
    unreachable when the caller also passed one observation, so the error named the
    wrong problem and the stricter rule never fired."""
    with pytest.raises(CriticError, match="hypothesis restated"):
        compare([_obs(0.5, 0.02, 0)], [0.01], mechanisms=(uptake_limited(),))
    with pytest.raises(CriticError, match="hypothesis restated"):
        compare([_obs(0.5, 0.02, 0)] * 4, [0.01] * 4,
                mechanisms=(uptake_limited(),))


def test_a_single_observation_is_refused():
    with pytest.raises(CriticError, match="at least two observations"):
        compare([_obs(0.5, 0.02, 0)], [0.01])


def test_bic_penalises_the_mechanism_with_more_freedom():
    conds = [_obs(o, 0.02, 0.0) for o in (0.2, 0.4, 0.6, 0.8)]
    truth = [0.02 * c["organ_fraction"] for c in conds]
    out = compare(conds, truth)
    byname = {r["mechanism"]: r for r in out["ranked"]}
    assert byname["uptake-limited"]["n_parameters"] < byname["release-limited"]["n_parameters"]


def test_the_discriminating_design_ranks_by_disagreement_not_by_fit():
    """Optimising for where the favoured mechanism looks best is how a critic
    becomes a confirmation engine."""
    conds = [_obs(0.0, 0.0, 0.0), _obs(0.9, 0.03, 0.0), _obs(0.9, 0.001, 48.0)]
    out = discriminating_design(DEFAULT_MECHANISMS, conds)
    assert out["best_condition_index"] != 0, "the all-zero condition is uninformative"
    assert out["ranked"][0]["prediction_spread"] >= out["ranked"][-1]["prediction_spread"]
    assert "confirmation engine" in out["note"]


def test_no_candidate_conditions_is_refused():
    with pytest.raises(CriticError, match="no candidate conditions"):
        discriminating_design(DEFAULT_MECHANISMS, [])
