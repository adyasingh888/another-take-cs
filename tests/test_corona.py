"""Corona composition is compositional, and the arithmetic must respect that."""
import numpy as np
import pytest

from atcs.corona import (CompositionError, SPECIES, apoe_c3_index, log_ratio,
                         softmax_composition, spurious_correlation_warning)


def test_softmax_output_is_always_on_the_simplex():
    for scale in (0.01, 1.0, 50.0):
        s = softmax_composition({k: scale * i for i, k in enumerate(SPECIES)})
        assert sum(s.values()) == pytest.approx(1.0)
        assert all(v >= 0 for v in s.values())


def test_the_simplex_holds_for_extreme_logits():
    """A share of 1.3 is a category error, not a small error."""
    s = softmax_composition({"ApoE": 800.0, "C3": -800.0, "other": 0.0})
    assert sum(s.values()) == pytest.approx(1.0)
    assert 0.0 <= s["C3"] <= 1.0 and s["ApoE"] == pytest.approx(1.0)


def test_the_state_group_invariant_accepts_softmax_output():
    """The Stage 1 output must satisfy x_id's declared invariant -- the registry
    checks it on write, so this is the contract both sides agree to."""
    from atcs.state import GROUPS
    GROUPS["x_id"].check(softmax_composition({k: 0.3 * i for i, k in enumerate(SPECIES)}))


def test_a_raw_regression_output_is_rejected_by_that_invariant():
    from atcs.state import GROUPS, InvariantViolation
    with pytest.raises(InvariantViolation, match="not 1"):
        GROUPS["x_id"].check({"ApoE": 0.7, "C3": 0.9})


def test_log_ratio_is_scale_free_where_a_difference_is_not():
    low = {"ApoE": 0.02, "C3": 0.01, "other": 0.97}
    high = {"ApoE": 0.60, "C3": 0.30, "other": 0.10}
    assert log_ratio(low, "ApoE", "C3") == pytest.approx(log_ratio(high, "ApoE", "C3"))
    assert (low["ApoE"] - low["C3"]) != pytest.approx(high["ApoE"] - high["C3"])


def test_a_zero_denominator_hits_a_stated_floor_not_a_silent_number():
    s = {"ApoE": 0.5, "C3": 0.0, "other": 0.5}
    assert log_ratio(s, "ApoE", "C3", floor=1e-6) == pytest.approx(np.log10(0.5 / 1e-6))


def test_an_unknown_species_is_refused():
    with pytest.raises(CompositionError, match="not a species"):
        log_ratio({"ApoE": 0.5, "other": 0.5}, "ApoE", "C3")


def test_the_targeting_index_reports_direction_not_just_magnitude():
    hep = apoe_c3_index({"ApoE": 0.45, "C3": 0.05, "other": 0.50})
    pha = apoe_c3_index({"ApoE": 0.03, "C3": 0.40, "other": 0.57})
    assert "hepatocyte" in hep["favours"] and hep["log10_apoe_c3"] > 0
    assert "phagocyte" in pha["favours"] and pha["log10_apoe_c3"] < 0


def test_a_balanced_corona_favours_neither():
    assert "neither strongly" in apoe_c3_index(
        {"ApoE": 0.2, "C3": 0.2, "other": 0.6})["favours"]


def test_negative_correlation_between_shares_is_a_constraint_artefact():
    """'ApoE and C3 compete for the surface' is what a raw correlation invites and
    cannot support. Constructed so the two are INDEPENDENT in log-ratio space; the
    raw share correlation must still come out negative."""
    rng = np.random.default_rng(0)
    comps = []
    for _ in range(300):
        comps.append(softmax_composition({
            "ApoE": rng.normal(0, 1), "C3": rng.normal(0, 1), "other": 0.0}))
    r = spurious_correlation_warning(comps, "ApoE", "C3")
    assert r["raw_share_correlation"] < -0.3, "the constraint forces this"
    assert abs(r["log_ratio_correlation"]) < abs(r["raw_share_correlation"])
    assert "whatever the biology" in r["note"]


def test_too_few_compositions_to_correlate_is_refused():
    with pytest.raises(CompositionError, match="at least 3"):
        spurious_correlation_warning([{"ApoE": 0.5, "C3": 0.5, "other": 0.0}], "ApoE", "C3")


def test_non_finite_logits_are_refused():
    with pytest.raises(CompositionError, match="non-finite"):
        softmax_composition({"ApoE": float("nan"), "other": 0.0})
