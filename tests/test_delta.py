"""The contrast target's algebra, its price, and when it is not worth paying."""
import numpy as np
import pytest

from atcs.delta import (Contrast, ContrastError, advantage, build_contrasts,
                        offset_cancellation)


def _panel(n_studies=40, arms=3, tau0=0.6, sigma_e=0.12, seed=0):
    rng = np.random.default_rng(seed)
    eff = {f"F{j}": rng.normal(0, 0.3) for j in range(arms)}
    study, arm, y, size, offs = [], [], [], [], {}
    for i in range(n_studies):
        b = rng.normal(0, tau0)
        offs[f"S{i}"] = b
        for j in range(arms):
            study.append(f"S{i}"); arm.append(f"F{j}")
            y.append(eff[f"F{j}"] + b + rng.normal(0, sigma_e))
            size.append(60 + 10 * j)
    return (np.array(study), np.array(arm), np.array(y), {"size_nm": np.array(size)},
            offs)


# ---- the algebra ----------------------------------------------------------
def test_differencing_removes_the_study_offset_exactly():
    """The method's whole justification, computed on data whose offsets are known."""
    s, a, y, u, offs = _panel()
    r = offset_cancellation(s, y, offs)
    assert r["offset_share_of_absolute"] > 0.7
    assert abs(r["offset_correlation_with_contrast"]) < 1e-6
    assert r["var_within_study_contrast"] < r["var_absolute_target"] / 3


def test_contrasts_are_built_against_the_named_reference_only():
    s, a, y, u, _ = _panel(n_studies=3, arms=3)
    cs = build_contrasts(s, a, y, u, reference={f"S{i}": "F0" for i in range(3)})
    assert len(cs) == 3 * 2
    assert all(c.reference_arm == "F0" for c in cs)
    assert all(c.arm != "F0" for c in cs)


def test_a_single_arm_study_contributes_nothing():
    """The method's standing limitation, not a configuration problem."""
    s = np.array(["S0", "S1", "S1"]); a = np.array(["F0", "F0", "F1"])
    y = np.array([1.0, 2.0, 2.5]); u = {"size_nm": np.array([60, 60, 80])}
    cs = build_contrasts(s, a, y, u, reference={"S0": "F0", "S1": "F0"})
    assert [c.study for c in cs] == ["S1"]


def test_a_study_with_no_named_reference_is_skipped_not_guessed():
    """Picking the first arm by row order would make the target depend on
    spreadsheet ordering, which is not a modelling choice anyone would defend."""
    s = np.array(["S0", "S0"]); a = np.array(["F0", "F1"])
    y = np.array([1.0, 1.4]); u = {"size_nm": np.array([60, 90])}
    with pytest.raises(ContrastError, match="no reference arm was named"):
        build_contrasts(s, a, y, u, reference={})
    assert len(build_contrasts(s, a, y, u, all_pairs=True)) == 1


def test_descriptor_differences_travel_with_the_target():
    s, a, y, u, _ = _panel(n_studies=2, arms=3)
    cs = build_contrasts(s, a, y, u, reference={"S0": "F0", "S1": "F0"})
    assert all("size_nm" in c.du for c in cs)
    assert cs[0].du["size_nm"] == pytest.approx(10.0)


def test_misaligned_descriptors_are_refused():
    s = np.array(["S0", "S0"]); a = np.array(["F0", "F1"]); y = np.array([1.0, 2.0])
    with pytest.raises(ContrastError, match="expected 2"):
        build_contrasts(s, a, y, {"bad": [1.0]}, reference={"S0": "F0"})


def test_all_pairs_yields_more_contrasts_but_they_share_arms():
    s, a, y, u, _ = _panel(n_studies=1, arms=4)
    ref = build_contrasts(s, a, y, u, reference={"S0": "F0"})
    allp = build_contrasts(s, a, y, u, all_pairs=True)
    assert len(allp) == 6 and len(ref) == 3
    assert all(c.study == "S0" for c in allp), "study label must survive for clustering"


# ---- the price ------------------------------------------------------------
def test_contrasting_helps_only_when_the_offset_exceeds_measurement_noise():
    """A condition, not a law. A quieter literature could fail it."""
    good = advantage(tau0_sq=0.36, sigma_e_sq=0.014)
    assert good["favourable"] and "reduces the noise" in good["verdict"]
    bad = advantage(tau0_sq=0.01, sigma_e_sq=0.25)
    assert not bad["favourable"] and "INCREASES the noise" in bad["verdict"]


def test_the_break_even_is_at_equal_variance_components():
    at = advantage(tau0_sq=0.2, sigma_e_sq=0.2)
    assert at["offset_share"] == pytest.approx(0.5)
    assert at["variance_ratio"] == pytest.approx(1.0)
    assert not at["favourable"], "strict inequality: equal components are no gain"


def test_at_the_briefs_reported_offset_share_the_trade_is_favourable():
    a = advantage(tau0_sq=0.26, sigma_e_sq=0.09)
    assert 0.70 < a["offset_share"] < 0.80
    assert a["favourable"] and a["variance_ratio"] < 1.0


def test_negative_variances_are_refused():
    with pytest.raises(ContrastError, match="cannot be negative"):
        advantage(-1.0, 0.1)
