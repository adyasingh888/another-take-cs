"""The two gates, tested against their own limits and against known ground truth."""
import numpy as np
import pytest

from atcs.calibrate import (CalibrationError, anchored_fold_width, bayes_shrink,
                            conformal_interval, coverage, zero_shot_fold_width)


# ---- BayesCalib: limiting behaviour ---------------------------------------
def test_shrinkage_factor_matches_the_briefs_estimator():
    s = bayes_shrink([1.0, 1.2, 0.8], [0.0, 0.0, 0.0], tau0_sq=0.30, sigma_e_sq=0.10)
    k = 3
    assert s.factor == pytest.approx(k * 0.30 / (k * 0.30 + 0.10))
    assert s.b_hat == pytest.approx(s.factor * 1.0)


def test_with_no_study_variance_nothing_is_attributed_to_the_laboratory():
    """tau0^2 -> 0 means there is no study effect to estimate, so the estimator
    must refuse to attribute the residual rather than absorb it."""
    s = bayes_shrink([5.0], [0.0], tau0_sq=0.0, sigma_e_sq=1.0)
    assert s.factor == 0.0 and s.b_hat == 0.0
    assert "measurement noise" in s.interpretation


def test_with_many_anchors_the_raw_residual_is_trusted():
    s = bayes_shrink([1.0] * 500, [0.0] * 500, tau0_sq=0.3, sigma_e_sq=0.1)
    assert s.factor > 0.99
    assert s.b_hat == pytest.approx(1.0, abs=0.01)
    assert "almost entirely" in s.interpretation


def test_more_anchors_shrink_less():
    f = [bayes_shrink([1.0] * k, [0.0] * k, 0.3, 0.1).factor for k in (1, 2, 3, 10)]
    assert f == sorted(f) and f[0] < f[-1]


def test_both_variance_components_are_required():
    """Defaulting sigma_e^2 is how a shrinkage factor becomes a tuning knob."""
    with pytest.raises(TypeError):
        bayes_shrink([1.0], [0.0], tau0_sq=0.3)      # type: ignore[call-arg]


def test_zero_anchors_is_refused_with_the_reason():
    with pytest.raises(CalibrationError, match="not identifiable"):
        bayes_shrink([], [], 0.3, 0.1)


def test_mismatched_anchor_arrays_are_refused():
    with pytest.raises(CalibrationError, match="same measurement"):
        bayes_shrink([1.0, 2.0], [0.0], 0.3, 0.1)


# ---- recovery of a known offset -------------------------------------------
def test_the_estimator_recovers_a_planted_offset():
    """Simulation with ground truth: 200 studies, known offset variance. The mean
    squared error of the shrunk estimate must beat the raw residual, which is the
    entire justification for shrinking."""
    rng = np.random.default_rng(0)
    tau0_sq, sigma_e_sq, k = 0.25, 0.09, 3
    b_true = rng.normal(0, np.sqrt(tau0_sq), 200)
    raw_err, shrunk_err = [], []
    for b in b_true:
        obs = b + rng.normal(0, np.sqrt(sigma_e_sq), k)
        s = bayes_shrink(obs, np.zeros(k), tau0_sq, sigma_e_sq)
        raw_err.append((s.raw_residual - b) ** 2)
        shrunk_err.append((s.b_hat - b) ** 2)
    assert np.mean(shrunk_err) < np.mean(raw_err), "shrinkage must reduce MSE"


# ---- the width argument, as a computation ---------------------------------
def test_zero_shot_width_reproduces_the_briefs_order_of_magnitude():
    """The brief reports 56.8x-158.1x zero-shot. With its own variance split that
    range should be reachable -- checked as a computation, not quoted."""
    w = zero_shot_fold_width(tau0_sq=0.26, sigma_e_sq=0.09, level=0.90)
    assert 40 < w["fold_width"] < 200
    assert 0.8 < w["half_width_log10"] < 1.2


def test_anchoring_narrows_the_interval_and_the_collapse_is_an_identity():
    """The 5-10x collapse follows from the posterior variance of the offset, so it
    is a consequence of the estimator rather than a separate empirical finding --
    worth knowing before attributing it to a model."""
    a = anchored_fold_width(tau0_sq=0.26, sigma_e_sq=0.09, k=3, level=0.90)
    assert a["fold_width"] < a["zero_shot_fold_width"]
    assert a["narrowing_factor"] > 3


def test_anchoring_with_more_references_narrows_further_but_saturates():
    widths = [anchored_fold_width(0.26, 0.09, k)["fold_width"] for k in (1, 2, 3, 10, 100)]
    assert widths == sorted(widths, reverse=True)
    assert widths[-1] / widths[-2] > 0.85, "returns saturate: sigma_e^2 is the floor"


def test_k_zero_is_refused():
    with pytest.raises(CalibrationError, match="at least 1"):
        anchored_fold_width(0.26, 0.09, 0)


# ---- ConformalCalib: the exchangeable unit --------------------------------
def test_row_level_calibration_on_clustered_residuals_is_too_narrow():
    """The conformal analogue of a row-level train/test split, and it is made for
    the same reason. Study-level calibration must give a wider interval."""
    rng = np.random.default_rng(1)
    studies = np.repeat([f"S{i}" for i in range(40)], 10)
    offs = {f"S{i}": rng.normal(0, 0.5) for i in range(40)}
    resid = np.array([offs[s] + rng.normal(0, 0.1) for s in studies])
    row = conformal_interval(resid, None, level=0.90)
    grp = conformal_interval(resid, studies, level=0.90)
    assert grp["half_width"] > row["half_width"]
    assert row["exchangeability_unit"] == "row"
    assert grp["exchangeability_unit"] == "study"
    assert "design effect" in row["note"]


def test_too_few_studies_for_the_requested_level_is_refused():
    """ceil((n+1)*level) <= n is the finite-sample requirement; below it the
    quantile does not exist and returning the max silently would understate."""
    studies = np.repeat(["A", "B", "C"], 5)
    with pytest.raises(CalibrationError, match="too few for a 90% conformal"):
        conformal_interval(np.random.randn(15), studies, level=0.90)


def test_empty_calibration_set_is_refused():
    with pytest.raises(CalibrationError, match="no calibration residuals"):
        conformal_interval([], None)


def test_coverage_is_reported_against_its_target():
    rng = np.random.default_rng(2)
    y = rng.normal(0, 1, 500)
    c = coverage(y, y - 1.96, y + 1.96)
    assert c["empirical_coverage"] == 1.0
    c2 = coverage(y, np.zeros(500) - 0.1, np.zeros(500) + 0.1)
    assert c2["empirical_coverage"] < 0.2


def test_conformal_coverage_holds_at_the_study_level_in_simulation():
    """The guarantee, exercised: calibrate on held-out studies, test on unseen
    ones, and empirical coverage should reach the nominal level."""
    rng = np.random.default_rng(3)
    def draw(n_studies):
        g, r = [], []
        for i in range(n_studies):
            b = rng.normal(0, 0.5)
            m = rng.integers(5, 15)
            g += [f"S{i}"] * m
            r += list(b + rng.normal(0, 0.1, m))
        return np.array(g), np.array(r)
    gc, rc = draw(60)
    half = conformal_interval(rc, gc, level=0.90)["half_width"]
    _, rt = draw(60)
    assert coverage(rt, -half * np.ones(rt.size), half * np.ones(rt.size))[
        "empirical_coverage"] >= 0.88
