"""Conformal intervals under grouped splits, and decision value."""
import numpy as np
import pandas as pd
import pytest

from atcs.benchmark import (BenchmarkError, _split_three, interval_scorecard,
                            interval_summary)
from atcs.datasets import TOTAL_LOG10_SD, generate_biodistribution
from atcs.multimodal import icc_oneway
from atcs.value import (Candidate, ValueError_, decision_now,
                        expected_value_of_information, value_per_cost)

FEATS = ["size_nm", "zeta_mv", "pdi", "dose_mg_per_kg"]


@pytest.fixture(scope="module")
def df():
    return generate_biodistribution(n_studies=200, seed=11)


# ---- the three-way split --------------------------------------------------
def test_calibration_gets_its_own_groups(df):
    """Split conformal guarantees coverage under exchangeability, and for clustered
    data the exchangeable unit is the STUDY. Calibrating on rows from training
    studies gives a quantile fitted to within-study residuals, which undercovers on
    a new study while reporting nominal coverage on its own calibration set."""
    g = df.study_id.astype(str).to_numpy()
    train, cal, test = _split_three(g, seed=0)
    gt, gc, gte = set(g[train]), set(g[cal]), set(g[test])
    assert not (gt & gc) and not (gt & gte) and not (gc & gte)
    assert len(train) + len(cal) + len(test) == len(g)


def test_a_corpus_too_small_to_spare_calibration_groups_is_refused():
    g = np.array(list("ABCD"))
    with pytest.raises(BenchmarkError, match="needs its own groups"):
        _split_three(g, frac_cal=0.5, frac_test=0.5)


# ---- the generator's absolute scale ---------------------------------------
def test_the_generator_reproduces_the_icc_and_the_absolute_scale(df):
    """Before v0.8 it drew each component at sqrt(variance SHARE), so the ICC was
    right and the absolute spread was twice too wide -- generated fold-widths were
    two orders larger than the brief's and not comparable to it."""
    r = icc_oneway(df["log10_pid_liver"], df.study_id)
    assert abs(r["icc"] - 0.748) < 0.08
    assert abs(float(df["log10_pid_liver"].std()) - TOTAL_LOG10_SD) < 0.12


def test_the_scale_is_a_parameter_and_preserves_the_icc():
    wide = generate_biodistribution(n_studies=150, seed=5, total_sd=1.0)
    narrow = generate_biodistribution(n_studies=150, seed=5, total_sd=0.25)
    icc_w = icc_oneway(wide["log10_pid_liver"], wide.study_id)["icc"]
    icc_n = icc_oneway(narrow["log10_pid_liver"], narrow.study_id)["icc"]
    assert abs(icc_w - icc_n) < 0.06, "scaling must not change the variance SHARE"
    assert wide["log10_pid_liver"].std() > 3 * narrow["log10_pid_liver"].std()


# ---- the interval claim, executed -----------------------------------------
def test_the_zero_shot_interval_is_in_the_briefs_reported_range(df):
    """Section 6.3 puts a zero-shot 90% interval at 57x to 158x fold-width."""
    iv = interval_scorecard(df, "log10_pid_liver", FEATS, n_repeats=6, seed=0)
    s = interval_summary(iv)
    zs = s["arms"]["zero-shot"]["median_fold_width"]
    assert 40 < zs < 400, f"zero-shot fold width {zs}"


def test_anchoring_narrows_the_interval(df):
    iv = interval_scorecard(df, "log10_pid_liver", FEATS, n_repeats=6, seed=0)
    s = interval_summary(iv)
    assert s["narrowing_factor_fold_width"] > 1.5
    assert s["narrowing_factor_log_width"] > 1.0


def test_the_fold_width_narrowing_is_not_scale_free_and_says_so(df):
    """Rescaling the generator moved the same anchoring effect from 10.5x to 3.3x
    with nothing about the method changed. The log-width ratio is the invariant."""
    wide = generate_biodistribution(n_studies=200, seed=11, total_sd=1.0)
    s_n = interval_summary(interval_scorecard(df, "log10_pid_liver", FEATS,
                                              n_repeats=5, seed=0))
    s_w = interval_summary(interval_scorecard(wide, "log10_pid_liver", FEATS,
                                              n_repeats=5, seed=0))
    assert s_w["narrowing_factor_fold_width"] > s_n["narrowing_factor_fold_width"] * 1.5
    assert abs(s_w["narrowing_factor_log_width"]
               - s_n["narrowing_factor_log_width"]) < 0.25
    assert "not scale-free" in s_n["scale_note"]


def test_coverage_has_three_outcomes_not_two(df):
    """An earlier version called anything other than 'contains nominal' NOT
    retained, and reported over-coverage with the verdict written for
    under-coverage. Opposite failures, opposite fixes."""
    iv = interval_scorecard(df, "log10_pid_liver", FEATS, n_repeats=6, seed=0)
    s = interval_summary(iv)
    assert s["coverage_status"] in {"undercovers", "calibrated", "conservative"}
    if s["coverage_status"] == "conservative":
        assert s["coverage_retained"] is True
        assert "safe but wider than it needs to be" in s["verdict"]
    if s["coverage_status"] == "undercovers":
        assert s["coverage_retained"] is False
        assert "is a lie" in s["verdict"] or "stopped covering" in s["verdict"]


def test_the_anchor_rows_are_excluded_from_the_interval_evaluation(df):
    iv = interval_scorecard(df, "log10_pid_liver", FEATS, n_repeats=3, seed=0)
    anc = iv[iv.arm.str.startswith("anchored")]
    zs = iv[iv.arm == "zero-shot"]
    assert (anc.anchor_rows_excluded > 0).all()
    assert anc.n_test_rows.sum() < zs.n_test_rows.sum()


def test_the_interval_summary_needs_both_arms(df):
    iv = interval_scorecard(df, "log10_pid_liver", FEATS, n_repeats=2, seed=0)
    with pytest.raises(BenchmarkError, match="both a zero-shot and an anchored"):
        interval_summary(iv[iv.arm == "zero-shot"])


def test_coverage_is_reported_with_its_own_interval(df):
    """A coverage number without its own uncertainty invites over-reading."""
    s = interval_summary(interval_scorecard(df, "log10_pid_liver", FEATS,
                                            n_repeats=4, seed=0))
    for arm in s["arms"].values():
        lo, hi = arm["coverage_95ci"]
        assert lo < arm["empirical_coverage"] < hi


# ---- decision value -------------------------------------------------------
CANDS = [Candidate("LNP-A", 0.62, 0.18), Candidate("LNP-B", 0.58, 0.22),
         Candidate("LNP-C", 0.31, 0.10)]


def test_a_close_contest_is_flagged_as_close():
    d = decision_now(CANDS)
    assert d["choice"] == "LNP-A" and d["runner_up"] == "LNP-B"
    assert d["contest_is_close"]
    assert "proceed rather than to measure" in d["note"]


def test_a_dominant_candidate_makes_measurement_pointless():
    d = decision_now([Candidate("X", 5.0, 0.01), Candidate("Y", 0.1, 0.01)])
    assert not d["contest_is_close"] and d["margin_in_sd"] > 10


def test_a_single_candidate_is_not_a_decision():
    with pytest.raises(ValueError_, match="at least two candidates"):
        decision_now(CANDS[:1])


def test_resolving_the_close_pair_captures_almost_all_the_value():
    v = expected_value_of_information(CANDS, resolves={"LNP-A": 0.9, "LNP-B": 0.9})
    assert v["fraction_of_evpi_captured"] > 0.8
    assert v["probability_the_decision_changes"] > 0.2
    assert v["evsi_exceeds_its_own_noise"]


def test_fully_resolving_an_also_ran_is_worth_nothing():
    """Information gain and decision value come apart here: this experiment removes
    ALL the uncertainty on a candidate and changes no decision."""
    v = expected_value_of_information(CANDS, resolves={"LNP-C": 1.0})
    assert not v["evsi_exceeds_its_own_noise"]
    assert v["probability_the_decision_changes"] < 0.01
    assert "worth NOTHING for this decision" in v["verdict"]


def test_evsi_is_gated_on_its_own_monte_carlo_noise():
    """The first version returned worth_running=True on an EVSI of 1e-5 -- the
    correct answer there is zero, and 1e-5 is the estimator's noise."""
    rows = value_per_cost(CANDS, {
        "resolve the close pair": ({"LNP-A": 0.9, "LNP-B": 0.9}, 66.0),
        "resolve the also-ran": ({"LNP-C": 1.0}, 20.0)})
    close, also = rows[0], rows[1]
    assert close["worth_running"] and not also["worth_running"]
    assert also["evsi"] == 0.0 and also["evsi_per_cost"] == 0.0
    assert "was not going to be chosen" in also["why_not"]


def test_evpi_bounds_every_experiment():
    v = expected_value_of_information(CANDS, resolves={"LNP-A": 1.0, "LNP-B": 1.0,
                                                       "LNP-C": 1.0})
    assert v["evsi"] <= v["evpi"] + 1e-9
    assert v["fraction_of_evpi_captured"] == pytest.approx(1.0, abs=0.02)


def test_a_determined_decision_has_zero_evpi():
    v = expected_value_of_information([Candidate("X", 5.0, 0.0),
                                       Candidate("Y", 0.1, 0.0)])
    assert v["evpi"] == pytest.approx(0.0, abs=1e-9)
    assert "Proceed rather" in v["verdict"]


def test_value_is_in_decision_units_not_nats():
    v = expected_value_of_information(CANDS, resolves={"LNP-A": 0.5})
    assert "not nats" in v["note"]


def test_an_unknown_candidate_in_resolves_is_refused():
    with pytest.raises(ValueError_, match="not in the decision"):
        expected_value_of_information(CANDS, resolves={"LNP-Z": 1.0})


def test_too_few_draws_is_refused():
    with pytest.raises(ValueError_, match="dominated by noise"):
        expected_value_of_information(CANDS, n_draws=10)


def test_a_nonpositive_cost_is_refused():
    with pytest.raises(ValueError_, match="cost must be positive"):
        value_per_cost(CANDS, {"free lunch": ({"LNP-A": 1.0}, 0.0)})
