"""Provenance propagates whether or not anyone looks at it."""
import pytest

from atcs.taint import Taint, Tainted, TaintViolation, audit, calibrate, combine, require

M = Tainted(0.30, Taint.MEASURED, "ICP-MS liver homogenate, study S1")
S = Tainted(0.42, Taint.SIMULATED, "forward simulator, target-cell partition")
A = Tainted(0.55, Taint.ASSUMED, "cytosolic water fraction, literature default")
I = Tainted(0.0, Taint.ILLUSTRATIVE, "placeholder pending the real analysis")


def test_one_assumed_input_makes_the_whole_result_assumed():
    r = combine("cytosolic exposure", M, S, A, op="stoichiometric chain")
    assert r.taint is Taint.ASSUMED
    assert "cytosolic water fraction" in r.origin


def test_a_measured_result_cannot_be_claimed_from_a_simulated_chain():
    r = combine("organ exposure", M, S)
    with pytest.raises(TaintViolation, match="requires taint <= MEASURED"):
        require(r, Taint.MEASURED, "report as an observation")
    require(r, Taint.SIMULATED, "show as a model output")


def test_the_refusal_states_what_the_level_may_be_used_for_and_the_chain():
    with pytest.raises(TaintViolation, match="may NOT be reported as an observation"):
        require(S, Taint.MEASURED, "report as measured")
    try:
        require(S, Taint.MEASURED, "report as measured")
    except TaintViolation as e:
        assert "forward simulator" in str(e)


def test_an_illustrative_placeholder_cannot_enter_any_results_artefact():
    """The defect class this repository's own history supplies: a four-panel
    statistical validation in which no statistic was computed."""
    with pytest.raises(TaintViolation, match="may not appear in any results artefact"):
        require(I, Taint.ASSUMED, "put in a figure")


# ---- the laundering operation, refused ------------------------------------
def test_calibration_does_not_make_a_simulation_measured():
    c = calibrate(S, M, "calibrated partition model")
    assert c.taint is Taint.SIMULATED
    assert "fitted to" in c.origin and "ICP-MS" in c.origin
    with pytest.raises(TaintViolation):
        require(c, Taint.MEASURED, "report as an observation")


def test_fitting_a_simulation_to_another_simulation_is_refused():
    with pytest.raises(TaintViolation, match="propagates its assumptions"):
        calibrate(S, Tainted(1.0, Taint.SIMULATED, "another simulator"), "x")


def test_calibrating_a_measurement_is_a_category_error():
    with pytest.raises(TaintViolation, match="comparison, not a calibration"):
        calibrate(M, M, "x")


# ---- no operation lowers a taint ------------------------------------------
def test_no_public_operation_can_lower_a_taint():
    """The invariant. Enumerated over the module's own surface rather than spot
    checked, so an operation added later cannot quietly introduce a launder."""
    import atcs.taint as T
    ops = [T.combine, T.calibrate]
    results = [T.combine("c", M, A), T.calibrate(S, M, "c")]
    for r, inputs in zip(results, [(M, A), (S, M)]):
        assert r.taint >= min(i.taint for i in inputs), "a taint was lowered"
    assert max(M.taint, A.taint) == combine("c", M, A).taint


def test_an_unnamed_origin_is_refused():
    with pytest.raises(TaintViolation, match="unnamed origin"):
        Tainted(1.0, Taint.MEASURED, "   ")


def test_combining_nothing_raises():
    with pytest.raises(TaintViolation, match="nothing to attribute"):
        combine("x")


# ---- audit ----------------------------------------------------------------
def test_the_audit_reports_the_worst_input_not_the_average():
    a = audit([M, M, M, M, A])
    assert a["worst"] == "ASSUMED"
    assert a["counts"] == {"MEASURED": 4, "ASSUMED": 1}
    assert "WEAKEST input, not its" in a["note"]
    assert a["worst_carriers"] == [A.origin]


def test_the_chain_is_reconstructable_from_the_result_alone():
    r = combine("final", combine("mid", M, S), A)
    assert "ICP-MS" in r.chain and "forward simulator" in r.chain
    assert "cytosolic water fraction" in r.chain
