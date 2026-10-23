"""A composition cannot be more certain than its least certain input."""
import math

import pytest

from atcs.propagate import (Dependence, MonotonicityViolation, Uncertain, chain,
                            product, reestimate_check)

ORGAN = Uncertain(0.30, 0.22, "organ fraction")          # measured, 22% RSD
PARTITION = Uncertain(0.42, 0.45, "target-cell fraction")  # assumed, 45% RSD
ESCAPE = Uncertain(0.02, 0.60, "endosomal escape")         # simulated, 60% RSD


def test_propagation_cannot_narrow_below_the_worst_input():
    out = product("exposure", ORGAN, PARTITION, Dependence.DEPENDENT)
    assert out.rel_sd >= max(ORGAN.rel_sd, PARTITION.rel_sd)


def test_quadrature_under_independence_still_respects_the_invariant():
    out = product("exposure", ORGAN, PARTITION, Dependence.INDEPENDENT)
    assert out.rel_sd == pytest.approx(math.hypot(0.22, 0.45))
    assert out.rel_sd >= 0.45


def test_dependence_is_positional_and_has_no_default():
    """Defaulting to independence is the commonest way a propagated interval comes
    out too narrow, and in this chain the stages share a study offset."""
    with pytest.raises(TypeError):
        product("exposure", ORGAN, PARTITION)      # type: ignore[call-arg]


def test_dependent_composition_is_wider_than_independent():
    dep = product("x", ORGAN, PARTITION, Dependence.DEPENDENT)
    ind = product("x", ORGAN, PARTITION, Dependence.INDEPENDENT)
    assert dep.rel_sd > ind.rel_sd


def test_the_full_chain_is_dominated_by_its_weakest_stage():
    out = chain("organ->response", [ORGAN, PARTITION, ESCAPE], Dependence.DEPENDENT)
    assert out.rel_sd == pytest.approx(0.22 + 0.45 + 0.60)
    assert out.rel_sd > ESCAPE.rel_sd
    assert out.fold_width > 10


def test_a_violation_names_the_stage_that_caused_it():
    """Synthesised directly, because a pipeline that re-estimates will produce
    exactly this and the error has to be legible."""
    bad = Uncertain(0.1, 0.05, "re-estimated stage")
    with pytest.raises(MonotonicityViolation, match="re-estimating uncertainty"):
        from atcs.propagate import _check_monotone
        _check_monotone(bad, (ORGAN, PARTITION))


def test_reestimation_audit_quantifies_the_understatement():
    """The audit to run on an existing multiscale pipeline: a stage declaring a
    tighter interval than its inputs permit has dropped its input's uncertainty."""
    propagated = chain("c", [ORGAN, PARTITION, ESCAPE], Dependence.DEPENDENT)
    r = reestimate_check(declared=0.15, propagated=propagated)
    assert r["understatement_factor"] > 8
    assert "too narrow" in r["verdict"]
    assert r["propagated_fold"] > r["declared_fold"]


def test_a_consistent_declaration_passes_the_audit():
    propagated = chain("c", [ORGAN, PARTITION], Dependence.DEPENDENT)
    r = reestimate_check(declared=0.80, propagated=propagated)
    assert "consistent with propagation" in r["verdict"]


def test_zero_value_with_relative_uncertainty_is_refused():
    with pytest.raises(ValueError, match="undefined"):
        Uncertain(0.0, 0.2, "bad")


def test_negative_uncertainty_is_refused():
    with pytest.raises(ValueError, match="cannot be negative"):
        Uncertain(1.0, -0.1, "bad")


def test_an_empty_chain_raises():
    with pytest.raises(ValueError, match="empty chain"):
        chain("c", [], Dependence.DEPENDENT)


def test_stage_names_are_retained_for_attribution():
    """Every stage must be findable in the attribution, including the first.

    This test caught a real bug: `chain` seeded its accumulator with `stages[0]`
    unchanged, whose own `.stages` is empty for a leaf value, so the FIRST stage
    silently vanished from every propagated result -- defeating the only purpose
    the field has. Asserting substring membership rather than exact equality,
    because intermediate names are prefixed with the chain's name.
    """
    out = chain("full", [ORGAN, PARTITION, ESCAPE], Dependence.UNKNOWN)
    joined = " | ".join(out.stages)
    for stage in (ORGAN, PARTITION, ESCAPE):
        assert stage.name in joined, f"{stage.name} lost from the attribution"
    assert out.stages[0] == ORGAN.name, "the first stage must appear unprefixed"
    assert len(out.stages) == 3


def test_an_interval_crossing_zero_reports_undefined_rather_than_infinite():
    """A relative SD above ~0.51 puts the 95% lower bound below zero for a positive
    quantity, at which point a MULTIPLICATIVE width is undefined rather than large.
    Reporting 'inf' as though it were a number is the error; saying which case you
    are in, and that the log scale is the fix, is the correct behaviour."""
    wide = chain("c", [ORGAN, PARTITION, ESCAPE], Dependence.DEPENDENT)
    assert wide.rel_sd > 0.51
    assert not wide.fold_width_defined
    assert wide.fold_width == float("inf")
    assert "undefined, not infinite" in wide.width_note
    assert "log scale" in wide.width_note

    narrow = Uncertain(0.3, 0.1, "tight")
    assert narrow.fold_width_defined and "multiplicative width" in narrow.width_note
