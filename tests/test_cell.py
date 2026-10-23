"""Escape, the virtual-cell interface, and the attribution that needs both."""
import numpy as np
import pytest

from atcs.cell import (EscapeError, IndirectResponseCell, IonizableLipid,
                       attribute_failure, escape_fraction)

LIPID = IonizableLipid(pka=6.35, delta_logd=2.1, name="MC3-like")
CELL = IndirectResponseCell()


# ---- Stage 3: the bell must emerge, not be assumed ------------------------
def test_escape_is_bell_shaped_in_pka_with_an_optimum_near_6_35():
    """The brief reports a hepatocyte optimum at pKa 6.2-6.5. Here it emerges from
    the PRODUCT of endosomal protonation (rising) and plasma neutrality (falling)
    rather than from a bell being written down."""
    pkas = np.arange(4.5, 9.01, 0.05)
    fr = np.array([escape_fraction(IonizableLipid(p, 2.1))["escape_fraction"]
                   for p in pkas])
    best = pkas[int(fr.argmax())]
    assert 6.0 <= best <= 6.7, f"optimum at pKa {best}"
    assert fr[0] < fr.max() * 0.2 and fr[-1] < fr.max() * 0.2, "must fall off both sides"


def test_the_two_competing_terms_are_reported_separately():
    low = escape_fraction(IonizableLipid(5.0, 2.1))
    high = escape_fraction(IonizableLipid(8.5, 2.1))
    assert low["limiting_term"] == "endosomal protonation"
    assert high["limiting_term"] == "plasma neutrality"
    assert low["side_of_optimum"] == "below" and high["side_of_optimum"] == "above"


def test_productive_escape_stays_within_a_few_percent():
    """A model permitting 50% escape would be describing a different phenomenon;
    the brief's Patisiran figure is about 2.8%."""
    best = max(escape_fraction(IonizableLipid(p, 2.5))["escape_fraction"]
               for p in np.arange(5, 9, 0.05))
    assert 0.0 < best <= 0.04


def test_insufficient_lipophilicity_gates_escape_even_at_optimal_pka():
    weak = escape_fraction(IonizableLipid(6.35, 0.2))
    assert weak["limiting_term"] == "lipophilicity"
    assert weak["escape_fraction"] < escape_fraction(LIPID)["escape_fraction"] / 5


def test_nonsense_pka_and_ph_are_refused():
    with pytest.raises(EscapeError, match="outside any physiological"):
        IonizableLipid(pka=15.0, delta_logd=2.0)
    with pytest.raises(EscapeError, match="outside 4.0-7.0"):
        escape_fraction(LIPID, endosomal_ph=2.0)


# ---- Stage 4: the cell ----------------------------------------------------
def test_response_is_monotone_saturating_in_cytosolic_concentration():
    r = [CELL.response(c, 168.0) for c in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)]
    assert r == sorted(r)
    assert r[-1] - r[-2] < 0.02, "a dose-response must plateau"
    assert r[-1] <= 0.95 + 1e-9


def test_risc_loading_saturates_so_more_cargo_stops_helping():
    assert CELL.loaded(1e6) == pytest.approx(CELL.risc_capacity_nM, rel=1e-3)


def test_response_builds_over_time_toward_the_turnover_limit():
    t = [CELL.response(10.0, h) for h in (1, 6, 24, 72, 168, 672)]
    assert t == sorted(t)
    assert t[0] < 0.2 * t[-1], "mRNA turnover sets the approach, not instant effect"


def test_any_object_with_a_response_method_satisfies_the_interface():
    """The boundary is typed and narrow, so a pre-trained perturbation model drops
    in without the delivery side knowing. Exercised with a stub."""
    class Stub:
        def response(self, cytosolic_nM, hours):
            return 0.5
    out = attribute_failure(Stub(), 10.0, 168.0, 0.2, target_response=0.7)
    assert out["verdict"] in {"transport-limited", "mechanism-limited", "not attributable"}
    assert out["local_slope_per_log10"] is None, "a stub has no sensitivity to report"


# ---- the attribution ------------------------------------------------------
def test_low_cytosolic_delivery_attributes_to_transport():
    out = attribute_failure(CELL, 0.5, 168.0, 0.18, target_response=0.70)
    assert out["verdict"] == "transport-limited"
    assert "carrier re-optimisation" in out["next_experiment"]
    assert "the target is not implicated" in out["next_experiment"]


def test_ample_delivery_with_no_response_attributes_to_mechanism():
    """Target set below the model's saturation ceiling of ~0.68, so the shortfall
    is genuinely a mechanism failure rather than an unreachable target."""
    out = attribute_failure(CELL, 60.0, 168.0, 0.05, target_response=0.60)
    assert out["verdict"] == "mechanism-limited"
    assert "target validation" in out["next_experiment"]
    assert "carrier changes will not help" in out["next_experiment"]


def test_a_below_threshold_observation_refuses_to_attribute():
    """The refusal branch, correctly scoped. Below the responding threshold the
    observation tests neither transport nor mechanism."""
    out = attribute_failure(CELL, 1e-4, 168.0, 0.0, target_response=0.70)
    assert out["verdict"] == "not attributable"
    assert "below the threshold" in out["reason"]
    assert "concentration series" in out["next_experiment"]


def test_flatness_at_saturation_strengthens_rather_than_blocks_attribution():
    """Written after a real logic bug: the first version refused wherever the curve
    was flat, including at saturation. That was backwards. At saturation flatness
    positively EXCLUDES transport -- more cargo demonstrably cannot help -- so it
    makes a mechanism attribution stronger, not impossible."""
    out = attribute_failure(CELL, 5000.0, 168.0, 0.10, target_response=0.60)
    assert out["verdict"] == "mechanism-limited"
    assert "positively excluded" in out["reason"]


def test_a_ceiling_below_the_target_is_reported_as_target_unreachable():
    """A third verdict, distinct from both and discovered by running the module:
    when the cell model's own saturation ceiling sits BELOW the requested target,
    no achievable concentration reaches it, and neither the carrier nor the dose is
    at fault. Attributing that to transport would send a chemist to re-optimise a
    carrier that was never the problem."""
    out = attribute_failure(CELL, 5000.0, 168.0, 0.10, target_response=0.95)
    assert out["verdict"] == "target-unreachable"
    assert "no achievable cytosolic concentration" in out["reason"]
    assert CELL.response(1e6, 168.0) < 0.95


def test_the_two_attributions_prescribe_opposite_experiments():
    """The reason the coupling is worth building: neither half alone separates
    these, and they prescribe opposite next steps."""
    t = attribute_failure(CELL, 0.5, 168.0, 0.18, target_response=0.60)
    m = attribute_failure(CELL, 60.0, 168.0, 0.05, target_response=0.60)
    assert t["next_experiment"] != m["next_experiment"]
    assert "carrier" in t["next_experiment"] and "target" in m["next_experiment"]


def test_a_met_target_has_nothing_to_attribute():
    out = attribute_failure(CELL, 50.0, 672.0, 0.85, target_response=0.70)
    assert out["verdict"] == "no failure to attribute"


def test_an_out_of_range_observation_is_refused():
    with pytest.raises(ValueError, match="fraction in"):
        attribute_failure(CELL, 10.0, 168.0, 1.4)


def test_attribution_is_correct_on_a_system_whose_truth_is_known():
    """Validation by construction: sweep concentrations through a cell whose
    dose-response is known, and check the attribution agrees with which side of
    the curve the observation sits on."""
    for c in (0.1, 0.3, 1.0):
        assert attribute_failure(CELL, c, 168.0, CELL.response(c, 168.0) * 0.9
                                 )["verdict"] in {"transport-limited", "not attributable"}
    for c in (40.0, 100.0):
        assert attribute_failure(CELL, c, 168.0, 0.02, target_response=0.60
                                 )["verdict"] == "mechanism-limited"
