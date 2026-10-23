"""Conservation must be an identity, not a fitted property."""
import numpy as np
import pytest

from atcs.pbpk import (COMPARTMENTS, ConstructionError, Generator, ORGANS,
                       build_generator, cleared_fraction, organ_shares)

UPTAKE = {"liver": 0.90, "spleen": 0.11, "kidney": 0.06,
          "lung": 0.05, "heart": 0.02, "tumour": 0.03}
RELEASE = {o: 0.02 for o in ORGANS}
T = [0.0, 0.25, 1.0, 4.0, 24.0, 72.0, 168.0, 672.0]


def gen(clearance=0.08):
    return build_generator(UPTAKE, RELEASE, clearance)


def test_mass_is_conserved_to_machine_precision_at_every_time():
    g = gen()
    assert g.residual(dose=100.0, t_h=T) < 1e-9


def test_conservation_holds_at_very_long_times_too():
    """A stepper's error accumulates; a matrix exponential's does not."""
    g = gen()
    assert g.residual(dose=100.0, t_h=[1e4]) < 1e-8


def test_no_compartment_ever_goes_negative():
    """Metzler structure makes exp(Kt) non-negative, so this is structural."""
    sol = gen().solve(100.0, T)
    for c, traj in sol.items():
        assert (traj >= -1e-12).all(), f"{c} went negative"


def test_the_sink_is_named_and_reported():
    """Conservation without an absorbing sink means nothing is ever eliminated --
    true of the mathematics and false of the biology. An audit of this programme's
    earlier work raised exactly that."""
    sol = gen().solve(100.0, T)
    assert "cleared" in sol
    cf = cleared_fraction(sol, 100.0)
    assert cf[0] == pytest.approx(0.0)
    assert np.all(np.diff(cf) >= -1e-12), "the sink must be absorbing (monotone)"
    assert cf[-1] > 0.5, "a substantial fraction eliminated by 672 h"


def test_terminal_elimination_depends_on_both_rates_not_one():
    """Written twice, and the first version was wrong both times.

    The test above originally asserted >90% cleared at 672 h; the model returned
    59.8%. I then asserted the shortfall was DEPOT-limited -- that plasma clearance
    barely matters because plasma residence is short. Measuring it refuted that
    too: 10x clearance takes 672 h elimination from 0.598 to 0.997, which is an
    enormous move.

    What is actually true, measured rather than reasoned: at base rates the system
    sits in a regime where BOTH the plasma clearance rate and the organ release
    rate are rate-limiting, and a 10x increase in either nearly completes
    elimination. Terminal behaviour cannot be attributed to one of them. That is
    worth keeping as a test because the intuition 'organs are a slow depot, so
    clearance is irrelevant' is natural, reaches for the wrong answer, and is
    exactly the kind of mechanistic story a results chapter states without
    checking.
    """
    base = cleared_fraction(gen(clearance=0.08).solve(100.0, [672.0]), 100.0)[0]
    faster_clearance = cleared_fraction(
        gen(clearance=0.80).solve(100.0, [672.0]), 100.0)[0]
    faster_release = cleared_fraction(
        build_generator(UPTAKE, {o: 0.20 for o in ORGANS}, 0.08).solve(100.0, [672.0]),
        100.0)[0]
    assert 0.55 < base < 0.65
    assert faster_clearance > 0.99, "10x clearance nearly completes elimination"
    assert faster_release > 0.99, "so does 10x organ release"
    assert cleared_fraction(gen().solve(100.0, [5.0e3]), 100.0)[0] > 0.99


def test_with_no_clearance_nothing_leaves_and_total_still_balances():
    g = gen(clearance=0.0)
    sol = g.solve(100.0, T)
    assert cleared_fraction(sol, 100.0)[-1] == pytest.approx(0.0)
    assert g.residual(100.0, T) < 1e-9


# ---- the construction cannot be bypassed ----------------------------------
def test_a_leaky_generator_is_refused():
    """The defect the construction prevents: a column that does not sum to zero
    means material vanishes, and conservation becomes something you fit."""
    K = np.zeros((3, 3))
    K[1, 0] = 0.5
    K[0, 0] = -0.9            # 0.4 of what leaves plasma goes nowhere
    with pytest.raises(ConstructionError, match="without arriving anywhere"):
        Generator(("plasma", "liver", "cleared"), K)


def test_a_negative_transfer_rate_is_refused():
    K = np.zeros((3, 3))
    K[1, 0] = -0.5
    K[0, 0] = 0.5
    with pytest.raises(ConstructionError, match="negative transfer rate"):
        Generator(("plasma", "liver", "cleared"), K)


def test_build_generator_cannot_produce_a_leaky_matrix():
    """Diagonals are SET from the column outflow, so there is no call that yields
    a non-conserving generator."""
    for cl in (0.0, 0.01, 1.0, 50.0):
        g = build_generator(UPTAKE, RELEASE, cl)
        assert np.abs(g.K.sum(axis=0)).max() < 1e-12


def test_negative_rates_are_rejected_at_construction():
    with pytest.raises(ConstructionError, match="cannot be negative"):
        build_generator(UPTAKE, RELEASE, -0.1)
    with pytest.raises(ConstructionError, match="negative rate"):
        build_generator({"liver": -1.0}, RELEASE, 0.1)


# ---- outputs --------------------------------------------------------------
def test_organ_shares_are_percent_of_injected_dose_and_sum_sensibly():
    sol = gen().solve(100.0, T)
    sh = organ_shares(sol, 100.0)
    total = sum(sh[o] for o in ORGANS)
    assert (total <= 100.0 + 1e-9).all()
    assert sh["liver"].max() > sh["heart"].max(), "liver uptake dominates here"


def test_liver_peaks_before_it_declines():
    sol = gen().solve(100.0, [0.0, 0.5, 2.0, 8.0, 48.0, 336.0])
    liver = sol["liver"]
    assert liver[0] == 0.0
    assert liver.argmax() not in (0, len(liver) - 1), "expected an interior peak"


def test_an_unknown_start_compartment_raises():
    with pytest.raises(ConstructionError, match="unknown start"):
        gen().solve(100.0, T, start="cytosol")


def test_negative_time_raises():
    with pytest.raises(ConstructionError, match="negative time"):
        gen().solve(100.0, [-1.0])
