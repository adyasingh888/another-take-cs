"""The chain composes, and the gate decides what it may be used for."""
import numpy as np
import pytest

from atcs.state import InvariantViolation, Process, Store, StateGroup, WiringError
from atcs.taint import Taint
from atcs.worldmodel import (CHAIN, CONTEXTS, ContextOfUseError, STAGE_TAINT,
                             build_world, credibility, predictive_width, run)

CTX = {"apoe_logit": 1.2, "c3_logit": -0.4, "pka": 6.35, "delta_logd": 2.1,
       "target_organ": "liver", "clearance_per_h": 0.08,
       "assumed_partition": 0.25, "target_cell_volume_L": 2.4e-4}


# ---- composition ----------------------------------------------------------
def test_the_four_stages_compose_and_every_invariant_holds():
    s = run(CTX).snapshot()
    assert sum(s["x_id"].values()) == pytest.approx(1.0)
    assert all(v >= 0 for v in s["x_organ"].values())
    assert 0 <= s["x_barrier"]["endosomal_escape"] <= 1
    assert s["x_cell"]["cytosolic_nM"] > 0
    assert 0 <= s["x_cell"]["knockdown"] <= 1


def test_mass_is_conserved_through_the_composed_chain():
    """Not only inside the PBPK stage: the quantity written into the store must
    still balance, because a stage can conserve mass internally and write out a
    rescaled copy."""
    s = run(CTX, dose_nmol=100.0).snapshot()["x_organ"]
    assert sum(s.values()) == pytest.approx(100.0, abs=1e-6)


def test_the_corona_coupling_has_the_direction_the_biology_implies():
    """Higher ApoE relative to C3 must raise hepatic uptake -- a mechanistic rate
    multiplier, inspectable, not a learned weight of unknown sign."""
    hi = run({**CTX, "apoe_logit": 2.5, "c3_logit": -1.0}).snapshot()["x_organ"]["liver"]
    lo = run({**CTX, "apoe_logit": -1.0, "c3_logit": 2.5}).snapshot()["x_organ"]["liver"]
    assert hi > lo


def test_a_stage_writing_an_invalid_quantity_is_caught_at_the_write():
    """The failure mode the registry exists for: a stage that is individually
    correct and jointly wrong. Caught at the write, not at the implausible result."""
    reg, store = build_world()
    store.write("x_c", CTX)
    bad = Process("bad_corona", ("x_c",), "x_id",
                  lambda x_c, **_: {"ApoE": 0.7, "C3": 0.9})
    reg.register(bad)
    with pytest.raises(InvariantViolation, match="not 1"):
        bad.run(store)


def test_a_process_reading_an_undeclared_group_is_refused():
    reg, store = build_world()
    p = Process("sneaky", ("x_nonexistent",), "x_cell", lambda **_: {})
    reg.register(p)
    with pytest.raises(WiringError, match="undeclared group"):
        p.run(store)


def test_a_zero_cell_volume_is_refused_rather_than_dividing():
    with pytest.raises(ValueError, match="volume must be positive"):
        run({**CTX, "target_cell_volume_L": 0.0})


# ---- extensibility, exercised rather than asserted ------------------------
def test_a_new_scale_can_be_added_without_editing_the_core():
    """The brief's architectural claim: new assays, organs or pathways plug in
    without rewriting the core. Tested by adding a whole new state group and a
    process over it at runtime."""
    reg, store = build_world()
    store.write("x_c", CTX)
    reg.run_chain(store, CHAIN, dose_nmol=100.0, t_h=168.0)
    store.add_group(StateGroup(
        "x_tissue", "spatial penetration depth",
        lambda v: ((all(x >= 0 for x in v.values()), "") if v else (True, "")),
        unit="um"))
    p = Process("stage5_penetration", ("x_cell",), "x_tissue",
                lambda x_cell, **_: {"depth_um": 40.0 * x_cell["knockdown"]})
    reg.register(p)
    assert p.run(store)["depth_um"] > 0
    assert "x_tissue" in store.snapshot()


def test_a_new_organ_needs_no_edit_to_the_state_module():
    from atcs.pbpk import build_generator
    organs = ("liver", "spleen", "brain")
    g = build_generator({o: 0.1 for o in organs}, {o: 0.02 for o in organs}, 0.08,
                        organs=organs)
    assert "brain" in g.compartments
    assert np.abs(g.K.sum(axis=0)).max() < 1e-12


def test_duplicate_registration_is_refused():
    reg, _ = build_world()
    with pytest.raises(WiringError, match="already registered"):
        reg.register(Process("stage1_corona", ("x_c",), "x_id", lambda **_: {}))


# ---- the credibility gate -------------------------------------------------
def test_a_comparative_claim_is_permitted_where_an_absolute_one_is_not():
    """The gate must DISCRIMINATE. A ranking is invariant to any factor common to
    every arm, so the assumed partition cannot change the order -- only the scale.
    A gate that refused this too would refuse everything and be ignored."""
    rank = credibility("rank_formulations_within_one_study")
    absolute = credibility("predict_target_cell_exposure")
    assert rank["permitted"] and rank["common_factors_excluded"]
    assert not absolute["permitted"]


def test_the_patient_facing_context_is_refused_by_construction():
    r = credibility("inform_a_patient_level_decision")
    assert not r["permitted"]
    assert any("MEASURED" in b for b in r["blockers"])


def test_the_unidentifiable_interface_blocks_target_cell_claims_and_names_the_fix():
    r = credibility("predict_target_cell_exposure")
    assert any("structurally unidentifiable" in b for b in r["blockers"])
    assert any("close the interface" in m for m in r["remedy"])


def test_the_anchor_requirement_is_satisfied_by_anchors_and_not_otherwise():
    none = credibility("predict_absolute_organ_delivery_in_a_new_laboratory", anchors=0)
    some = credibility("predict_absolute_organ_delivery_in_a_new_laboratory", anchors=3)
    assert any("reference anchor" in b for b in none["blockers"])
    assert not any("reference anchor" in b for b in some["blockers"])


def test_an_undeclared_context_of_use_is_refused_not_defaulted():
    """A context that has not been written down cannot have its evidence
    requirements checked, and defaulting to permitted is how a model escapes."""
    with pytest.raises(ContextOfUseError, match="cannot have its evidence"):
        credibility("whatever_the_user_wants")


def test_the_gate_reads_taint_rather_than_a_declared_maturity_label():
    """An agent can choose what to compute; it cannot relabel a stage into
    credibility. Verified by checking the gate's verdict tracks STAGE_TAINT."""
    assert max(STAGE_TAINT.values()) is Taint.ASSUMED
    assert credibility("predict_target_cell_exposure")["chain_taint"] == "ASSUMED"


def test_every_declared_context_states_its_rationale():
    for name, spec in CONTEXTS.items():
        assert len(spec["why"]) > 50, f"{name} does not justify its requirement"
        assert credibility(name)["requirement_rationale"] == spec["why"]


# ---- the width the chain is entitled to ------------------------------------
def test_anchoring_narrows_the_entitled_interval():
    zero = predictive_width(0.26, 0.09, anchors=0)
    three = predictive_width(0.26, 0.09, anchors=3)
    assert three["fold_width"] < zero["fold_width"]
    assert three["narrowing_factor"] > 3
    assert "zero-shot" in zero["basis"]
