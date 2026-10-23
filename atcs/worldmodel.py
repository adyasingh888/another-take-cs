"""
The dual-loop world model: the four stages wired into the registry, with the
credibility gate that decides what each may be used for.

WHAT A "WORLD MODEL" MEANS HERE, AND WHAT IT DOES NOT
-----------------------------------------------------
The brief's core thesis is "A Dual-Loop Biological Intervention World Model
Coupling Multi-Scale Physical Delivery with Intracellular Perturbation Dynamics
under Empirical Bayes and Conformal Credibility Gates". Three commitments in that
sentence are worth separating, because only two of them are delivered by code:

  COUPLING  -- the stages compose through typed state with checked invariants.
               Delivered: `build_world()` wires them and the registry refuses a
               write that violates the target group's contract.
  GATES     -- Empirical Bayes shrinkage and conformal calibration decide what a
               prediction may claim. Delivered: `atcs.calibrate`, and `credibility()`
               applies them per context of use.
  WORLD MODEL -- a model that predicts the consequences of interventions it has
               not seen. NOT delivered, and nothing here pretends otherwise. The
               organ-to-cytosol handoff is structurally unidentifiable from the
               measurements the field makes (`atcs.observability`), so the chain
               cannot be validated end to end on current evidence.

A world model whose central handoff is unidentifiable is an ARCHITECTURE, and
calling it that is the honest description. What the architecture buys is not
prediction: it is that the chain states which stage is unsupported, refuses the
compositions it cannot justify, and names the measurement that would change that.

THE AGENTIC CONSTRAINT
----------------------
The brief is explicit that agents "act strictly as compilers and auditors" while
"equations enforce physical constraints". That ordering is implemented rather than
promised: no function here accepts a model-generated quantity that has not passed
the state-group invariant, and `credibility()` reads taint rather than trusting a
label. An agent in this design can choose WHAT to compute; it cannot choose
whether mass is conserved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .calibrate import anchored_fold_width, bayes_shrink, zero_shot_fold_width
from .cell import IndirectResponseCell, IonizableLipid, attribute_failure, escape_fraction
from .corona import apoe_c3_index, softmax_composition
from .observability import StructuralModel
from .models import ORGAN_TO_CYTOSOL, CYTOSOL_CANDIDATES
from .propagate import Dependence, Uncertain, chain
from .pbpk import ORGANS, build_generator, cleared_fraction, organ_shares
from .state import GROUPS, Process, Registry, Store
from .taint import Taint, Tainted, audit


class ContextOfUseError(RuntimeError):
    """A prediction was used outside the context its credibility supports."""


# --------------------------------------------------------------- the wiring
def build_world() -> tuple[Registry, Store]:
    """Register the four stages as processes over the typed state.

    Each stage declares what it reads and writes; the store checks the invariant
    on every write. Adding a fifth stage is a `register` call, which is the
    extensibility claim and is exercised in the tests rather than asserted here.
    """
    reg, store = Registry(), Store()

    def stage1_corona(x_c, **_):
        logits = {"ApoE": float(x_c.get("apoe_logit", 0.0)),
                  "C3": float(x_c.get("c3_logit", 0.0)),
                  "IgG": -0.5, "albumin": 0.2, "clusterin": -0.2, "other": 0.0}
        return softmax_composition(logits)

    def stage2_organ(x_id, x_c, dose_nmol=100.0, t_h=168.0, **_):
        # the corona's ApoE:C3 index modulates hepatic uptake -- a MECHANISTIC
        # coupling, written as a rate multiplier rather than a learned weight, so
        # the direction is inspectable
        idx = apoe_c3_index(x_id)["log10_apoe_c3"]
        liver_uptake = 0.90 * float(np.clip(10 ** (0.25 * idx), 0.2, 3.0))
        up = {"liver": liver_uptake, "spleen": 0.11, "kidney": 0.06,
              "lung": 0.05, "heart": 0.02, "tumour": 0.03}
        g = build_generator(up, {o: 0.02 for o in ORGANS},
                            float(x_c.get("clearance_per_h", 0.08)))
        sol = g.solve(dose_nmol, [t_h])
        out = {o: float(sol[o][0]) for o in ORGANS}
        out["plasma"] = float(sol["plasma"][0])
        out["cleared"] = float(sol["cleared"][0])
        return out

    def stage3_barrier(x_c, **_):
        lip = IonizableLipid(pka=float(x_c.get("pka", 6.35)),
                             delta_logd=float(x_c.get("delta_logd", 2.1)))
        e = escape_fraction(lip)
        return {"endosomal_escape": e["escape_fraction"],
                # NOT measured: the target-cell partition is the unidentifiable one
                "target_cell_partition": float(x_c.get("assumed_partition", 0.25))}

    def stage4_cell(x_organ, x_barrier, x_c, **_):
        organ = float(x_organ.get(str(x_c.get("target_organ", "liver")), 0.0))
        cyto_nmol = organ * x_barrier["target_cell_partition"] * \
            x_barrier["endosomal_escape"]
        vol_L = float(x_c.get("target_cell_volume_L", 2.4e-4))
        if vol_L <= 0:
            raise ValueError("target-cell compartment volume must be positive")
        c_nM = cyto_nmol / vol_L
        cell = IndirectResponseCell()
        return {"cytosolic_nM": c_nM,
                "knockdown": cell.response(c_nM, float(x_c.get("t_h", 168.0)))}

    reg.register(Process("stage1_corona", ("x_c",), "x_id", stage1_corona,
                         "protein corona identity on the simplex"))
    reg.register(Process("stage2_organ", ("x_id", "x_c"), "x_organ", stage2_organ,
                         "multi-organ PBPK, mass-conserving by construction"))
    reg.register(Process("stage3_barrier", ("x_c",), "x_barrier", stage3_barrier,
                         "endosomal escape and target-cell partition"))
    reg.register(Process("stage4_cell", ("x_organ", "x_barrier", "x_c"), "x_cell",
                         stage4_cell, "cytosolic concentration and target knockdown"))
    return reg, store


CHAIN = ("stage1_corona", "stage2_organ", "stage3_barrier", "stage4_cell")


def run(context: Mapping[str, float], dose_nmol: float = 100.0,
        t_h: float = 168.0) -> Store:
    """Run the full chain for one context. Every intermediate write is checked."""
    reg, store = build_world()
    store.write("x_c", dict(context))
    return reg.run_chain(store, CHAIN, dose_nmol=dose_nmol, t_h=t_h)


# ------------------------------------------------------------- credibility
#: The chain's stages, each with the taint of the quantity it contributes.
STAGE_TAINT = {
    "stage1_corona": Taint.SIMULATED,
    "stage2_organ": Taint.SIMULATED,
    "stage3_barrier": Taint.ASSUMED,      # target_cell_partition is a chosen number
    "stage4_cell": Taint.SIMULATED,
}

#: Stages whose weakest quantity enters the chain as a COMMON MULTIPLICATIVE
#: FACTOR -- identical across arms of one study. Such a factor cancels from any
#: ratio or ranking, so it taints an absolute claim but not a comparative one.
#: Listed explicitly rather than inferred: whether a parameter is shared across
#: arms is a fact about the experiment, not about the code.
COMMON_FACTOR_STAGES = frozenset({"stage3_barrier"})

#: What each context of use requires, in the ICH M15 sense: model influence and
#: decision consequence set the credibility evidence needed.
CONTEXTS = {
    "rank_formulations_within_one_study": {
        "max_taint": Taint.SIMULATED,
        "requires_identifiable_chain": False,
        # A ranking is invariant to any factor shared by all arms. The assumed
        # target-cell partition is such a factor -- the same constant multiplies
        # every formulation in the study -- so it cannot change the ORDER, only the
        # scale. Declaring that here is what makes the gate discriminating instead
        # of uniformly negative, and a gate that refuses everything gets ignored.
        "invariant_to_common_factors": True,
        "why": ("a within-study ranking cancels both the study offset and any factor "
                "common to all arms, so an assumed constant partition cannot change "
                "the order and a simulated chain can support it"),
    },
    "predict_absolute_organ_delivery_in_a_new_laboratory": {
        "max_taint": Taint.SIMULATED,
        "requires_identifiable_chain": False,
        "requires_anchor": True,
        "why": ("absolute cross-laboratory prediction needs the target laboratory's "
                "offset, which is identifiable only from a reference arm measured in "
                "the same analytical run"),
    },
    "predict_target_cell_exposure": {
        "max_taint": Taint.SIMULATED,
        "requires_identifiable_chain": True,
        "why": ("the organ-to-cytosol handoff must be identifiable from achievable "
                "measurements, and it is not"),
    },
    "inform_a_patient_level_decision": {
        "max_taint": Taint.MEASURED,
        "requires_identifiable_chain": True,
        "why": ("a patient-facing decision requires measured evidence through every "
                "handoff; no stage of this chain meets that"),
    },
}


def credibility(context_of_use: str, *, anchors: int = 0,
                interface: StructuralModel = ORGAN_TO_CYTOSOL) -> dict:
    """May the chain be used for this purpose? Decided, not asserted.

    Reads the STAGE TAINTS and the INTERFACE IDENTIFIABILITY rather than a declared
    maturity level, so a stage cannot be promoted by relabelling it. The returned
    `blockers` list is what has to change, which is the only part a research plan
    can act on.
    """
    if context_of_use not in CONTEXTS:
        raise ContextOfUseError(
            f"unknown context of use {context_of_use!r}; declared contexts are "
            f"{sorted(CONTEXTS)}. A context of use that has not been written down "
            f"cannot have its evidence requirements checked.")
    spec = CONTEXTS[context_of_use]
    worst = max(STAGE_TAINT.values())
    carriers = [s for s, t in STAGE_TAINT.items() if t == worst]
    blockers: list[str] = []

    # A comparative claim is invariant to factors common to every arm, so stages
    # contributing only such a factor are excluded before the taint is taken.
    if spec.get("invariant_to_common_factors"):
        relevant = {s_: t for s_, t in STAGE_TAINT.items()
                    if s_ not in COMMON_FACTOR_STAGES}
        if relevant:
            worst = max(relevant.values())
            carriers = [s_ for s_, t in relevant.items() if t == worst]

    if worst > spec["max_taint"]:
        blockers.append(
            f"chain taint is {worst.name} (from {', '.join(carriers)}) but this "
            f"context permits at most {spec['max_taint'].name}")
    if spec.get("requires_identifiable_chain"):
        rep = interface.report()
        if rep["unidentifiable"]:
            blockers.append(
                f"{len(rep['unidentifiable'])} of {rep['n_unknowns']} interface "
                f"unknowns are structurally unidentifiable from achievable "
                f"measurements ({', '.join(rep['unidentifiable'][:3])}...)")
    if spec.get("requires_anchor") and anchors < 1:
        blockers.append(
            "no reference anchor in the target laboratory: the study offset is not "
            "identifiable, so an absolute prediction has no calibrated scale")

    return {"context_of_use": context_of_use, "permitted": not blockers,
            "chain_taint": worst.name, "taint_carriers": carriers,
            "common_factors_excluded": bool(spec.get("invariant_to_common_factors")),
            "blockers": blockers, "requirement_rationale": spec["why"],
            "remedy": _remedy(blockers, interface)}


def _remedy(blockers: Sequence[str], interface: StructuralModel) -> list[str]:
    out = []
    for b in blockers:
        if "unidentifiable" in b:
            sets = interface.minimal_additional_observations(
                "f_parenchymal", CYTOSOL_CANDIDATES)
            out.append(f"add one of {sets} to close the interface")
        elif "reference anchor" in b:
            out.append("measure a known reference formulation in the same analytical run")
        elif "taint" in b:
            out.append("replace the assumed stage with a measurement, or restrict the "
                       "claim to a context that permits a simulated chain")
    return out


def predictive_width(tau0_sq: float, sigma_e_sq: float, anchors: int) -> dict:
    """The interval the chain is entitled to, zero-shot and anchored."""
    zs = zero_shot_fold_width(tau0_sq, sigma_e_sq)
    if anchors < 1:
        return {"anchors": 0, "fold_width": zs["fold_width"],
                "basis": "zero-shot: variance components only", **zs}
    an = anchored_fold_width(tau0_sq, sigma_e_sq, anchors)
    return {"anchors": anchors, "fold_width": an["fold_width"],
            "zero_shot_fold_width": zs["fold_width"],
            "narrowing_factor": an["narrowing_factor"],
            "basis": f"posterior offset variance after {anchors} anchors"}
