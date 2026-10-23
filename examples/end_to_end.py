"""
The hepatic siRNA case, run through every layer.

    python examples/end_to_end.py

Seven steps. The last three are the ones a conventional pipeline cannot do:
decide what the chain may be used for, name the measurement that would change
that, and refuse the rest.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from atcs.calibrate import bayes_shrink, conformal_interval
from atcs.cell import IndirectResponseCell, attribute_failure
from atcs.corona import apoe_c3_index
from atcs.delta import advantage
from atcs.models import CYTOSOL_CANDIDATES, ORGAN_TO_CYTOSOL
from atcs.propagate import Dependence, Uncertain, chain, reestimate_check
from atcs.taint import Taint, Tainted, audit, combine, require, TaintViolation
from atcs.worldmodel import credibility, predictive_width, run

LINE = "-" * 78
CTX = {"apoe_logit": 1.2, "c3_logit": -0.4, "pka": 6.35, "delta_logd": 2.1,
       "target_organ": "liver", "clearance_per_h": 0.08,
       "assumed_partition": 0.25, "target_cell_volume_L": 2.4e-4}


def step1_run_the_chain():
    print(f"{LINE}\n1. The four stages compose, and every invariant is checked\n{LINE}")
    s = run(CTX, dose_nmol=100.0, t_h=168.0).snapshot()
    idx = apoe_c3_index(s["x_id"])
    print(f"   corona   ApoE {idx['apoe_share']:.3f} / C3 {idx['c3_share']:.3f} "
          f"-> log-ratio {idx['log10_apoe_c3']:+.2f}, favours {idx['favours']}")
    print(f"   organ    liver {s['x_organ']['liver']:.1f} nmol of 100 injected; "
          f"cleared {s['x_organ']['cleared']:.1f}; total "
          f"{sum(s['x_organ'].values()):.6f} (conserved by construction)")
    print(f"   barrier  escape {s['x_barrier']['endosomal_escape']*100:.2f}%, "
          f"partition {s['x_barrier']['target_cell_partition']:.2f} (ASSUMED)")
    print(f"   cell     cytosolic {s['x_cell']['cytosolic_nM']:.0f} nM "
          f"-> knockdown {s['x_cell']['knockdown']*100:.1f}%")
    return s


def step2_taint(s):
    print(f"\n{LINE}\n2. What is that knockdown made of?\n{LINE}")
    organ = Tainted(s["x_organ"]["liver"], Taint.SIMULATED, "PBPK stage 2")
    escape = Tainted(s["x_barrier"]["endosomal_escape"], Taint.SIMULATED, "escape stage 3")
    part = Tainted(s["x_barrier"]["target_cell_partition"], Taint.ASSUMED,
                   "target-cell partition, chosen because a number was needed")
    kd = combine("knockdown", organ, escape, part, op="stoichiometric chain")
    print(f"   chain taint: {kd.taint.name}")
    print(f"   {audit([organ, escape, part])['artefact_standing']}")
    try:
        require(kd, Taint.SIMULATED, "report as a model prediction")
    except TaintViolation as e:
        print(f"   REFUSED even as a model output:\n      {str(e)[:150]}...")


def step3_identifiability():
    print(f"\n{LINE}\n3. Why the partition is assumed: it cannot be measured this way\n{LINE}")
    r = ORGAN_TO_CYTOSOL.report()
    print(f"   {r['n_unknowns']} unknowns, {r['n_equations']} achievable equation "
          f"-> {r['degrees_of_freedom']} degrees of freedom")
    print(f"   {r['verdict']}")
    print(f"   f_parenchymal: {r['reasons']['f_parenchymal'][:120]}...")
    sets = ORGAN_TO_CYTOSOL.minimal_additional_observations(
        "f_parenchymal", CYTOSOL_CANDIDATES)
    print(f"   minimal fixes: {sets}")
    print("   -> this is decided from the WIRING, before any data, at zero cost")


def step4_uncertainty():
    print(f"\n{LINE}\n4. The interval the chain is entitled to\n{LINE}")
    stages = [Uncertain(0.67, 0.22, "organ fraction"),
              Uncertain(0.25, 0.45, "target-cell partition"),
              Uncertain(0.029, 0.60, "endosomal escape")]
    out = chain("cytosolic exposure", stages, Dependence.DEPENDENT)
    print(f"   propagated relative SD {out.rel_sd:.2f} -> {out.width_note}")
    print(f"   stages retained: {', '.join(out.stages)}")
    r = reestimate_check(declared=0.15, propagated=out)
    print(f"   a stage declaring 0.15: {r['verdict'][:100]}")


def step5_calibration():
    print(f"\n{LINE}\n5. What anchoring buys, and why contrasting works\n{LINE}")
    for k in (0, 1, 3, 10):
        w = predictive_width(0.26, 0.09, anchors=k)
        print(f"   {k:>2} anchors -> {w['fold_width']:>7.1f}x fold width")
    s = bayes_shrink([0.82, 0.71, 0.90], [0.0, 0.0, 0.0], 0.26, 0.09)
    print(f"   shrinkage factor {s.factor:.3f}: {s.interpretation}")
    a = advantage(tau0_sq=0.26, sigma_e_sq=0.09)
    print(f"   contrast target: offset share {a['offset_share']:.2f}, "
          f"favourable = {a['favourable']}")
    print(f"      {a['verdict'][:100]}")


def step6_attribution(s):
    print(f"\n{LINE}\n6. If it fails, whose fault is it?\n{LINE}")
    cell = IndirectResponseCell()
    for label, nM, obs, tgt in [
            ("poor delivery", 0.5, 0.18, 0.60),
            ("good delivery, no response", 60.0, 0.05, 0.60),
            ("nothing delivered", 1e-4, 0.0, 0.60),
            ("target above the model's ceiling", 5000.0, 0.10, 0.95)]:
        r = attribute_failure(cell, nM, 168.0, obs, target_response=tgt)
        print(f"   {label:34s} -> {r['verdict']}")
        print(f"      next: {r['next_experiment'][:95]}")


def step7_what_may_be_claimed():
    print(f"\n{LINE}\n7. So what may this chain actually be used for?\n{LINE}")
    for c in ("rank_formulations_within_one_study",
              "predict_absolute_organ_delivery_in_a_new_laboratory",
              "predict_target_cell_exposure",
              "inform_a_patient_level_decision"):
        r = credibility(c, anchors=0)
        mark = "PERMITTED" if r["permitted"] else "REFUSED  "
        print(f"   {mark}  {c}")
        for b in r["blockers"]:
            print(f"               - {b[:95]}")
        for m in r["remedy"]:
            print(f"               fix: {m[:95]}")
    print("\n   One of four is permitted, and it is the comparative one. That is the")
    print("   honest scope of the architecture today -- stated by the code rather")
    print("   than discovered by a reviewer.")


if __name__ == "__main__":
    s = step1_run_the_chain()
    step2_taint(s)
    step3_identifiability()
    step4_uncertainty()
    step5_calibration()
    step6_attribution(s)
    step7_what_may_be_claimed()
