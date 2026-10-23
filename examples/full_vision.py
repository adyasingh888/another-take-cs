"""
The whole programme: both loops, the bridge, the data layer and the mixing rules.

    python examples/full_vision.py

Nine sections. Sections 5-7 are the ones a chain-shaped repository cannot express:
what the corpora can and cannot be mixed into, which rival mechanism the data
actually支持, and why the patient loop is a different operation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from atcs.agents import RefusalCase, contract_table, versus_baselines
from atcs.critic import DEFAULT_MECHANISMS, compare, discriminating_design
from atcs.datasets import (as_measurement, generate_biodistribution,
                           reported_ablation, reported_variance)
from atcs.multimodal import (early_fusion, icc_oneway, inventory, late_fusion,
                             MixingError, naive_total, partial_pooling,
                             shared_parameter, technique_table)
from atcs.patient import (companion_diagnostic, context_coverage, effective_n,
                          GovernanceError, reverse_translate, simulate_regimens,
                          state_estimate)
from atcs.phases import Phase, PhaseResult, loop_a_table, validate_inference, PhaseError
from atcs.taint import Taint
from atcs.worldmodel import credibility

L = "=" * 78


def s1_loop_a():
    print(f"{L}\n1. LOOP A -- the five-phase bench chain\n{L}")
    for r in loop_a_table():
        sim = "SIM" if r["simulation"] != "none" else "   "
        print(f"   P{r['phase']} [{sim}] {r['name']:22s} revises: {r['failure_revises']}")
    print("\n   Simulation happens TWICE, and the two are different:")
    p2 = PhaseResult(Phase.TARGET_VALIDATION, True, {"response": 0.82}, Taint.SIMULATED)
    ok, why = p2.may_claim_reachability()
    print(f"   Phase 2 may claim reachability: {ok}")
    print(f"      {why[:150]}")
    p4 = PhaseResult(Phase.DELIVERY_SIMULATION, False, {}, Taint.SIMULATED)
    tgt, why = p4.revision_target()
    print(f"\n   A Phase 4 failure revises -> Phase {int(tgt)}")
    print(f"      {why[:150]}")


def s2_phase5():
    print(f"\n{L}\n2. Phase 5 must compare rivals, not confirm one\n{L}")
    try:
        validate_inference(PhaseResult(Phase.STRESS_AND_INFERENCE, True, {},
                                       Taint.SIMULATED, mechanisms_compared=1))
    except PhaseError as e:
        print(f"   REFUSED: {str(e)[:170]}")


def s3_critic():
    print(f"\n{L}\n3. The mechanistic critic -- four rivals, one posterior\n{L}")
    conds = [{"organ_fraction": 0.6, "escape_fraction": e, "delay_h": 0.0}
             for e in (0.005, 0.015, 0.030, 0.015, 0.005)]
    truth = [0.6 * c["escape_fraction"] for c in conds]
    out = compare(conds, truth)
    for r in out["ranked"]:
        print(f"   {r['mechanism']:26s} posterior {r['posterior']:.3f}  "
              f"dBIC {r['delta_bic']:>8.1f}  params {r['n_parameters']}")
    print(f"\n   {out['verdict'][:160]}")
    dd = discriminating_design(DEFAULT_MECHANISMS, conds)
    print(f"   best discriminating condition: index {dd['best_condition_index']} "
          f"(spread {dd['ranked'][0]['prediction_spread']:.4f})")


def s4_data():
    print(f"\n{L}\n4. The data layer -- reported, generated, and the wall\n{L}")
    v = reported_variance()
    print("   reported between-study variance share:")
    for _, r in v.iterrows():
        print(f"      {r.organ:8s} {r.between_study_share:.3f}  "
              f"-> max R2 from formulation alone {r.max_r2_from_formulation_alone:.3f}")
    st = reported_ablation().attrs["sign_test"]
    print(f"\n   ablation sign test recomputed: p = {st['exact_binomial_p']} "
          f"(brief reports {st['brief_reports']}, matches: {st['matches']})")
    gen = generate_biodistribution(n_studies=300, seed=7)
    r = icc_oneway(gen["log10_pid_liver"], gen.study_id)
    print(f"\n   generated table reproduces the STRUCTURE: ICC {r['icc']:.3f}")
    print(f"      naive ratio would say {r['naive_ratio_of_group_mean_variance']:.3f} "
          f"-- overstated by {r['naive_overstatement']:.3f}")
    try:
        as_measurement(gen, "organ delivery R2")
    except Exception as e:
        print(f"\n   REFUSED as measurement: {str(e)[:165]}")


def s5_mixing():
    print(f"\n{L}\n5. Multimodal data -- what may be mixed, and how\n{L}")
    for row in inventory():
        print(f"   {row['modality']:22s} {row['n_rows']:>5} rows / "
              f"{row['n_groups']:>3} groups  {row['unit'][:38]}")
    nt = naive_total()
    print(f"\n   {nt['verdict'][:200]}")
    print(f"\n   Techniques, by what each assumes:")
    for t in technique_table():
        print(f"      {t['technique']:24s} assumes: {t['assumes'][:52]}")
    try:
        early_fusion("corona_proteomics", "hepg2_response", "patisiran_clinical")
    except MixingError as e:
        print(f"\n   REFUSED: {str(e)[:200]}")
    sp = shared_parameter({"corona": (0.20, 0.02), "pbpk": (0.90, 0.02)},
                          "apoe_affinity")
    print(f"\n   shared-parameter test: {sp['verdict'][:175]}")


def s6_loop_b():
    print(f"\n{L}\n6. LOOP B -- the patient loop is a different operation\n{L}")
    e = effective_n(29, 215, icc=0.5)
    print(f"   {e['note']}")
    print(f"      design effect {e['design_effect']}, effective n {e['effective_n']}")
    cov = context_coverage(["complement_c3", "crp", "egfr", "hepatic_fibrosis_stage"])
    cover = ", ".join(f"{k} {v['available']}/{v['n']}"
                      for k, v in sorted(cov["by_stage"].items()))
    print(f"\n   c_patient coverage by stage: {cover}")
    print(f"      cannot be personalised: {cov['stages_without_any_available_modulator']}")
    st = state_estimate(0.60, 0.25, [0.74, 0.71, 0.77], 0.06)
    print(f"\n   patient-state posterior {st['posterior_mean']:.3f} "
          f"+/- {st['posterior_sd']:.3f}  (population model untouched: "
          f"{st['population_model_untouched']})")
    for r in simulate_regimens(
            {"0.3 q3w": {"dose_mg_per_kg": 0.3, "interval_weeks": 3},
             "0.4 q3w": {"dose_mg_per_kg": 0.4, "interval_weeks": 3},
             "0.3 q6w": {"dose_mg_per_kg": 0.3, "interval_weeks": 6}},
            st["posterior_mean"], st["posterior_sd"]):
        flag = "distinguishable" if r["distinguishable_from_next"] else "overlaps next"
        print(f"      {r['regimen']:9s} response {r['predicted_response']:.3f} "
              f"[{r['lo']:.3f}, {r['hi']:.3f}]  {flag}")
    cd = companion_diagnostic(["apoe_plasma", "complement_c3", "hepatic_fibrosis_stage"],
                              st["posterior_sd"],
                              {"apoe_plasma": 0.55, "complement_c3": 0.20,
                               "hepatic_fibrosis_stage": 0.30})
    print(f"\n   best measurement: {cd['best_overall']} "
          f"| best DEPLOYABLE: {cd['best_deployable']}")


def s7_reverse():
    print(f"\n{L}\n7. Reverse translation yields a hypothesis, never a model update\n{L}")
    try:
        reverse_translate("x", 29, 215, ["differential exposure"], True)
    except GovernanceError as e:
        print(f"   REFUSED (one explanation): {str(e)[:165]}")
    r = reverse_translate("biomarker trajectory outside the predicted range",
                          29, 215,
                          ["differential exposure", "assay variation", "adherence"],
                          provenance_cleared=True)
    print(f"\n   outcome: {r['outcome']}  (effective n {r['effective_n']})")
    print(f"   motivates: {r['motivates']}")
    print(f"   does NOT: {r['does_not']}")


def s8_agents():
    print(f"\n{L}\n8. Agents -- the one axis on which they can differ\n{L}")
    for row in contract_table():
        print(f"   {row['role'][:34]:34s} may not: {row['forbidden'][:60]}")
    def cases(recall, n=40, fr=0):
        out = []
        for i in range(n // 2):
            out.append(RefusalCase(f"u{i}", True, i < int(recall * (n // 2))))
        for i in range(n - n // 2):
            out.append(RefusalCase(f"s{i}", False, i < fr))
        return out
    out = versus_baselines(cases(0.95, 40, 3), cases(0.45, 40, 0))
    print(f"\n   {out['verdict']}")
    print(f"   (no single score: {out['arms']['ours']['no_single_score'][:110]})")


def s9_experiment():
    print(f"\n{L}\n9. The next experiment, as a checkable object\n{L}")
    from atcs.experiment import ExperimentSpec, Resource, rank
    from atcs.models import CYTOSOL_CANDIDATES, ORGAN_TO_CYTOSOL
    cand = {c.name: c for c in CYTOSOL_CANDIDATES}
    specs = [
        ExperimentSpec("flow-sorted parenchymal fraction",
                       (cand["flow_sorted_parenchymal"],), "f_parenchymal",
                       0.30, 12, 4, 0.75, 0.35,
                       Resource(animals=48, assay_runs=12, weeks=6.0),
                       "closes the parenchymal fraction, which no whole-organ assay can"),
        ExperimentSpec("another homogenate series",
                       (ORGAN_TO_CYTOSOL.observations[0],), "f_parenchymal",
                       0.30, 30, 6, 0.80, 0.35,
                       Resource(animals=180, assay_runs=30, weeks=10.0),
                       "more of the measurement already in hand, at larger n"),
    ]
    out = rank(specs, ORGAN_TO_CYTOSOL)
    for r in (out["ranked"] or out["candidates"]):
        print(f"   {r['experiment']:34s} closes={str(r['closes']):5s} "
              f"power={r['power']:.2f} cost={r['cost_units']:>6.1f} "
              f"blocker={r['binding_constraint']}")
    print(f"\n   {out['headline'][:190]}")
    a = specs[0].assess(ORGAN_TO_CYTOSOL)
    p = a["power"]
    print(f"\n   power at n_eff {p['n_effective']:.0f} = {p['power']:.2f}; "
          f"the raw row count would say {p['power_if_rows_were_independent']:.2f} "
          f"({p['overstatement_factor']}x)")


def s10_execution():
    print(f"\n{L}\n10. Execution: attributable runs and honest joint uncertainty\n{L}")
    import numpy as np
    from atcs.execution import Engine
    from atcs.propagate import Dependence, Uncertain
    e = Engine(); e.register("organ_rf", "2.1.0"); e.register("stochastic", "0.1")
    r = e.run("organ_rf", lambda size_nm: 0.42, {"size_nm": 95.0},
              taint=Taint.DERIVED, dependence=Dependence.DEPENDENT)
    print(f"   run {r['run_id']} attributed to organ_rf@2.1.0")
    chk = e.reproduce_check("stochastic", lambda n: float(np.random.rand()), {"n": 1},
                            taint=Taint.SIMULATED, dependence=Dependence.UNKNOWN)
    print(f"   reproducibility: {chk['verdict'][:140]}")
    stages = [Uncertain(0.30, 0.22, "organ fraction"),
              Uncertain(0.42, 0.45, "target-cell fraction")]
    unk = e.compose("chain", stages, Dependence.UNKNOWN)
    print(f"\n   undeclared dependence -> {unk['interpretation'][:150]}")


def s11_m15():
    print(f"\n{L}\n11. ICH M15: a credibility PLAN, not a qualification\n{L}")
    from atcs.icm15 import (PROPOSED_FIRST_CONTEXT, credibility_plan, model_risk,
                           required_activities)
    r = model_risk(PROPOSED_FIRST_CONTEXT.influence, PROPOSED_FIRST_CONTEXT.consequence)
    print(f"   influence {r['influence']} x consequence {r['consequence']} "
          f"-> risk {r['risk_score']} ({r['risk_band']})")
    plan = credibility_plan(PROPOSED_FIRST_CONTEXT,
                            ["code_verification", "applicability_bounds",
                             "sensitivity_analysis", "numerical_verification"])
    print(f"   required: {len(plan['required_activities'])} activities")
    print(f"   missing : {plan['missing']}")
    print(f"\n   {plan['verdict'][:190]}")


def s12_planner():
    print(f"\n{L}\n12. The planner: the rejection log is the output\n{L}")
    from atcs.models import ORGAN_TO_CYTOSOL
    from atcs.planner import admissible_plans, plan_advantage
    from atcs.registry import ContextOfUse, RegisteredModel, Registry
    from atcs.status import Evidenced, Status
    reg = Registry()
    IND = Evidenced(Status.INDEPENDENT, "executed study-grouped diagnostic, Oct 2026")
    REP = Evidenced(Status.REPORTED, "project-reported benchmark, 20 Oct brief",
                    what_would_raise_it="independent reproduction on withheld data")
    for mid, ev, comp in [("organ_rf", IND, "whole_liver_homogenate"),
                          ("organ_gbm", REP, "whole_liver_homogenate"),
                          ("cyto_a", REP, "hepatocyte_cytosol_risc")]:
        reg.register(RegisteredModel(
            mid, ev, ContextOfUse(species="mouse", compartment=comp,
                                  question="disposition of lipid carriers",
                                  descriptor_ranges={"size_nm": (20.0, 200.0)},
                                  decision="research prioritisation of formulations"),
            ("size_nm",), "log10 delivery fraction"))
    taint = {"organ_rf": Taint.MEASURED, "organ_gbm": Taint.MEASURED,
             "cyto_a": Taint.SIMULATED}
    res = admissible_plans(
        reg, [["organ_rf", "organ_gbm"], ["cyto_a"]],
        stage_contexts=[{"species": "mouse", "compartment": "whole_liver_homogenate"},
                        {"species": "mouse", "compartment": "hepatocyte_cytosol_risc"}],
        decision="target-cell exposure", stage_taint=taint, max_taint=Taint.DERIVED,
        structural=ORGAN_TO_CYTOSOL, target="f_parenchymal")
    print(f"   enumerated {res['n_enumerated']}, admissible {res['n_admissible']}")
    print(f"   rejections by condition: {res['rejections_by_condition']}")
    print(f"   binding on EVERY candidate: "
          f"{res['conditions_binding_on_every_candidate']}")
    adv = plan_advantage(res, ["organ_rf", "cyto_a"], taint, Taint.DERIVED)
    print(f"\n   {adv['verdict'][:190]}")


def s9_scope():
    print(f"\n{L}\n9. So what may the chain be used for, today\n{L}")
    for c in ("rank_formulations_within_one_study",
              "predict_absolute_organ_delivery_in_a_new_laboratory",
              "predict_target_cell_exposure",
              "inform_a_patient_level_decision"):
        r = credibility(c, anchors=0)
        print(f"   {'PERMITTED' if r['permitted'] else 'REFUSED  '}  {c}")
    print("\n   Both loops are built, both are gated, and one of four contexts is")
    print("   permitted. The architecture is complete; the evidence is not, and the")
    print("   code says which is which.")


if __name__ == "__main__":
    s1_loop_a(); s2_phase5(); s3_critic(); s4_data()
    s5_mixing(); s6_loop_b(); s7_reverse(); s8_agents()
    s9_experiment(); s10_execution(); s11_m15(); s12_planner(); s9_scope()
