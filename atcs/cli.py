"""Command line entry points. A prototype someone should clone must be runnable."""
from __future__ import annotations

import argparse
import json
import sys


def _vision(_: argparse.Namespace) -> int:
    """Both loops with their gates.

    An earlier version imported `assessment()` from `atcs.worldmodel`, which does not
    exist here -- it lives in the sibling `mesoloop` package. That is the second
    cross-repository reference to slip in, and both were caught by a test that
    exercises every subcommand rather than by reading the code. The CLI is now built
    only from this package's own surface.
    """
    from .phases import loop_a_table
    from .patient import loop_b_table
    from .worldmodel import CONTEXTS

    print("LOOP A -- the bench chain")
    for r in loop_a_table():
        sim = "SIM" if r["simulation"] != "none" else "   "
        print(f"  P{r['phase']} [{sim}] {r['name']:24s} "
              f"failure revises: {r['failure_revises']}")
    print("\nLOOP B -- the patient chain")
    for r in loop_b_table():
        print(f"  S{r['step']}       {r['name']:24s} {r['does']}")
    print(f"\nCONTEXTS OF USE -- {len(CONTEXTS)} declared; run `atcs scope` for "
          f"which are permitted")
    return 0


def _scope(args: argparse.Namespace) -> int:
    from .worldmodel import credibility
    for c in ("rank_formulations_within_one_study",
              "predict_absolute_organ_delivery_in_a_new_laboratory",
              "predict_target_cell_exposure", "inform_a_patient_level_decision"):
        r = credibility(c, anchors=args.anchors)
        mark = "PERMITTED" if r["permitted"] else "REFUSED  "
        print(f"{mark}  {c}")
        for b in r["blockers"]:
            print(f"             - {b}")
    return 0


def _identifiability(_: argparse.Namespace) -> int:
    from .models import REGISTRY
    for key, (model, cands) in REGISTRY.items():
        r = model.report()
        print(f"\n{key}: {r['n_unknowns']} unknowns, {r['n_equations']} equation(s), "
              f"{r['degrees_of_freedom']} unresolved")
        print(f"  {r['verdict']}")
        for p in sorted(model.unknowns):
            if not model.identifiable(p)[0]:
                sets = model.minimal_additional_observations(p, cands)
                if sets:
                    print(f"  {p:22s} closed by {[list(s) for s in sets]}")
    return 0


def _data(args: argparse.Namespace) -> int:
    from .datasets import write_all
    made = write_all(args.outdir)
    for k, v in sorted(made.items()):
        print(f"{k:36s} {v['rows']:>5} rows x {v['cols']:>2} cols  "
              f"synthetic={v['synthetic']}")
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    from .benchmark import leak_and_remedy, scorecard, sibling_leakage
    from .datasets import generate_biodistribution
    df = generate_biodistribution(n_studies=args.studies, seed=args.seed)
    g = df.study_id.astype(str).to_numpy()
    lk = sibling_leakage(g)
    print(f"sibling leakage under a random split: "
          f"{lk['mean_sibling_fraction']:.1%} (sd {lk['sd']:.1%})\n")
    sc = scorecard(df, f"log10_pid_{args.organ}",
                   ["size_nm", "zeta_mv", "pdi", "dose_mg_per_kg"],
                   n_splits=args.splits, seed=args.seed)
    cols = ["protocol", "arm", "r2", "median_fold_error", "within_2fold", "n"]
    print(sc[cols].to_string(index=False))
    print()
    out = leak_and_remedy(sc)
    print(out["verdict"])
    print(f"\nexercises_protocol={out['exercises_protocol']}  "
          f"measures_performance={out['measures_performance']}")
    if not out["measures_performance"]:
        print("  -> generated substrate: this exercises the PROTOCOL and does not "
              "measure any model's real performance")
    return 0


def _sensitivity(args: argparse.Namespace) -> int:
    from .models import CYTOSOL_CANDIDATES, ORGAN_TO_CYTOSOL
    from .sensitivity import Parameter, measurement_priority, sobol_indices

    def chain(p):
        return (p["organ_fraction"] * p["f_parenchymal"]
                * p["phi_intact"] * p["k_escape"])

    params = [Parameter("organ_fraction", 0.05, 0.60, "NTDB liver %ID range"),
              Parameter("f_parenchymal", 0.10, 0.80, "unmeasured; literature spread"),
              Parameter("phi_intact", 0.20, 0.95, "cargo stability spread"),
              Parameter("k_escape", 0.005, 0.040, "reported escape fractions")]
    s = sobol_indices(chain, params, n_base=args.n_base, seed=args.seed)
    for r in s["indices"]:
        print(f"  {r['parameter']:16s} first-order {r['first_order']:>7.3f}  "
              f"total {r['total_effect']:>7.3f}  interaction {r['interaction']:>7.3f}")
    print(f"\n  interaction share {s['interaction_share']} over "
          f"{s['n_evaluations']:,} evaluations")
    mp = measurement_priority(s, ORGAN_TO_CYTOSOL, candidates=CYTOSOL_CANDIDATES)
    print()
    for r in mp["rows"]:
        print(f"  {r['parameter']:16s} total {r['total_effect']:>6.3f}  "
              f"identifiable={str(r['identifiable']):5s}  -> {r['quadrant']}")
    print(f"\n  {mp['verdict']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="atcs",
        description="Another Take (CS): the bench-bedside programme as a verified "
                    "dataflow program with evidential status enforced in code.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("vision", help="the two loops with their evidential asymmetry"
                   ).set_defaults(fn=_vision)
    s = sub.add_parser("scope", help="what the chain may be used for today")
    s.add_argument("--anchors", type=int, default=0)
    s.set_defaults(fn=_scope)
    sub.add_parser("identifiability", help="unknowns, equations and what closes them"
                   ).set_defaults(fn=_identifiability)
    d = sub.add_parser("data", help="write the reported and generated tables")
    d.add_argument("--outdir", default="data")
    d.set_defaults(fn=_data)
    b = sub.add_parser("benchmark", help="split protocols and the anchored remedy")
    b.add_argument("--studies", type=int, default=120)
    b.add_argument("--organ", default="liver")
    b.add_argument("--splits", type=int, default=5)
    b.add_argument("--seed", type=int, default=11)
    b.set_defaults(fn=_benchmark)
    v = sub.add_parser("sensitivity", help="Sobol indices and measurement priority")
    v.add_argument("--n-base", dest="n_base", type=int, default=2048)
    v.add_argument("--seed", type=int, default=3)
    v.set_defaults(fn=_sensitivity)

    a = p.parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":                        # pragma: no cover
    sys.exit(main())
