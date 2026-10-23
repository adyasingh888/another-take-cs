"""
Figure 1 -- what the architecture establishes before any new data is collected.

Three panels, each computed from a module rather than typed:
  (a) identifiability by interface: unknowns against achievable equations
  (b) the clustering penalty: groups needed to detect an effect, against rho
  (c) the admissible-composition space, with rejections by condition

Run: python figures/make_figure1.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "axes.axisbelow": True,
})

GREEN, ORANGE, GREY, BAND, RED = "#2d6a4f", "#b5651d", "#6b6b70", "#e8e8ea", "#9b2226"


def build(out="figures/atcs_figure1.png"):
    from atcs.experiment import ExperimentSpec, Resource
    from atcs.models import CYTOSOL_CANDIDATES, REGISTRY
    from atcs.planner import admissible_plans
    from atcs.registry import ContextOfUse, RegisteredModel, Registry
    from atcs.status import Evidenced, Status
    from atcs.taint import Taint

    fig, (aA, aB, aC) = plt.subplots(
        1, 3, figsize=(7.4, 2.85), gridspec_kw={"width_ratios": [1.15, 1.2, 1.1]})

    # ---- (a) identifiability by interface -------------------------------
    names, unk, eqs = [], [], []
    for key, (model, _) in REGISTRY.items():
        r = model.report()
        names.append(key.replace("_", "\n"))
        unk.append(r["n_unknowns"])
        eqs.append(r["n_equations"])
    y = np.arange(len(names))[::-1]
    aA.barh(y, unk, height=0.52, color=BAND, edgecolor="none", label="unknowns")
    aA.barh(y, eqs, height=0.52, color=GREEN, edgecolor="none",
            label="achievable equations")
    for yi, u, e in zip(y, unk, eqs):
        aA.annotate(f"{u - e} unresolved", (u + 0.12, yi), va="center",
                    fontsize=6.3, color=RED if u > e else GREY)
    aA.set_yticks(y); aA.set_yticklabels(names, fontsize=6.3)
    aA.set_xlim(0, max(unk) * 1.72)
    aA.set_xlabel("count")
    aA.set_title("every interface is underdetermined\nbefore new measurement",
                 loc="left")
    aA.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
              frameon=False)

    # ---- (b) the clustering penalty -------------------------------------
    rhos = np.linspace(0.0, 0.95, 40)
    for m, col, ls in ((2, GREY, ":"), (4, ORANGE, "--"), (8, GREEN, "-")):
        need = []
        for rho in rhos:
            s = ExperimentSpec(
                name="x", measures=(CYTOSOL_CANDIDATES[0],),
                targets="f_parenchymal", effect_size=0.30, n_groups=12,
                m_per_group=m, rho=float(rho), residual_sd=0.35,
                resource=Resource(animals=1),
                rationale="power curve against the intraclass correlation")
            need.append(s.groups_needed(0.80))
        aB.plot(rhos, need, color=col, ls=ls, lw=1.5,
                label=f"{m} measurements per group")
    aB.axvspan(0.748, 0.882, color=BAND, zorder=0)
    aB.annotate("reported $\\rho$\n0.75–0.88", (0.815, 0.04), xycoords=("data",
                "axes fraction"), ha="center", va="bottom", fontsize=6.3, color=GREY)
    aB.set_xlabel("intraclass correlation $\\rho$ (between studies)")
    aB.set_ylabel("independent groups needed\nfor 80% power")
    aB.set_title("repeated measures stop buying power\nonce $\\rho$ is high", loc="left")
    aB.legend(loc="upper left", frameon=False)

    # ---- (c) the admissible-composition space ---------------------------
    reg = Registry()
    IND = Evidenced(Status.INDEPENDENT, "executed study-grouped diagnostic, Oct 2026")
    REP = Evidenced(Status.REPORTED, "project-reported benchmark, 20 Oct brief",
                    what_would_raise_it="independent reproduction on withheld data")
    for mid, ev, comp in [("organ_rf", IND, "whole_liver_homogenate"),
                          ("organ_gbm", REP, "whole_liver_homogenate"),
                          ("cyto_a", REP, "hepatocyte_cytosol_risc"),
                          ("cyto_b", REP, "hepatocyte_cytosol_risc")]:
        reg.register(RegisteredModel(
            mid, ev, ContextOfUse(species="mouse", compartment=comp,
                                  question="disposition of lipid carriers",
                                  descriptor_ranges={"size_nm": (20.0, 200.0)},
                                  decision="research prioritisation of formulations"),
            required_inputs=("size_nm",), output="log10 delivery fraction"))
    taint = {"organ_rf": Taint.MEASURED, "organ_gbm": Taint.MEASURED,
             "cyto_a": Taint.SIMULATED, "cyto_b": Taint.SIMULATED}
    octx = {"species": "mouse", "compartment": "whole_liver_homogenate"}
    cctx = {"species": "mouse", "compartment": "hepatocyte_cytosol_risc"}
    res = admissible_plans(
        reg, [["organ_rf", "organ_gbm"], ["cyto_a", "cyto_b"]],
        stage_contexts=[octx, cctx], decision="target-cell exposure",
        stage_taint=taint, max_taint=Taint.DERIVED,
        structural=REGISTRY["organ_to_cytosol"][0], target="f_parenchymal")
    conds = res["rejections_by_condition"]
    order = ["context_of_use", "provenance", "identifiability"]
    vals = [conds.get(k, 0) for k in order]
    binding = res["conditions_binding_on_every_candidate"]
    labs = [k.replace("_", " ") for k in order]
    bars = aC.bar(range(len(order)), vals,
                  color=[GREY, ORANGE, RED], width=0.6)
    aC.bar([len(order)], [res["n_admissible"]], color=GREEN, width=0.6)
    for i, v in enumerate(vals + [res["n_admissible"]]):
        aC.annotate(str(v), (i, v + 0.08), ha="center", fontsize=7, color=GREY)
    aC.set_xticks(range(len(order) + 1))
    aC.set_xticklabels(labs + ["admissible"], rotation=20, ha="right", fontsize=6.3)
    aC.set_ylabel(f"compositions (of {res['n_enumerated']} enumerated)")
    aC.set_ylim(0, max(vals + [res["n_admissible"], 1]) * 1.3)
    # The title is COMPUTED from the result, not typed. An earlier version asserted
    # that identifiability emptied the space while the data showed provenance --
    # which was itself caused by the planner short-circuiting on the first failing
    # condition. Both are fixed; the title can no longer disagree with the bars.
    if binding:
        pretty = " and ".join(b.replace("_", " ") for b in binding)
        aC.set_title(f"every composition is blocked by\n{pretty}", loc="left")
    else:
        aC.set_title(f"{res['n_admissible']} of {res['n_enumerated']} compositions\n"
                     f"are admissible", loc="left")

    for ax, lab in ((aA, "a"), (aB, "b"), (aC, "c")):
        ax.annotate(lab, xy=(-0.12, 1.06), xycoords="axes fraction",
                    fontweight="bold", fontsize=9)

    fig.tight_layout(pad=0.5, w_pad=1.6)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    return out, res


if __name__ == "__main__":
    p, res = build()
    print("written:", p)
    print("admissible:", res["n_admissible"], "of", res["n_enumerated"],
          "| rejections:", res["rejections_by_condition"])
