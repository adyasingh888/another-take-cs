"""
Figure 2 -- the leak, the remedy, and what the remedy costs in coverage.

Three panels, every number computed at build time:
  (a) R2 by split protocol, with the train-mean floor
  (b) interval width against coverage, zero-shot vs anchored
  (c) expected value of information: uncertainty reduction is not decision value

Run: python figures/make_figure2.py
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


def build(out="figures/atcs_figure2.png", n_studies=200, seed=11):
    from atcs.benchmark import (interval_scorecard, interval_summary,
                                leak_and_remedy, scorecard, sibling_leakage)
    from atcs.datasets import generate_biodistribution
    from atcs.value import Candidate, expected_value_of_information

    df = generate_biodistribution(n_studies=n_studies, seed=seed)
    feats = ["size_nm", "zeta_mv", "pdi", "dose_mg_per_kg"]
    sc = scorecard(df, "log10_pid_liver", feats, n_splits=5, seed=seed)
    lr = leak_and_remedy(sc)
    lk = sibling_leakage(df.study_id.astype(str).to_numpy())
    iv = interval_scorecard(df, "log10_pid_liver", feats, n_repeats=8, seed=0)
    isum = interval_summary(iv)

    fig, (aA, aB, aC) = plt.subplots(
        1, 3, figsize=(7.4, 2.85), gridspec_kw={"width_ratios": [1.1, 1.15, 1.1]})

    # ---- (a) the leak and the remedy -----------------------------------
    labels = ["random\n(published\npractice)", "grouped\n(the real\nquestion)",
              f"grouped +\n1 anchor"]
    vals = [lr["random_r2"], lr["grouped_r2"], lr["anchored_r2"]]
    cols = [ORANGE, RED, GREEN]
    aA.bar(range(3), vals, color=cols, width=0.62)
    aA.axhline(lr["grouped_train_mean_r2"], color=GREY, ls="--", lw=1.0,
               label="train-mean floor")
    for i, v in enumerate(vals):
        aA.annotate(f"{v:.3f}", (i, v + 0.025 if v >= 0 else v - 0.05),
                    ha="center", fontsize=6.8, color=GREY)
    aA.set_xticks(range(3)); aA.set_xticklabels(labels, fontsize=6.2)
    aA.set_ylabel("$R^2$ on held-out rows")
    aA.set_ylim(min(vals) - 0.12, max(vals) + 0.14)
    aA.set_title(f"{lk['mean_sibling_fraction']:.0%} of test rows have a\n"
                 f"same-study sibling in training", loc="left")
    aA.legend(loc="upper left", frameon=False)

    # ---- (b) width against coverage ------------------------------------
    nominal = isum["nominal_coverage"]
    for arm, col, mk in (("zero-shot", ORANGE, "o"),
                         (next(k for k in isum["arms"] if k.startswith("anchored")),
                          GREEN, "s")):
        sub = iv[iv.arm == arm]
        aB.scatter(10 ** (2 * sub.half_width), sub.empirical_coverage,
                   s=26, color=col, marker=mk, alpha=0.75, label=arm, zorder=3)
    aB.axhline(nominal, color=GREY, ls="--", lw=1.0)
    aB.annotate(f"nominal {nominal:.0%}", (0.02, nominal), xycoords=("axes fraction",
                "data"), ha="left", va="bottom", fontsize=6.3, color=GREY)
    aB.set_xscale("log")
    # major decade ticks only: matplotlib's log minor labels ran into each other
    aB.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=6))
    aB.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    aB.set_xlabel("interval width (fold)")
    aB.set_ylabel("empirical coverage")
    aB.set_title(f"narrows {isum['narrowing_factor_log_width']:.2f}x on log-width;\n"
                 f"coverage is {isum['coverage_status']}", loc="left")
    aB.legend(loc="upper right", frameon=False)

    # ---- (c) uncertainty reduction is not decision value ----------------
    cands = [Candidate("LNP-A", 0.62, 0.18), Candidate("LNP-B", 0.58, 0.22),
             Candidate("LNP-C", 0.31, 0.10)]
    arms = [("close\npair", {"LNP-A": 0.9, "LNP-B": 0.9}),
            ("leader\nonly", {"LNP-A": 1.0}),
            ("also-ran\n(resolved)", {"LNP-C": 1.0})]
    ev, var_removed, pswitch = [], [], []
    for _, res in arms:
        v = expected_value_of_information(cands, resolves=res, seed=0)
        ev.append(v["evsi"] if v["evsi_exceeds_its_own_noise"] else 0.0)
        var_removed.append(sum(res.get(c.name, 0.0) * c.sd ** 2 for c in cands))
        pswitch.append(v["probability_the_decision_changes"])
    x = np.arange(len(arms))
    w = 0.36
    vr = np.array(var_removed) / max(var_removed)
    evn = np.array(ev) / max(max(ev), 1e-12)
    aC.bar(x - w / 2, vr, w, color=BAND, edgecolor="none",
           label="variance removed (scaled)")
    aC.bar(x + w / 2, evn, w, color=GREEN, edgecolor="none",
           label="decision value (scaled)")
    for i, p in enumerate(pswitch):
        aC.annotate(f"P(switch)\n{p:.2f}", (i, 1.04), ha="center", va="bottom",
                    fontsize=6.0, color=GREY)
    aC.set_xticks(x); aC.set_xticklabels([a for a, _ in arms], fontsize=6.5)
    aC.set_xlabel("which candidate the experiment resolves")
    aC.set_ylim(0, 1.52)
    aC.set_ylabel("scaled to each panel's maximum")
    aC.set_title("removing uncertainty is not the\nsame as changing a decision",
                 loc="left")
    aC.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2,
              frameon=False)

    for ax, lab in ((aA, "a"), (aB, "b"), (aC, "c")):
        ax.annotate(lab, xy=(-0.14, 1.06), xycoords="axes fraction",
                    fontweight="bold", fontsize=9)

    fig.tight_layout(pad=0.5, w_pad=1.8)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    return out, {"leak": lr, "interval": isum, "evsi": ev, "p_switch": pswitch}


if __name__ == "__main__":
    p, info = build()
    print("written:", p)
    print("  leak gap", info["leak"]["leak_gap"], "| remedy gap", info["leak"]["remedy_gap"])
    print("  narrowing", info["interval"]["narrowing_factor_log_width"],
          "on log-width |", info["interval"]["coverage_status"])
    print("  EVSI", [round(e, 5) for e in info["evsi"]],
          "| P(switch)", info["p_switch"])
