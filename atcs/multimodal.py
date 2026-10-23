"""
Multimodal data: the modalities, their units of observation, and how they may and
may not be mixed.

THE PROBLEM THIS MODULE EXISTS FOR
----------------------------------
The brief's §08 states it exactly: "Five useful corpora, but not five paired
stages. The sizes in the meeting material describe different units of observation;
the total must not be treated as an end-to-end experimental cohort." And §17:
"Do not create synthetic 'ground truth' to fill the gap."

Those two sentences bound every mixing technique below. The naive move -- stack
six tables, add a modality column, train one model -- fails not because of scale
or architecture but because the ROWS MEAN DIFFERENT THINGS. A corona row is a
protein-species share for one formulation in one plasma; a PBPK row is an organ
amount for one animal at one timepoint; a HepG2 row is a cellular endpoint for a
cell line that never saw an animal. Concatenating them produces a model of the
concatenation.

SO WHAT CAN BE DONE
-------------------
Five techniques, in increasing order of what they assume, each implemented with
its assumption stated and checkable:

  1. LINKAGE          join only where identity genuinely matches. The honest
                      baseline, and usually returns very little.
  2. LATE FUSION      train per modality, combine PREDICTIONS. Requires only that
                      the modalities predict the same target, not that their rows
                      are comparable. The correct default when units differ.
  3. PARTIAL POOLING  one hierarchical model with modality-level random effects.
                      Shares statistical strength without asserting the rows are
                      exchangeable -- the honest version of "mixing".
  4. SHARED-PARAMETER a mechanistic parameter appears in two modalities' likelihoods.
                      Strong, correct when the mechanism is real, and the only
                      technique that genuinely couples stages.
  5. REWEIGHTING      correct covariate shift between a source and target corpus.
                      Requires overlapping support, which is checked and usually
                      fails for these corpora.

And one that is REFUSED: early fusion of rows across modalities with different
units of observation. `early_fusion` exists only to raise, with the reason.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping, Sequence

import numpy as np

from .taint import Taint


class MixingError(RuntimeError):
    """A mixing technique was applied where its assumption does not hold."""


class Unit(Enum):
    """The unit of observation. Rows are comparable only within a unit."""

    FORMULATION_IN_PLASMA = "one formulation in one plasma sample"
    ANIMAL_ORGAN_TIMEPOINT = "one organ of one animal at one timepoint"
    FORMULATION_IN_STUDY = "one formulation within one study's protocol"
    LIPID_PANEL_MEMBER = "one ionizable lipid in a characterisation panel"
    CELL_LINE_WELL = "one well of an immortalised cell line"
    PATIENT_VISIT = "one patient at one scheduled visit"


@dataclass(frozen=True)
class Modality:
    """One corpus: what it measures, on what unit, and what it can support."""

    name: str
    unit: Unit
    n_rows: int
    n_groups: int
    group_key: str
    measures: str
    stage: str
    taint: Taint
    caution: str

    @property
    def rows_per_group(self) -> float:
        return self.n_rows / self.n_groups if self.n_groups else float("nan")


#: The brief's own inventory, with its own reported sizes and cautions.
MODALITIES: dict[str, Modality] = {m.name: m for m in [
    Modality("corona_proteomics", Unit.FORMULATION_IN_PLASMA, 1265, 83, "study_id",
             "adsorbed protein species shares (ApoE, albumin, C3, IgG)", "x_id",
             Taint.MEASURED,
             "593 + 672 rows over 83 studies and 62 formulations; not measured for "
             "all MesoBench carriers"),
    Modality("ntdb_biodistribution", Unit.ANIMAL_ORGAN_TIMEPOINT, 1682, 86, "study_id",
             "six-organ %ID/g", "x_organ", Taint.MEASURED,
             "external table; source and protocol heterogeneity; 74.8-88.2% of "
             "variance is between-study"),
    Modality("mesobench_contrasts", Unit.FORMULATION_IN_STUDY, 2135, 288, "study_id",
             "within-study formulation contrasts", "x_organ", Taint.MEASURED,
             "288 studies of which 120 are multi-formulation; only those 120 "
             "contribute a contrast"),
    Modality("lnp_escape_panel", Unit.LIPID_PANEL_MEMBER, 412, 1, "panel",
             "ionizable lipid pKa and delta log D against escape", "x_barrier",
             Taint.MEASURED,
             "a matched 412-row subset of a 19.2k panel; panel membership is not a "
             "random sample of lipid space"),
    Modality("hepg2_response", Unit.CELL_LINE_WELL, 1675, 1, "cell_line",
             "cellular endpoints in an immortalised line", "x_cell", Taint.MEASURED,
             "largely UNPAIRED with the disposition rows; a cell line never saw an "
             "animal, so no row here shares an animal with any PBPK row"),
    Modality("perturbation_atlas", Unit.CELL_LINE_WELL, 481, 42, "study_id",
             "delivery-coupled perturbation transcriptomics", "x_cell", Taint.MEASURED,
             "42 studies; the only corpus that couples delivery to omics at all"),
    Modality("patisiran_clinical", Unit.PATIENT_VISIT, 215, 29, "patient_id",
             "longitudinal TTR knockdown and biomarkers", "x_cell", Taint.MEASURED,
             "215 observations from 29 PATIENTS; not 215 independent samples"),
]}


def inventory() -> list[dict]:
    """The corpora with their units, so the mismatch is visible at a glance."""
    return [{"modality": m.name, "stage": m.stage, "unit": m.unit.value,
             "n_rows": m.n_rows, "n_groups": m.n_groups,
             "rows_per_group": round(m.rows_per_group, 2),
             "group_key": m.group_key, "caution": m.caution}
            for m in MODALITIES.values()]


def naive_total() -> dict:
    """The number that must not be reported, and why.

    Computed here precisely so it can be refused with its own arithmetic attached:
    summing rows across incommensurable units produces a figure that looks like a
    cohort size and is not one.
    """
    total = sum(m.n_rows for m in MODALITIES.values())
    units = {m.unit for m in MODALITIES.values()}
    return {"naive_row_total": total, "distinct_units_of_observation": len(units),
            "units": sorted(u.value for u in units),
            "verdict": (f"{total} rows across {len(units)} different units of "
                        f"observation. This total is not a cohort size and must not "
                        f"be reported as one: no row set here measures the same "
                        f"intervention through consecutive stages."),
            "largest_genuinely_pooled": max(m.n_rows for m in MODALITIES.values())}


# -------------------------------------------------------- 1. LINKAGE
def linkage(a: str, b: str, keys_a: Sequence[Mapping[str, str]],
            keys_b: Sequence[Mapping[str, str]],
            require: Sequence[str] = ("formulation_id", "batch_id")) -> dict:
    """Join two modalities only where identity genuinely matches.

    `require` defaults to formulation AND batch, because the brief's own adequacy
    criterion is "the same carrier/cargo/formulation and batch". Matching on
    formulation alone pairs a particle with a differently-manufactured namesake,
    and batch-to-batch variation in this field is not small.
    """
    for name in (a, b):
        if name not in MODALITIES:
            raise MixingError(f"unknown modality {name!r}")
    ma, mb = MODALITIES[a], MODALITIES[b]
    sa = {tuple(k.get(r, "") for r in require) for k in keys_a}
    sb = {tuple(k.get(r, "") for r in require) for k in keys_b}
    sa.discard(tuple("" for _ in require)); sb.discard(tuple("" for _ in require))
    shared = sa & sb
    return {"left": a, "right": b, "match_keys": list(require),
            "n_left": len(sa), "n_right": len(sb), "n_linked": len(shared),
            "linkage_rate": round(len(shared) / max(len(sa), 1), 4),
            "units_differ": ma.unit is not mb.unit,
            "verdict": ("linked on full identity" if shared else
                        "no shared identity: these corpora describe different "
                        "materials and cannot be joined at the row level")}


# -------------------------------------------------------- 2. LATE FUSION
def late_fusion(predictions: Mapping[str, Sequence[float]],
                weights: Mapping[str, float] | None = None,
                variances: Mapping[str, float] | None = None) -> dict:
    """Combine per-modality PREDICTIONS, not rows.

    The correct default when units differ, because it assumes only that each
    modality predicts the same target -- never that their rows are comparable.
    With `variances` the weights are inverse-variance, which is optimal under
    independence; the independence assumption is flagged rather than buried,
    because these modalities share studies and are therefore correlated.
    """
    if not predictions:
        raise MixingError("nothing to fuse")
    names = list(predictions)
    arrs = {k: np.asarray(v, float) for k, v in predictions.items()}
    n = {len(v) for v in arrs.values()}
    if len(n) != 1:
        raise MixingError(
            f"modalities predict {sorted(n)} items respectively; late fusion "
            f"combines predictions FOR THE SAME ITEMS, so they must align")
    if variances is not None:
        missing = [k for k in names if k not in variances]
        if missing:
            raise MixingError(f"no variance given for {missing}")
        w = {k: 1.0 / max(variances[k], 1e-12) for k in names}
        basis = "inverse-variance (optimal under INDEPENDENCE, which is not checked)"
    elif weights is not None:
        w = {k: float(weights[k]) for k in names}
        basis = "caller-supplied weights"
    else:
        w = {k: 1.0 for k in names}
        basis = "equal weights"
    tot = sum(w.values())
    fused = sum(w[k] * arrs[k] for k in names) / tot
    return {"fused": fused, "weights": {k: round(w[k] / tot, 4) for k in names},
            "basis": basis, "n_modalities": len(names),
            "caveat": ("these corpora share studies, so their errors are correlated "
                       "and inverse-variance weighting will be overconfident; the "
                       "fused interval needs a study-clustered calibration")}


# -------------------------------------------------------- 3. PARTIAL POOLING
def partial_pooling(group_means: Mapping[str, float],
                    group_ns: Mapping[str, int],
                    within_var: float) -> dict:
    """One hierarchical model with group-level random effects.

    The honest version of "mixing": strength is shared, but each group keeps its
    own level, so the model never asserts that rows from different groups are
    exchangeable. Returns the shrunk group estimates and the between-group variance
    estimated by method of moments, with `pooling_factor` per group -- which is the
    quantity that says how much each group actually borrowed.
    """
    ks = list(group_means)
    if len(ks) < 2:
        raise MixingError("partial pooling needs at least two groups")
    if within_var <= 0:
        raise MixingError("within-group variance must be positive")
    y = np.array([group_means[k] for k in ks], float)
    nn = np.array([group_ns[k] for k in ks], float)
    if (nn < 1).any():
        raise MixingError("every group needs at least one observation")
    grand = float(np.average(y, weights=nn))
    tot_var = float(np.var(y, ddof=1))
    tau2 = max(tot_var - within_var / nn.mean(), 0.0)
    out = {}
    for k, yi, ni in zip(ks, y, nn):
        se2 = within_var / ni
        lam = tau2 / (tau2 + se2) if (tau2 + se2) > 0 else 0.0
        out[k] = {"raw": round(float(yi), 5),
                  "shrunk": round(float(lam * yi + (1 - lam) * grand), 5),
                  "pooling_factor": round(float(1 - lam), 4), "n": int(ni)}
    return {"grand_mean": round(grand, 5), "between_group_var": round(tau2, 6),
            "within_group_var": within_var, "groups": out,
            "note": ("each group keeps its own level; pooling_factor is how much it "
                     "borrowed. A factor near 1 means that group's data were ignored "
                     "in favour of the grand mean, which is worth seeing")}


# -------------------------------------------------- 4. SHARED PARAMETER
def shared_parameter(modality_estimates: Mapping[str, tuple[float, float]],
                     parameter: str) -> dict:
    """Combine estimates of the SAME mechanistic parameter from several modalities.

    The only technique here that genuinely couples stages, because it asserts
    something real: that a physical parameter appearing in two likelihoods is the
    same number. That assertion is testable, and the test is heterogeneity -- if the
    modality-specific estimates disagree beyond their stated precision, the shared-
    parameter assumption is refuted and combining them is wrong.

    Cochran's Q with its chi-square tail is reported, and a significant result is
    returned as a REFUSAL rather than a wider interval.
    """
    from scipy.stats import chi2
    ks = list(modality_estimates)
    if len(ks) < 2:
        raise MixingError("a shared parameter needs at least two modalities")
    est = np.array([modality_estimates[k][0] for k in ks], float)
    se = np.array([modality_estimates[k][1] for k in ks], float)
    if (se <= 0).any():
        raise MixingError("standard errors must be positive")
    w = 1.0 / se ** 2
    pooled = float(np.sum(w * est) / np.sum(w))
    pooled_se = float(math.sqrt(1.0 / np.sum(w)))
    q = float(np.sum(w * (est - pooled) ** 2))
    dof = len(ks) - 1
    p = float(chi2.sf(q, dof))
    i2 = max(0.0, (q - dof) / q) if q > 0 else 0.0
    consistent = p >= 0.05
    return {"parameter": parameter, "modalities": ks,
            "pooled_estimate": round(pooled, 5) if consistent else None,
            "pooled_se": round(pooled_se, 5) if consistent else None,
            "cochran_q": round(q, 4), "dof": dof, "p_heterogeneity": round(p, 5),
            "i_squared": round(i2, 4), "shared_parameter_supported": consistent,
            "verdict": ("the modality-specific estimates are consistent, so treating "
                        "this as one shared parameter is supported"
                        if consistent else
                        f"estimates disagree beyond their stated precision "
                        f"(Q = {q:.2f}, p = {p:.4f}, I2 = {i2:.0%}). The shared-"
                        f"parameter assumption is REFUTED: these modalities are not "
                        f"measuring the same quantity, and pooling them would hide "
                        f"that. No pooled estimate is returned.")}


# -------------------------------------------------------- 5. REWEIGHTING
def covariate_shift_weights(source: Mapping[str, Sequence[float]],
                            target: Mapping[str, Sequence[float]],
                            clip: float = 20.0) -> dict:
    """Importance weights to correct covariate shift from source to target.

    Gaussian density ratio per feature, which is crude but transparent. The
    substantive part is the SUPPORT CHECK: importance weighting is only valid where
    the target's support is contained in the source's, and for these corpora it
    frequently is not -- a clinical cohort contains patients no mouse study covers.
    Where support fails the function says so and the weights are not usable,
    because a huge clipped weight is extrapolation wearing a weight's clothing.
    """
    feats = sorted(set(source) & set(target))
    if not feats:
        raise MixingError("source and target share no features")
    rows, logw = [], np.zeros(len(next(iter(target.values()))))
    for f in feats:
        s = np.asarray(source[f], float); t = np.asarray(target[f], float)
        ms, ss = s.mean(), s.std(ddof=1) or 1e-9
        mt, st = t.mean(), t.std(ddof=1) or 1e-9
        out_of_support = float(np.mean((t < s.min()) | (t > s.max())))
        rows.append({"feature": f, "source_mean": round(float(ms), 4),
                     "target_mean": round(float(mt), 4),
                     "standardised_shift": round(float(abs(mt - ms) / ss), 4),
                     "target_fraction_outside_source_range": round(out_of_support, 4)})
        logw += (-0.5 * ((t - mt) / st) ** 2 - np.log(st)) - \
                (-0.5 * ((t - ms) / ss) ** 2 - np.log(ss))
    w = np.exp(logw - logw.max())
    clipped = float(np.mean(w > clip * w.mean()))
    worst = max(rows, key=lambda r: r["target_fraction_outside_source_range"])
    usable = worst["target_fraction_outside_source_range"] < 0.05
    return {"features": rows, "weights": np.clip(w, 0, clip * w.mean()),
            "ess_fraction": round(float(w.sum() ** 2 / (len(w) * (w ** 2).sum())), 4),
            "fraction_clipped": round(clipped, 4),
            "support_adequate": usable,
            "verdict": ("support overlaps; weights are usable" if usable else
                        f"{worst['target_fraction_outside_source_range']:.0%} of target "
                        f"rows fall outside the source range on {worst['feature']!r}. "
                        f"Importance weighting cannot correct for a region the source "
                        f"never covered -- a large clipped weight is extrapolation "
                        f"wearing a weight's clothing.")}


def icc_oneway(values: Sequence[float], groups: Sequence[str]) -> dict:
    """One-way random-effects ICC by method of moments.

    WHY NOT var(group means) / var(all)
    -----------------------------------
    Because that ratio is biased UPWARD, and by a knowable amount. The variance of
    a group mean over m observations is tau^2 + sigma^2/m, so the naive ratio
    estimates (tau^2 + sigma^2/m) / (tau^2 + sigma^2), which exceeds the true
    tau^2 / (tau^2 + sigma^2) for every m > 1. With four arms per study the
    overstatement is substantial -- this repository's own generator was measured at
    0.861 against a true 0.748 by the naive route, which is how the bias was found.

    The unbiased route subtracts the within-group mean square:

        ICC = (MSB - MSW) / (MSB + (m0 - 1) * MSW)

    with m0 the standard unbalanced-design correction. This is the estimator any
    reported variance decomposition should be using, and the gap between the two is
    worth reporting because a reader cannot tell from a bare percentage which was
    computed.
    """
    y = np.asarray(values, float)
    g = np.asarray(groups)
    if y.size != g.size:
        raise MixingError("values and groups must align")
    uniq = np.unique(g)
    k = uniq.size
    if k < 2:
        raise MixingError("ICC needs at least two groups")
    n = y.size
    ns = np.array([(g == u).sum() for u in uniq], float)
    means = np.array([y[g == u].mean() for u in uniq])
    grand = y.mean()
    ssb = float(np.sum(ns * (means - grand) ** 2))
    ssw = float(np.sum([(y[g == u] - means[i]) ** 2 for i, u in enumerate(uniq)][0]
                       if False else sum(((y[g == u] - means[i]) ** 2).sum()
                                         for i, u in enumerate(uniq))))
    dfb, dfw = k - 1, n - k
    if dfw <= 0:
        raise MixingError("every group has one observation: within variance is undefined")
    msb, msw = ssb / dfb, ssw / dfw
    m0 = (n - float(np.sum(ns ** 2)) / n) / dfb
    tau2 = max((msb - msw) / m0, 0.0)
    icc = tau2 / (tau2 + msw) if (tau2 + msw) > 0 else 0.0
    naive = float(np.var(means, ddof=1) / np.var(y, ddof=1)) if np.var(y, ddof=1) > 0 else 0.0
    return {"icc": round(float(icc), 4), "tau2_between": round(float(tau2), 6),
            "sigma2_within": round(float(msw), 6), "n_groups": k, "n": n,
            "mean_group_size": round(float(ns.mean()), 2), "m0": round(float(m0), 3),
            "naive_ratio_of_group_mean_variance": round(naive, 4),
            "naive_overstatement": round(naive - float(icc), 4),
            "note": ("the naive ratio estimates (tau2 + sigma2/m)/(tau2 + sigma2) and "
                     "therefore exceeds the ICC for every m > 1; the gap above is the "
                     "bias, and a bare percentage does not say which was computed")}


# ------------------------------------------------------------ THE REFUSAL
def early_fusion(*modalities: str) -> None:
    """Row-level concatenation across modalities. Always refuses.

    Present as a function so the refusal is reachable and documented, rather than
    being a rule in a README that a future contributor has not read.
    """
    units = {}
    for m in modalities:
        if m not in MODALITIES:
            raise MixingError(f"unknown modality {m!r}")
        units.setdefault(MODALITIES[m].unit, []).append(m)
    raise MixingError(
        "early fusion across these modalities is refused. Their units of observation "
        f"are {[u.value for u in units]}, so a concatenated row set has no consistent "
        "meaning and a model trained on it is a model of the concatenation. Use "
        "late_fusion (combine predictions), partial_pooling (share strength, keep "
        "levels) or shared_parameter (couple one mechanism, and test that it IS "
        "shared). If the goal is an end-to-end trajectory, the honest answer is that "
        "no corpus here contains one -- and inventing it is the specific thing the "
        "20 Oct brief forbids.")


def technique_table() -> list[dict]:
    """The five techniques with what each assumes, for a methods section."""
    return [
        {"technique": "linkage", "assumes": "identity genuinely matches",
         "couples_stages": "yes, where it matches",
         "typical_yield": "very little, honestly reported"},
        {"technique": "late_fusion", "assumes": "modalities predict the same target",
         "couples_stages": "no", "typical_yield": "robust; the correct default"},
        {"technique": "partial_pooling",
         "assumes": "groups are exchangeable AT THE GROUP LEVEL",
         "couples_stages": "no", "typical_yield": "shares strength without pooling rows"},
        {"technique": "shared_parameter",
         "assumes": "one physical parameter is common -- and TESTS it",
         "couples_stages": "yes", "typical_yield": "strong when heterogeneity passes"},
        {"technique": "covariate_shift_weights",
         "assumes": "target support is contained in source support",
         "couples_stages": "no", "typical_yield": "often fails the support check"},
        {"technique": "early_fusion", "assumes": "rows are comparable -- FALSE here",
         "couples_stages": "n/a", "typical_yield": "refused"},
    ]
