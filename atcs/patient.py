"""
Loop B: the six-step patient-facing chain, conditioned on c_patient.

WHAT MAKES THIS A DIFFERENT OPERATION
-------------------------------------
The brief's §05 is unambiguous: the patient loop's "immediate task is state
estimation and supported response modelling; patient-facing clinical decisions are
a longer-term, higher-evidence application". And §03: "Updating a patient-state
posterior is not the same as retraining or deploying a population model."

Those two sentences are the module's entire design. A patient loop that silently
updates a population model is the failure this architecture exists to prevent, so
`reverse_translate` returns a RESEARCH HYPOTHESIS and has no code path that
touches a model version. There is nothing to misconfigure.

THE CONTEXT VECTOR, AND WHY IT IS NOT A FEATURE VECTOR
------------------------------------------------------
The brief's §3.2 conditions the mechanistic backbone on "an individual patient's
clinical, proteomic, and genomic context vector c_patient", and RQ2 names what is
in it: plasma proteome opsonisation (ApoE, complement C3/C4, CRP, anti-PEG
antibodies), hepatic fibrosis, altered vascular clearance.

The design decision here is that c_patient entries are MECHANISM-TYPED rather than
pooled into a feature vector. Each field declares which stage of the delivery chain
it modulates and in which direction. Two reasons, both substantive:

  1. A field that modulates Stage 1 (corona) and a field that modulates Stage 2
     (clearance) enter the model at different places. Concatenating them into one
     vector and learning weights discards that, and with 29 patients there is no
     prospect of recovering it from data.
  2. The direction is a prior commitment that can be checked. If fibrosis is
     declared to reduce hepatocyte access and the fitted effect goes the other way,
     that is a finding -- but only if the direction was declared in advance.

THE PATIENT COUNT PROBLEM, ENFORCED
-----------------------------------
The brief states it twice, and it is the kind of error that survives into print:
the patisiran phase II enrolled 29 patients; 215 repeated observations are not 215
independent patients. `effective_n()` computes the design-effect-adjusted sample
size, and `require_independent_patients()` refuses any inference that has used the
observation count as though it were a patient count.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Mapping, Sequence

import numpy as np

from .taint import Taint


class GovernanceError(RuntimeError):
    """An operation was attempted that patient-loop governance forbids."""


class Step(IntEnum):
    PATIENT_CONTEXT = 1       # assemble and provenance-check c_patient
    STATE_ESTIMATION = 2      # posterior over this patient's delivery state
    REGIMEN_SIMULATION = 3    # candidate regimens under that posterior
    CREDIBILITY_GATE = 4      # ICH M15: what may this support?
    COMPANION_DIAGNOSTIC = 5  # which measurement would most reduce the uncertainty
    REVERSE_TRANSLATION = 6   # unexpected observation -> research hypothesis


# ------------------------------------------------- the context vector, typed
@dataclass(frozen=True)
class ContextField:
    """One entry of c_patient, typed by the mechanism it modulates."""

    name: str
    modulates_stage: str          # which state group it enters
    direction: int                # +1 raises the modulated quantity, -1 lowers it
    units: str
    rationale: str
    routinely_measured: bool      # is this available in standard clinical care?

    def __post_init__(self) -> None:
        if self.direction not in (-1, 0, +1):
            raise GovernanceError("direction must be -1, 0 or +1")
        if len(self.rationale) < 30:
            raise GovernanceError(
                f"{self.name}: a context field must state WHY it modulates its stage. "
                f"An undeclared mechanism cannot be checked against a fitted effect, "
                f"which is the only way a 29-patient cohort yields anything.")


#: c_patient as the brief's RQ2 specifies it. Stage assignment is the claim.
C_PATIENT: dict[str, ContextField] = {
    f.name: f for f in [
        ContextField("apoe_plasma", "x_id", +1, "mg/dL",
                     "plasma ApoE competes for the particle surface and raises the "
                     "ApoE share of the corona, favouring LDLR-mediated hepatocyte uptake",
                     routinely_measured=False),
        ContextField("complement_c3", "x_id", -1, "mg/dL",
                     "C3 opsonisation shifts the corona toward complement-mediated "
                     "phagocyte clearance and away from hepatocyte delivery",
                     routinely_measured=True),
        ContextField("complement_c4", "x_id", -1, "mg/dL",
                     "C4 participates in the classical pathway and co-varies with C3 "
                     "opsonisation of the particle surface",
                     routinely_measured=True),
        ContextField("crp", "x_id", -1, "mg/L",
                     "CRP marks acute-phase proteome remodelling, which changes which "
                     "proteins are available to adsorb and shifts the corona",
                     routinely_measured=True),
        ContextField("anti_peg_igg", "x_organ", -1, "titre",
                     "pre-existing anti-PEG antibodies drive accelerated blood "
                     "clearance, removing particles before organ uptake can occur",
                     routinely_measured=False),
        ContextField("hepatic_fibrosis_stage", "x_barrier", -1, "METAVIR F0-F4",
                     "fibrosis increases sinusoidal capillarisation and collagen "
                     "deposition, reducing the parenchymal fraction of hepatic uptake",
                     routinely_measured=True),
        ContextField("egfr", "x_organ", -1, "mL/min/1.73m2",
                     "reduced glomerular filtration alters systemic clearance and "
                     "therefore the plasma residence driving organ uptake",
                     routinely_measured=True),
        ContextField("albumin", "x_id", 0, "g/L",
                     "albumin is the dominant dysopsonin but its net directional "
                     "effect on hepatocyte delivery is not established, so the "
                     "direction is declared as zero rather than guessed",
                     routinely_measured=True),
        ContextField("ttr_genotype", "x_cell", 0, "variant",
                     "the TTR variant determines baseline aggregation propensity and "
                     "the disease endpoint, but acts downstream of delivery rather "
                     "than on transport",
                     routinely_measured=True),
    ]
}


def context_coverage(available: Sequence[str]) -> dict:
    """Which parts of c_patient are obtainable in routine care?

    The practical constraint on Loop B: a context field that requires a research
    assay cannot condition a model intended for clinical use, however predictive it
    is. Reported per stage, because a stage with NO routinely available modulator
    cannot be personalised at all.
    """
    have = set(available)
    rows = []
    for f in C_PATIENT.values():
        rows.append({"field": f.name, "stage": f.modulates_stage,
                     "available": f.name in have,
                     "routine": f.routinely_measured,
                     "direction": f.direction})
    by_stage: dict[str, dict] = {}
    for r in rows:
        s = by_stage.setdefault(r["stage"], {"n": 0, "available": 0, "routine": 0})
        s["n"] += 1
        s["available"] += int(r["available"])
        s["routine"] += int(r["routine"])
    unpersonalisable = [s for s, v in by_stage.items() if v["available"] == 0]
    return {"fields": rows, "by_stage": by_stage,
            "stages_without_any_available_modulator": sorted(unpersonalisable),
            "note": ("a stage with no available modulator cannot be personalised; "
                     "the model still runs there, on population parameters, and "
                     "should say so rather than implying patient-specificity")}


# --------------------------------------------------- the sample-size problem
def effective_n(n_patients: int, n_observations: int,
                icc: float = 0.5) -> dict:
    """Design-effect-adjusted sample size for repeated measures.

        DEFF = 1 + (m - 1) * ICC,    n_eff = n_obs / DEFF

    with m the mean observations per patient. At ICC = 0.5 and 7.4 observations
    per patient the effective sample size is barely above the patient count, which
    is the quantitative form of the brief's warning. Reported with both the naive
    and adjusted figures so the gap is visible rather than inferred.
    """
    if n_patients < 1:
        raise GovernanceError("no patients")
    if n_observations < n_patients:
        raise GovernanceError("fewer observations than patients")
    if not 0.0 <= icc <= 1.0:
        raise GovernanceError("ICC must lie in [0, 1]")
    m = n_observations / n_patients
    deff = 1.0 + (m - 1.0) * icc
    return {"n_patients": n_patients, "n_observations": n_observations,
            "obs_per_patient": round(m, 2), "icc": icc,
            "design_effect": round(deff, 3),
            "effective_n": round(n_observations / deff, 1),
            "naive_overstatement": round(n_observations / (n_observations / deff), 2),
            "note": (f"{n_observations} observations from {n_patients} patients is "
                     f"not {n_observations} independent samples; at ICC {icc} the "
                     f"effective size is {n_observations / deff:.0f}")}


def require_independent_patients(claimed_n: int, n_patients: int,
                                 claim: str) -> None:
    """Refuse an inference that used observations as though they were patients."""
    if claimed_n > n_patients:
        raise GovernanceError(
            f"claim {claim!r} asserts n = {claimed_n} but the cohort contains "
            f"{n_patients} patients. Repeated observations are not independent "
            f"samples; use effective_n() and report the design effect.")


# --------------------------------------------------------------- the steps
def state_estimate(prior_mean: float, prior_sd: float,
                   observations: Sequence[float], obs_sd: float) -> dict:
    """Step 2: a posterior over THIS patient's delivery state.

    Conjugate normal update, deliberately simple, because the point is not the
    inference machinery -- it is that the result is a patient-state POSTERIOR that
    lives and dies with this patient. `population_model_untouched` is returned as
    an explicit True to make the architectural separation legible at the call site.
    """
    if prior_sd <= 0 or obs_sd <= 0:
        raise GovernanceError("prior and observation SDs must be positive")
    y = np.asarray(observations, float)
    if y.size == 0:
        return {"posterior_mean": prior_mean, "posterior_sd": prior_sd,
                "n_observations": 0, "shrinkage": 0.0,
                "population_model_untouched": True,
                "note": "no observations: the posterior is the prior"}
    prec_prior, prec_obs = 1 / prior_sd ** 2, y.size / obs_sd ** 2
    post_mean = (prec_prior * prior_mean + prec_obs * y.mean()) / (prec_prior + prec_obs)
    post_sd = math.sqrt(1 / (prec_prior + prec_obs))
    return {"posterior_mean": round(post_mean, 5), "posterior_sd": round(post_sd, 5),
            "n_observations": int(y.size),
            "shrinkage": round(prec_obs / (prec_prior + prec_obs), 4),
            "population_model_untouched": True,
            "note": ("a patient-state posterior. Updating it is not retraining: no "
                     "population parameter has changed and no model version has moved")}


def simulate_regimens(candidates: Mapping[str, Mapping[str, float]],
                      posterior_mean: float, posterior_sd: float) -> list[dict]:
    """Step 3: rank candidate regimens under the patient-state posterior.

    Returns a RANKING with overlap flags rather than a winner. With a 29-patient
    evidence base the posterior is wide, and two regimens whose intervals overlap
    are not distinguishable for this patient -- saying so is the useful output.
    """
    rows = []
    for name, spec in candidates.items():
        dose = float(spec.get("dose_mg_per_kg", 0.0))
        interval_w = float(spec.get("interval_weeks", 3.0))
        if dose <= 0 or interval_w <= 0:
            raise GovernanceError(f"regimen {name!r} has a non-positive dose or interval")
        exposure = posterior_mean * dose * (3.0 / interval_w)
        sd = posterior_sd * dose * (3.0 / interval_w)
        rows.append({"regimen": name, "dose_mg_per_kg": dose,
                     "interval_weeks": interval_w,
                     "predicted_response": round(exposure, 4),
                     "sd": round(sd, 4),
                     "lo": round(exposure - 1.96 * sd, 4),
                     "hi": round(exposure + 1.96 * sd, 4)})
    rows.sort(key=lambda r: -r["predicted_response"])
    for i, r in enumerate(rows):
        r["distinguishable_from_next"] = (
            i + 1 < len(rows) and r["lo"] > rows[i + 1]["hi"])
    return rows


def companion_diagnostic(fields: Sequence[str], posterior_sd: float,
                         variance_explained: Mapping[str, float]) -> dict:
    """Step 5: which single measurement most reduces this patient's uncertainty?

    Ranked by expected variance reduction, and restricted to fields that are
    actually obtainable -- a companion diagnostic requiring a research assay is a
    research proposal, not a companion diagnostic.
    """
    out = []
    for f in fields:
        if f not in C_PATIENT:
            raise GovernanceError(f"{f!r} is not a declared context field")
        cf = C_PATIENT[f]
        frac = float(variance_explained.get(f, 0.0))
        if not 0.0 <= frac <= 1.0:
            raise GovernanceError(f"variance explained by {f!r} must be a fraction")
        out.append({"field": f, "stage": cf.modulates_stage,
                    "routinely_measured": cf.routinely_measured,
                    "variance_explained": frac,
                    "posterior_sd_after": round(posterior_sd * math.sqrt(1 - frac), 5),
                    "deployable_as_companion_dx": cf.routinely_measured})
    out.sort(key=lambda r: -r["variance_explained"])
    best_deployable = next((r for r in out if r["deployable_as_companion_dx"]), None)
    return {"ranked": out,
            "best_overall": out[0]["field"] if out else None,
            "best_deployable": best_deployable["field"] if best_deployable else None,
            "note": ("the best measurement and the best DEPLOYABLE measurement differ "
                     "when the most informative field needs a research assay")}


def reverse_translate(observation: str, n_patients: int, n_observations: int,
                      confounders_assessed: Sequence[str],
                      provenance_cleared: bool,
                      minimum_confounders: int = 3) -> dict:
    """Step 6: an unexpected observation becomes a RESEARCH HYPOTHESIS.

    There is no code path from here to a model version, and that absence is the
    design. The brief: governed aggregated evidence "does not automatically change
    a deployed clinical model".

    The confounder requirement is not bureaucratic. The brief names four candidate
    explanations for an unexpected trajectory -- differential exposure, disease
    biology, assay variation, adherence -- and promoting one before the others are
    assessed is how a laboratory artefact becomes a mechanism.
    """
    if not provenance_cleared:
        raise GovernanceError(
            "provenance, privacy and confounding assessment has not cleared; no "
            "clinical observation may enter the research loop before it does")
    eff = effective_n(n_patients, n_observations)
    if len(confounders_assessed) < minimum_confounders:
        raise GovernanceError(
            f"{len(confounders_assessed)} confounder(s) assessed; at least "
            f"{minimum_confounders} are required. The candidate explanations are "
            f"differential exposure, disease biology, assay variation and adherence; "
            f"promoting one before the others are assessed is how an artefact "
            f"becomes a mechanism.")
    return {"step": int(Step.REVERSE_TRANSLATION),
            "observation": observation,
            "effective_n": eff["effective_n"],
            "design_effect": eff["design_effect"],
            "confounders_assessed": list(confounders_assessed),
            "outcome": "research hypothesis",
            "motivates": "a new experiment in Loop A, entering at Phase 5",
            "does_not": ("update a deployed model, change a population parameter, or "
                         "constitute evidence of a causal mechanism"),
            "taint": Taint.DERIVED.name}


def loop_b_table() -> list[dict]:
    desc = {
        Step.PATIENT_CONTEXT: "assemble c_patient; check provenance and availability",
        Step.STATE_ESTIMATION: "posterior over this patient's delivery state",
        Step.REGIMEN_SIMULATION: "rank candidate regimens under that posterior",
        Step.CREDIBILITY_GATE: "ICH M15: what decision may this support?",
        Step.COMPANION_DIAGNOSTIC: "which measurement most reduces the uncertainty",
        Step.REVERSE_TRANSLATION: "unexpected observation -> research hypothesis",
    }
    return [{"step": int(s), "name": s.name, "does": desc[s]} for s in Step]
