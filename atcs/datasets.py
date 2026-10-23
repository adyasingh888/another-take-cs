"""
The data layer: reported figures, generated exercise data, and the wall between them.

TWO KINDS OF NUMBER, NEVER MIXED
--------------------------------
This module holds two things that must not be confused, so they are separated by
type rather than by convention:

  REPORTED   figures transcribed from the 20 October brief. They are the project's
             own measurements. This repository has NOT reproduced them -- it has no
             access to the substrate -- so they carry Taint.MEASURED with an origin
             naming the brief, and `reported()` returns them with that provenance
             attached. Transcribed, not recomputed, and the distinction is in the
             origin string of every row.

  GENERATED  tables this module synthesises with fixed seeds so the pipeline can be
             exercised end to end. They carry Taint.SIMULATED and `is_synthetic`
             is True in every row.

THE PROHIBITION, ENFORCED
-------------------------
The brief's §17 gate is explicit: "Do not create synthetic 'ground truth' to fill
the gap." Generated tables here exist to exercise code, never to stand in for
measurement, and that is enforced rather than promised:

  - every generated frame carries an `is_synthetic` column and a SIMULATED taint;
  - `as_measurement()` raises on any frame containing a synthetic row;
  - the generators reproduce the STRUCTURE the brief reports (group sizes, variance
    split, the pKa optimum) without reproducing its VALUES, so a reader cannot
    mistake a generated row for a replication.

Why generate at all: because a pipeline that has never run on a table with 86
study groups and a 74.8% between-study variance share has not been tested on the
shape of the problem. Structure is the thing worth reproducing; values are not.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .taint import Taint


class DataProvenanceError(RuntimeError):
    """Generated data was used where measured data is required."""


BRIEF = ("MesoReach supervisor briefing, 20 October 2026 (project-reported; "
         "transcribed, not reproduced by this repository)")


# ------------------------------------------------------- REPORTED figures
#: §6.1 variance decomposition. Between-study share of total variance per organ.
VARIANCE_DECOMPOSITION = {
    "liver":  {"sigma2_study": 0.748, "sigma2_chem": 0.252},
    "spleen": {"sigma2_study": 0.793, "sigma2_chem": 0.207},
    "lung":   {"sigma2_study": 0.841, "sigma2_chem": 0.159},
    "kidney": {"sigma2_study": 0.816, "sigma2_chem": 0.184},
    "heart":  {"sigma2_study": 0.882, "sigma2_chem": 0.118},
    "tumour": {"sigma2_study": 0.774, "sigma2_chem": 0.226},
}

#: §6.2 the Edition A feature-inflation ablation: 67 features against 22.
EDITION_A_ABLATION = {
    "liver":  {"inflated_67": 0.271, "pruned_22": 0.312},
    "spleen": {"inflated_67": 0.148, "pruned_22": 0.194},
    "kidney": {"inflated_67": 0.238, "pruned_22": 0.281},
    "lung":   {"inflated_67": 0.092, "pruned_22": 0.149},
    "heart":  {"inflated_67": 0.064, "pruned_22": 0.118},
    "tumour": {"inflated_67": 0.219, "pruned_22": 0.264},
}


def reported_variance() -> pd.DataFrame:
    """The variance split, with the sign test recomputed from the ablation table.

    The shares are transcribed. The sign test is the one thing this repository
    COMPUTES from them, because it is arithmetic over the transcribed values rather
    than a claim about the substrate -- and it is worth checking that the reported
    p = 0.0312 is the exact binomial figure it is said to be.
    """
    rows = []
    for organ, v in VARIANCE_DECOMPOSITION.items():
        s = v["sigma2_study"]
        rows.append({
            "organ": organ,
            "between_study_share": s,
            "within_study_share": v["sigma2_chem"],
            "sums_to_one": abs(s + v["sigma2_chem"] - 1.0) < 1e-9,
            "max_r2_from_formulation_alone": round(1.0 - s, 4),
            "taint": Taint.MEASURED.name,
            "origin": BRIEF,
            "is_synthetic": False,
        })
    return pd.DataFrame(rows)


def reported_ablation() -> pd.DataFrame:
    """The 67-vs-22 feature ablation, with the sign test computed."""
    from scipy.stats import binomtest
    rows = []
    for organ, v in EDITION_A_ABLATION.items():
        d = v["pruned_22"] - v["inflated_67"]
        rows.append({"organ": organ, "r2_inflated_67_features": v["inflated_67"],
                     "r2_pruned_22_features": v["pruned_22"],
                     "delta_r2": round(d, 4), "pruning_improved": d > 0,
                     "taint": Taint.MEASURED.name, "origin": BRIEF,
                     "is_synthetic": False})
    df = pd.DataFrame(rows)
    k = int(df.pruning_improved.sum())
    p = binomtest(k, len(df), 0.5, alternative="two-sided").pvalue
    df.attrs["sign_test"] = {
        "n_organs": len(df), "n_improved": k, "exact_binomial_p": round(float(p), 5),
        "brief_reports": 0.0312,
        "matches": abs(float(p) - 0.0312) < 5e-4,
        "note": ("the shares and R2 values are transcribed; THIS p-value is computed "
                 "here from them, because it is arithmetic over the transcribed "
                 "numbers rather than a claim about the substrate"),
    }
    return df


def corpus_registry() -> pd.DataFrame:
    from .multimodal import inventory
    df = pd.DataFrame(inventory())
    df["taint"] = Taint.MEASURED.name
    df["origin"] = BRIEF
    df["is_synthetic"] = False
    return df


# ------------------------------------------------------ GENERATED tables
@dataclass(frozen=True)
class GenerationSpec:
    """What a generated table reproduces -- structure only, never values."""

    name: str
    n_rows: int
    n_groups: int
    reproduces: str
    does_not_reproduce: str
    seed: int


def _stamp(df: pd.DataFrame, spec: GenerationSpec) -> pd.DataFrame:
    df = df.copy()
    df["is_synthetic"] = True
    df["taint"] = Taint.SIMULATED.name
    df["origin"] = (f"atcs.datasets.{spec.name} (generated, seed {spec.seed}): "
                    f"reproduces {spec.reproduces}; does NOT reproduce "
                    f"{spec.does_not_reproduce}")
    return df


def generate_biodistribution(n_studies: int = 86, seed: int = 20261023
                             ) -> pd.DataFrame:
    """A study-clustered organ table with the reported between-study variance share.

    Reproduces the SHAPE of the problem: 86 study groups, several arms each, and a
    per-organ between-study variance share matching the reported decomposition --
    so a grouped-CV pipeline is tested against the clustering it will actually
    meet. Does not reproduce any reported R2, because that would be a fabricated
    replication.
    """
    spec = GenerationSpec("generate_biodistribution", 0, n_studies,
                         "the reported between-study variance share and group structure",
                         "any reported R2 or absolute %ID value", seed)
    rng = np.random.default_rng(seed)
    organs = list(VARIANCE_DECOMPOSITION)
    rows = []
    for s in range(n_studies):
        n_arms = int(rng.integers(1, 8))
        study = f"SYN-S{s:03d}"
        offsets = {o: rng.normal(0, np.sqrt(VARIANCE_DECOMPOSITION[o]["sigma2_study"]))
                   for o in organs}
        for a in range(n_arms):
            size = float(rng.normal(95, 30)); size = max(size, 20.0)
            zeta = float(rng.normal(-12, 14))
            pdi = float(np.clip(rng.beta(2, 8), 0.01, 0.6))
            rec = {"study_id": study, "arm_id": f"{study}-F{a}",
                   "size_nm": round(size, 1), "zeta_mv": round(zeta, 1),
                   "pdi": round(pdi, 3),
                   "dose_mg_per_kg": round(float(rng.choice([0.1, 0.3, 1.0, 3.0])), 2),
                   "timepoint_h": int(rng.choice([1, 4, 24, 72, 168]))}
            for o in organs:
                chem = (-0.004 * (size - 95) + 0.002 * zeta - 1.2 * pdi)
                noise = rng.normal(0, np.sqrt(
                    VARIANCE_DECOMPOSITION[o]["sigma2_chem"]))
                rec[f"log10_pid_{o}"] = round(float(offsets[o] + chem + noise), 4)
            rows.append(rec)
    df = pd.DataFrame(rows)
    return _stamp(df, GenerationSpec(spec.name, len(df), n_studies, spec.reproduces,
                                     spec.does_not_reproduce, seed))


def generate_corona(n_studies: int = 83, seed: int = 20261024) -> pd.DataFrame:
    """Corona compositions on the simplex, with study-level plasma effects."""
    from .corona import SPECIES, round_composition, softmax_composition
    spec = GenerationSpec("generate_corona", 0, n_studies,
                          "simplex-valid compositions with study-level plasma effects",
                          "the reported R2 = 0.71-0.86 for species prediction", seed)
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_studies):
        study = f"SYN-C{s:03d}"
        plasma = {k: rng.normal(0, 0.8) for k in SPECIES}
        for a in range(int(rng.integers(1, 4))):
            size = max(float(rng.normal(95, 30)), 20.0)
            peg = float(np.clip(rng.beta(2, 5), 0, 1))
            logits = {k: plasma[k] + (0.6 if k == "albumin" else 0.0) * peg
                      - (0.9 if k == "ApoE" else 0.0) * peg
                      + rng.normal(0, 0.3) for k in SPECIES}
            # round_composition, NOT round(): naive rounding of a composition
            # breaks the sum-to-one constraint and would write a simplex-violating
            # table to disk. Caught by the assertion below.
            shares = round_composition(softmax_composition(logits), dp=5)
            rows.append({"study_id": study, "arm_id": f"{study}-F{a}",
                         "size_nm": round(size, 1), "peg_density": round(peg, 3),
                         **{f"share_{k}": v for k, v in shares.items()}})
    df = pd.DataFrame(rows)
    share_cols = [c for c in df.columns if c.startswith("share_")]
    assert np.allclose(df[share_cols].sum(axis=1), 1.0, atol=1e-9), (
        "a generated corona table must satisfy the simplex invariant exactly at the "
        "precision it is written, or a consumer checking it will reject the artefact")
    return _stamp(df, GenerationSpec(spec.name, len(df), n_studies, spec.reproduces,
                                     spec.does_not_reproduce, seed))


def generate_escape_panel(n: int = 412, seed: int = 20261025) -> pd.DataFrame:
    """An ionizable-lipid panel spanning the reported pKa optimum."""
    from .cell import IonizableLipid, escape_fraction
    spec = GenerationSpec("generate_escape_panel", n, 1,
                          "the bell-shaped pKa dependence with an optimum near 6.35",
                          "the reported escape-to-toxicity ratio", seed)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        pka = float(np.clip(rng.normal(6.4, 0.9), 4.2, 9.5))
        dlogd = float(np.clip(rng.normal(2.0, 0.8), 0.0, 4.5))
        e = escape_fraction(IonizableLipid(pka, dlogd))
        rows.append({"lipid_id": f"SYN-L{i:04d}", "pka": round(pka, 2),
                     "delta_log_d": round(dlogd, 2),
                     "escape_fraction": round(e["escape_fraction"], 6),
                     "limiting_term": e["limiting_term"],
                     "side_of_optimum": e["side_of_optimum"]})
    return _stamp(pd.DataFrame(rows), spec)


def generate_patient_series(n_patients: int = 29, seed: int = 20261026
                            ) -> pd.DataFrame:
    """A longitudinal series with the reported patient and observation counts.

    29 patients, about 7.4 visits each. Generated specifically so the
    observations-are-not-patients error can be DEMONSTRATED on a table rather than
    only described: any analysis treating these 215 rows as independent will
    overstate its precision by the design effect.
    """
    spec = GenerationSpec("generate_patient_series", 0, n_patients,
                          "29 patients with ~7.4 repeated visits each (215 rows)",
                          "any patisiran efficacy or safety outcome", seed)
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        pid = f"SYN-P{p:02d}"
        sens = rng.normal(0, 0.35)
        base = {"complement_c3": float(rng.normal(120, 22)),
                "crp": float(np.clip(rng.gamma(2, 1.8), 0.1, 40)),
                "hepatic_fibrosis_stage": int(rng.integers(0, 5)),
                "egfr": float(np.clip(rng.normal(78, 20), 15, 130))}
        for v in range(int(rng.integers(5, 11))):
            rows.append({"patient_id": pid, "visit": v,
                         "weeks_since_first_dose": 3 * v,
                         **{k: round(x, 2) for k, x in base.items()},
                         "ttr_knockdown_pct": round(float(np.clip(
                             100 * (0.78 + sens) * (1 - np.exp(-0.35 * (v + 1)))
                             + rng.normal(0, 4), 0, 100)), 2)})
    return _stamp(pd.DataFrame(rows), GenerationSpec(
        spec.name, len(rows), n_patients, spec.reproduces,
        spec.does_not_reproduce, seed))


# --------------------------------------------------------------- the wall
def as_measurement(df: pd.DataFrame, claim: str) -> pd.DataFrame:
    """Assert a frame may support a measurement claim. Raises on synthetic rows.

    The enforcement behind the brief's prohibition. There is no flag to override
    it, because the entire value of the rule is that it cannot be overridden by a
    caller in a hurry.
    """
    if "is_synthetic" not in df.columns:
        raise DataProvenanceError(
            f"frame has no `is_synthetic` column, so its provenance is unknown and "
            f"the claim {claim!r} cannot be supported. Every table in this module "
            f"carries one.")
    n = int(df["is_synthetic"].sum())
    if n:
        raise DataProvenanceError(
            f"claim {claim!r} requires measured data, but {n} of {len(df)} rows are "
            f"GENERATED. Generated tables exist to exercise the pipeline, never to "
            f"stand in for measurement -- creating synthetic ground truth to fill an "
            f"evidence gap is the specific thing the 20 Oct brief forbids.")
    return df


def write_all(outdir: str | Path = "data") -> dict:
    """Materialise every table, reported and generated, with its provenance."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    made: dict[str, dict] = {}
    frames = {
        "reported_variance_decomposition": reported_variance(),
        "reported_edition_a_ablation": reported_ablation(),
        "corpus_registry": corpus_registry(),
        "generated_biodistribution": generate_biodistribution(),
        "generated_corona": generate_corona(),
        "generated_escape_panel": generate_escape_panel(),
        "generated_patient_series": generate_patient_series(),
    }
    for name, df in frames.items():
        p = out / f"{name}.csv"
        df.to_csv(p, index=False)
        made[name] = {"path": str(p), "rows": len(df), "cols": df.shape[1],
                      "synthetic": bool(df["is_synthetic"].any())}
    return made
