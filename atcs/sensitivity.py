"""
Variance-based sensitivity, and the one product that makes it actionable.

TWO REASONS THIS MODULE EXISTS
------------------------------
1. ICH M15 lists sensitivity analysis as required credibility evidence from the
   moderate risk band upward, and `atcs.icm15.ACTIVITIES` names it. Naming an
   activity a package cannot perform is the defect a test already caught once here.

2. More usefully: sensitivity multiplied by identifiability is a MEASUREMENT
   PRIORITY. A parameter the output barely depends on is not worth measuring even
   if it is unidentifiable; a parameter the output depends on strongly and which is
   already identifiable needs no new assay. The intersection -- high total effect
   AND unidentifiable -- is a short list, and it is the list a research plan wants.

   `measurement_priority()` computes that intersection. It is the only function in
   this package that joins two different analyses into a decision, and the join is
   the point.

WHY SOBOL AND NOT ONE-AT-A-TIME
-------------------------------
Because the chain is multiplicative. Local one-at-a-time perturbation reports the
derivative at a point, and for a product of uncertain factors the derivative at the
mean is a poor guide to the variance contribution -- a factor near zero suppresses
everything downstream of it regardless of its own local slope. Variance-based
indices integrate over the input distribution and so answer the question actually
being asked: how much of the output variance would be removed by learning this
input exactly?

First-order and TOTAL indices are both returned, because their difference is the
interaction. In a multiplicative chain that difference is large, and a report of
first-order indices alone will understate every parameter.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np


class SensitivityError(ValueError):
    """A sensitivity analysis was requested that cannot be interpreted."""


@dataclass(frozen=True)
class Parameter:
    """One uncertain input, with the range the analysis integrates over."""

    name: str
    low: float
    high: float
    basis: str = ""

    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise SensitivityError(
                f"{self.name}: range [{self.low}, {self.high}] is empty. A parameter "
                f"with no range contributes no variance and should be a constant, "
                f"which is a different claim and should be made explicitly.")


def sobol_indices(fn: Callable[[Mapping[str, float]], float],
                  parameters: Sequence[Parameter], *, n_base: int = 2048,
                  seed: int = 0) -> dict:
    """Saltelli estimator of first-order and total-effect indices.

    Cost is n_base * (k + 2) model evaluations. The estimator is the standard one:

        S_i   = E[ f(B) * (f(A_B^i) - f(A)) ] / Var(f)
        S_Ti  = E[ (f(A) - f(A_B^i))^2 ] / (2 Var(f))

    Negative first-order estimates are reported as computed rather than clipped to
    zero. A negative S_i means the estimator's noise exceeds that parameter's real
    contribution, which is information about the sample size -- clipping it to zero
    hides exactly that and makes an underpowered analysis look clean.
    """
    k = len(parameters)
    if k < 2:
        raise SensitivityError(
            "variance decomposition needs at least two parameters; with one, the "
            "first-order index is 1 by construction")
    if n_base < 64:
        raise SensitivityError("n_base below 64 gives indices dominated by noise")
    rng = np.random.default_rng(seed)
    lo = np.array([p.low for p in parameters])
    hi = np.array([p.high for p in parameters])
    names = [p.name for p in parameters]

    A = lo + (hi - lo) * rng.random((n_base, k))
    B = lo + (hi - lo) * rng.random((n_base, k))

    def ev(M: np.ndarray) -> np.ndarray:
        return np.array([fn(dict(zip(names, row))) for row in M], float)

    fA, fB = ev(A), ev(B)
    var = float(np.var(np.concatenate([fA, fB]), ddof=1))
    if var <= 0:
        raise SensitivityError(
            "the output has zero variance over the parameter ranges, so no "
            "decomposition exists. Either the function ignores its inputs or the "
            "ranges are too narrow to matter.")

    rows = []
    for i, p in enumerate(parameters):
        AB = A.copy(); AB[:, i] = B[:, i]
        fAB = ev(AB)
        s1 = float(np.mean(fB * (fAB - fA)) / var)
        st = float(np.mean((fA - fAB) ** 2) / (2 * var))
        rows.append({"parameter": p.name, "first_order": round(s1, 4),
                     "total_effect": round(st, 4),
                     "interaction": round(st - s1, 4),
                     "range": [p.low, p.high], "basis": p.basis})
    rows.sort(key=lambda r: -r["total_effect"])
    s1_sum = sum(r["first_order"] for r in rows)
    st_sum = sum(r["total_effect"] for r in rows)
    return {"indices": rows, "output_variance": round(var, 6),
            "n_evaluations": n_base * (k + 2),
            "sum_first_order": round(s1_sum, 4),
            "sum_total_effect": round(st_sum, 4),
            "interaction_share": round(max(st_sum - s1_sum, 0.0), 4),
            "note": ("first-order indices sum to less than one when inputs interact; "
                     "the gap to the total-effect sum is the interaction, which in a "
                     "multiplicative chain is large -- so a report of first-order "
                     "indices alone understates every parameter. Negative estimates "
                     "are shown as computed, not clipped: a negative index means the "
                     "estimator's noise exceeds that parameter's contribution, which "
                     "is information about n_base rather than about the parameter.")}


def measurement_priority(sens: Mapping[str, object], model,
                         *, candidates: Sequence = (),
                         total_effect_floor: float = 0.05) -> dict:
    """Sensitivity x identifiability: the short list a research plan wants.

    Four quadrants, and only one of them is a measurement priority:

      high effect, unidentifiable  -> MEASURE THIS. The output depends on it and
                                      nothing in hand determines it.
      high effect, identifiable    -> already determined; no new assay needed
      low effect, unidentifiable   -> not worth measuring however open it is
      low effect, identifiable     -> ignore

    The quadrant that is usually acted on in practice is the second, because a
    parameter the model is sensitive to feels important -- but if it is already
    identifiable, measuring it again buys nothing.
    """
    rows = []
    for ent in sens["indices"]:                    # type: ignore[index]
        name = ent["parameter"]
        if name in model.unknowns:
            ident, why = model.identifiable(name)
            closers = model.minimal_additional_observations(name, candidates) \
                if (not ident and candidates) else []
        else:
            ident, why, closers = True, "not a free parameter of this model", []
        high = ent["total_effect"] >= total_effect_floor
        quadrant = ("MEASURE" if high and not ident else
                    "already determined" if high and ident else
                    "not worth measuring" if not high and not ident else "ignore")
        rows.append({"parameter": name, "total_effect": ent["total_effect"],
                     "identifiable": ident, "quadrant": quadrant,
                     "closed_by": [list(c) for c in closers],
                     "why_unidentifiable": "" if ident else why})
    priority = [r for r in rows if r["quadrant"] == "MEASURE"]
    priority.sort(key=lambda r: -r["total_effect"])
    return {"rows": rows, "priority": [r["parameter"] for r in priority],
            "n_priority": len(priority),
            "verdict": (f"{len(priority)} parameter(s) carry total effect >= "
                        f"{total_effect_floor} AND are unidentifiable: "
                        f"{', '.join(r['parameter'] for r in priority)}. These are "
                        f"the measurement priorities; a sensitive parameter that is "
                        f"already identifiable needs no new assay, which is the "
                        f"quadrant most often acted on by mistake."
                        if priority else
                        f"no parameter is both sensitive (>= {total_effect_floor}) "
                        f"and unidentifiable. Either the open parameters do not "
                        f"matter to this output, or the ranges understate them.")}
