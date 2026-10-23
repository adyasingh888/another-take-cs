"""
DeltaPBPK: predict differences, not absolute values.

THE IDEA, AND WHY IT IS THE RIGHT ONE
-------------------------------------
The brief's §12 states it: train on WITHIN-STUDY CONTRASTS

    dy_{s,i} = y_{s,i} - y_{s,ref}      with    du_{s,i} = u_{s,i} - u_{s,ref}

rather than on absolute values. The argument is one line of algebra and it is
worth writing out, because it is the only place in this programme where a
modelling choice provably removes the dominant error term rather than fitting
around it.

Suppose every observation decomposes as

    y_{s,i} = f(x_i) + alpha_s + e_{s,i}

with alpha_s the study offset. Differencing two arms of the SAME study gives

    dy_{s,i} = f(x_i) - f(x_ref) + (e_{s,i} - e_{s,ref})

and alpha_s is GONE -- exactly, for every study, with no estimate of it required.
That is why the contrast target works where absolute prediction does not: it
cancels the term that holds 74.8%-88.2% of the variance, instead of trying to
predict it.

THE PRICE, WHICH MUST BE STATED
-------------------------------
Three costs, all real:

  1. Differencing DOUBLES the residual variance: var(e_i - e_ref) = 2*sigma_e^2
     under independence. The contrast has a better signal-to-offset ratio and a
     worse signal-to-noise ratio, and whether that trade is favourable depends on
     tau0^2 / sigma_e^2 -- which is why `advantage()` computes it rather than
     assuming it.
  2. It requires multi-arm studies. A single-formulation paper contributes
     NOTHING, and the brief's own figure is 120 studies with >= 3 carriers each.
  3. It predicts a difference. Recovering an absolute value needs the reference
     arm's absolute value, so the anchor is still required -- the contrast target
     and reference anchoring are complements, not alternatives.

WHAT IS NOT CLAIMED
-------------------
The algebra and the variance accounting are exact and tested. The brief's reported
DeltaPBPK performance is not reproduced here; that needs the substrate.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np


class ContrastError(ValueError):
    """A contrast was requested where none exists."""


@dataclass(frozen=True)
class Contrast:
    """One within-study pair."""

    study: str
    arm: str
    reference_arm: str
    dy: float
    du: dict[str, float]


def build_contrasts(study: Sequence[str], arm: Sequence[str], y: Sequence[float],
                    u: Mapping[str, Sequence[float]],
                    reference: Mapping[str, str] | None = None,
                    all_pairs: bool = False) -> list[Contrast]:
    """Form within-study contrasts.

    `reference` names each study's reference arm. Where it is absent and
    `all_pairs` is False the study is SKIPPED rather than given an arbitrary
    reference: picking the first arm by row order would make the target depend on
    spreadsheet ordering, which is not a modelling choice anyone would defend.

    `all_pairs=True` forms every ordered pair within a study. That extracts more
    contrasts from the same data but makes them DEPENDENT -- each arm appears in
    several -- so any inference over them must be clustered by study, and the
    returned objects carry the study label so that is possible.
    """
    s = np.asarray(study); a = np.asarray(arm); yy = np.asarray(y, float)
    if not (s.size == a.size == yy.size):
        raise ContrastError("study, arm and y must align")
    for k, v in u.items():
        if len(v) != yy.size:
            raise ContrastError(f"descriptor {k!r} has {len(v)} rows, expected {yy.size}")
    out: list[Contrast] = []
    for st in dict.fromkeys(s.tolist()):
        idx = np.where(s == st)[0]
        if idx.size < 2:
            continue                       # a single-arm study yields no contrast
        if all_pairs:
            pairs = [(i, j) for i, j in combinations(idx, 2)]
        else:
            ref = (reference or {}).get(st)
            if ref is None or ref not in a[idx]:
                continue
            r = idx[np.where(a[idx] == ref)[0][0]]
            pairs = [(i, r) for i in idx if i != r]
        for i, r in pairs:
            out.append(Contrast(
                study=str(st), arm=str(a[i]), reference_arm=str(a[r]),
                dy=float(yy[i] - yy[r]),
                du={k: float(np.asarray(v, float)[i] - np.asarray(v, float)[r])
                    for k, v in u.items()}))
    if not out:
        raise ContrastError(
            "no within-study contrasts could be formed: every study has one arm, or "
            "no reference arm was named. Single-formulation papers contribute nothing "
            "to a contrast target -- that is the method's standing limitation, not a "
            "configuration problem")
    return out


def offset_cancellation(study: Sequence[str], y_absolute: Sequence[float],
                        offsets: Mapping[str, float]) -> dict:
    """Demonstrate, numerically, that differencing removes the study offset.

    Takes data whose offsets are KNOWN -- a simulation -- and shows the offset
    variance present in the absolute target is absent from the contrast target.
    This is the method's whole justification, so it is computed rather than cited.
    """
    s = np.asarray(study); y = np.asarray(y_absolute, float)
    off_col = np.array([offsets[str(x)] for x in s])
    var_abs = float(np.var(y, ddof=1))
    centred = y - np.array([y[s == x].mean() for x in s])
    return {
        "var_absolute_target": round(var_abs, 6),
        "var_offset_component": round(float(np.var(off_col, ddof=1)), 6),
        "offset_share_of_absolute": round(float(np.var(off_col, ddof=1)) / var_abs, 4)
                                    if var_abs > 0 else None,
        "var_within_study_contrast": round(float(np.var(centred, ddof=1)), 6),
        "offset_correlation_with_contrast": round(
            float(np.corrcoef(off_col, centred)[0, 1]), 6)
            if np.std(centred) > 0 else 0.0,
        "note": ("the contrast target is uncorrelated with the study offset by "
                 "construction: differencing within a study removes alpha_s exactly, "
                 "for every study, with no estimate of it required"),
    }


def advantage(tau0_sq: float, sigma_e_sq: float) -> dict:
    """Is the contrast trade favourable? Computed, not assumed.

    Absolute target  : signal competes with tau0^2 + sigma_e^2
    Contrast target  : offset gone, but residual variance is 2*sigma_e^2

    So contrasting helps when tau0^2 + sigma_e^2 > 2*sigma_e^2, i.e. when
    tau0^2 > sigma_e^2 -- when the study offset exceeds measurement noise. At the
    brief's reported 74.8%-88.2% offset share that is comfortably satisfied, but it
    is a condition rather than a law and a quieter literature could fail it.
    """
    if tau0_sq < 0 or sigma_e_sq < 0:
        raise ContrastError("variance components cannot be negative")
    abs_noise = tau0_sq + sigma_e_sq
    con_noise = 2.0 * sigma_e_sq
    share = tau0_sq / abs_noise if abs_noise > 0 else 0.0
    return {
        "offset_share": round(share, 4),
        "absolute_target_noise": round(abs_noise, 6),
        "contrast_target_noise": round(con_noise, 6),
        "variance_ratio": round(con_noise / abs_noise, 4) if abs_noise > 0 else None,
        "favourable": tau0_sq > sigma_e_sq,
        "break_even_offset_share": 0.5,
        "verdict": ("contrasting reduces the noise the signal competes against"
                    if tau0_sq > sigma_e_sq else
                    "contrasting INCREASES the noise: the doubled measurement variance "
                    "exceeds the offset variance removed, so the absolute target is "
                    "preferable here"),
    }
