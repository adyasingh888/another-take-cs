"""
Stage 1: protein corona identity as a composition.

WHY THE SIMPLEX IS THE WHOLE POINT
----------------------------------
Corona composition is a COMPOSITIONAL quantity: the shares of adsorbed protein
species sum to one, and only ratios between them carry information. That is not a
technicality, it determines the arithmetic:

  - A share cannot be modelled by unconstrained regression, because nothing stops
    the prediction leaving the simplex, and a "share" of 1.3 is not a small error
    but a category error.
  - Differences between shares are not interpretable; LOG-RATIOS are. An increase
    of 0.05 in ApoE means something different at a baseline of 0.02 than at 0.60,
    whereas a log-ratio shift is the same quantity either way.
  - Correlations between shares are spuriously negative by construction, because
    the parts sum to a constant. Reading a negative ApoE-C3 correlation as
    competition is reading the constraint.

So the model outputs a softmax over species logits -- on the simplex by
construction, in the brief's own formulation -- and the biologically meaningful
read-out is the ApoE:C3 LOG-RATIO, which the brief identifies as the hepatocyte-
versus-phagocyte targeting signal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


class CompositionError(ValueError):
    """A compositional quantity was used as though it were unconstrained."""


#: The species the brief's corona stage resolves. ApoE drives LDLR-mediated
#: hepatocyte uptake; C3 and IgG drive complement- and Fc-mediated phagocyte
#: clearance; albumin and clusterin are the dysopsonin fraction.
SPECIES = ("ApoE", "C3", "IgG", "albumin", "clusterin", "other")


def softmax_composition(logits: Mapping[str, float]) -> dict[str, float]:
    """Species logits -> shares on the simplex, by construction.

    A softmax rather than a normalised regression: normalising an unconstrained
    predictor after the fact can still produce negative parts, and the gradient of
    a post-hoc normalisation does not respect the constraint during fitting.
    """
    if not logits:
        raise CompositionError("no species to compose")
    ks = list(logits)
    z = np.asarray([logits[k] for k in ks], float)
    if not np.isfinite(z).all():
        raise CompositionError("non-finite logit")
    e = np.exp(z - z.max())
    return {k: float(v) for k, v in zip(ks, e / e.sum())}


def round_composition(shares: Mapping[str, float], dp: int = 5) -> dict[str, float]:
    """Round a composition so the parts STILL sum to one at the given precision.

    Naive rounding breaks the simplex: six species at 5 dp can drift by 3e-5, which
    is enough to fail the state-group invariant. This is not pedantry -- it was
    caught by an assertion in this repository's own data generator, which would
    otherwise have written a simplex-violating table to disk and shipped it.

    Largest-remainder method: round everything down to the grid, then distribute the
    deficit one unit at a time to the parts with the largest discarded remainder.
    This is the standard apportionment rule, and unlike "adjust the biggest part to
    absorb the error" it does not systematically bias one component.
    """
    if not shares:
        raise CompositionError("no composition to round")
    ks = list(shares)
    scale = 10 ** dp
    exact = np.asarray([float(shares[k]) for k in ks]) * scale
    floor = np.floor(exact)
    deficit = int(round(scale - floor.sum()))
    order = np.argsort(-(exact - floor))
    for i in range(max(deficit, 0)):
        floor[order[i % len(ks)]] += 1
    out = {k: float(v / scale) for k, v in zip(ks, floor)}
    if abs(sum(out.values()) - 1.0) > 1e-12:
        raise CompositionError(
            f"rounding failed to preserve the simplex: sum = {sum(out.values())!r}")
    return out


def log_ratio(shares: Mapping[str, float], numerator: str, denominator: str,
              floor: float = 1e-6) -> float:
    """log10 of a ratio between two shares -- the interpretable contrast.

    `floor` guards a zero denominator and is APPLIED EXPLICITLY rather than
    silently: a species absent from the corona gives an infinite log-ratio, and
    quietly returning a large finite number would let that propagate as a real
    measurement. The floor is a stated detection limit.
    """
    for k in (numerator, denominator):
        if k not in shares:
            raise CompositionError(f"{k!r} is not a species of this composition")
    num = max(float(shares[numerator]), floor)
    den = max(float(shares[denominator]), floor)
    return float(math.log10(num / den))


def apoe_c3_index(shares: Mapping[str, float]) -> dict:
    """The hepatocyte-versus-phagocyte targeting read-out.

    Returned with its interpretation band rather than as a bare number, because the
    sign is what matters operationally: a positive log-ratio favours LDLR-mediated
    hepatocyte uptake, a negative one favours complement-mediated clearance.
    """
    lr = log_ratio(shares, "ApoE", "C3")
    return {"log10_apoe_c3": round(lr, 4),
            "apoe_share": round(float(shares.get("ApoE", 0.0)), 4),
            "c3_share": round(float(shares.get("C3", 0.0)), 4),
            "favours": ("LDLR-mediated hepatocyte uptake" if lr > 0.3 else
                        "complement-mediated phagocyte clearance" if lr < -0.3 else
                        "neither strongly"),
            "note": ("a log-ratio, not a difference: a 0.05 share change means "
                     "different things at baseline 0.02 and 0.60")}


def spurious_correlation_warning(shares_matrix: Sequence[Mapping[str, float]],
                                 a: str, b: str) -> dict:
    """Compositional parts are negatively correlated BY CONSTRUCTION.

    Computes the observed correlation between two shares and compares it against
    the correlation between their log-ratios to a common reference, which is the
    compositionally valid comparison. The gap is the artefact, and it is reported
    because "ApoE and C3 compete for the surface" is exactly the conclusion a raw
    correlation invites and cannot support.
    """
    if len(shares_matrix) < 3:
        raise CompositionError("need at least 3 compositions to correlate")
    A = np.array([float(s[a]) for s in shares_matrix])
    B = np.array([float(s[b]) for s in shares_matrix])
    raw = float(np.corrcoef(A, B)[0, 1]) if A.std() > 0 and B.std() > 0 else float("nan")
    la = np.array([log_ratio(s, a, "other") for s in shares_matrix])
    lb = np.array([log_ratio(s, b, "other") for s in shares_matrix])
    clr = float(np.corrcoef(la, lb)[0, 1]) if la.std() > 0 and lb.std() > 0 else float("nan")
    return {"raw_share_correlation": round(raw, 4),
            "log_ratio_correlation": round(clr, 4),
            "artefact_magnitude": round(abs(raw - clr), 4) if raw == raw and clr == clr else None,
            "note": ("parts of a composition sum to a constant, so they are forced "
                     "toward negative correlation whatever the biology. Only the "
                     "log-ratio correlation supports a claim about competition.")}
