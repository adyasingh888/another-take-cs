"""
BayesCalib and ConformalCalib: the two gates that make a cross-lab prediction
usable, written as the brief states them.

THE PROBLEM, QUANTIFIED
-----------------------
The brief's §6.1 decomposes biodistribution variance and finds 74.8%-88.2% of it
is inter-laboratory protocol offset. §6.3 draws the consequence: a zero-shot 90%
conformal interval on log10(%ID) must span about +/-1.96*sqrt(tau0^2 + sigma_e^2),
which is 0.88 to 1.10 log10 units -- a 56.8x to 158.1x multiplicative fold-width.
Honest, and useless for ranking formulations.

THE TWO FIXES, AND WHY THEY ARE DIFFERENT
-----------------------------------------
`bayes_shrink` estimates the target laboratory's offset from k reference anchor
measurements, by the brief's own estimator

    b_hat = [k*tau0^2 / (k*tau0^2 + sigma_e^2)] * [ybar_anchor - fbar_theta(x_anchor)]

This is the Empirical Bayes posterior mean of a random study effect given k noisy
observations, and the shrinkage factor is exactly the signal-to-total ratio. The
two limits are the useful intuition: with k -> infinity the factor goes to 1 and
the raw anchor residual is trusted completely; with tau0 -> 0 there is no study
effect to estimate and the factor goes to 0, correctly refusing to attribute
anything to the laboratory.

`conformal_interval` is a SEPARATE guarantee and the distinction matters. Shrinkage
changes the POINT prediction; conformal calibration bounds the ERROR. A conformal
interval has finite-sample coverage under exchangeability alone, with no
distributional assumption -- but exchangeability must hold at the right unit, and
for clustered data that unit is the STUDY, not the row. Row-level conformal
calibration on study-clustered data produces intervals that are too narrow by
roughly the square root of the design effect, which is the same mistake as a
row-level train/test split and is made for the same reason.

WHAT IS NOT CLAIMED
-------------------
These are the estimators, implemented and tested against their own limiting
behaviour and against a simulation in which the ground-truth offset is known. They
are not a reproduction of the brief's reported R-squared values; that requires the
substrate, and this module is the machinery rather than the result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


class CalibrationError(ValueError):
    """A calibration was requested under conditions that void its guarantee."""


# --------------------------------------------------------------- BayesCalib
@dataclass(frozen=True)
class Shrinkage:
    """The estimated laboratory offset and the factor that produced it."""

    b_hat: float
    factor: float
    k: int
    raw_residual: float
    tau0_sq: float
    sigma_e_sq: float

    @property
    def interpretation(self) -> str:
        if self.factor < 0.25:
            return ("most of the anchor residual is attributed to measurement noise "
                    "rather than to a laboratory offset")
        if self.factor > 0.85:
            return ("the anchor residual is attributed almost entirely to a "
                    "laboratory offset")
        return "the anchor residual is split between offset and noise"


def bayes_shrink(anchor_observed: Sequence[float],
                 anchor_predicted: Sequence[float],
                 tau0_sq: float, sigma_e_sq: float) -> Shrinkage:
    """Empirical Bayes estimate of a study's offset from k anchor measurements.

        b_hat = [k*tau0^2 / (k*tau0^2 + sigma_e^2)] * mean(y_anchor - f(x_anchor))

    Both variances are required. Defaulting either is how a shrinkage factor
    becomes a tuning knob: with sigma_e^2 set small enough the factor goes to 1 and
    the estimator stops shrinking at all, which is indistinguishable from not
    having an estimator.
    """
    y = np.asarray(anchor_observed, float)
    f = np.asarray(anchor_predicted, float)
    if y.shape != f.shape:
        raise CalibrationError(
            f"{y.size} observed anchors against {f.size} predictions: an anchor "
            f"residual requires a prediction for the same measurement")
    k = int(y.size)
    if k == 0:
        raise CalibrationError(
            "zero anchors: with no reference measurement in the target laboratory "
            "the offset is not identifiable and no shrinkage is defined")
    if tau0_sq < 0 or sigma_e_sq < 0:
        raise CalibrationError("variance components cannot be negative")
    if tau0_sq == 0 and sigma_e_sq == 0:
        raise CalibrationError("both variance components zero: the estimator is 0/0")
    resid = float(np.mean(y - f))
    factor = (k * tau0_sq) / (k * tau0_sq + sigma_e_sq)
    return Shrinkage(b_hat=factor * resid, factor=factor, k=k, raw_residual=resid,
                     tau0_sq=tau0_sq, sigma_e_sq=sigma_e_sq)


def zero_shot_fold_width(tau0_sq: float, sigma_e_sq: float, level: float = 0.90
                         ) -> dict:
    """The width a zero-shot interval MUST have, from the variance components.

    Reproduces the brief's §6.3 argument as a computation rather than a quoted
    number: the interval spans +/- z*sqrt(tau0^2 + sigma_e^2) on log10, and the
    multiplicative fold-width is 10^(2*half-width). It is reported here because an
    interval this wide is the honest starting point, and any narrower interval that
    has not anchored the laboratory is narrower than the data permits.
    """
    from scipy.stats import norm
    z = float(norm.ppf(0.5 + level / 2))
    half = z * float(np.sqrt(tau0_sq + sigma_e_sq))
    return {"level": level, "z": round(z, 4),
            "half_width_log10": round(half, 4),
            "fold_width": round(float(10 ** (2 * half)), 1),
            "note": ("the width implied by the variance components alone; narrower "
                     "than this without anchoring the laboratory is narrower than "
                     "the data permits")}


def anchored_fold_width(tau0_sq: float, sigma_e_sq: float, k: int,
                        level: float = 0.90) -> dict:
    """Width after k anchors, from the POSTERIOR variance of the offset.

        var(b | k anchors) = tau0^2 * sigma_e^2 / (k*tau0^2 + sigma_e^2)

    so the predictive variance becomes var(b|k) + sigma_e^2 instead of
    tau0^2 + sigma_e^2. The collapse is therefore a consequence of the estimator,
    not a separate empirical finding -- which is worth knowing before attributing
    it to a model.
    """
    from scipy.stats import norm
    if k < 1:
        raise CalibrationError("k must be at least 1 to anchor a laboratory")
    post = tau0_sq * sigma_e_sq / (k * tau0_sq + sigma_e_sq)
    z = float(norm.ppf(0.5 + level / 2))
    half = z * float(np.sqrt(post + sigma_e_sq))
    zs = zero_shot_fold_width(tau0_sq, sigma_e_sq, level)
    fold = float(10 ** (2 * half))
    return {"k": k, "posterior_var_offset": round(post, 6),
            "half_width_log10": round(half, 4), "fold_width": round(fold, 1),
            "zero_shot_fold_width": zs["fold_width"],
            "narrowing_factor": round(zs["fold_width"] / fold, 2)}


# ------------------------------------------------------------ ConformalCalib
def conformal_interval(residuals: Sequence[float], groups: Sequence[str] | None,
                       level: float = 0.90) -> dict:
    """Split-conformal half-width at the given level.

    `groups` is REQUIRED and may be None only to assert explicitly that the
    calibration residuals are exchangeable at the row level. Passing study labels
    switches the quantile to a study-level one: the calibration set becomes one
    value per study, because rows within a study are not exchangeable with rows
    across studies and using them as though they were is the conformal analogue of
    a row-level train/test split.
    """
    r = np.abs(np.asarray(residuals, float))
    if r.size == 0:
        raise CalibrationError("no calibration residuals")
    if groups is None:
        n = r.size
        q_idx = min(n - 1, int(np.ceil((n + 1) * level)) - 1)
        half = float(np.sort(r)[q_idx])
        unit, n_eff = "row", n
    else:
        g = np.asarray(groups)
        if g.size != r.size:
            raise CalibrationError("groups and residuals must align")
        per = np.array([r[g == u].max() for u in np.unique(g)])
        n = per.size
        if n < 10:
            raise CalibrationError(
                f"{n} calibration STUDIES is too few for a {level:.0%} conformal "
                f"quantile: the finite-sample guarantee needs ceil((n+1)*level) <= n, "
                f"i.e. at least {int(np.ceil(1 / (1 - level)))} studies")
        q_idx = min(n - 1, int(np.ceil((n + 1) * level)) - 1)
        half = float(np.sort(per)[q_idx])
        unit, n_eff = "study", n
    return {"half_width": round(half, 5), "level": level,
            "exchangeability_unit": unit, "n_calibration": n_eff,
            "fold_width": round(float(10 ** (2 * half)), 2),
            "note": ("row-level calibration on study-clustered residuals is too "
                     "narrow by roughly the square root of the design effect"
                     if unit == "row" else
                     "calibrated at the study level, which is the exchangeable unit")}


def coverage(y: Sequence[float], lo: Sequence[float], hi: Sequence[float]) -> dict:
    """Empirical coverage, reported with the target so the gap is visible."""
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    if not (y.shape == lo.shape == hi.shape):
        raise CalibrationError("y, lo and hi must align")
    inside = (y >= lo) & (y <= hi)
    return {"n": int(y.size), "empirical_coverage": round(float(inside.mean()), 4),
            "median_fold_width": round(float(np.median(10 ** (hi - lo))), 2)}
