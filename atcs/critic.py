"""
The mechanistic critic: comparing competing explanations rather than confirming one.

THE JOB, AS THE BRIEF DEFINES IT
--------------------------------
§14 lists the critic's task: "Compare uptake-limited, release-limited,
degradation-limited and target-response explanations." Four rival mechanisms for
the same observed shortfall. §17's gate adds the standard: a single mechanism
reported without rivals is not an inference.

So this module computes a POSTERIOR OVER MECHANISMS, and its most useful output is
often that the posterior is flat -- that the available observations do not
distinguish the rivals. A critic that always returns a winner is a confirmation
engine.

WHY MODEL EVIDENCE AND NOT A FIT STATISTIC
------------------------------------------
Each mechanism predicts a different pattern across observations. Comparing them by
goodness-of-fit alone favours whichever has more freedom, so the comparison uses
BIC-approximated model evidence -- fit penalised by parameter count. Approximate,
stated as such, and adequate for the real purpose: identifying when two mechanisms
fit equally well, which is a statement about the DATA rather than about the models.

The discriminating-observation search is the actionable part. Given a flat
posterior, which measurement would separate the rivals most? That turns "we cannot
tell" into "here is the experiment", which is what a critic owes a research plan.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np


class CriticError(ValueError):
    """A comparison was requested that cannot be made."""


@dataclass(frozen=True)
class Mechanism:
    """One candidate explanation for an observed shortfall."""

    name: str
    n_parameters: int
    predict: Callable[[Mapping[str, float]], float]
    signature: str          # the pattern that would distinguish it
    next_measurement: str   # what would test it directly

    def __post_init__(self) -> None:
        if self.n_parameters < 0:
            raise CriticError("negative parameter count")


def uptake_limited(scale: float = 1.0) -> Mechanism:
    """Too little material reaches the organ at all."""
    return Mechanism(
        "uptake-limited", 1,
        lambda c: scale * float(c.get("organ_fraction", 0.0)),
        signature=("response tracks ORGAN exposure and is insensitive to lipid pKa "
                   "or cargo chemistry"),
        next_measurement="organ-level quantification across a dose series")


def release_limited(scale: float = 1.0) -> Mechanism:
    """Material reaches the organ but does not escape the endosome."""
    return Mechanism(
        "release-limited", 2,
        lambda c: scale * float(c.get("organ_fraction", 0.0)) *
                  float(c.get("escape_fraction", 0.0)),
        signature=("response tracks ionizable-lipid pKa with organ exposure held "
                   "constant -- the bell-shaped pKa dependence is the fingerprint"),
        next_measurement="a pKa series at matched organ exposure")


def degradation_limited(scale: float = 1.0, half_life_h: float = 6.0) -> Mechanism:
    """Cargo is degraded before it acts."""
    return Mechanism(
        "degradation-limited", 2,
        lambda c: scale * float(c.get("organ_fraction", 0.0)) *
                  math.exp(-math.log(2) * float(c.get("delay_h", 0.0)) / half_life_h),
        signature=("response decays with the delay between delivery and read-out, "
                   "independently of how much was delivered"),
        next_measurement="a time-course with chemically stabilised cargo as comparator")


def target_response_limited(ceiling: float = 0.3) -> Mechanism:
    """Cargo arrives and acts, but the target does not control the endpoint."""
    return Mechanism(
        "target-response-limited", 1,
        lambda c: ceiling,
        signature=("response is FLAT in every delivery variable -- a ceiling that "
                   "moves with neither exposure nor chemistry"),
        next_measurement=("genetic knockdown of the target without any carrier, which "
                          "removes delivery from the question entirely"))


DEFAULT_MECHANISMS = (uptake_limited(), release_limited(),
                      degradation_limited(), target_response_limited())


def compare(observations: Sequence[Mapping[str, float]],
            observed: Sequence[float],
            mechanisms: Sequence[Mechanism] = DEFAULT_MECHANISMS,
            noise_sd: float | None = None) -> dict:
    """Posterior over mechanisms by BIC-approximated model evidence.

    `noise_sd` is required in spirit: if omitted it is estimated from the best fit,
    which makes the comparison relative rather than absolute. That is adequate for
    ranking rivals and is flagged, because an estimated noise level also absorbs
    model misspecification and will make every mechanism look better than it is.
    """
    y = np.asarray(observed, float)
    if len(observations) != y.size:
        raise CriticError("observations and observed values must align")
    # Argument validity before data adequacy. An earlier version checked the
    # observation count first, which made the single-mechanism refusal unreachable
    # whenever the caller also passed one observation -- so the error named the
    # wrong problem and the stricter rule never fired.
    if len(mechanisms) < 2:
        raise CriticError(
            "a single mechanism is a hypothesis restated, not an inference. Supply "
            "at least two rivals.")
    if y.size < 2:
        raise CriticError(
            "a mechanism comparison needs at least two observations; with one, every "
            "mechanism with a free scale fits exactly")

    rows = []
    for m in mechanisms:
        pred = np.array([m.predict(c) for c in observations], float)
        rss = float(np.sum((y - pred) ** 2))
        rows.append({"mechanism": m.name, "n_parameters": m.n_parameters,
                     "rss": rss, "pred": pred})
    sd = noise_sd
    if sd is None:
        best = min(r["rss"] for r in rows)
        sd = math.sqrt(max(best / y.size, 1e-12))
        sd_basis = "estimated from the best fit (relative comparison only)"
    else:
        if sd <= 0:
            raise CriticError("noise_sd must be positive")
        sd_basis = "supplied"

    n = y.size
    for r in rows:
        ll = -0.5 * r["rss"] / sd ** 2 - n * math.log(sd * math.sqrt(2 * math.pi))
        r["log_likelihood"] = ll
        r["bic"] = -2 * ll + r["n_parameters"] * math.log(n)
    best_bic = min(r["bic"] for r in rows)
    for r in rows:
        r["delta_bic"] = round(r["bic"] - best_bic, 3)
    wts = np.array([math.exp(-0.5 * r["delta_bic"]) for r in rows])
    wts = wts / wts.sum()
    for r, w in zip(rows, wts):
        r["posterior"] = round(float(w), 4)
        r.pop("pred")
        r["log_likelihood"] = round(r["log_likelihood"], 3)
        r["bic"] = round(r["bic"], 3)
        r["rss"] = round(r["rss"], 6)
    rows.sort(key=lambda r: -r["posterior"])

    top, second = rows[0], rows[1]
    decisive = second["delta_bic"] >= 6.0          # conventional "strong" threshold
    return {"ranked": rows, "n_observations": n, "noise_sd": round(sd, 6),
            "noise_basis": sd_basis,
            "top": top["mechanism"], "top_posterior": top["posterior"],
            "delta_bic_to_next": second["delta_bic"],
            "decisive": decisive,
            "verdict": (f"{top['mechanism']} is favoured decisively "
                        f"(delta BIC {second['delta_bic']:.1f} to the next)"
                        if decisive else
                        f"NOT DISTINGUISHABLE: {top['mechanism']} and "
                        f"{second['mechanism']} differ by only "
                        f"{second['delta_bic']:.1f} BIC. These observations do not "
                        f"separate the rivals, which is a statement about the DATA "
                        f"rather than about the mechanisms."),
            "next_measurement": (None if decisive else
                                 _discriminating(mechanisms, top["mechanism"],
                                                 second["mechanism"]))}


def _discriminating(mechanisms: Sequence[Mechanism], a: str, b: str) -> dict:
    ma = next(m for m in mechanisms if m.name == a)
    mb = next(m for m in mechanisms if m.name == b)
    return {"rivals": [a, b],
            "signature_a": ma.signature, "signature_b": mb.signature,
            "measurement": ma.next_measurement,
            "why": (f"{a} and {b} make the same prediction over the observations in "
                    f"hand but differ in their signature; the measurement above "
                    f"varies the quantity on which they disagree")}


def discriminating_design(mechanisms: Sequence[Mechanism],
                          candidate_conditions: Sequence[Mapping[str, float]]
                          ) -> dict:
    """Which experimental condition best separates the rivals?

    Ranks candidate conditions by the SPREAD of mechanism predictions at that
    condition. A condition where all mechanisms agree is uninformative however
    cheap it is, and this is the quantity an experiment-selection agent should be
    optimising -- not the condition where the favoured mechanism looks best.
    """
    if len(mechanisms) < 2:
        raise CriticError("need at least two mechanisms to separate")
    if not candidate_conditions:
        raise CriticError("no candidate conditions")
    rows = []
    for i, c in enumerate(candidate_conditions):
        preds = np.array([m.predict(c) for m in mechanisms], float)
        rows.append({"condition_index": i, "condition": dict(c),
                     "prediction_spread": round(float(preds.max() - preds.min()), 6),
                     "predictions": {m.name: round(float(p), 5)
                                     for m, p in zip(mechanisms, preds)}})
    rows.sort(key=lambda r: -r["prediction_spread"])
    uninformative = [r["condition_index"] for r in rows
                     if r["prediction_spread"] < 1e-9]
    return {"ranked": rows, "best_condition_index": rows[0]["condition_index"],
            "uninformative_conditions": uninformative,
            "note": ("ranked by how much the mechanisms DISAGREE at each condition. A "
                     "condition where they agree is uninformative however cheap, and "
                     "optimising for where the favoured mechanism looks best is how a "
                     "critic becomes a confirmation engine.")}
