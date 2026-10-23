"""
Decision value: what an experiment is worth, in the units of the decision.

WHY THIS IS THE MISSING SCORE
-----------------------------
The brief's §14 lists six scoring criteria, and `atcs.agents` implements one of
them properly (correct refusal). "Decision value" is the one that cannot be scored
without knowing what decision is being made, and it is the one that distinguishes a
useful experiment from an informative one.

The distinction matters. An experiment can reduce uncertainty a great deal and
change no decision -- if the candidate it would rule out was never going to be
chosen, learning that it is bad is worth nothing. Conversely a small reduction in
uncertainty about a close contest can be decisive. Information gain measures the
first; expected value of information measures the second, and only the second is
what a research plan should allocate on.

THE MODEL
---------
A decision over candidates, each with an uncertain value. The decision-maker picks
the candidate with the highest expected value. An experiment reduces the variance of
one or more candidates' values. Its EVPI/EVSI is the expected improvement in the
value of the decision taken AFTER seeing the result, over the decision taken now.

Computed by Monte Carlo rather than closed form, because the closed forms exist only
for a normal-normal single-candidate case and the interesting structure -- three
candidates where two are close -- is exactly what those forms cannot express.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


class ValueError_(ValueError):
    """A decision was specified in a way whose value cannot be computed."""


@dataclass(frozen=True)
class Candidate:
    """One option in the decision, with the uncertainty on its value."""

    name: str
    mean: float
    sd: float

    def __post_init__(self) -> None:
        if self.sd < 0:
            raise ValueError_(f"{self.name}: negative standard deviation")


def decision_now(candidates: Sequence[Candidate]) -> dict:
    """The decision taken on present knowledge, and how close it is.

    The margin is reported because it determines whether any experiment can be worth
    anything: when one candidate dominates by several standard deviations, no
    feasible measurement changes the choice and the correct research action is to
    proceed, not to measure.
    """
    if len(candidates) < 2:
        raise ValueError_("a decision needs at least two candidates")
    order = sorted(candidates, key=lambda c: -c.mean)
    best, second = order[0], order[1]
    pooled = math.sqrt(best.sd ** 2 + second.sd ** 2) or 1e-12
    return {"choice": best.name, "expected_value": round(best.mean, 5),
            "runner_up": second.name,
            "margin": round(best.mean - second.mean, 5),
            "margin_in_sd": round((best.mean - second.mean) / pooled, 3),
            "contest_is_close": (best.mean - second.mean) / pooled < 1.0,
            "note": ("when the margin is several SD, no feasible measurement changes "
                     "the choice and the correct research action is to proceed "
                     "rather than to measure")}


def expected_value_of_information(candidates: Sequence[Candidate],
                                  *, resolves: Mapping[str, float] | None = None,
                                  n_draws: int = 40_000, seed: int = 0) -> dict:
    """EVSI for an experiment that reduces named candidates' SDs by a given fraction.

    `resolves` maps candidate name -> fraction of its variance the experiment would
    remove (1.0 = resolved exactly, 0.0 = learns nothing). Omitting it computes EVPI,
    the value of resolving everything, which is the ceiling on any experiment.

    Returns the value in the SAME UNITS as the candidate values, because a decision
    value in nats or bits cannot be compared against the cost of an experiment, and
    comparing them is the whole purpose.
    """
    if n_draws < 1000:
        raise ValueError_("Monte Carlo EVSI below 1000 draws is dominated by noise")
    rng = np.random.default_rng(seed)
    names = [c.name for c in candidates]
    mu = np.array([c.mean for c in candidates])
    sd = np.array([c.sd for c in candidates])

    frac = np.zeros(len(candidates))
    if resolves is not None:
        unknown = [k for k in resolves if k not in names]
        if unknown:
            raise ValueError_(f"resolves names candidates not in the decision: {unknown}")
        for i, n in enumerate(names):
            f = float(resolves.get(n, 0.0))
            if not 0.0 <= f <= 1.0:
                raise ValueError_(f"{n}: variance reduction must be a fraction")
            frac[i] = f
    else:
        frac[:] = 1.0

    truth = mu + sd * rng.standard_normal((n_draws, len(candidates)))

    # value of the decision taken NOW: always the prior-best candidate
    i_now = int(np.argmax(mu))
    now_draws = truth[:, i_now]
    v_now = float(now_draws.mean())

    # value of the decision taken AFTER the experiment. The signal is the posterior
    # mean given a partially informative observation: mu + frac * (truth - mu),
    # which interpolates between the prior mean (frac 0) and the truth (frac 1).
    signal = mu + frac * (truth - mu)
    i_after = np.argmax(signal, axis=1)
    after_draws = truth[np.arange(n_draws), i_after]
    v_after = float(after_draws.mean())

    evsi = v_after - v_now
    # Monte Carlo standard error of the DIFFERENCE, computed on the paired draws.
    # An EVSI indistinguishable from zero must not be reported as positive: the
    # first version of this module returned worth_running=True on an EVSI of 1e-5
    # for an experiment that fully resolved a candidate which was never going to be
    # chosen -- the correct answer there is zero, and 1e-5 is the estimator's noise.
    evsi_se = float(np.std(after_draws - now_draws, ddof=1) / math.sqrt(n_draws))
    significant = evsi > 2.0 * evsi_se
    # the ceiling: perfect information about everything
    i_perfect = np.argmax(truth, axis=1)
    evpi = float(truth[np.arange(n_draws), i_perfect].mean()) - v_now
    switch = float(np.mean(i_after != i_now))
    return {"candidates": names,
            "value_of_decision_now": round(v_now, 5),
            "value_after_experiment": round(v_after, 5),
            "evsi": round(evsi, 5), "evsi_monte_carlo_se": round(evsi_se, 6),
            "evsi_exceeds_its_own_noise": bool(significant),
            "evpi": round(evpi, 5),
            "fraction_of_evpi_captured": round(evsi / evpi, 4) if evpi > 0 else None,
            "probability_the_decision_changes": round(switch, 4),
            "n_draws": n_draws,
            "note": ("value is in the units of the candidate values, not nats, "
                     "because a decision value must be comparable with the cost of "
                     "the experiment -- which is the point of computing it"),
            "verdict": (
                (f"the experiment is worth {evsi:.4g} +/- {evsi_se:.2g} against an "
                 f"EVPI ceiling of {evpi:.4g} ({evsi / evpi:.0%} captured) and "
                 f"changes the decision with probability {switch:.2f}"
                 if significant else
                 f"EVSI {evsi:.3g} does not exceed its own Monte Carlo noise "
                 f"({evsi_se:.2g}); this experiment is worth NOTHING for this "
                 f"decision, however much uncertainty it removes, because the "
                 f"candidate it informs was not going to be chosen")
                if evpi > 0 else
                "the decision is already determined: perfect information has zero "
                "value, so no experiment can be worth anything here. Proceed rather "
                "than measure.")}


def value_per_cost(candidates: Sequence[Candidate],
                   experiments: Mapping[str, tuple[Mapping[str, float], float]],
                   *, n_draws: int = 40_000, seed: int = 0) -> list[dict]:
    """Rank experiments by EVSI per unit cost.

    `experiments` maps name -> (variance reductions, cost). An experiment with zero
    EVSI is ranked last whatever its cost, for the same reason `experiment.rank`
    refuses to order non-closing candidates: a cheap measurement that changes no
    decision is not cheap.
    """
    rows = []
    for name, (resolves, cost) in experiments.items():
        if cost <= 0:
            raise ValueError_(f"{name}: cost must be positive")
        v = expected_value_of_information(candidates, resolves=resolves,
                                          n_draws=n_draws, seed=seed)
        worth = bool(v["evsi_exceeds_its_own_noise"])
        rows.append({"experiment": name,
                     "evsi": v["evsi"] if worth else 0.0,
                     "evsi_monte_carlo_se": v["evsi_monte_carlo_se"],
                     "cost": cost,
                     "evsi_per_cost": round(v["evsi"] / cost, 8) if worth else 0.0,
                     "p_decision_changes": v["probability_the_decision_changes"],
                     "worth_running": worth,
                     "why_not": "" if worth else
                                "EVSI does not exceed its Monte Carlo noise: the "
                                "candidate this informs was not going to be chosen"})
    rows.sort(key=lambda r: (-int(r["worth_running"]), -r["evsi_per_cost"]))
    return rows
