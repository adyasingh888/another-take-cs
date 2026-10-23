"""
The experiment specification: identifiability, power and cost as one checkable object.

WHAT THIS CLOSES
----------------
Phase 5 and reverse translation both terminate in "a proposed next experiment", and
in every version of this programme so far that output has been a sentence. A
sentence cannot be checked. An experiment specification can, and the three things
worth checking are:

  IDENTIFIABILITY  does this measurement actually close the unknown it targets?
                   Answerable from `atcs.observability` before anything is run.
  POWER            at the achievable sample size, can it detect the effect it is
                   proposed to detect? Answerable from the variance components.
  DECISIVENESS     would its result separate the rival mechanisms, or is it
                   consistent with all of them? Answerable from `atcs.critic`.

An experiment that fails any one of these is not worth running, and the three fail
for different reasons with different fixes. A proposal that has not stated all
three is not a proposal.

WHY POWER IS COMPUTED FROM THE VARIANCE COMPONENTS
--------------------------------------------------
Because in this field the design effect dominates everything. A study-clustered
design with m arms per study and intraclass correlation rho has effective sample
size n / (1 + (m-1)rho), and at the reported rho of 0.75-0.88 that is a reduction
by a factor of three to six. A power calculation that uses the raw row count --
which is the usual practice -- overstates by that factor, and the overstatement is
larger than most effects anyone is looking for.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .observability import Observation, StructuralModel


class SpecificationError(ValueError):
    """An experiment was specified in a way that cannot be assessed."""


@dataclass(frozen=True)
class Resource:
    """What an experiment costs. Units are whatever the lab counts in."""

    animals: int = 0
    assay_runs: int = 0
    weeks: float = 0.0
    requires_new_capability: str = ""

    @property
    def is_feasible_in_house(self) -> bool:
        return not self.requires_new_capability


@dataclass(frozen=True)
class ExperimentSpec:
    """A proposed experiment, stated so it can be assessed before it is run."""

    name: str
    #: The equations this experiment adds. A TUPLE, because a real experiment
    #: usually yields several at once -- a flow-sorting run measures every sorted
    #: fraction in the same sample, not one of them. An earlier version took a
    #: single Observation and could not express that, which made every multi-readout
    #: assay look insufficient.
    measures: tuple[Observation, ...]
    targets: str                     # the unknown it is meant to close
    effect_size: float               # the effect it is powered to detect, on log10
    n_groups: int                    # independent studies/animals/sites
    m_per_group: int                 # repeated measures within each
    rho: float                       # intraclass correlation at the group level
    residual_sd: float               # within-group SD on the same scale
    resource: Resource = field(default_factory=Resource)
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.n_groups < 2:
            raise SpecificationError(
                f"{self.name}: {self.n_groups} group(s). A clustered design needs at "
                f"least two independent groups or there is no between-group contrast "
                f"to test.")
        if self.m_per_group < 1:
            raise SpecificationError("at least one measurement per group")
        if not 0.0 <= self.rho <= 1.0:
            raise SpecificationError("rho must lie in [0, 1]")
        if self.residual_sd <= 0:
            raise SpecificationError("residual SD must be positive")
        if isinstance(self.measures, Observation):
            raise SpecificationError(
                f"{self.name}: `measures` is a TUPLE of observations. A real assay "
                f"usually yields several equations from the same sample; pass "
                f"(obs,) for a single-readout experiment.")
        if not self.measures:
            raise SpecificationError(f"{self.name}: an experiment must measure something")
        if len(self.rationale) < 25:
            raise SpecificationError(
                f"{self.name}: an experiment specification must state why this "
                f"measurement rather than another; an unargued proposal cannot be "
                f"compared against rivals")

    # ------------------------------------------------------------- power
    @property
    def design_effect(self) -> float:
        return 1.0 + (self.m_per_group - 1) * self.rho

    @property
    def n_raw(self) -> int:
        return self.n_groups * self.m_per_group

    @property
    def n_effective(self) -> float:
        return self.n_raw / self.design_effect

    def power(self, alpha: float = 0.05) -> dict:
        """Two-sided power at the EFFECTIVE sample size.

        The naive figure at the raw row count is returned alongside, because the
        gap is the whole point: at the reported clustering it is a factor of three
        to six, which is larger than most effects in this literature.
        """
        from scipy.stats import norm
        z_a = float(norm.ppf(1 - alpha / 2))

        def _pow(n: float) -> float:
            if n <= 1:
                return 0.0
            se = self.residual_sd * math.sqrt(2.0 / n)
            return float(norm.sf(z_a - abs(self.effect_size) / se) +
                         norm.cdf(-z_a - abs(self.effect_size) / se))

        p_eff, p_raw = _pow(self.n_effective), _pow(self.n_raw)
        return {"power": round(p_eff, 4), "power_if_rows_were_independent": round(p_raw, 4),
                "n_raw": self.n_raw, "n_effective": round(self.n_effective, 1),
                "design_effect": round(self.design_effect, 3),
                "overstatement_factor": round(p_raw / p_eff, 2) if p_eff > 0 else None,
                "adequate": p_eff >= 0.80,
                "note": ("power is computed at the effective sample size; the "
                         "independent-rows figure is what a raw row count would give "
                         "and is reported only to show the gap")}

    def groups_needed(self, target_power: float = 0.80, alpha: float = 0.05) -> int:
        """How many independent groups would reach the target power?

        Solved on GROUPS rather than rows, because adding repeated measures within a
        group buys almost nothing once rho is high -- which is the practical lesson
        and the one a row-count calculation hides.
        """
        from scipy.stats import norm
        if not 0 < target_power < 1:
            raise SpecificationError("target power must lie in (0, 1)")
        z_a, z_b = norm.ppf(1 - alpha / 2), norm.ppf(target_power)
        n_eff_needed = 2 * ((z_a + z_b) * self.residual_sd / self.effect_size) ** 2
        return int(math.ceil(n_eff_needed * self.design_effect / self.m_per_group))

    # ------------------------------------------------- identifiability
    def closes(self, model: StructuralModel) -> dict:
        """Does adding this measurement make the target unknown identifiable?"""
        if self.targets not in model.unknowns:
            raise SpecificationError(
                f"{self.targets!r} is not an unknown of {model.name!r}; an experiment "
                f"cannot close a parameter the model does not contain")
        before = model.identifiable(self.targets)[0]
        after_model = StructuralModel(model.unknowns,
                                      model.observations + tuple(self.measures),
                                      model.constraints, model.name)
        after = after_model.identifiable(self.targets)[0]
        return {"target": self.targets, "identifiable_before": before,
                "identifiable_after": after,
                "closes_it": (not before) and after,
                "redundant": before and after,
                "insufficient": (not before) and (not after),
                "verdict": ("closes the target unknown" if (not before) and after else
                            "redundant: the target was already identifiable" if before
                            else "insufficient: the target remains unidentifiable even "
                                 "with this measurement, so further equations are needed")}

    def assess(self, model: StructuralModel, target_power: float = 0.80) -> dict:
        """All three checks, with the binding constraint named.

        The useful output is not the verdict but WHICH check failed: an
        unidentifiable target needs a different measurement, an underpowered design
        needs more groups, and an indecisive experiment needs a different condition.
        Three failures, three different fixes.
        """
        ident = self.closes(model)
        pwr = self.power()
        blockers = []
        if not ident["closes_it"]:
            blockers.append(("identifiability", ident["verdict"]))
        if not pwr["adequate"]:
            blockers.append(("power", f"power {pwr['power']:.2f} at "
                                      f"n_eff {pwr['n_effective']:.0f}; "
                                      f"{self.groups_needed(target_power)} groups needed"))
        if not self.resource.is_feasible_in_house:
            blockers.append(("feasibility",
                             f"requires {self.resource.requires_new_capability}"))
        return {"experiment": self.name, "worth_running": not blockers,
                "identifiability": ident, "power": pwr,
                "resource": {"animals": self.resource.animals,
                             "assay_runs": self.resource.assay_runs,
                             "weeks": self.resource.weeks,
                             "new_capability": self.resource.requires_new_capability},
                "blockers": blockers,
                "binding_constraint": blockers[0][0] if blockers else None}


def rank(specs: Sequence[ExperimentSpec], model: StructuralModel,
         cost_weight: float = 1.0) -> list[dict]:
    """Rank candidate experiments by what they close per unit cost.

    Experiments that do not close their target are ranked last regardless of cost,
    because a cheap measurement that resolves nothing is not a cheap experiment --
    it is a wasted one. Cost is animals plus assay runs, which is crude and is
    labelled as such; the ordering it produces is robust to the exact weighting
    whenever one candidate closes a target and another does not.
    """
    rows = []
    for s in specs:
        a = s.assess(model)
        cost = s.resource.animals + s.resource.assay_runs + cost_weight * s.resource.weeks
        rows.append({"experiment": s.name, "targets": s.targets,
                     "closes": a["identifiability"]["closes_it"],
                     "power": a["power"]["power"],
                     "adequate_power": a["power"]["adequate"],
                     "cost_units": round(cost, 1),
                     "worth_running": a["worth_running"],
                     "binding_constraint": a["binding_constraint"],
                     "value_per_cost": round(
                         (1.0 if a["identifiability"]["closes_it"] else 0.0) *
                         a["power"]["power"] / max(cost, 1.0), 6)})
    rows.sort(key=lambda r: (-int(r["closes"]), -r["value_per_cost"]))
    n_closing = sum(1 for r in rows if r["closes"])
    if n_closing == 0:
        # Refuse to produce an ordering. A ranked list of equally-useless
        # experiments invites picking the top one, and the top one resolves
        # nothing. This was caught by a test that expected an order and got an
        # arbitrary one: with every value_per_cost at zero the sort was stable on
        # input order, which is not a ranking.
        return {"ranked": None, "candidates": rows, "n_closing": 0,
                "headline": (f"NONE of the {len(rows)} candidate experiments closes "
                             f"{rows[0]['targets']!r}. No ordering is returned, because "
                             f"ranking equally-uninformative experiments by cost invites "
                             f"running the cheapest one, which resolves nothing. Either "
                             f"combine candidates into one experiment with several "
                             f"readouts, or accept that this unknown is not reachable "
                             f"with the assays on the list.")}
    return {"ranked": rows, "candidates": rows, "n_closing": n_closing,
            "headline": (f"{n_closing} of {len(rows)} candidates close "
                         f"{rows[0]['targets']!r}; non-closing candidates are listed "
                         f"last regardless of cost, because a cheap measurement that "
                         f"resolves nothing is a wasted experiment rather than a cheap "
                         f"one.")}
