"""
Stages 3 and 4: endosomal escape and the matter-aware virtual cell.

WHAT THIS LAYER IS FOR
----------------------
The brief's §1 states the structural disconnect precisely: virtual cell models
(PerturbODE, scGPT, GEARS) "predict how genetic perturbations alter transcription
ASSUMING THE THERAPEUTIC IS ALREADY INSIDE THE TARGET CELL", while delivery models
predict bulk organ biodistribution "without resolving intracellular
bioavailability". Each half is well developed. The join is empty.

A foundation model of the cell is not what closes that gap, and saying so is the
substantive point of this module. A perturbation foundation model answers: given
that this gene is knocked down by this much, what happens to the transcriptome?
It cannot answer: how much knockdown does 0.3 mg/kg produce in a cirrhotic liver?
The second is a transport question, and no amount of transcriptomic pre-training
contains the answer because the training data never varied the delivery.

So the interface this module provides is deliberately narrow: delivery computes a
CYTOSOLIC CONCENTRATION, the cell model consumes it, and the boundary is typed.
`VirtualCell` is a protocol with one method, so a mechanistic indirect-response
model and a pre-trained perturbation model are interchangeable behind it -- and
`attribute_failure` then does the thing neither half can do alone.

TRANSPORT-LIMITED VERSUS MECHANISM-LIMITED
------------------------------------------
This is the capability that justifies coupling the stages, and the brief names it
as Stage 4's empirical test. A candidate that fails can fail two ways:

    transport-limited   insufficient cargo reached the cytosol; the mechanism
                        would have worked. Fix the carrier.
    mechanism-limited   ample cargo reached the cytosol and the response did not
                        follow. Fix the target or the payload.

These prescribe opposite experiments, and a model of either half alone cannot
distinguish them -- which is why the coupling is worth building even though the
end-to-end chain is not validated. `attribute_failure` returns the attribution
WITH the cytosolic concentration that drives it, and refuses to attribute at all
when the concentration sits where the dose-response is flat, because there the
data cannot distinguish the two.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


class EscapeError(ValueError):
    """Endosomal escape was requested outside the regime the model describes."""


# --------------------------------------------------------------- Stage 3
@dataclass(frozen=True)
class IonizableLipid:
    """The parameters the brief's Stage 3 panel varies."""

    pka: float
    delta_logd: float
    name: str = ""

    def __post_init__(self) -> None:
        if not 2.0 <= self.pka <= 12.0:
            raise EscapeError(f"pKa {self.pka} outside any physiological regime")


def escape_fraction(lipid: IonizableLipid, endosomal_ph: float = 5.5,
                    plasma_ph: float = 7.4, optimum: float = 6.35,
                    width: float = 0.55, ceiling: float = 0.04) -> dict:
    """Productive cytosolic escape as a function of ionizable-lipid pKa.

    The brief reports a bell-shaped hepatocyte optimum around pKa 6.2-6.5, and the
    mechanism is a genuine trade-off between two requirements that pull in opposite
    directions:

      too LOW a pKa  -> the lipid stays neutral even at endosomal pH, so it never
                        forms the hexagonal HII ion pairs that disrupt the membrane
      too HIGH a pKa -> the lipid is already cationic at plasma pH 7.4, which drives
                        opsonisation and non-target uptake before the endosome

    So escape is modelled as the PRODUCT of a protonation term rising with pKa and
    a plasma-neutrality term falling with pKa, which produces the bell without the
    bell being assumed. `ceiling` is the maximum productive fraction: the brief's
    Patisiran figure is ~2.8%, and escape efficiencies in this class are a few
    percent at best -- a model permitting 50% would be describing a different
    phenomenon.

    Returned as a dict rather than a float so the two competing terms are visible;
    a single number hides which side of the optimum a candidate sits on, and that
    is precisely what a formulation chemist needs.
    """
    if not 4.0 <= endosomal_ph <= 7.0:
        raise EscapeError(f"endosomal pH {endosomal_ph} outside 4.0-7.0")
    protonated_endosome = 1.0 / (1.0 + 10.0 ** (endosomal_ph - lipid.pka))
    neutral_in_plasma = 1.0 / (1.0 + 10.0 ** (lipid.pka - plasma_ph))
    # lipophilicity gate: insufficient delta log D cannot destabilise the membrane
    lipophilic = 1.0 / (1.0 + math.exp(-(lipid.delta_logd - 1.5) / 0.6))
    raw = protonated_endosome * neutral_in_plasma * lipophilic
    norm = (1.0 / (1.0 + 10.0 ** (endosomal_ph - optimum))) * \
           (1.0 / (1.0 + 10.0 ** (optimum - plasma_ph)))
    frac = ceiling * raw / norm if norm > 0 else 0.0
    return {"escape_fraction": min(frac, ceiling),
            "protonated_at_endosomal_ph": round(protonated_endosome, 4),
            "neutral_at_plasma_ph": round(neutral_in_plasma, 4),
            "lipophilicity_gate": round(lipophilic, 4),
            "side_of_optimum": ("below" if lipid.pka < optimum - 0.05 else
                                "above" if lipid.pka > optimum + 0.05 else "at"),
            "limiting_term": ("endosomal protonation" if protonated_endosome <
                              min(neutral_in_plasma, lipophilic) else
                              "plasma neutrality" if neutral_in_plasma < lipophilic
                              else "lipophilicity")}


# --------------------------------------------------------------- Stage 4
class VirtualCell(Protocol):
    """The narrow interface between delivery and the cell.

    One method, taking a cytosolic CONCENTRATION. A pre-trained perturbation model
    and a mechanistic indirect-response model are interchangeable behind it, which
    is the point: the coupling does not depend on which cell model is used, and a
    foundation model can be dropped in without the delivery side knowing.
    """

    def response(self, cytosolic_nM: float, hours: float) -> float:
        """Fractional target knockdown in [0, 1]."""
        ...


@dataclass(frozen=True)
class IndirectResponseCell:
    """A mechanistic RISC-loading and indirect-response model.

    Saturable effector loading (Hill, capacity-limited) driving first-order mRNA
    turnover. Deliberately mechanistic rather than learned, so the coupling can be
    exercised and audited without a trained model -- and so the transport-versus-
    mechanism attribution below can be validated against a system whose ground
    truth is known by construction.
    """

    ic50_nM: float = 2.0
    hill: float = 1.3
    risc_capacity_nM: float = 25.0
    k_deg_mrna_per_h: float = 0.08
    e_max: float = 0.95

    def loaded(self, cytosolic_nM: float) -> float:
        """Saturable RISC occupancy. Above capacity more cargo achieves nothing,
        which is the mechanistic reason a dose-response plateaus."""
        c = max(cytosolic_nM, 0.0)
        return self.risc_capacity_nM * c / (self.ic50_nM + c)

    def response(self, cytosolic_nM: float, hours: float) -> float:
        occ = self.loaded(cytosolic_nM) / self.risc_capacity_nM
        inhibition = self.e_max * occ ** self.hill / (0.5 ** self.hill + occ ** self.hill)
        approach = 1.0 - math.exp(-self.k_deg_mrna_per_h * max(hours, 0.0))
        return float(min(max(inhibition * approach, 0.0), 1.0))

    def sensitivity(self, cytosolic_nM: float, hours: float, eps: float = 1e-3) -> float:
        """d(response)/d(log10 cytosolic). The local slope of the dose-response,
        which is what decides whether a failure can be attributed at all."""
        lo = self.response(cytosolic_nM * (1 - eps), hours)
        hi = self.response(cytosolic_nM * (1 + eps), hours)
        return float((hi - lo) / (2 * eps / math.log(10)))


# ----------------------------------------------- the attribution that matters
def attribute_failure(cell: VirtualCell, cytosolic_nM: float, hours: float,
                      observed_response: float, *,
                      target_response: float = 0.70,
                      flat_slope: float = 0.05) -> dict:
    """Transport-limited or mechanism-limited? And may the question be answered?

    The refusal branch is the substantive part. If the dose-response is locally
    FLAT at the achieved concentration -- saturated, or below threshold -- then
    transport and mechanism are not distinguishable from this observation, and
    returning one of them would be a guess dressed as an attribution. The
    prescribed experiment differs in each case and is returned with the verdict,
    because an attribution that does not name the next experiment has no decision
    value.
    """
    if not 0.0 <= observed_response <= 1.0:
        raise ValueError("observed response must be a fraction in [0, 1]")
    predicted = cell.response(cytosolic_nM, hours)
    slope = (cell.sensitivity(cytosolic_nM, hours)
             if hasattr(cell, "sensitivity") else float("nan"))
    met = observed_response >= target_response
    base = {"cytosolic_nM": cytosolic_nM, "hours": hours,
            "observed_response": observed_response,
            "predicted_response": round(predicted, 4),
            "local_slope_per_log10": round(slope, 4) if slope == slope else None,
            "target_response": target_response}

    if met:
        return {**base, "verdict": "no failure to attribute",
                "next_experiment": "none required"}

    flat = slope == slope and abs(slope) < flat_slope
    saturated = flat and predicted > target_response * 0.5

    # The refusal applies to the BELOW-THRESHOLD flat region only.
    #
    # An earlier version refused wherever the curve was flat, which was wrong in a
    # way worth recording: at saturation, flatness positively EXCLUDES transport --
    # more cargo demonstrably cannot change the outcome -- so it strengthens a
    # mechanism attribution rather than blocking it. The genuinely unanswerable
    # case is the opposite end, where response is near zero because nothing has
    # been meaningfully tested.
    if flat and not saturated:
        return {**base, "verdict": "not attributable",
                "reason": ("the achieved concentration is below the threshold where "
                           "this cell responds at all, so the observation tests "
                           "neither transport nor mechanism"),
                "next_experiment": ("a concentration series spanning the steep region "
                                    "of the dose-response, which this observation is "
                                    "not in")}

    if predicted >= target_response:
        return {**base, "verdict": "mechanism-limited",
                "reason": ("ample cargo reached the cytosol -- the cell model predicts "
                           f"{predicted:.2f} against {observed_response:.2f} observed"
                           + (", and the dose-response is saturated here, so more "
                              "cargo is positively excluded as the explanation"
                              if saturated else
                              ", so the shortfall is not explained by transport")),
                "next_experiment": ("target validation: confirm the target is required "
                                    "in this cell type and that the payload engages it; "
                                    "carrier changes will not help")}

    if saturated:
        return {**base, "verdict": "target-unreachable",
                "reason": (f"the cell model's own ceiling at saturation is "
                           f"{predicted:.2f}, below the {target_response:.2f} target: "
                           f"no achievable cytosolic concentration reaches it, so the "
                           f"shortfall is a property of the target or payload rather "
                           f"than of either the carrier or this particular dose"),
                "next_experiment": ("revise the target response, change the payload, or "
                                    "test the cell model's ceiling against a positive "
                                    "control that is known to reach the target")}

    return {**base, "verdict": "transport-limited",
            "reason": ("the cell model predicts the observed shortfall from the "
                       "cytosolic concentration achieved, and the dose-response is "
                       "still rising here: the mechanism would have worked given "
                       "more cargo"),
            "next_experiment": ("carrier re-optimisation -- raise cytosolic delivery "
                                "and re-measure; the target is not implicated")}
