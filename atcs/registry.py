"""
The model registry: context of use is part of the model.

WHY THIS IS NOT A LOOKUP TABLE
------------------------------
The brief's §06 asks for a registry of "models, supported contexts, required inputs,
output definitions and validated questions". The design decision is that the context
is not metadata attached to a model -- it is PART of the model. A model is not a
function that may be applied to any input of the right shape; it is a function plus
the species, compartment, descriptor range, question and decision it was validated
for.

Applying it outside that is therefore not a judgement call to be made at the call
site. It is a refusal the registry performs, because the call site is exactly where
the context gets forgotten -- the caller has an input of the right shape and a model
that accepts it, and nothing in the types objects.

THE ERROR THIS PREVENTS
-----------------------
"No better model exists" is the most common reason a model gets used outside its
validated range, and it is not a credential. `admissible()` returns an empty list
rather than the least-bad option, and an empty list is the correct answer to "which
model may I use here" when none may.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .status import Evidenced, Status


class ContextViolation(RuntimeError):
    """A model was requested outside the context it was validated for."""


@dataclass(frozen=True)
class ContextOfUse:
    """Where a model may be applied, and for what decision.

    `decision` is required and not decorative: the same model can be adequate for
    ranking research candidates and inadequate for anything a patient experiences,
    and a registry that records only species and compartment cannot express that.
    """

    species: str
    compartment: str
    question: str
    descriptor_ranges: Mapping[str, tuple[float, float]]
    decision: str

    def __post_init__(self) -> None:
        for name, (lo, hi) in self.descriptor_ranges.items():
            if lo >= hi:
                raise ContextViolation(
                    f"descriptor range for {name!r} is empty: [{lo}, {hi}]")
        if len(self.decision) < 15:
            raise ContextViolation(
                "a context of use must name the DECISION the model was validated "
                "for; the same model can be adequate for research ranking and "
                "inadequate for anything a patient experiences")

    def covers(self, *, species: str, compartment: str,
               descriptors: Mapping[str, float] | None = None) -> tuple[bool, list[str]]:
        why: list[str] = []
        if species != self.species:
            why.append(f"validated on {self.species!r}, requested for {species!r}")
        if compartment != self.compartment:
            why.append(f"validated in {self.compartment!r}, requested for "
                       f"{compartment!r}")
        for k, v in (descriptors or {}).items():
            if k in self.descriptor_ranges:
                lo, hi = self.descriptor_ranges[k]
                if not lo <= v <= hi:
                    why.append(f"{k} = {v} lies outside the validated range "
                               f"[{lo}, {hi}]")
        return not why, why


@dataclass(frozen=True)
class RegisteredModel:
    """A model with its evidential status, context, inputs and output definition."""

    model_id: str
    evidence: Evidenced
    context: ContextOfUse
    required_inputs: tuple[str, ...]
    output: str

    def check(self, *, species: str, compartment: str,
              descriptors: Mapping[str, float] | None = None,
              inputs: Sequence[str] = ()) -> None:
        ok, why = self.context.covers(species=species, compartment=compartment,
                                      descriptors=descriptors)
        if not ok:
            raise ContextViolation(
                f"{self.model_id} is outside its context of use: {'; '.join(why)}. "
                f"Scarcity is not a credential: that no better model exists is not a "
                f"reason to apply this one here.")
        missing = [i for i in self.required_inputs if i not in set(inputs)]
        if inputs and missing:
            raise ContextViolation(
                f"{self.model_id} requires {missing} which were not supplied")


class Registry:
    """What may be used where, with refusal as the default for anything else."""

    def __init__(self) -> None:
        self._m: dict[str, RegisteredModel] = {}

    def register(self, model: RegisteredModel) -> None:
        self._m[model.model_id] = model

    def get(self, model_id: str) -> RegisteredModel:
        if model_id not in self._m:
            raise KeyError(
                f"{model_id!r} is not registered. Registered: {sorted(self._m)}")
        return self._m[model_id]

    def admissible(self, *, species: str, compartment: str,
                   descriptors: Mapping[str, float] | None = None,
                   minimum_status: Status = Status.ILLUSTRATIVE) -> list[str]:
        """Every model admissible here. An empty list is a correct answer.

        Returns no least-bad option, deliberately. "Nothing may be used here" is
        the honest answer when nothing was validated here, and offering a fallback
        is how a model leaves its range.
        """
        out = []
        for mid, m in sorted(self._m.items()):
            ok, _ = m.context.covers(species=species, compartment=compartment,
                                     descriptors=descriptors)
            if ok and m.evidence.status >= minimum_status:
                out.append(mid)
        return out

    def table(self) -> list[dict]:
        return [{"model": m.model_id, "status": m.evidence.status.name,
                 "species": m.context.species, "compartment": m.context.compartment,
                 "decision": m.context.decision, "output": m.output,
                 "inputs": ", ".join(m.required_inputs)}
                for m in sorted(self._m.values(), key=lambda x: x.model_id)]

    def __len__(self) -> int:
        return len(self._m)
