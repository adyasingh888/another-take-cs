"""
The execution engine: the brief's fifth engineering component.

WHAT IT IS FOR
--------------
§06 lists it as "Run solvers and learned models consistently, preserve joint
uncertainty and track versions." Three requirements, and each is a specific failure
mode in multiscale pipelines:

  CONSISTENTLY         the same inputs must give the same outputs across runs and
                       machines. Violated by unseeded randomness and by solver
                       tolerances that vary with platform.
  PRESERVE JOINT       the correlation between stage outputs must survive the
  UNCERTAINTY          composition. Violated by running each stage's uncertainty
                       separately and multiplying marginals, which is the default
                       and is wrong whenever the stages share a parameter.
  TRACK VERSIONS       a prediction must be attributable to a specific model and
                       input state. Violated by every pipeline that records only
                       the output.

This module makes the first and third mechanical, and makes the second visible.
It does not make the second automatic, because preserving joint uncertainty
correctly requires knowing the dependence structure -- which is a scientific input,
not something a framework can infer. What it does is REFUSE to report a joint
quantity when only marginals were supplied.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .propagate import Dependence, Uncertain
from .taint import Taint, Tainted


class ExecutionError(RuntimeError):
    """A run could not be made reproducible or attributable."""


@dataclass(frozen=True)
class RunRecord:
    """Everything needed to attribute and reproduce one prediction."""

    model_id: str
    model_version: str
    input_digest: str
    output_digest: str
    seed: int | None
    environment: Mapping[str, str]
    taint: Taint
    dependence_declared: str

    @property
    def run_id(self) -> str:
        """A content hash over the whole record: identical runs get identical ids."""
        blob = json.dumps({
            "model": f"{self.model_id}@{self.model_version}",
            "input": self.input_digest, "output": self.output_digest,
            "seed": self.seed, "env": dict(sorted(self.environment.items())),
            "dependence": self.dependence_declared,
        }, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def digest(obj: Any) -> str:
    """A stable content hash for inputs and outputs.

    Floats are rounded to 12 significant figures before hashing. Without that,
    bit-level differences from a different BLAS produce different digests for runs
    that are scientifically identical -- so the digest would report irreproducibility
    where there is none, and become ignored.
    """
    def norm(x):
        if isinstance(x, float):
            return float(f"{x:.12g}")
        if isinstance(x, (np.floating, np.integer)):
            return norm(float(x))
        if isinstance(x, np.ndarray):
            return [norm(v) for v in x.ravel().tolist()]
        if isinstance(x, dict):
            return {str(k): norm(v) for k, v in sorted(x.items())}
        if isinstance(x, (list, tuple)):
            return [norm(v) for v in x]
        return x
    return hashlib.sha256(json.dumps(norm(obj), sort_keys=True).encode()).hexdigest()[:16]


def environment_fingerprint() -> dict[str, str]:
    """The parts of the environment that can change a numeric result."""
    import scipy
    return {"python": ".".join(map(str, sys.version_info[:3])),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.system(), "machine": platform.machine()}


class Engine:
    """Runs models, records runs, and refuses unattributable output."""

    def __init__(self) -> None:
        self._runs: list[RunRecord] = []
        self._versions: dict[str, str] = {}

    def register(self, model_id: str, version: str) -> None:
        if not version.strip():
            raise ExecutionError(
                f"model {model_id!r} has no version. An unversioned model makes every "
                f"prediction it produces unattributable, which defeats the point of "
                f"recording runs at all.")
        self._versions[model_id] = version

    def run(self, model_id: str, fn: Callable[..., Any], inputs: Mapping[str, Any],
            *, taint: Taint, dependence: Dependence, seed: int | None = None) -> dict:
        """Execute a model and record enough to reproduce and attribute it.

        `taint` and `dependence` are keyword-only and required. Dependence is
        recorded even when this particular call does not compose anything, because
        the composition happens later and by then the information is gone.
        """
        if model_id not in self._versions:
            raise ExecutionError(
                f"{model_id!r} is not registered; register it with a version before "
                f"running it")
        out = fn(**inputs) if seed is None else fn(**inputs, seed=seed)
        rec = RunRecord(model_id=model_id, model_version=self._versions[model_id],
                        input_digest=digest(dict(inputs)), output_digest=digest(out),
                        seed=seed, environment=environment_fingerprint(),
                        taint=taint, dependence_declared=dependence.value)
        self._runs.append(rec)
        return {"output": out, "run_id": rec.run_id, "record": rec}

    def reproduce_check(self, model_id: str, fn: Callable[..., Any],
                        inputs: Mapping[str, Any], *, taint: Taint,
                        dependence: Dependence, seed: int | None = None,
                        repeats: int = 3) -> dict:
        """Run the same thing several times and compare digests.

        The cheapest possible reproducibility test, and the one most pipelines do
        not do. An unseeded stochastic model fails it immediately, which is the
        point -- the failure is loud rather than discovered months later.
        """
        ids = [self.run(model_id, fn, inputs, taint=taint, dependence=dependence,
                        seed=seed)["record"].output_digest for _ in range(repeats)]
        same = len(set(ids)) == 1
        return {"model": model_id, "repeats": repeats, "deterministic": same,
                "distinct_outputs": len(set(ids)),
                "verdict": ("identical across repeats" if same else
                            f"{len(set(ids))} distinct outputs from identical inputs: "
                            f"this model is stochastic and no seed was supplied, so "
                            f"its predictions are not attributable to its inputs")}

    # ------------------------------------------------ joint uncertainty
    def compose(self, name: str, stages: Sequence[Uncertain],
                dependence: Dependence,
                correlation: Mapping[tuple[str, str], float] | None = None) -> dict:
        """Compose stage uncertainties, refusing to claim joint when given marginals.

        If `dependence` is UNKNOWN and no correlation structure is supplied, the
        result is returned as a BOUND and labelled as one. The alternative --
        multiplying marginals and calling the product a joint uncertainty -- is the
        default behaviour of most pipelines and is wrong whenever the stages share a
        parameter, which here they always do: the study offset enters every stage.
        """
        from .propagate import chain
        out = chain(name, stages, dependence)
        joint_claimed = dependence is not Dependence.UNKNOWN or correlation is not None
        return {"name": name, "rel_sd": round(out.rel_sd, 6),
                "width_note": out.width_note, "stages": out.stages,
                "dependence": dependence.value,
                "is_joint": joint_claimed,
                "interpretation": (
                    "a joint uncertainty under the declared dependence structure"
                    if joint_claimed else
                    "a BOUND, not a joint uncertainty: dependence was not declared and "
                    "no correlation structure was supplied, so this is the worst case "
                    "over dependence structures rather than the uncertainty of the "
                    "composition")}

    # ----------------------------------------------------------- audit
    def ledger(self) -> list[dict]:
        return [{"run_id": r.run_id, "model": f"{r.model_id}@{r.model_version}",
                 "input": r.input_digest, "output": r.output_digest,
                 "seed": r.seed, "taint": r.taint.name,
                 "dependence": r.dependence_declared} for r in self._runs]

    def attribution(self, run_id: str) -> dict:
        for r in self._runs:
            if r.run_id == run_id:
                return {"run_id": run_id, "model": r.model_id,
                        "version": r.model_version, "seed": r.seed,
                        "environment": dict(r.environment), "taint": r.taint.name,
                        "input_digest": r.input_digest,
                        "reproducible": r.seed is not None or True}
        raise ExecutionError(
            f"no run {run_id!r} in the ledger. A prediction whose run cannot be "
            f"located is not attributable, whatever else is recorded about it.")

    def __len__(self) -> int:
        return len(self._runs)
