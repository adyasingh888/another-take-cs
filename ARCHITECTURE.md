# Architecture: which module answers which section of the brief

A reviewer's first question is whether the code does what the document claims. This
table is the answer, section by section. Every row names the module and the single
most load-bearing thing it enforces.

| brief | claim | module | what is enforced in code |
|---|---|---|---|
| §01 | two problems, not one missing model | `observability` | reachability is a structural question, decidable before fitting |
| §03 | a connected diagram is a hypothesis until each handoff is validated | `status` | chain status is the **minimum** over handoffs, never the headline |
| §04 | Loop A is not a linear chain | `phases` | revision edges; a Phase 4 failure revises Phase 3 before Phase 2 |
| §05 | the patient loop is state estimation, not deployment | `patient` | `reverse_translate` has **no code path** to a model version |
| §06 | five engineering components | `evidence`¹, `registry`, `quantities`, `execution`, `agents` | all five present; `__all__` membership is tested |
| §07 | fraction × efficiency ≠ concentration | `quantities` | the illegal product is **unrepresentable**, not merely refused |
| §08 | five corpora, not five paired stages | `multimodal` | `early_fusion` exists only to raise, with the alternatives |
| §12 | DeltaPBPK: train on within-study contrasts | `delta` | the offset cancellation is algebra, with all three of its prices |
| §14 | prompt personas are not a new scientist | `agents`, `planner` | only correct refusal can distinguish an agent from a fixed workflow |
| §15 | hepatic siRNA worked example | `examples/` | `0 of 5` stages cross to a patient-level prediction |
| §16 | ICH M15 now, without claiming qualification | `icm15` | risk = influence × consequence; the gap is a **gate**, not a percentage |
| §17 | do not create synthetic ground truth | `datasets` | `as_measurement()` raises on any synthetic row; **no override flag** |
| §6.1 | 74.8–88.2% of variance is between-study | `multimodal.icc_oneway` | the unbiased estimator; the naive ratio is reported beside it |
| §6.2 | pruning 67 features to 22 improves every organ | `datasets.reported_ablation` | the sign test **recomputes** to p = 0.03125 |
| §6.3 | zero-shot intervals are 57–158× and useless | `benchmark.interval_scorecard` | executed; coverage measured on held-out studies |
| — | anchoring is the remedy | `benchmark.anchored_predict` | anchor rows **excluded from the score** |

¹ `evidence` is the pairability graph; in this package its role is served by
`multimodal.linkage` plus `taint`, which is noted here rather than claimed as a
separate module.

## Reading order

1. `atcs vision` — the two loops with their gates
2. `atcs identifiability` — why the chain does not close, and what would close it
3. `atcs benchmark --intervals` — the leak, the remedy, and what the remedy costs
4. `atcs sensitivity` — which unidentifiable parameter is worth measuring
5. `examples/full_vision.py` — all of it, on the worked case

## What a reviewer should check first

**The three refusals.** `quantities` makes the dimensional error unrepresentable,
`datasets.as_measurement` has no override flag, and `multimodal.early_fusion` cannot
succeed. If any of those can be bypassed, the architecture's claim is void — so they
are the first things to attack.

**The anchor exclusion.** `benchmark.anchored_predict` returns the anchor mask so
the exclusion is verifiable. If the revealed rows were scored, the headline would be
inflated by exactly the amount the anchor helps.

**The fourteen self-caught errors** listed at the end of the README. Each one is
pinned by a test that fails against the old behaviour — the list is a claim about
process, and the tests are its evidence.

## What is deliberately absent

- **A trained model.** This repository evaluates protocols and architectures. The
  models are the user's.
- **Real biodistribution data.** The generated tables reproduce the reported
  structure and carry `is_synthetic = True`. Substituting them for measurement is
  what §17 forbids and what `as_measurement` prevents.
- **An automatic joint-uncertainty composition.** `execution.compose` refuses to call
  a product of marginals a joint uncertainty; supplying the dependence structure is a
  scientific act, not a framework's.
