# Another Take (CS)

**The bench↔bedside programme rewritten as a computer-science problem: a verified
dataflow program over typed biological quantities, in which the mathematics
enforces the physics and the type system enforces the honesty.**

```
pip install -e .
python examples/end_to_end.py      # the hepatic siRNA case through all seven layers
pytest                            # 128 tests
```

---

## The take

The 20 October brief sets out a dual-loop biological intervention world model:
corona identity → multi-organ PBPK → endosomal barrier → intracellular network,
with Empirical Bayes and conformal credibility gates, and a controlled bridge to a
patient loop. The science is in the brief. This repository asks a different
question about it:

> If the chain is a *program*, which of its claims are **properties of the
> program** — true by construction, checkable before any data — and which are
> empirical claims that need evidence?

That split turns out to be unusually clean here, and it is the contribution:

| claim | kind | where |
|---|---|---|
| mass is conserved exactly | structural — an algebraic identity | `pbpk.py` |
| corona shares lie on the simplex | structural — by construction | `corona.py` |
| the organ→cytosol handoff is unidentifiable | structural — decidable from wiring | `observability.py` |
| a simulation calibrated to data is still a simulation | structural — a lattice property | `taint.py` |
| a composition is never more certain than its inputs | structural — an invariant | `propagate.py` |
| anchoring collapses the interval 5–10× | **follows from the estimator** | `calibrate.py` |
| the contrast target removes the study offset | **algebra, exactly** | `delta.py` |
| ...and whether that trade is worth it | **conditional**, computed | `delta.advantage` |
| any reported R², any fold-width on real data | **empirical** — not in this repo | — |

The practical consequence: five of the programme's central guarantees need no
experiment, and one of its central obstacles can be diagnosed **at zero cost,
before a year of fitting**.

---

## What each module does

### `observability.py` — is the quantity recoverable at all?

The field asks "can we predict organ delivery better?". The sharper question is
whether the parameter is *determined* by the measurements an experiment can
produce, or whether a whole family of values produces identical observations. That
is decidable from the model's wiring — control theory calls it observability — and
it is answerable **before data exists**.

Run it on the organ→cytosol interface and you get: **6 unknowns, 1 achievable
equation, 5 degrees of freedom, nothing identifiable.** Whole-organ homogenate
reports the *sum* of parenchymal, macrophage, endothelial and extracellular
material; the interface needs the *split*. No model capacity helps, because the
shortfall is in the measurement set. The module then returns the minimal
measurement sets that would close it — `flow_sorted_parenchymal` alone suffices —
which is the actionable half.

The same formalism explains why reference anchoring works, as one equation:

```
y = f(x) + α_study            one equation, two unknowns → unidentifiable
+ reference arm, same run     pins α_study                → identifiable
+ a second NOVEL arm          adds an equation AND an unknown → still not
```

That last line is why "more formulations per paper" never fixed cross-study
transfer. And the clinical fix — a shared reference material at every site — is
*structurally the same operation*, which is the one genuine piece of shared science
between the two loops. It is derived here, not asserted.

### `taint.py` — measuredness does not launder

Provenance as an information-flow lattice, `MEASURED < DERIVED < SIMULATED <
ASSUMED < ILLUSTRATIVE`, where join is max and **no operation lowers a taint**.
One assumed parameter in a chain of measurements makes the output assumed.

`calibrate()` exists specifically to refuse the most tempting laundering operation
in the field: a simulator tuned to data *feels* like data. It is not. The
calibrated object is a new simulated object whose parameters were fitted, and the
measurement was consumed as a target rather than reported as a result.

### `propagate.py` — uncertainty cannot shrink

A monotonicity invariant: no composition may be more certain than its least certain
input. Obvious stated that way, violated constantly in practice, because each stage
of a multiscale pipeline is fitted independently and reports the uncertainty of *its
own residuals* — discarding its input's. `reestimate_check()` is the audit to run on
an existing pipeline, and it quantifies the understatement.

`Dependence` is **positional with no default**: in this chain the stages share a
study offset, so assuming independence is how every interval comes out too narrow.

### `pbpk.py` — conservation as an identity, not a penalty

The generator is built Metzler with zero column sums, so `1ᵀK = 0` and total amount
is conserved **exactly**, as algebra. Non-negative off-diagonals make `exp(Kt)`
non-negative, so a compartment cannot go negative. Solved by matrix exponential
rather than a stepper: conservation then holds to machine precision at every time,
with no dependence on tolerance — the difference between a guarantee and a setting.

Elimination is a transfer into a **named absorbing sink**, not a leak, because
`Σ Aₘ + A_cleared = D₀` cannot otherwise hold for a drug that is actually
eliminated. `build_generator` *sets* the diagonals from each column's outflow, so
there is no way to call it and get a leaky generator.

**And conservation is not a result.** Reporting "0.0% mass-balance violation" as an
empirical finding is reporting that arithmetic works.

### `calibrate.py` — the two gates, and why they differ

`bayes_shrink` implements the brief's Empirical Bayes estimator
`b̂ = [kτ₀²/(kτ₀²+σₑ²)]·(ȳ_anchor − f̄)`, tested against both limits: `k→∞` trusts
the anchor residual completely, `τ₀²→0` correctly attributes nothing to the
laboratory. Validated by recovering a planted offset and beating the raw residual
on MSE — the entire justification for shrinking.

`conformal_interval` is a **separate** guarantee: shrinkage moves the point
prediction, conformal calibration bounds the error. Exchangeability must hold at the
right unit, and for clustered data that unit is the **study**, not the row. Row-level
calibration on study-clustered residuals is the conformal analogue of a row-level
train/test split, made for the same reason. Below `⌈1/(1−level)⌉` studies the
quantile does not exist and the function refuses rather than silently returning the
maximum.

The 5–10× collapse from anchoring **follows from the posterior variance of the
offset** — worth knowing before attributing it to a model.

### `delta.py` — predict differences, not values

Within-study contrasts remove the study offset *exactly*:

```
y_{s,i} = f(x_i) + α_s + e_{s,i}
Δy      = f(x_i) − f(x_ref) + (e_i − e_ref)        α_s is gone
```

No estimate of `α_s` required, for every study. Three prices, all stated and all
tested: differencing **doubles** residual variance; single-arm studies contribute
**nothing**; and recovering an absolute value still needs the reference arm — so the
contrast target and anchoring are complements, not alternatives.

`advantage()` computes whether the trade is favourable rather than assuming it:
contrasting helps only when `τ₀² > σₑ²`. At the brief's reported offset share that
holds comfortably — but it is a *condition*, not a law, and a quieter literature
would fail it.

### `corona.py` — composition is compositional

Softmax to the simplex by construction, because a "share" of 1.3 is a category error
rather than a small one. Only **log-ratios** are interpretable: a 0.05 share change
means different things at baseline 0.02 and 0.60. And parts of a composition are
forced toward negative correlation *whatever the biology* — so "ApoE and C3 compete
for the surface" is exactly what a raw correlation invites and cannot support.
`spurious_correlation_warning` measures the artefact.

### `cell.py` — the interface a foundation model plugs into

Perturbation foundation models (PerturbODE, scGPT, GEARS) answer: *given* this
knockdown, what happens to the transcriptome. They cannot answer: how much knockdown
does 0.3 mg/kg produce in a cirrhotic liver. That is a transport question, and no
amount of transcriptomic pre-training contains the answer **because the training
data never varied the delivery**.

So `VirtualCell` is a protocol with one method taking a cytosolic *concentration*. A
mechanistic indirect-response model and a pre-trained model are interchangeable
behind it — the coupling does not depend on which.

What the coupling buys is `attribute_failure`, which neither half can do alone:

| verdict | meaning | next experiment |
|---|---|---|
| **transport-limited** | mechanism would have worked given more cargo | fix the carrier |
| **mechanism-limited** | ample cargo, no response | fix the target |
| **target-unreachable** | the model's own ceiling is below the target | fix the payload or the target spec |
| **not attributable** | below the responding threshold | a concentration series |

Escape is modelled as the **product** of endosomal protonation (rising with pKa) and
plasma neutrality (falling with pKa), so the hepatocyte optimum near pKa 6.35 *emerges*
rather than being written down, and the limiting term is reported — which is what a
formulation chemist actually needs.

### `state.py` + `worldmodel.py` — the architecture, and its honest scope

Typed state groups with invariants checked **on every write**, because the failure
mode is a stage that is individually correct and jointly wrong: a corona model
returning probabilities and a PBPK model returning amounts are each fine; feeding
one into the other as though probabilities were amounts is the
fraction-as-concentration defect. Putting the invariant on the *store* catches it at
the write, not at the eventual implausible result.

The brief's extensibility claim — new assays, organs and pathways plug in without
rewriting the core — is **tested rather than asserted**: `test_worldmodel.py` adds a
new state group, a new process and a new organ at runtime.

`credibility()` then decides what the chain may be used for, reading taint and
identifiability rather than a declared maturity label, so a stage cannot be promoted
by relabelling it:

| context of use | verdict |
|---|---|
| rank formulations within one study | **permitted** |
| predict absolute delivery in a new laboratory | refused — assumed stage, no anchor |
| predict target-cell exposure | refused — interface unidentifiable |
| inform a patient-level decision | refused — requires measured evidence throughout |

One of four, and it is the comparative one. The gate **discriminates**: a ranking is
invariant to any factor common to every arm, so an assumed constant partition cannot
change the order — only the scale. A gate that refused that too would refuse
everything and be ignored.

---

## On the words "world model"

Three commitments in "dual-loop biological intervention world model", and only two
are delivered by code.

- **Coupling** — delivered. The stages compose through typed state with checked
  invariants.
- **Gates** — delivered. Empirical Bayes and conformal calibration decide what a
  prediction may claim.
- **World model** — *not* delivered, and nothing here pretends otherwise. A world
  model predicts the consequences of interventions it has not seen; the central
  handoff of this chain is structurally unidentifiable from the measurements the
  field makes.

A world model whose central handoff is unidentifiable is an **architecture**. What
the architecture buys is not prediction. It is that the chain states which stage is
unsupported, refuses the compositions it cannot justify, and names the measurement
that would change that.

## The agentic constraint, implemented

The brief requires that agents "act strictly as compilers and auditors" while
"equations enforce physical constraints". That ordering is enforced, not promised: no
function accepts a model-generated quantity that has not passed its state group's
invariant, and `credibility()` reads taint rather than trusting a label. **An agent
here can choose what to compute; it cannot choose whether mass is conserved.**

## Errors this repository found in itself

Kept visible, each pinned by a test, because the discipline is the point.

1. **A definitional simplex counted as information.** The four cell fractions are
   fractions *of organ-associated material*, so "they sum to one" *is* "the
   homogenate measures their total" — a definition restated. Declaring both made the
   system look one degree of freedom better than it is. Caught because equations are
   compared as unknown-*sets* rather than counted.
2. **A mechanistic story asserted without measuring it.** Having found 59.8% cleared
   at 672 h, I claimed the shortfall was depot-limited and clearance irrelevant.
   Measuring it refuted that: 10× clearance takes elimination to 0.997. Both rates
   are limiting; the intuition reaches for the wrong answer.
3. **An attribution that refused where flatness was informative.** `attribute_failure`
   originally refused wherever the dose-response was flat — including at saturation,
   where flatness *positively excludes* transport and therefore **strengthens** a
   mechanism attribution. Fixing it surfaced a fourth verdict, `target-unreachable`.
4. **The first stage lost from its own attribution.** `chain()` seeded from a leaf
   whose stage list is empty, so the first stage vanished from every propagated
   result — defeating the only purpose the field has.
5. **An infinity reported as a number.** A relative SD above ~0.51 puts the lower
   bound below zero, at which point a multiplicative fold-width is *undefined*, not
   large — and the quantity belongs on the log scale.

## What this is not

Not a reproduction of the brief's reported performance figures: those need the
substrate, and this is the machinery rather than the result. Not a validated model of
anything — `IndirectResponseCell` is mechanistic so the coupling can be *exercised*,
not because it is calibrated. Not a replacement for the brief's science. It is the
argument that the science has a structure, and that the structure can be checked by
running it.

---

# v0.5 — beyond the chain

v0.1 implemented the delivery chain. v0.5 implements the **programme**: both loops,
the bridge, the data layer, the mixing rules and the agent contract. The version is
0.5 rather than 1.0 for a reason stated at the bottom of this section.

## `phases.py` — Loop A is a loop, not a pipeline

The brief's §04: "not just a linear target → drug chain: evidence can force revision
of the target, delivery system, or entire mechanistic hypothesis." A pipeline cannot
express that, so each phase declares **revision edges** — which earlier phases its
failure can send you back to. A Phase 4 failure revises Phase 3 before Phase 2:
latest first, because that discards the least work, and because exhausting delivery
explanations before abandoning a target is the right scientific order.

**Simulation happens twice, and conflating the two is a real error.**

| | Phase 2 | Phase 4 |
|---|---|---|
| what it simulates | the virtual cell | the full delivery chain |
| delivery | **assumed perfect** | **is the question** |
| answers | would suppressing this target matter? | does this formulation reach it? |

`PhaseResult.may_claim_reachability()` refuses a Phase 2 result read as reachability
evidence — which is precisely the "assumes the therapeutic is already inside the
target cell" error the brief names in its executive summary. And `validate_inference`
refuses a Phase 5 that compared fewer than two mechanisms, because a single mechanism
without rivals is a hypothesis restated.

## `patient.py` — Loop B is a different operation

Two sentences from the brief are this module's whole design: the patient loop's task
is "state estimation and supported response modelling", and "updating a patient-state
posterior is not the same as retraining or deploying a population model."

So `reverse_translate` returns a **research hypothesis** and has *no code path* that
touches a model version. There is nothing to misconfigure.

**`c_patient` is mechanism-typed, not a feature vector.** Each field declares which
stage it modulates and in which direction:

| field | stage | direction | routine? |
|---|---|---|---|
| `apoe_plasma` | corona | +1 | no |
| `complement_c3` / `c4` / `crp` | corona | −1 | yes |
| `anti_peg_igg` | organ | −1 | no |
| `hepatic_fibrosis_stage` | barrier | −1 | yes |
| `egfr` | organ | −1 | yes |
| `albumin` | corona | **0** — direction not established | yes |

Two reasons this matters. Fields modulating different stages enter the model at
different places, and with 29 patients there is no prospect of recovering that from
data. And a declared direction is checkable: if fibrosis is declared to reduce
parenchymal access and the fitted effect goes the other way, *that is a finding* —
but only because the direction was committed in advance. Where the direction is
genuinely unknown (`albumin`) it is declared zero rather than guessed.

`context_coverage` reports which stages have **no** routinely available modulator and
therefore cannot be personalised at all — the model still runs there, on population
parameters, and should say so rather than implying patient-specificity.

**The patient-count problem is enforced, not noted.** The brief states it twice:
215 observations from 29 patients are not 215 samples. At ICC 0.5 the design effect
is 4.2 and the effective n is **51**. `require_independent_patients` raises on any
claim that used the observation count as a sample size.

## `multimodal.py` — what may be mixed, and how

Seven corpora, **six different units of observation**. The naive move — stack the
tables, add a modality column, train one model — fails not on scale or architecture
but because *the rows mean different things*. A corona row is a protein share for one
plasma; a PBPK row is an organ amount for one animal at one timepoint; a HepG2 row is
a well of a cell line that never saw an animal.

Five techniques, in increasing order of what they assume:

| technique | assumes | couples stages? |
|---|---|---|
| **linkage** | identity genuinely matches (formulation **and batch**) | yes, where it matches |
| **late fusion** | modalities predict the same target | no — the correct default |
| **partial pooling** | groups exchangeable *at the group level* | no |
| **shared parameter** | one physical parameter is common — **and tests it** | yes |
| **reweighting** | target support ⊆ source support | no |

`shared_parameter` is the only one that genuinely couples stages, and it is the only
one that can **refuse**: if two modalities' estimates disagree beyond their stated
precision (Cochran's Q), the shared-parameter assumption is refuted and *no pooled
estimate is returned*. Widening the interval instead would hide the refutation.

`covariate_shift_weights` checks support before weighting, because a clinical cohort
contains patients no mouse study covers — and a large clipped weight is extrapolation
wearing a weight's clothing.

`early_fusion` exists only to **raise**, with the reason and the alternatives, so the
rule lives in code rather than in a README a future contributor has not read.

**`icc_oneway`** was added after a test failure. The naive ratio
var(group means)/var(all) estimates (τ²+σ²/m)/(τ²+σ²) and so **exceeds** the ICC for
every m > 1 — measured here at 0.865 against a true 0.739 with four arms per study.
The function reports both and the gap between them, because a bare percentage does
not say which was computed.

## `critic.py` — four rivals, one posterior

The brief's §14: compare uptake-limited, release-limited, degradation-limited and
target-response explanations. Each has a **signature** — the pattern that would
distinguish it — and a direct next measurement.

Comparison is by BIC-approximated model evidence, so a mechanism is not favoured for
having more freedom. The most useful output is often **"not distinguishable"**: a
statement about the *data* rather than about the models, returned with the
discriminating experiment attached. `discriminating_design` ranks conditions by how
much the mechanisms *disagree* — never by where the favoured one looks best, which is
how a critic becomes a confirmation engine.

## `datasets.py` — two kinds of number, separated by type

**Reported** figures are transcribed from the brief and carry its origin string;
this repository has not reproduced them. The one thing it *computes* is the ablation
sign test, because that is arithmetic over the transcribed values — and it recomputes
to **p = 0.03125**, matching the reported 0.0312 exactly.

**Generated** tables reproduce the *structure* of the problem (86 study groups, the
reported ICC, the pKa optimum, 29 patients × ~7.4 visits) and deliberately **not** its
values. Every row carries `is_synthetic = True` and an origin naming what it does not
reproduce.

The brief's §17 gate — "Do not create synthetic 'ground truth' to fill the gap" — is
enforced by `as_measurement()`, which raises on any frame containing a synthetic row.
**There is no override flag**, and a test asserts there is none: the entire value of
the rule is that a caller in a hurry cannot bypass it.

Shipped under `data/`: 7 tables, 3 reported and 4 generated.

## `agents.py` — the one axis on which an agent layer can differ

The brief's originality boundary: "Building six prompt personas does not make a new
scientist." So this is not an agent framework — it is the **scoring protocol** that
would settle the question, plus five roles each declaring *what it may not do* (which
is where agent architectures actually fail: roles that quietly acquire each other's
powers), and the module that can check it.

Of the brief's six criteria, only **correct refusal** can distinguish an
evidence-aware agent from a fixed workflow with the same tools — a good fixed
workflow gets the valid links right too; what it cannot do is decline a link that
looks computable and is not supported. `score_refusals` returns a confusion table and
**no single score**: a false permit propagates an unsupported number into a
conclusion, a false refusal costs an analysis, and an F1 that treats them as
interchangeable is the wrong summary. `versus_baselines` reports "no measured
advantage" as a *finding about agent architectures*, not a null to bury.

## Further errors this repository found in itself (v0.5)

6. **Rounding broke the simplex.** The corona generator rounded shares to 5 dp, and
   six parts at that precision drift by up to 3 × 10⁻⁵ — enough to fail the invariant
   the module enforces. The artefact would have shipped simplex-violating. Fixed with
   largest-remainder apportionment, which unlike "adjust the biggest part" does not
   bias one component.
7. **A biased variance estimator in my own test.** The generated ICC "failed" at
   0.861 against a 0.748 target. The generator was right; the test used
   var(group means)/var(all), which is the same upward bias this session had already
   flagged for group-count effect sizes.
8. **An unreachable refusal.** `critic.compare` checked the observation count before
   the mechanism count, so the single-mechanism refusal never fired when the caller
   also passed one observation — the error named the wrong problem. Argument validity
   now precedes data adequacy.

## Why 0.5 and not 1.0

Everything the architecture needs is implemented and tested: both loops, the bridge,
the typed state, the gates, the mixing rules, the data wall, the agent contract. What
is missing is not code.

**1.0 requires one measurement.** The organ→cytosol handoff is structurally
unidentifiable from whole-organ data (`observability`), and
`credibility("predict_target_cell_exposure")` will refuse until it is closed. The
minimal fix the module returns is a flow-sorted parenchymal measurement. Until a
paired dataset exists, no version of this repository can honestly be 1.0 — and
raising the number without the measurement would be the one thing every module here
is built to prevent.
