"""The harness, the sensitivity decomposition, the cycle closure and the CLI."""
import numpy as np
import pandas as pd
import pytest

from atcs.benchmark import (BASELINES, BenchmarkError, anchored_predict,
                            leak_and_remedy, point_metrics, predict_ridge,
                            predict_train_mean, scorecard, sibling_leakage,
                            split_grouped, split_random)
from atcs.datasets import generate_biodistribution
from atcs.experiment import ExperimentSpec, Resource
from atcs.loop import CycleError, close_cycle
from atcs.models import CYTOSOL_CANDIDATES, ORGAN_TO_CYTOSOL
from atcs.sensitivity import (Parameter, SensitivityError, measurement_priority,
                              sobol_indices)

FEATS = ["size_nm", "zeta_mv", "pdi", "dose_mg_per_kg"]


@pytest.fixture(scope="module")
def df():
    return generate_biodistribution(n_studies=120, seed=11)


# ---- protocols ------------------------------------------------------------
def test_a_grouped_split_never_leaves_a_sibling_in_training(df):
    g = df.study_id.astype(str).to_numpy()
    for train, test in split_grouped(g):
        assert not (set(g[train]) & set(g[test]))


def test_a_random_split_leaves_a_sibling_for_most_test_rows(df):
    """The measurable form of the leak, and why a random-split score answers a
    question nobody asked."""
    g = df.study_id.astype(str).to_numpy()
    lk = sibling_leakage(g)
    assert lk["mean_sibling_fraction"] > 0.80
    assert "nobody asked" in lk["note"]


def test_every_split_partitions_the_rows(df):
    g = df.study_id.astype(str).to_numpy()
    for proto in (split_random, split_grouped):
        for train, test in proto(g):
            assert not set(train) & set(test)
            assert len(train) + len(test) == len(g)


# ---- baselines ------------------------------------------------------------
def test_the_train_mean_arm_is_not_optional():
    assert "train_mean" in BASELINES


def test_ridge_does_not_penalise_the_intercept():
    """A penalised intercept shrinks predictions toward zero rather than toward the
    training mean, which silently degrades every arm that uses it."""
    y = np.full(50, 7.0)
    X = np.random.default_rng(0).normal(size=(50, 3))
    p = predict_ridge(y, X, X, alpha=1e6)
    assert np.allclose(p, 7.0, atol=1e-3)


def test_the_train_mean_arm_predicts_the_train_mean():
    y = np.array([1.0, 2.0, 3.0])
    assert np.allclose(predict_train_mean(y, None, [0, 0]), 2.0)


# ---- the remedy -----------------------------------------------------------
def test_anchored_prediction_excludes_the_revealed_rows_from_the_score():
    """Scoring a row whose answer was revealed is the leak this package exists to
    catch, and it would inflate the result by exactly the amount the anchor helps."""
    rng = np.random.default_rng(0)
    g = np.repeat(["A", "B", "C"], 6)
    y = np.concatenate([rng.normal(m, 0.1, 6) for m in (3.0, -2.0, 0.5)])
    X = rng.normal(size=(18, 2))
    mask: list = []
    anchored_predict(y, X, X, groups_test=g, y_test=y, n_anchors=1, seed=0,
                     anchor_mask_out=mask)
    assert mask[0].sum() == 3, "one anchor per group"
    for grp in np.unique(g):
        assert mask[0][g == grp].sum() == 1


def test_anchoring_recovers_a_large_group_offset(df):
    """The central empirical claim, made runnable rather than asserted."""
    sc = scorecard(df, "log10_pid_liver", FEATS, n_splits=5, seed=11)
    out = leak_and_remedy(sc)
    assert out["anchored_r2"] > out["grouped_r2"] + 0.2
    assert out["beats_train_mean_when_anchored"]
    assert "Reporting either alone is" in out["verdict"]


def test_a_singleton_group_has_no_spare_row_to_anchor_on():
    g = np.array(["only"])
    mask: list = []
    p = anchored_predict(np.array([1.0, 2.0]), np.zeros((2, 1)), np.zeros((1, 1)),
                         groups_test=g, y_test=np.array([5.0]),
                         anchor_mask_out=mask)
    assert mask[0].sum() == 0 and len(p) == 1


# ---- the scorecard --------------------------------------------------------
def test_the_scorecard_cannot_drop_its_provenance_columns(df):
    sc = scorecard(df, "log10_pid_liver", FEATS, n_splits=3)
    assert sc.exercises_protocol.all()
    assert not sc.measures_performance.any(), "generated substrate"
    assert (sc.substrate == "generated").all()


def test_a_measured_frame_would_be_marked_as_measuring_performance():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"study_id": np.repeat(list("ABCDEFGH"), 5),
                      "y": rng.normal(size=40), "x": rng.normal(size=40),
                      "is_synthetic": False})
    sc = scorecard(d, "y", ["x"], n_splits=4)
    assert sc.measures_performance.all() and (sc.substrate == "measured").all()


def test_a_missing_column_is_refused(df):
    with pytest.raises(BenchmarkError, match="columns absent"):
        scorecard(df, "nonexistent", FEATS)


def test_leak_and_remedy_needs_an_anchored_arm(df):
    sc = scorecard(df, "log10_pid_liver", FEATS, n_splits=3)
    with pytest.raises(BenchmarkError, match="no anchored protocol"):
        leak_and_remedy(sc[~sc.protocol.str.contains("anchor")])


def test_fold_error_is_reported_alongside_r2():
    """R2 is not comparable across targets and is dominated by the tail on skewed
    data, so a scorecard carrying it alone is not interpretable."""
    m = point_metrics([1.0, 2.0, 3.0], [1.1, 2.1, 2.9])
    assert set(m) >= {"r2", "median_fold_error", "within_2fold", "mae"}


# ---- sensitivity ----------------------------------------------------------
def _chain(p):
    return p["a"] * p["b"] * p["c"]


PARAMS = [Parameter("a", 0.05, 0.60, "wide"), Parameter("b", 0.10, 0.80, "wide"),
          Parameter("c", 0.005, 0.040, "narrow")]


def test_a_multiplicative_chain_has_large_interaction():
    """First-order indices alone understate every parameter when inputs interact,
    which in a product they always do."""
    s = sobol_indices(_chain, PARAMS, n_base=2048, seed=3)
    assert s["sum_first_order"] < 0.95
    assert s["interaction_share"] > 0.15
    assert all(r["total_effect"] >= r["first_order"] - 0.05 for r in s["indices"])


def test_a_parameter_the_function_ignores_has_near_zero_total_effect():
    s = sobol_indices(lambda p: p["a"] * p["b"], PARAMS, n_base=2048, seed=1)
    c = next(r for r in s["indices"] if r["parameter"] == "c")
    assert abs(c["total_effect"]) < 0.05


def test_an_empty_parameter_range_is_refused():
    with pytest.raises(SensitivityError, match="should be a constant"):
        Parameter("x", 1.0, 1.0)


def test_a_single_parameter_decomposition_is_refused():
    with pytest.raises(SensitivityError, match="first-order index is 1"):
        sobol_indices(_chain, PARAMS[:1])


def test_a_constant_output_has_no_decomposition():
    with pytest.raises(SensitivityError, match="zero variance"):
        sobol_indices(lambda p: 1.0, PARAMS, n_base=128)


def test_too_few_samples_is_refused():
    with pytest.raises(SensitivityError, match="dominated by noise"):
        sobol_indices(_chain, PARAMS, n_base=8)


# ---- the join that makes sensitivity actionable ---------------------------
def test_the_measurement_priority_is_the_intersection_not_the_ranking():
    """The quadrant most often acted on by mistake: a sensitive parameter that is
    ALREADY identifiable needs no new assay, however high its index."""
    def chain(p):
        return (p["organ_fraction"] * p["f_parenchymal"] * p["phi_intact"]
                * p["k_escape"])
    params = [Parameter("organ_fraction", 0.05, 0.60), Parameter("f_parenchymal", 0.1, 0.8),
              Parameter("phi_intact", 0.2, 0.95), Parameter("k_escape", 0.005, 0.04)]
    s = sobol_indices(chain, params, n_base=1024, seed=3)
    mp = measurement_priority(s, ORGAN_TO_CYTOSOL, candidates=CYTOSOL_CANDIDATES)
    top = s["indices"][0]["parameter"]
    assert top == "organ_fraction"
    quad = {r["parameter"]: r["quadrant"] for r in mp["rows"]}
    assert quad["organ_fraction"] == "already determined", (
        "the highest-effect parameter is identifiable and is NOT a priority")
    assert "f_parenchymal" in mp["priority"]
    assert all(r["closed_by"] for r in mp["rows"] if r["quadrant"] == "MEASURE")


def test_no_priority_is_reported_honestly():
    s = sobol_indices(_chain, PARAMS, n_base=512, seed=0)
    mp = measurement_priority(s, ORGAN_TO_CYTOSOL)
    assert mp["n_priority"] == 0
    assert "no parameter is both sensitive" in mp["verdict"]


# ---- closing the cycle ----------------------------------------------------
def _proposal(target="f_parenchymal", groups=12):
    cand = {c.name: c for c in CYTOSOL_CANDIDATES}
    return ExperimentSpec(
        "flow-sorted parenchymal fraction", (cand["flow_sorted_parenchymal"],),
        target, 0.30, groups, 4, 0.75, 0.35,
        Resource(animals=48, assay_runs=12, weeks=6.0),
        "closes the parenchymal fraction, which no whole-organ assay can")


def test_the_cycle_closes_and_re_enters_at_phase_5():
    """A clinical surprise does not restart the programme; it enters at the
    inference phase, where it is compared against rival explanations."""
    out = close_cycle(
        observation="biomarker trajectory outside the predicted range",
        n_patients=29, n_observations=215,
        confounders_assessed=["differential exposure", "assay variation", "adherence"],
        provenance_cleared=True, hypothesis_parameter="f_parenchymal",
        model=ORGAN_TO_CYTOSOL, proposed=_proposal(groups=60))
    assert out["cycle_closed"] and out["reenters_at"] == 5
    assert "never Phase 1" in out["reentry_rationale"]
    assert len(out["gates"]) >= 3


def test_the_clinical_gate_fires_first():
    with pytest.raises(CycleError, match="gate 1-2"):
        close_cycle(observation="x", n_patients=29, n_observations=215,
                    confounders_assessed=["only one"], provenance_cleared=True,
                    hypothesis_parameter="f_parenchymal", model=ORGAN_TO_CYTOSOL,
                    proposed=_proposal())


def test_a_hypothesis_naming_no_modelled_quantity_is_refused():
    with pytest.raises(CycleError, match="needs a different model first"):
        close_cycle(observation="x", n_patients=29, n_observations=215,
                    confounders_assessed=["a", "b", "c"], provenance_cleared=True,
                    hypothesis_parameter="patient_mood", model=ORGAN_TO_CYTOSOL,
                    proposed=_proposal())


def test_an_experiment_measuring_something_else_is_not_a_test_of_the_hypothesis():
    with pytest.raises(CycleError, match="is not a test of this hypothesis"):
        close_cycle(observation="x", n_patients=29, n_observations=215,
                    confounders_assessed=["a", "b", "c"], provenance_cleared=True,
                    hypothesis_parameter="k_escape", model=ORGAN_TO_CYTOSOL,
                    proposed=_proposal(target="f_parenchymal"))


def test_an_underpowered_proposal_stops_the_cycle_with_the_power_gate():
    with pytest.raises(CycleError, match="power"):
        close_cycle(observation="x", n_patients=29, n_observations=215,
                    confounders_assessed=["a", "b", "c"], provenance_cleared=True,
                    hypothesis_parameter="f_parenchymal", model=ORGAN_TO_CYTOSOL,
                    proposed=_proposal(groups=3))


def test_an_insensitive_parameter_stops_the_cycle():
    """Closing an unknown the output does not depend on is a cheaper mistake than
    the alternatives, but still a mistake."""
    fake = {"indices": [{"parameter": "f_parenchymal", "total_effect": 0.001}]}
    with pytest.raises(CycleError, match="gate 6"):
        close_cycle(observation="x", n_patients=29, n_observations=215,
                    confounders_assessed=["a", "b", "c"], provenance_cleared=True,
                    hypothesis_parameter="f_parenchymal", model=ORGAN_TO_CYTOSOL,
                    proposed=_proposal(groups=60), sensitivity=fake)


def test_a_parameter_absent_from_the_sensitivity_analysis_is_not_assessed():
    out = close_cycle(observation="x", n_patients=29, n_observations=215,
                      confounders_assessed=["a", "b", "c"], provenance_cleared=True,
                      hypothesis_parameter="f_parenchymal", model=ORGAN_TO_CYTOSOL,
                      proposed=_proposal(groups=60), sensitivity={"indices": []})
    g = next(x for x in out["gates"] if "depends on" in x["gate"])
    assert g["passed"] is None and "not assessed" in g["detail"]


# ---- the CLI --------------------------------------------------------------
@pytest.mark.parametrize("argv", [
    ["vision"], ["scope"], ["scope", "--anchors", "3"], ["identifiability"],
    ["sensitivity", "--n-base", "128"],
])
def test_every_cli_subcommand_exits_zero(argv, capsys):
    from atcs.cli import main
    assert main(argv) == 0
    assert capsys.readouterr().out.strip()


def test_the_benchmark_subcommand_labels_its_substrate(capsys):
    from atcs.cli import main
    assert main(["benchmark", "--studies", "40", "--splits", "3"]) == 0
    out = capsys.readouterr().out
    assert "measures_performance=False" in out
    assert "exercises the PROTOCOL" in out


def test_no_module_imports_from_the_sibling_package():
    """Two cross-repository references slipped in during development -- a registry
    and a vision function that live in `mesoloop`. Both were caught by running the
    code rather than by reading it. This catches the next one at import time.
    """
    import pkgutil
    import re
    from pathlib import Path
    import atcs
    root = Path(atcs.__file__).parent
    offenders = []
    for p in sorted(root.glob("*.py")):
        src = p.read_text()
        for m in re.finditer(r"^\s*(?:from|import)\s+(mesoloop[\w.]*)", src,
                             re.MULTILINE):
            offenders.append(f"{p.name}: {m.group(1)}")
    assert not offenders, (
        f"cross-package imports would break a standalone install: {offenders}")


def test_every_declared_module_actually_imports():
    """__all__ naming a module that does not import is the same class of defect as a
    contract naming a checker that does not resolve."""
    import importlib
    import atcs
    for name in atcs.__all__:
        importlib.import_module(f"atcs.{name}")
