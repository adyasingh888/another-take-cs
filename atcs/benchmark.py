"""
The evaluation harness: split protocols, baselines, and the anchored remedy, runnable.

WHY THIS MODULE HAD TO EXIST
----------------------------
Every other module in this package makes a claim checkable. This one makes the
programme's CENTRAL claim runnable: that a reference measurement from the held-out
study collapses the predictive interval, and that nothing else does.

Until now that claim lived in `calibrate` (the estimator), `delta` (the algebra) and
the README (the prose). None of those is a demonstration. A reviewer asked to
believe "anchoring works" is entitled to run it.

WHAT IT RUNS ON, AND WHY THAT IS STATED LOUDLY
----------------------------------------------
On the GENERATED table from `atcs.datasets`, whose rows carry `is_synthetic = True`.
Every result this module returns therefore carries `exercises_protocol = True` and
`measures_performance = False`, and `scorecard()` refuses to drop those keys.

That is not a disclaimer, it is the correct description. The generated table
reproduces the reported variance structure, so the harness genuinely exercises the
protocol against the clustering it will meet -- and the protocol is what is being
evaluated here. What it cannot do is tell you how well any model predicts real
biodistribution, because that requires the real substrate, and inventing it is what
the 20 October brief forbids.

THE PROTOCOLS, AND WHY THE COMPARISON IS THE POINT
--------------------------------------------------
  random      rows split at random -- the published practice, and the one that
              leaves a same-study sibling in training for most test rows
  grouped     whole studies held out -- what the deployment question actually is
  anchored    whole studies held out, but ONE measurement from the held-out study
              is revealed, and used to estimate that study's offset

The gap between the first two is the leak. The gap between the second and third is
the remedy. Reporting any one alone is how this literature produced numbers nobody
could reproduce.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd


class BenchmarkError(RuntimeError):
    """A benchmark was configured in a way whose result could not be interpreted."""


# ----------------------------------------------------------------- protocols
def split_random(groups: np.ndarray, *, n_splits: int = 5, seed: int = 0):
    """Rows split at random. Yields (train_idx, test_idx)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(groups))
    for fold in np.array_split(idx, n_splits):
        test = np.sort(fold)
        train = np.setdiff1d(np.arange(len(groups)), test)
        yield train, test


def split_grouped(groups: np.ndarray, *, n_splits: int = 5, seed: int = 0):
    """Whole groups held out."""
    rng = np.random.default_rng(seed)
    uniq = rng.permutation(np.unique(groups))
    for chunk in np.array_split(uniq, n_splits):
        test = np.where(np.isin(groups, chunk))[0]
        train = np.setdiff1d(np.arange(len(groups)), test)
        yield train, test


def sibling_leakage(groups: np.ndarray, *, n_splits: int = 5, seed: int = 0) -> dict:
    """What fraction of test rows share a group with a training row?

    The quantity that explains why random and grouped splits disagree. Computed
    rather than asserted, because it is the measurable form of the leak.
    """
    frac = []
    for train, test in split_random(groups, n_splits=n_splits, seed=seed):
        seen = set(groups[train])
        frac.append(float(np.mean([g in seen for g in groups[test]])))
    return {"mean_sibling_fraction": round(float(np.mean(frac)), 4),
            "sd": round(float(np.std(frac)), 4), "n_splits": n_splits,
            "note": ("the fraction of test rows with a same-group row in training. "
                     "Under a random split this is near 1 for a clustered corpus, "
                     "which is why a random-split score answers a question nobody "
                     "asked")}


# ------------------------------------------------------------------ baselines
def predict_train_mean(y_train, X_train, X_test, **_) -> np.ndarray:
    """The only baseline that cannot be beaten by accident."""
    return np.full(len(X_test), float(np.mean(y_train)))


def predict_ridge(y_train, X_train, X_test, *, alpha: float = 1.0, **_) -> np.ndarray:
    """Closed-form ridge, so the harness has no sklearn dependency to pin."""
    Xtr = np.c_[np.ones(len(X_train)), np.asarray(X_train, float)]
    Xte = np.c_[np.ones(len(X_test)), np.asarray(X_test, float)]
    n_f = Xtr.shape[1]
    pen = alpha * np.eye(n_f)
    pen[0, 0] = 0.0                       # never penalise the intercept
    beta = np.linalg.solve(Xtr.T @ Xtr + pen, Xtr.T @ np.asarray(y_train, float))
    return Xte @ beta


#: The arms every scorecard runs. A result without the train-mean arm is not
#: interpretable, so it is not optional.
BASELINES: dict[str, Callable] = {
    "train_mean": predict_train_mean,
    "ridge_descriptors": predict_ridge,
}


# -------------------------------------------------------------------- metrics
def _r2(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def point_metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = y - p
    return {"r2": round(_r2(y, p), 4),
            "mae": round(float(np.mean(np.abs(err))), 4),
            "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4),
            "median_fold_error": round(float(np.median(10 ** np.abs(err))), 3),
            "within_2fold": round(float(np.mean(np.abs(err) <= math.log10(2))), 4),
            "n": int(y.size)}


# ------------------------------------------------------------- the remedy
def anchored_predict(y_train, X_train, X_test, *, groups_test: np.ndarray,
                     y_test: np.ndarray, base: Callable = predict_ridge,
                     n_anchors: int = 1, seed: int = 0,
                     anchor_mask_out: list | None = None) -> np.ndarray:
    """Predict with ONE revealed measurement per held-out group.

    The anchor is a row of the test group whose true value is revealed; the group's
    offset is estimated as the residual on that row and added to every other
    prediction in the group.

    Two things this does NOT do, deliberately:

      - it does not score the anchor rows themselves. Including a row whose answer
        was revealed in the score is the leak this whole package exists to catch,
        and it would inflate the result by exactly the amount the anchor helps.
        `anchor_mask_out` returns the mask so the caller can verify the exclusion.
      - it does not choose the anchor to be favourable. Anchors are drawn at
        random within each group with a fixed seed, because choosing the anchor
        that helps most is a form of test-set selection.
    """
    rng = np.random.default_rng(seed)
    base_pred = np.asarray(base(y_train, X_train, X_test), float)
    y_test = np.asarray(y_test, float)
    out = base_pred.copy()
    is_anchor = np.zeros(len(y_test), bool)
    for g in np.unique(groups_test):
        where = np.where(groups_test == g)[0]
        k = min(n_anchors, max(len(where) - 1, 0))
        if k == 0:
            continue                      # a singleton group has no spare row
        picked = rng.choice(where, size=k, replace=False)
        is_anchor[picked] = True
        offset = float(np.mean(y_test[picked] - base_pred[picked]))
        rest = np.setdiff1d(where, picked)
        out[rest] = base_pred[rest] + offset
    if anchor_mask_out is not None:
        anchor_mask_out.append(is_anchor)
    return out


# ------------------------------------------------------------------ scorecard
def scorecard(df: pd.DataFrame, target: str, features: Sequence[str],
              group_col: str = "study_id", *, n_splits: int = 5, seed: int = 0,
              n_anchors: int = 1) -> pd.DataFrame:
    """Every arm under every protocol, in one table.

    Returns a frame carrying `exercises_protocol` and `measures_performance` on
    every row; the second is False whenever the input contains synthetic rows, and
    the columns are not droppable because a scorecard read without them says
    something this harness cannot support.
    """
    missing = [c for c in [target, group_col, *features] if c not in df.columns]
    if missing:
        raise BenchmarkError(f"columns absent from the frame: {missing}")
    synthetic = bool(df.get("is_synthetic", pd.Series([False])).any())
    y = df[target].to_numpy(float)
    X = df[list(features)].to_numpy(float)
    g = df[group_col].astype(str).to_numpy()

    rows = []
    for proto_name, proto in (("random", split_random), ("grouped", split_grouped)):
        for arm, fn in BASELINES.items():
            preds, truths = [], []
            for train, test in proto(g, n_splits=n_splits, seed=seed):
                preds.append(fn(y[train], X[train], X[test]))
                truths.append(y[test])
            rows.append({"protocol": proto_name, "arm": arm,
                         **point_metrics(np.concatenate(truths),
                                         np.concatenate(preds))})

    # the remedy, scored ONLY on the non-anchor rows
    preds, truths, n_anchor_rows = [], [], 0
    for train, test in split_grouped(g, n_splits=n_splits, seed=seed):
        mask: list = []
        p = anchored_predict(y[train], X[train], X[test], groups_test=g[test],
                             y_test=y[test], n_anchors=n_anchors, seed=seed,
                             anchor_mask_out=mask)
        keep = ~mask[0]
        n_anchor_rows += int(mask[0].sum())
        preds.append(p[keep]); truths.append(y[test][keep])
    rows.append({"protocol": f"grouped + {n_anchors} anchor",
                 "arm": "ridge_descriptors",
                 **point_metrics(np.concatenate(truths), np.concatenate(preds)),
                 "anchor_rows_excluded_from_score": n_anchor_rows})

    out = pd.DataFrame(rows)
    out["exercises_protocol"] = True
    out["measures_performance"] = not synthetic
    out["substrate"] = "generated" if synthetic else "measured"
    return out


def leak_and_remedy(sc: pd.DataFrame, arm: str = "ridge_descriptors") -> dict:
    """The two gaps that matter, read off a scorecard.

    Reported as a pair, because either alone is misleading: the leak alone invites
    "so the model does not work", and the remedy alone invites "so the model works".
    The honest statement needs both.
    """
    def _get(proto: str) -> float:
        m = sc[(sc.arm == arm) & (sc.protocol == proto)]
        if m.empty:
            raise BenchmarkError(f"no {arm!r} row for protocol {proto!r}")
        return float(m.r2.iloc[0])

    anchored_proto = next((p for p in sc.protocol if "anchor" in p), None)
    if anchored_proto is None:
        raise BenchmarkError("the scorecard contains no anchored protocol")
    rnd, grp, anc = _get("random"), _get("grouped"), _get(anchored_proto)
    base = float(sc[(sc.arm == "train_mean") & (sc.protocol == "grouped")].r2.iloc[0])
    return {"random_r2": rnd, "grouped_r2": grp, "anchored_r2": anc,
            "grouped_train_mean_r2": base,
            "leak_gap": round(rnd - grp, 4),
            "remedy_gap": round(anc - grp, 4),
            "beats_train_mean_when_anchored": anc > base,
            "verdict": (f"random {rnd:.3f} -> grouped {grp:.3f} is the leak "
                        f"({rnd - grp:+.3f}); grouped -> anchored {anc:.3f} is the "
                        f"remedy ({anc - grp:+.3f}). Reporting either alone is "
                        f"misleading: the leak alone invites 'the model does not "
                        f"work', the remedy alone invites 'the model works'."),
            "exercises_protocol": True,
            "measures_performance": bool(sc.measures_performance.all())}
