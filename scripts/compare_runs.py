#!/usr/bin/env python
"""Paired bootstrap comparison of two groups of runs on the same test set.

    python scripts/compare_runs.py --a "runs/hybrid_bs16_lr1e-04_br0.30_seed*" \
                                   --b "runs/codebert_bs16_lr1e-04_br0.30_seed*" [--n_boot 2000] [--threshold tuned|0.5]
                                   [--metric f1|macro_f1|accuracy]

Each run directory must contain test_preds.npz (probs, labels, idx) and metrics.json
(written by run.py). Samples are aligned across runs by ``idx``; every run is
scored at its own validation-tuned threshold (or 0.5). The statistic is the
difference of the seed-averaged metric (group A minus group B); its 95% interval
comes from resampling test samples with replacement (paired, i.e. the same
resample is applied to every run).

Metrics: ``f1`` (default) is the positive-class F1 of a binary run (1-D probs,
thresholded). Multi-class runs store probs of shape [N, C]; their predictions
are the argmax (no threshold) and the metric must be ``macro_f1`` (mean of the
per-class F1 over all C classes) or ``accuracy``, both also usable on binary
runs (thresholded predictions, C = 2).
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

METRICS = ("f1", "macro_f1", "accuracy", "auc")


def load_group(pattern, threshold):
    runs = []
    for d in sorted(glob.glob(pattern)):
        d = Path(d)
        if not (d / "test_preds.npz").exists():
            continue
        z = np.load(d / "test_preds.npz")
        m = json.loads((d / "metrics.json").read_text())
        probs = z["probs"]
        if probs.ndim == 2:
            t = None                                  # multi-class: argmax, no threshold
        else:
            t = m["test_tuned"]["threshold"] if threshold == "tuned" else float(threshold)
        runs.append({"name": d.name, "probs": probs, "labels": z["labels"], "idx": z["idx"], "thr": t})
    if not runs:
        raise SystemExit(f"no runs match {pattern}")
    return runs


def f1_from_counts(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else 2 * tp / denom


def align(groups):
    """Restrict every run to the test items (idx) all runs share, in one order.

    Refuses inputs on which a paired comparison is meaningless: duplicate or
    missing (-1) idx within a run, no common idx, or runs that disagree on the
    label of a shared idx (idx are renumbered from 0 by every preparation
    script, so runs of two different datasets overlap in idx but not in labels)."""
    common = None
    for g in groups:
        for r in g:
            idx = r["idx"].tolist()
            if len(set(idx)) != len(idx) or any(v < 0 for v in idx):
                raise SystemExit(f"{r['name']}: test_preds.npz has duplicate or missing (-1) idx; "
                                 f"regenerate the inputs with an idx column before comparing")
            s = set(idx)
            common = s if common is None else common & s
    if not common:
        raise SystemExit("the groups share no test idx (different datasets / data_root?)")
    common = np.array(sorted(common), dtype=np.int64)
    sizes = {}
    for g in groups:
        for r in g:
            sizes[r["name"]] = len(r["idx"])
            pos = {v: i for i, v in enumerate(r["idx"].tolist())}
            sel = np.array([pos[v] for v in common], dtype=np.int64)
            r["probs"], r["labels"], r["idx"] = r["probs"][sel], r["labels"][sel], r["idx"][sel]
    ref = groups[0][0]
    for g in groups:
        for r in g:
            if not np.array_equal(r["labels"], ref["labels"]):
                bad = int((r["labels"] != ref["labels"]).sum())
                raise SystemExit(f"{r['name']} and {ref['name']} disagree on the label of {bad} of {len(common)} shared "
                                 f"test idx: they were not evaluated on the same test set")
    partial = {k: v for k, v in sizes.items() if v != len(common)}
    if partial:
        print(f"[WARN] only {len(common)} test idx are shared; full test sizes: {partial}")
    return common


def num_classes_of(run):
    return int(run["probs"].shape[1]) if run["probs"].ndim == 2 else 2


def predictions(run):
    """Integer class predictions: argmax for [N, C] probs, thresholded for 1-D probs."""
    if run["probs"].ndim == 2:
        return run["probs"].argmax(axis=1)
    return (run["probs"] >= run["thr"]).astype(int)


def group_f1(group, sample_idx=None):
    """Positive-class F1 (binary runs), seed-averaged. Unchanged from the original script."""
    vals = []
    for r in group:
        p = r["probs"] >= r["thr"]
        y = r["labels"].astype(bool)
        if sample_idx is not None:
            p, y = p[sample_idx], y[sample_idx]
        tp = int((p & y).sum()); fp = int((p & ~y).sum()); fn = int((~p & y).sum())
        vals.append(f1_from_counts(tp, fp, fn))
    return float(np.mean(vals)), vals


def macro_f1(pred, y, C):
    """Mean over all C classes of the per-class F1 (a class absent from both
    pred and y contributes 0, as run.py's metrics_multiclass does)."""
    tp = np.bincount(y[pred == y], minlength=C)[:C]
    pred_n = np.bincount(pred, minlength=C)[:C]
    true_n = np.bincount(y, minlength=C)[:C]
    f1s = [f1_from_counts(int(tp[c]), int(pred_n[c] - tp[c]), int(true_n[c] - tp[c])) for c in range(C)]
    return float(np.mean(f1s))


def roc_auc(scores, y):
    """ROC-AUC by the rank statistic (ties get average ranks); None if y holds one class."""
    y = np.asarray(y).astype(bool)
    n_pos = int(y.sum()); n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    s_sorted = np.asarray(scores)[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def group_metric(group, metric, sample_idx=None):
    """Seed-averaged ``metric`` of a group, optionally on the resampled items ``sample_idx``."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric}; choose from {METRICS}")
    if metric == "f1":
        for r in group:
            if r["probs"].ndim == 2:
                raise SystemExit(f"--metric f1 is the positive-class F1 of a binary task, but {r['name']} holds "
                                 f"[N, {num_classes_of(r)}] multi-class probabilities; use --metric macro_f1 or accuracy")
        return group_f1(group, sample_idx)
    if metric == "auc":
        vals = []
        for r in group:
            if r["probs"].ndim == 2:
                raise SystemExit(f"--metric auc is the ROC-AUC of P(y=1) of a binary task; {r['name']} is multi-class")
            p, y = r["probs"], r["labels"]
            if sample_idx is not None:
                p, y = p[sample_idx], y[sample_idx]
            a = roc_auc(p, y)
            vals.append(0.5 if a is None else a)       # a resample with one class carries no ranking information
        return float(np.mean(vals)), vals
    vals = []
    for r in group:
        pred = predictions(r)
        y = r["labels"].astype(int)
        if sample_idx is not None:
            pred, y = pred[sample_idx], y[sample_idx]
        if metric == "accuracy":
            vals.append(float((pred == y).mean()) if len(y) else 0.0)
        else:
            vals.append(macro_f1(pred, y, num_classes_of(r)))
    return float(np.mean(vals)), vals


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="glob of run dirs (group A, e.g. hybrid seeds)")
    ap.add_argument("--b", required=True, help="glob of run dirs (group B, e.g. baseline seeds)")
    ap.add_argument("--threshold", default="tuned")
    ap.add_argument("--metric", choices=METRICS, default="f1",
                    help="f1 = positive-class F1 (binary only); auc = ROC-AUC of P(y=1) (binary only, threshold-free; "
                         "the MoleculeNet metric); macro_f1 / accuracy work for binary and multi-class runs")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    A, B = load_group(args.a, args.threshold), load_group(args.b, args.threshold)
    common = align([A, B])
    n = len(common)
    classes = {num_classes_of(r) for r in A + B}
    if len(classes) > 1:
        raise SystemExit(f"runs disagree on the number of classes: {sorted(classes)}")
    C = classes.pop()
    fa, fa_each = group_metric(A, args.metric)
    fb, fb_each = group_metric(B, args.metric)
    label = {"f1": "F1", "macro_f1": "macro-F1", "accuracy": "accuracy", "auc": "ROC-AUC"}[args.metric]
    if C == 2:
        print(f"test samples aligned by idx: {n} (positives {int(A[0]['labels'].sum())})")
    else:
        print(f"test samples aligned by idx: {n} ({C} classes, argmax predictions)")
    print(f"A ({len(A)} runs): mean {label} {fa:.4f} +/- {np.std(fa_each, ddof=1) if len(fa_each) > 1 else 0:.4f}  {[round(v, 4) for v in fa_each]}")
    print(f"B ({len(B)} runs): mean {label} {fb:.4f} +/- {np.std(fb_each, ddof=1) if len(fb_each) > 1 else 0:.4f}  {[round(v, 4) for v in fb_each]}")
    rng = np.random.default_rng(args.seed)
    diffs = np.empty(args.n_boot)
    for i in range(args.n_boot):
        s = rng.integers(0, n, n)
        diffs[i] = group_metric(A, args.metric, s)[0] - group_metric(B, args.metric, s)[0]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"A - B: {fa - fb:+.4f}  95% paired-bootstrap CI [{lo:+.4f}, {hi:+.4f}]  P(A>B)={float((diffs > 0).mean()):.3f}")
    if lo <= 0 <= hi:
        print("=> the difference is not distinguishable from zero at this sample size / number of seeds")


if __name__ == "__main__":
    main()
