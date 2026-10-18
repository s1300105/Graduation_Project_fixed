#!/usr/bin/env python
"""Paired bootstrap comparison of two groups of runs on the same test set.

    python scripts/compare_runs.py --a "runs/hybrid_bs16_lr1e-04_br0.30_seed*" \
                                   --b "runs/codebert_bs16_lr1e-04_br0.30_seed*" [--n_boot 2000] [--threshold tuned|0.5]

Each run directory must contain test_preds.npz (probs, labels, idx) and metrics.json
(written by run.py). Samples are aligned across runs by ``idx``; every run is
scored at its own validation-tuned threshold (or 0.5). The statistic is the
difference of the seed-averaged F1 (group A minus group B); its 95% interval
comes from resampling test samples with replacement (paired, i.e. the same
resample is applied to every run).
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np


def load_group(pattern, threshold):
    runs = []
    for d in sorted(glob.glob(pattern)):
        d = Path(d)
        if not (d / "test_preds.npz").exists():
            continue
        z = np.load(d / "test_preds.npz")
        m = json.loads((d / "metrics.json").read_text())
        t = m["test_tuned"]["threshold"] if threshold == "tuned" else float(threshold)
        runs.append({"name": d.name, "probs": z["probs"], "labels": z["labels"], "idx": z["idx"], "thr": t})
    if not runs:
        raise SystemExit(f"no runs match {pattern}")
    return runs


def f1_from_counts(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else 2 * tp / denom


def align(groups):
    common = None
    for g in groups:
        for r in g:
            s = set(r["idx"].tolist())
            common = s if common is None else common & s
    common = np.array(sorted(common))
    for g in groups:
        for r in g:
            pos = {v: i for i, v in enumerate(r["idx"].tolist())}
            sel = np.array([pos[v] for v in common])
            r["probs"], r["labels"], r["idx"] = r["probs"][sel], r["labels"][sel], r["idx"][sel]
    return common


def group_f1(group, sample_idx=None):
    vals = []
    for r in group:
        p = r["probs"] >= r["thr"]
        y = r["labels"].astype(bool)
        if sample_idx is not None:
            p, y = p[sample_idx], y[sample_idx]
        tp = int((p & y).sum()); fp = int((p & ~y).sum()); fn = int((~p & y).sum())
        vals.append(f1_from_counts(tp, fp, fn))
    return float(np.mean(vals)), vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="glob of run dirs (group A, e.g. hybrid seeds)")
    ap.add_argument("--b", required=True, help="glob of run dirs (group B, e.g. baseline seeds)")
    ap.add_argument("--threshold", default="tuned")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    A, B = load_group(args.a, args.threshold), load_group(args.b, args.threshold)
    common = align([A, B])
    n = len(common)
    fa, fa_each = group_f1(A)
    fb, fb_each = group_f1(B)
    print(f"test samples aligned by idx: {n} (positives {int(A[0]['labels'].sum())})")
    print(f"A ({len(A)} runs): mean F1 {fa:.4f} +/- {np.std(fa_each, ddof=1) if len(fa_each) > 1 else 0:.4f}  {[round(v, 4) for v in fa_each]}")
    print(f"B ({len(B)} runs): mean F1 {fb:.4f} +/- {np.std(fb_each, ddof=1) if len(fb_each) > 1 else 0:.4f}  {[round(v, 4) for v in fb_each]}")
    rng = np.random.default_rng(args.seed)
    diffs = np.empty(args.n_boot)
    for i in range(args.n_boot):
        s = rng.integers(0, n, n)
        diffs[i] = group_f1(A, s)[0] - group_f1(B, s)[0]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"A - B: {fa - fb:+.4f}  95% paired-bootstrap CI [{lo:+.4f}, {hi:+.4f}]  P(A>B)={float((diffs > 0).mean()):.3f}")
    if lo <= 0 <= hi:
        print("=> the difference is not distinguishable from zero at this sample size / number of seeds")


if __name__ == "__main__":
    main()
