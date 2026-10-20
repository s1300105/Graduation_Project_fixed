#!/usr/bin/env python
"""Non-neural floor for the chemistry tasks: ECFP4 (Morgan radius 2, 2048 bits) + random forest.

    python scripts/fingerprint_baseline.py --data_root data_bace --dataset ogb_molbace \
        --runs_dir runs_bace --run_prefix bace_ --seeds 0 1 2 3 4

Reads the same jsonl splits as run.py (<data_root>/cleaned_data/<dataset>_{train,valid,test}.jsonl),
trains one forest per seed on train, and writes a run directory per seed in run.py's format
(metrics.json, val_preds.npz, test_preds.npz, test_preds_final.npz, args.json) so that
scripts/compare_runs.py can pair it with the neural arms on the same test molecules. Rows whose
SMILES RDKit cannot parse are dropped exactly as run.py --molgraph drops them.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fingerprints(smiles, radius=2, n_bits=2048):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out, ok = [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None or m.GetNumAtoms() == 0:
            ok.append(False)
            continue
        out.append(gen.GetFingerprintAsNumPy(m).astype(np.uint8))
        ok.append(True)
    return (np.stack(out) if out else np.zeros((0, n_bits), np.uint8)), np.array(ok)


def load_split(data_root: Path, dataset: str, split: str):
    df = pd.read_json(data_root / "cleaned_data" / f"{dataset}_{split}.jsonl", lines=True)
    X, ok = fingerprints(df["func"].tolist())
    df = df[ok]
    return X, df["target"].to_numpy().astype(int), df["idx"].to_numpy().astype(int)


def main(argv=None):
    import run as R
    from sklearn.ensemble import RandomForestClassifier
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--run_prefix", default="")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n_estimators", type=int, default=500)
    ap.add_argument("--n_jobs", type=int, default=-1)
    args = ap.parse_args(argv)
    root = Path(args.data_root)
    Xtr, ytr, _ = load_split(root, args.dataset, "train")
    Xva, yva, iva = load_split(root, args.dataset, "valid")
    Xte, yte, ite = load_split(root, args.dataset, "test")
    for seed in args.seeds:
        rf = RandomForestClassifier(n_estimators=args.n_estimators, random_state=seed, n_jobs=args.n_jobs)
        rf.fit(Xtr, ytr)
        pv, pt = rf.predict_proba(Xva)[:, 1], rf.predict_proba(Xte)[:, 1]
        d = Path(args.runs_dir) / f"{args.run_prefix}ecfp_rf_seed{seed}"
        d.mkdir(parents=True, exist_ok=True)
        vt = R.best_threshold(pv, yva)
        out = {"val_05": R.metrics_at(pv, yva, 0.5), "val_tuned": vt,
               "test_05": R.metrics_at(pt, yte, 0.5), "test_tuned": R.metrics_at(pt, yte, vt["threshold"]),
               "num_classes": 2, "select_on": "none (no epochs)", "best_val_selection_metric": None}
        out["val_auc"], out["val_pr_auc"] = R.binary_auc(pv, yva)
        out["test_auc"], out["test_pr_auc"] = R.binary_auc(pt, yte)
        out["final_epoch"] = {"test_auc": out["test_auc"], "test_pr_auc": out["test_pr_auc"]}
        np.savez(d / "val_preds.npz", probs=pv, labels=yva, idx=iva)
        np.savez(d / "test_preds.npz", probs=pt, labels=yte, idx=ite)
        np.savez(d / "test_preds_final.npz", probs=pt, labels=yte, idx=ite)
        (d / "metrics.json").write_text(json.dumps(out, indent=1))
        (d / "args.json").write_text(json.dumps({"model": "ECFP4(2048)+RandomForest", "n_estimators": args.n_estimators,
                                                 "seed": seed, "dataset": args.dataset,
                                                 "data_root": str(root.resolve())}, indent=1))
        print(f"[DONE] {d.name}: test ROC-AUC {out['test_auc']:.4f}  PR-AUC {out['test_pr_auc']:.4f}  (val ROC {out['val_auc']:.4f})")


if __name__ == "__main__":
    main()
