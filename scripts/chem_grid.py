#!/usr/bin/env python
"""Run the pre-registered chemistry grid (docs/CHEM_PROTOCOL_ja.md) and summarise it.

    python scripts/chem_grid.py --root experiments/chem --datasets ogb_molbace ogb_molbbbp \
        --seeds 0 1 2 3 4 --workers 2 --threads 2          # this CPU box
    python scripts/chem_grid.py --root experiments/chem --datasets ogb_molhiv --seeds 0 1 2 3 4 5 6 7 8 9 \
        --workers 1 --extra "--epochs 100 --patience 100"  # GPU
    python scripts/chem_grid.py --root experiments/chem --summarise_only

Resumable: a (dataset, arm, seed) whose run directory already holds metrics.json is skipped, so
the same command continues an interrupted grid. Data are prepared on first use
(data/scripts/make_molnet_json.py, then run.py --molgraph) under <root>/<dataset>/.
Writes <root>/summary.json and <root>/summary.md (per-arm mean +/- sd of test ROC-AUC at the
best-validation epoch and at the final epoch, plus the paired / two-level bootstrap contrasts).
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
TEXT_MODEL = "DeepChem/ChemBERTa-77M-MTR"
RECIPE = ["--loss", "ce", "--select_on", "auc", "--epochs", "30", "--patience", "30", "--batch_sizes", "16",
          "--lrs", "1e-4", "--bert_lr_ratios", "0.3", "--eval_batch_size", "64", "--max_length", "256"]
# arm name -> run.py model arguments (A0, the fingerprint floor, is handled separately)
ARMS = {
    "graph_only": ["--model_type", "graph_only"],
    "graph_only-no_edges": ["--model_type", "graph_only", "--ablation", "no_edges"],
    "codebert": ["--model_type", "codebert"],
    "hybrid": ["--model_type", "hybrid"],
    "hybrid-no_edges": ["--model_type", "hybrid", "--ablation", "no_edges"],
    "hybrid-shuffle_edges": ["--model_type", "hybrid", "--ablation", "shuffle_edges"],
}
ARM_LABELS = {"ecfp_rf": "A0 ECFP4+RF", "codebert": "A1 text only", "graph_only": "A2 graph only",
              "graph_only-no_edges": "A2e graph, no edges", "hybrid": "A3 hybrid",
              "hybrid-shuffle_edges": "A3s hybrid, shuffled edges", "hybrid-no_edges": "A3e hybrid, no edges"}
CONTRASTS = [("hybrid", "codebert"), ("hybrid", "hybrid-no_edges"), ("hybrid", "hybrid-shuffle_edges"),
             ("graph_only", "graph_only-no_edges"), ("hybrid", "ecfp_rf"), ("codebert", "ecfp_rf"),
             ("hybrid", "graph_only")]


def run_dir_name(ds, arm, seed):
    prefix = f"{ds}_"
    return f"{prefix}ecfp_rf_seed{seed}" if arm == "ecfp_rf" else f"{prefix}{arm}_bs16_lr1.0e-04_br0.30_seed{seed}"


def prepare(root: Path, ds: str, log):
    data_root = root / ds
    if not (data_root / "input" / "manifest.json").exists():
        cleaned = data_root / "cleaned_data"
        subprocess.run([PY, "-I", str(ROOT / "data/scripts/make_molnet_json.py"), "--dataset", ds,
                        "--input", str(root / "raw"), "--download", "--out_dir", str(cleaned)],
                       check=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        subprocess.run([PY, str(ROOT / "run.py"), "--molgraph", "--dataset", ds, "--data_root", str(data_root),
                        "--text_model", TEXT_MODEL], check=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    return data_root


def job(root: Path, ds: str, arm: str, seed: int, threads: int, extra: list):
    data_root, runs = root / ds, root / ds / "runs"
    d = runs / run_dir_name(ds, arm, seed)
    if (d / "metrics.json").exists():
        return f"skip {d.name}"
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), TOKENIZERS_PARALLELISM="false")
    logf = root / "logs" / f"{d.name}.log"
    logf.parent.mkdir(parents=True, exist_ok=True)
    if arm == "ecfp_rf":
        cmd = [PY, str(ROOT / "scripts/fingerprint_baseline.py"), "--data_root", str(data_root), "--dataset", ds,
               "--runs_dir", str(runs), "--run_prefix", f"{ds}_", "--seeds", str(seed), "--n_jobs", str(threads)]
    else:
        cmd = [PY, "-u", str(ROOT / "run.py"), "--mode", "train", "--data_root", str(data_root), "--dataset", ds,
               "--runs_dir", str(runs), "--run_prefix", f"{ds}_", "--seeds", str(seed), "--overwrite"] \
              + ARMS[arm] + RECIPE + extra
    with logf.open("w") as fh:
        r = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return f"{'done' if r.returncode == 0 else 'FAILED(' + str(r.returncode) + ')'} {d.name}"


def summarise(root: Path, datasets, seeds):
    out = {"datasets": {}}
    md = ["# Chemistry grid summary", "",
          "Protocol: docs/CHEM_PROTOCOL_ja.md. Test ROC-AUC, mean +/- sd over seeds; 'best' = epoch with the best "
          "validation ROC-AUC, 'final' = last epoch.", ""]
    for ds in datasets:
        runs = root / ds / "runs"
        if not runs.exists():
            continue
        arms = {}
        for arm in ["ecfp_rf"] + list(ARMS):
            vals = []
            for seed in seeds:
                p = runs / run_dir_name(ds, arm, seed) / "metrics.json"
                if not p.exists():
                    continue
                m = json.loads(p.read_text())
                hist = runs / run_dir_name(ds, arm, seed) / "history.jsonl"
                gate = None
                if hist.exists():
                    h = [json.loads(l) for l in hist.read_text().splitlines() if l.strip()]
                    g = [r.get("gate_graph") for r in h if r.get("gate_graph") is not None]
                    gate = float(g[-1]) if g else None
                vals.append({"seed": seed, "test_auc": m.get("test_auc"), "test_pr_auc": m.get("test_pr_auc"),
                             "val_auc": m.get("val_auc"), "final_test_auc": (m.get("final_epoch") or {}).get("test_auc"),
                             "best_epoch": m.get("best_epoch"), "gate_graph_last_epoch": gate})
            if vals:
                def ms(k):
                    a = np.array([v[k] for v in vals if v[k] is not None], dtype=float)
                    return (float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0, int(len(a))) if len(a) else None
                arms[arm] = {"runs": vals, "test_auc": ms("test_auc"), "final_test_auc": ms("final_test_auc"),
                             "test_pr_auc": ms("test_pr_auc"), "val_auc": ms("val_auc")}
        contrasts = []
        for a, b in CONTRASTS:
            if a not in arms or b not in arms:
                continue
            for preds in ("best", "final"):
                ga = str(runs / (run_dir_name(ds, a, 0).replace("seed0", "seed*")))
                gb = str(runs / (run_dir_name(ds, b, 0).replace("seed0", "seed*")))
                r = subprocess.run([PY, str(ROOT / "scripts/compare_runs.py"), "--a", ga, "--b", gb, "--metric", "auc",
                                    "--preds", preds, "--seed_bootstrap", "--n_boot", "2000"],
                                   cwd=ROOT, capture_output=True, text=True)
                line = next((l for l in r.stdout.splitlines() if l.startswith("A - B")), r.stderr.strip()[-300:])
                contrasts.append({"a": a, "b": b, "preds": preds, "result": line})
        out["datasets"][ds] = {"arms": arms, "contrasts": contrasts}
        md += [f"## {ds}", "", "| arm | n | test ROC-AUC (best) | test ROC-AUC (final) | test PR-AUC | val ROC-AUC |",
               "|---|---|---|---|---|---|"]
        f = lambda t: "-" if t is None else f"{t[0]:.3f} +/- {t[1]:.3f}"
        for arm, v in arms.items():
            md.append(f"| {ARM_LABELS.get(arm, arm)} | {v['test_auc'][2] if v['test_auc'] else 0} | {f(v['test_auc'])} | "
                      f"{f(v['final_test_auc'])} | {f(v['test_pr_auc'])} | {f(v['val_auc'])} |")
        md += ["", "Contrasts (two-level bootstrap over seeds x test molecules, 2000 resamples):", ""]
        for c in contrasts:
            md.append(f"- {ARM_LABELS.get(c['a'], c['a'])} vs {ARM_LABELS.get(c['b'], c['b'])} ({c['preds']}): `{c['result']}`")
        md.append("")
    (root / "summary.json").write_text(json.dumps(out, indent=1))
    (root / "summary.md").write_text("\n".join(md) + "\n")
    return root / "summary.md"


def export(root: Path, datasets, dest: Path):
    """Copy what the summary and compare_runs need into a git-trackable directory: per run
    metrics.json, args.json, history (as JSON) and the val / test predictions as JSON. A lost
    container can then be restored with --restore_from without retraining finished runs."""
    n = 0
    for ds in datasets:
        runs = root / ds / "runs"
        if not runs.exists():
            continue
        for d in sorted(runs.iterdir()):
            if not (d / "metrics.json").exists():
                continue
            o = dest / ds / d.name
            o.mkdir(parents=True, exist_ok=True)
            for f in ("metrics.json", "args.json"):
                if (d / f).exists():
                    (o / f).write_text((d / f).read_text())
            if (d / "history.jsonl").exists():
                hist = [json.loads(l) for l in (d / "history.jsonl").read_text().splitlines() if l.strip()]
                (o / "history.json").write_text(json.dumps(hist))
            preds = {}
            for name in ("val_preds", "test_preds", "test_preds_final"):
                if (d / f"{name}.npz").exists():
                    z = np.load(d / f"{name}.npz")
                    preds[name] = {"probs": z["probs"].tolist(), "labels": z["labels"].tolist(), "idx": z["idx"].tolist()}
            (o / "preds.json").write_text(json.dumps(preds))
            n += 1
    return n


def restore(src: Path, root: Path):
    n = 0
    for o in sorted(src.glob("*/*")):
        if not (o / "metrics.json").exists():
            continue
        d = root / o.parent.name / "runs" / o.name
        if (d / "metrics.json").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        for f in ("metrics.json", "args.json"):
            if (o / f).exists():
                (d / f).write_text((o / f).read_text())
        if (o / "history.json").exists():
            (d / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in json.loads((o / "history.json").read_text())))
        for name, z in json.loads((o / "preds.json").read_text()).items():
            np.savez(d / f"{name}.npz", probs=np.asarray(z["probs"]), labels=np.asarray(z["labels"]), idx=np.asarray(z["idx"]))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/chem")
    ap.add_argument("--datasets", nargs="+", default=["ogb_molbace", "ogb_molbbbp"])
    ap.add_argument("--arms", nargs="+", default=["ecfp_rf"] + list(ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--extra", default="", help="extra run.py arguments, e.g. '--epochs 100 --patience 100'")
    ap.add_argument("--summarise_only", action="store_true")
    ap.add_argument("--export", default=None, help="after the grid, copy compact results to this (git-tracked) directory")
    ap.add_argument("--restore_from", default=None, help="recreate finished run directories from an --export directory first")
    args = ap.parse_args()
    root = (ROOT / args.root) if not Path(args.root).is_absolute() else Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    if args.restore_from:
        print(f"restored {restore(Path(args.restore_from), root)} run(s) from {args.restore_from}")
    if not args.summarise_only:
        with (root / "prepare.log").open("a") as log:
            for ds in args.datasets:
                prepare(root, ds, log)
        # seed-major order: every arm of seed 0 (both datasets) before seed 1, so partial grids are balanced
        jobs = [(ds, arm, seed) for seed in args.seeds for ds in args.datasets for arm in args.arms]
        extra = args.extra.split() if args.extra else []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(job, root, ds, arm, seed, args.threads, extra) for ds, arm, seed in jobs]
            for fu in as_completed(futs):
                print(fu.result(), flush=True)
    summary = summarise(root, args.datasets, args.seeds)
    print("summary:", summary)
    if args.export:
        dest = Path(args.export)
        dest.mkdir(parents=True, exist_ok=True)
        print(f"exported {export(root, args.datasets, dest)} run(s) to {dest}")
        for f in ("summary.md", "summary.json"):
            (dest / f).write_text((root / f).read_text())


if __name__ == "__main__":
    main()
