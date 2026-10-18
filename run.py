#!/usr/bin/env python
"""
Corrected training / evaluation driver for the BDCA (R-GCN + CodeBERT) thesis model.

Stages
  --cpg    : jsonl splits -> source files -> Joern CPG -> graph JSON -> <split>_<slice>_cpg.pkl
  --embed  : cpg.pkl -> PyG Data with frozen-CodeBERT node features -> <split>_<slice>_input.pkl
  --mode train : train one or more (config x seed) runs, each written to runs/<run_name>/
  --mode test  : evaluate a finished run on the test split

Changes relative to the thesis code are listed in README.md (mapped to audit ids).
"""
import argparse
import gc
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data import datamanager as dm                       # noqa: E402
from utils.process import cpg_generator as cpgproc              # noqa: E402
from utils.functions.cpg import parse_to_functions, RELATIONS   # noqa: E402
from utils.process.embeddings import nodes_to_input, NUM_RELATIONS  # noqa: E402
from utils.functions.parse import clean_gadget                  # noqa: E402
from models.LMGNN import build_model                            # noqa: E402


# ============================================================================
# reproducibility
# ============================================================================
def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


# ============================================================================
# loss
# ============================================================================
class FocalLoss(nn.Module):
    """alpha_t * (1 - p_t)^gamma * (-log p_t) with p_t from the UNWEIGHTED
    cross-entropy (thesis eq. 12). The original passed ``weight=alpha`` into
    cross_entropy before exponentiating, which made p_t = p_t^alpha_t (audit M01)."""

    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = (1.0 - pt) ** self.gamma * ce
        if self.alpha is not None:
            loss = self.alpha.to(logits.device)[targets] * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_criterion(name: str, class_counts, gamma: float, device):
    cnt0, cnt1 = class_counts
    total = max(1, cnt0 + cnt1)
    if cnt0 > 0 and cnt1 > 0:
        alpha = torch.tensor([0.5 / (cnt0 / total), 0.5 / (cnt1 / total)], dtype=torch.float32)
    else:
        alpha = torch.tensor([1.0, 1.0], dtype=torch.float32)
    if name == "focal":
        return FocalLoss(alpha=alpha, gamma=gamma).to(device), alpha
    if name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=alpha.to(device)), alpha
    if name == "ce":
        return nn.CrossEntropyLoss(), torch.tensor([1.0, 1.0])
    raise ValueError(name)


# ============================================================================
# data stream
# ============================================================================
def split_files(input_dir: Path, split: str):
    return sorted(str(p) for p in Path(input_dir).glob(f"{split}_*_input.pkl"))


class StreamGraphDataset(IterableDataset):
    """Streams PyG graphs from <split>_*_input.pkl.

    Fixed relative to the thesis code (audit M02): the shuffle RNG depends on
    (seed, epoch) via ``set_epoch`` and the file order is shuffled too, so every
    epoch sees a different order and -- with ``max_samples`` -- a different
    subset instead of the same prefix."""

    def __init__(self, files, *, buffer_size=256, shuffle=True, seed=0, max_samples=None):
        super().__init__()
        self.files = sorted(files)
        self.buffer_size = int(buffer_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.max_samples = int(max_samples) if max_samples else None
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed * 1_000_003 + self.epoch)
        files = list(self.files)
        if self.shuffle:
            rng.shuffle(files)
        yielded = 0
        buffer = []

        def flush():
            nonlocal yielded
            if self.shuffle:
                rng.shuffle(buffer)
            for item in buffer:
                if self.max_samples is not None and yielded >= self.max_samples:
                    return
                yielded += 1
                yield item

        for fn in files:
            if self.max_samples is not None and yielded >= self.max_samples:
                break
            df = pd.read_pickle(fn)
            try:
                has = set(df.columns)
                for row in df.itertuples(index=False):
                    g = getattr(row, "input", None)
                    if g is None:
                        continue
                    g.y = torch.tensor([int(getattr(row, "target", 0))], dtype=torch.long)
                    if "func" in has:
                        g.func = row.func
                    if "idx" in has and pd.notna(row.idx):
                        g.idx = int(row.idx)
                    buffer.append(g)
                    if len(buffer) >= self.buffer_size:
                        yield from flush()
                        buffer = []
            finally:
                del df
                gc.collect()
        if buffer:
            yield from flush()


def make_loader(input_dir, split, *, batch_size, shuffle, seed, max_samples=None, buffer_size=256):
    files = split_files(input_dir, split)
    if not files:
        return None, None
    ds = StreamGraphDataset(files, buffer_size=buffer_size, shuffle=shuffle, seed=seed, max_samples=max_samples)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=Batch.from_data_list, num_workers=0)
    return loader, ds


def count_targets(files):
    cnt0 = cnt1 = 0
    for fn in files:
        df = pd.read_pickle(fn)
        if "target" in df.columns:
            m = df["input"].notna() if "input" in df.columns else slice(None)
            vals = df.loc[m, "target"].astype("int64").clip(0, 1)
            cnt1 += int(vals.sum())
            cnt0 += int(len(vals) - vals.sum())
        del df
    return cnt0, cnt1


# ============================================================================
# evaluation
# ============================================================================
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels, idxs = [], [], []
    loss_sum, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        y = batch.y.view(-1).long()
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        n += y.numel()
        probs.extend(F.softmax(logits, dim=-1)[:, 1].cpu().tolist())
        labels.extend(y.cpu().tolist())
        bidx = getattr(batch, "idx", None)
        if bidx is None:
            idxs.extend([-1] * y.numel())
        else:
            idxs.extend([int(v) for v in (bidx.tolist() if torch.is_tensor(bidx) else bidx)])
    return np.asarray(probs), np.asarray(labels), np.asarray(idxs), loss_sum / max(1, n)


def metrics_at(probs, labels, t: float):
    preds = (probs >= t).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    return {
        "threshold": float(t),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        "n": int(len(labels)), "positives": int(labels.sum()),
    }


THRESHOLDS = np.linspace(0.05, 0.95, 19)


def best_threshold(probs, labels):
    best = None
    for t in THRESHOLDS:
        m = metrics_at(probs, labels, float(t))
        if best is None or m["f1"] > best["f1"]:
            best = m
    return best


# ============================================================================
# training
# ============================================================================
def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, clip_grad):
    model.train()
    loss_sum, nb, oom = 0.0, 0, 0
    gate_code = []
    params = [p for p in model.parameters() if p.requires_grad]
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        try:
            logits = model(batch)
            loss = criterion(logits, batch.y.view(-1).long())
            loss.backward()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                oom += 1
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise
        if clip_grad and clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(params, clip_grad)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss_sum += float(loss.item())
        nb += 1
        gate = getattr(model, "last_gate", None)
        if gate is not None:
            gate_code.append(float(gate[:, 0].mean().item()))
    return {
        "train_loss": loss_sum / max(1, nb),
        "batches": nb,
        "oom_skipped": oom,
        "gate_code": float(np.mean(gate_code)) if gate_code else None,
        "gate_graph": float(1.0 - np.mean(gate_code)) if gate_code else None,
    }


def make_optimizer(model, lr, bert_lr, weight_decay):
    decay, no_decay, bert_decay, bert_no_decay = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bert = "func_encoder.bert" in name
        is_nd = name.endswith(".bias") or "LayerNorm" in name or "layer_norm" in name.lower() or ".norm" in name
        (bert_no_decay if is_bert and is_nd else bert_decay if is_bert else no_decay if is_nd else decay).append(p)
    groups = []
    if decay: groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
    if no_decay: groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    if bert_decay: groups.append({"params": bert_decay, "lr": bert_lr, "weight_decay": weight_decay})
    if bert_no_decay: groups.append({"params": bert_no_decay, "lr": bert_lr, "weight_decay": 0.0})
    return torch.optim.AdamW(groups)


def run_name_for(args, batch_size, lr, ratio, seed):
    tag = args.model_type + ("" if args.ablation == "none" else f"-{args.ablation}")
    return f"{tag}_bs{batch_size}_lr{lr:.0e}_br{ratio:.2f}_seed{seed}"


def plot_curves(run_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    hist = [json.loads(l) for l in (run_dir / "history.jsonl").read_text().splitlines() if l.strip()]
    if not hist:
        return
    ep = [h["epoch"] for h in hist]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(ep, [h["train_loss"] for h in hist], label="train (criterion)")
    axes[0].plot(ep, [h["val_loss"] for h in hist], label="val (CE)")
    axes[0].set_title("loss"); axes[0].legend()
    axes[1].plot(ep, [h["val_f1_tuned"] for h in hist], label="val F1 (tuned thr)")
    axes[1].plot(ep, [h["val_f1_05"] for h in hist], label="val F1 @0.5")
    axes[1].plot(ep, [h["val_precision_tuned"] for h in hist], label="val P (tuned)")
    axes[1].plot(ep, [h["val_recall_tuned"] for h in hist], label="val R (tuned)")
    axes[1].set_title("validation"); axes[1].legend()
    if any(h.get("gate_code") is not None for h in hist):
        axes[2].plot(ep, [h.get("gate_code") for h in hist], marker="o", label="gate: code")
        axes[2].plot(ep, [h.get("gate_graph") for h in hist], marker="x", label="gate: graph")
        axes[2].set_ylim(0, 1); axes[2].legend()
    axes[2].set_title("gate weights (train mean)")
    for a in axes:
        a.set_xlabel("epoch"); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "curves.png", dpi=120)
    plt.close(fig)


def train_run(args, device, *, batch_size, lr, ratio, seed, num_relations, node_feat_dim):
    set_seed(seed, deterministic=not args.no_deterministic)
    input_dir = Path(args.data_root) / "input"
    run_dir = Path(args.runs_dir) / run_name_for(args, batch_size, lr, ratio, seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_files = split_files(input_dir, "train")
    if not train_files:
        raise RuntimeError(f"no train_*_input.pkl under {input_dir}; run --cpg and --embed first")
    cnt0, cnt1 = count_targets(train_files)
    total = cnt0 + cnt1
    max_samples = None if args.subset_frac >= 1.0 else max(1, int(total * args.subset_frac))

    train_loader, train_ds = make_loader(input_dir, "train", batch_size=batch_size, shuffle=True, seed=seed,
                                         max_samples=max_samples, buffer_size=max(256, batch_size * 16))
    val_loader, _ = make_loader(input_dir, "valid", batch_size=args.eval_batch_size, shuffle=False, seed=seed)
    test_loader, _ = make_loader(input_dir, "test", batch_size=args.eval_batch_size, shuffle=False, seed=seed)
    if val_loader is None:
        raise RuntimeError("valid split is empty; model selection needs it")

    model = build_model(args.model_type, device, num_relations=num_relations, node_feat_dim=node_feat_dim,
                        hidden_dim=args.hidden_dim, num_rgcn_layers=args.rgcn_layers,
                        fusion_layers=args.fusion_layers, tune_last_n_layers=args.tune_last_n_layers,
                        max_length=args.max_length, ablation=args.ablation,
                        gate_temperature=args.gate_temperature)
    n_new = sum(p.numel() for n, p in model.named_parameters() if "func_encoder" not in n)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion, alpha = build_criterion(args.loss, (cnt0, cnt1), args.focal_gamma, device)
    optimizer = make_optimizer(model, lr, lr * ratio, args.weight_decay)
    steps_per_epoch = math.ceil((max_samples or total) / batch_size)
    total_steps = steps_per_epoch * args.epochs
    scheduler = None
    if total_steps > 0:
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)

    meta = {
        "args": vars(args), "batch_size": batch_size, "lr": lr, "bert_lr_ratio": ratio, "seed": seed,
        "git_commit": git_commit(), "class_counts": {"0": cnt0, "1": cnt1}, "alpha": alpha.tolist(),
        "train_samples_per_epoch": max_samples or total, "num_relations": num_relations,
        "node_feat_dim": node_feat_dim, "params_trainable": n_train, "params_new_non_codebert": n_new,
        "device": str(device), "torch": torch.__version__,
    }
    (run_dir / "args.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    hist_path = run_dir / "history.jsonl"
    hist_path.write_text("")
    print(f"[RUN] {run_dir.name}: train {cnt0}+{cnt1} (per epoch {max_samples or total}), "
          f"new params {n_new:,}, trainable {n_train:,}, alpha={alpha.tolist()}")

    best_f1, best_epoch, wait = -1.0, -1, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(epoch)
        tr = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, args.clip_grad)
        probs, labels, _, val_loss = predict(model, val_loader, device)
        m05 = metrics_at(probs, labels, 0.5)
        mt = best_threshold(probs, labels)
        rec = {
            "epoch": epoch, "train_loss": tr["train_loss"], "val_loss": val_loss,
            "val_f1_05": m05["f1"], "val_precision_05": m05["precision"], "val_recall_05": m05["recall"],
            "val_f1_tuned": mt["f1"], "val_precision_tuned": mt["precision"], "val_recall_tuned": mt["recall"],
            "val_threshold": mt["threshold"], "gate_code": tr["gate_code"], "gate_graph": tr["gate_graph"],
            "oom_skipped": tr["oom_skipped"], "lr": optimizer.param_groups[0]["lr"],
            "mean_p1": float(probs.mean()), "frac_p1_mid": float(((probs > 0.05) & (probs < 0.95)).mean()),
            "elapsed_s": time.time() - t0,
        }
        with hist_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[E{epoch:03d}] loss {tr['train_loss']:.4f} | val CE {val_loss:.4f} F1@0.5 {m05['f1']:.4f} "
              f"F1*={mt['f1']:.4f}@{mt['threshold']:.2f} P {mt['precision']:.3f} R {mt['recall']:.3f}"
              + (f" | gate code {tr['gate_code']:.2f}" if tr["gate_code"] is not None else ""))
        selection = mt["f1"] if args.select_on == "tuned" else m05["f1"]
        if selection > best_f1 + 1e-4:
            best_f1, best_epoch, wait = selection, epoch, 0
            torch.save(model.state_dict(), run_dir / "best.pth")
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[EarlyStopping] no improvement for {args.patience} epochs (best epoch {best_epoch})")
                break

    plot_curves(run_dir)
    model.load_state_dict(torch.load(run_dir / "best.pth", map_location=device))
    result = evaluate_run(model, run_dir, val_loader, test_loader, device)
    result.update({"best_epoch": best_epoch, "best_val_selection_f1": best_f1, "select_on": args.select_on})
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=1))
    print(f"[DONE] {run_dir.name}: test F1 {result['test_tuned']['f1']:.4f} (thr {result['test_tuned']['threshold']:.2f}), "
          f"F1@0.5 {result['test_05']['f1']:.4f}, best epoch {best_epoch}")
    return result


def evaluate_run(model, run_dir: Path, val_loader, test_loader, device):
    """Threshold chosen on validation, applied to test; both the fixed 0.5 and
    the tuned threshold are reported, and per-sample test probabilities are
    saved for paired comparisons (scripts/compare_runs.py)."""
    vp, vl, vi, vloss = predict(model, val_loader, device)
    vt = best_threshold(vp, vl)
    out = {"val_05": metrics_at(vp, vl, 0.5), "val_tuned": vt, "val_loss": vloss}
    np.savez(run_dir / "val_preds.npz", probs=vp, labels=vl, idx=vi)
    if test_loader is not None:
        tp, tl, ti, tloss = predict(model, test_loader, device)
        out.update({"test_05": metrics_at(tp, tl, 0.5), "test_tuned": metrics_at(tp, tl, vt["threshold"]),
                    "test_loss": tloss})
        np.savez(run_dir / "test_preds.npz", probs=tp, labels=tl, idx=ti)
    return out


# ============================================================================
# CPG / embedding generation
# ============================================================================
def read_jsonl(fp: str) -> pd.DataFrame:
    try:
        return pd.read_json(fp, lines=True)
    except ValueError:
        return pd.read_json(fp, lines=False)


def normalise_for_graph(code: str) -> str:
    """Thesis section 3.2.2 (FUN_i / VAR_j) applied to the graph branch only.
    Uses the legacy regex normaliser; optional because it is fragile."""
    return "\n".join(clean_gadget(code.splitlines()))


def cpg_generate(args):
    data_root = Path(args.data_root)
    cpg_dir, joern_dir = data_root / "cpg", data_root / "joern_tmp"
    if cpg_dir.exists():
        shutil.rmtree(cpg_dir)
    cpg_dir.mkdir(parents=True)
    manifest = {"args": vars(args), "splits": {}}
    for split, path in (("train", args.train_jsonl), ("valid", args.valid_jsonl), ("test", args.test_jsonl)):
        df = read_jsonl(path)
        n_in = len(df)
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        if args.max_chars:
            df = df[df.func.str.len() < args.max_chars]
        if args.min_chars:
            df = df[df.func.str.len() >= args.min_chars]
        df = dm.clean(df)
        if "idx" not in df.columns:
            df["idx"] = df.index
        stats = {"rows_in": n_in, "rows_after_filter": len(df), "positives_after_filter": int(df.target.sum()),
                 "slices": 0, "joern_failed_slices": 0, "graphs_written": 0}
        for s, slice_df in dm.slice_frame(df, args.slice_size):
            slice_df = slice_df.copy()
            slice_df["Index"] = slice_df.index
            stats["slices"] += 1
            if joern_dir.exists():
                shutil.rmtree(joern_dir)
            src_df = slice_df.copy()
            if args.normalize_graph_identifiers:
                src_df["func"] = src_df["func"].map(normalise_for_graph)
            dm.to_files(src_df, str(joern_dir))
            stem = f"{split}_{s}_cpg"
            try:
                cpgproc.joern_parse(args.joern_cli_dir, str(joern_dir), str(cpg_dir) + "/", stem)
            except Exception as e:
                print(f"[ERROR] joern-parse failed on {stem}: {e}")
                stats["joern_failed_slices"] += 1
                continue
            res = cpgproc.joern_create(args.joern_cli_dir, str(cpg_dir), str(cpg_dir), [stem + ".bin"])
            json_file = res.get(stem + ".bin")
            if not json_file:
                stats["joern_failed_slices"] += 1
                continue
            graphs = cpgproc.json_process(str(cpg_dir), json_file, stats)
            if not graphs:
                print(f"[{split}] slice {s}: no graphs")
                continue
            right = pd.DataFrame(graphs, columns=["Index", "cpg"]).set_index("Index")
            left = slice_df.set_index("Index", drop=False)
            joined = left.join(right, how="inner")
            dm.write(joined, str(cpg_dir), f"{stem}.pkl")
            stats["graphs_written"] += len(joined)
            if not args.keep_intermediate:
                for f in (cpg_dir / (stem + ".bin"), Path(json_file)):
                    f.unlink(missing_ok=True)
        stats["rows_lost_in_joern"] = stats["rows_after_filter"] - stats["graphs_written"]
        if stats["rows_lost_in_joern"]:
            print(f"[WARN] {split}: {stats['rows_lost_in_joern']} of {stats['rows_after_filter']} functions produced no "
                  f"parsable method in Joern (fragments / unsupported syntax) and are excluded from ALL models.")
        manifest["splits"][split] = stats
        print(f"[{split}] {stats}")
    shutil.rmtree(joern_dir, ignore_errors=True)
    (cpg_dir / "cpg_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))


def embed_generate(args):
    data_root = Path(args.data_root)
    cpg_dir, input_dir = data_root / "cpg", data_root / "input"
    if not args.resume and input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    done = {p.name.replace("_input.pkl", "") for p in input_dir.glob("*_input.pkl")}
    counts = {"rows": 0, "graphs": 0, "rows_without_graph": 0, "rows_with_extra_graphs": 0,
              "node_counts": [], "edge_counts": []}
    for pkl in sorted(cpg_dir.glob("*_cpg.pkl")):
        base = pkl.name.replace("_cpg.pkl", "")
        if args.resume and base in done:
            print(f"[RESUME] skip {base}")
            continue
        df = pd.read_pickle(pkl)
        rows = []
        for _, row in df.iterrows():
            counts["rows"] += 1
            funcs = parse_to_functions(row.cpg, max_nodes=args.max_nodes)
            if not funcs:
                counts["rows_without_graph"] += 1
                continue
            if len(funcs) > 1:
                counts["rows_with_extra_graphs"] += 1
            nodes = funcs[0]
            g = nodes_to_input(nodes, row.target, args.max_nodes, mode=args.node_embedding,
                               reverse_edges=args.reverse_edges, max_len=args.node_max_len)
            g.func = row.func
            g.idx = int(row["idx"]) if "idx" in row and pd.notna(row["idx"]) else -1
            counts["graphs"] += 1
            counts["node_counts"].append(int(g.num_nodes))
            counts["edge_counts"].append(int(g.edge_index.size(1)))
            rows.append({"input": g, "target": int(row.target), "func": row.func, "idx": g.idx,
                         "commit_id": row.get("commit_id", None), "project": row.get("project", None)})
        out = pd.DataFrame(rows, columns=["input", "target", "func", "idx", "commit_id", "project"])
        dm.write(out, str(input_dir), f"{base}_input.pkl")
        del df
        gc.collect()
    nc, ec = counts.pop("node_counts"), counts.pop("edge_counts")
    manifest = {
        "reverse_edges": bool(args.reverse_edges),
        "num_relations": NUM_RELATIONS * (2 if args.reverse_edges else 1),
        "relations": list(RELATIONS), "node_feat_dim": 768, "max_nodes": args.max_nodes,
        "node_embedding": args.node_embedding, "node_max_len": args.node_max_len, "counts": counts,
        "nodes_mean": float(np.mean(nc)) if nc else None, "edges_mean": float(np.mean(ec)) if ec else None,
    }
    (input_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("[EMBED]", json.dumps(manifest))


def load_input_manifest(args):
    p = Path(args.data_root) / "input" / "manifest.json"
    if p.exists():
        m = json.loads(p.read_text())
        return int(m.get("num_relations", NUM_RELATIONS)), int(m.get("node_feat_dim", 768))
    return args.num_relations, 768


# ============================================================================
# main
# ============================================================================
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cpg", action="store_true", help="generate CPGs from the jsonl splits")
    p.add_argument("--embed", action="store_true", help="generate PyG inputs from the CPGs")
    p.add_argument("--resume", action="store_true", help="(--embed) keep already generated input files")
    p.add_argument("--mode", choices=["train", "test", "none"], default="none")
    # data
    p.add_argument("--data_root", default="data")
    p.add_argument("--dataset", default="bigvul", help="prefix of data/cleaned_data/<dataset>_<split>.jsonl")
    p.add_argument("--train_jsonl"); p.add_argument("--valid_jsonl"); p.add_argument("--test_jsonl")
    p.add_argument("--max_chars", type=int, default=None, help="drop functions with >= this many chars (thesis code: 1200, undisclosed). Default: no filter")
    p.add_argument("--min_chars", type=int, default=0)
    p.add_argument("--slice_size", type=int, default=500)
    p.add_argument("--joern_cli_dir", default="joern/joern-cli/")
    p.add_argument("--normalize_graph_identifiers", action="store_true", help="apply FUN_i/VAR_j normalisation to the Joern input (thesis 3.2.2); text branch keeps raw code")
    p.add_argument("--keep_intermediate", action="store_true")
    p.add_argument("--max_nodes", type=int, default=500)
    p.add_argument("--node_embedding", choices=["codebert", "fast"], default="codebert")
    p.add_argument("--node_max_len", type=int, default=128)
    p.add_argument("--reverse_edges", action="store_true", help="add reversed edges as relations 3..5")
    p.add_argument("--num_relations", type=int, default=3, help="used only when input/manifest.json is absent")
    # model
    p.add_argument("--model_type", choices=["hybrid", "codebert", "graph_only"], default="hybrid")
    p.add_argument("--ablation", choices=["none", "no_gate", "no_cls_skip", "shuffle_edges", "no_input_norm", "no_residual"], default="none")
    p.add_argument("--hidden_dim", type=int, default=200)
    p.add_argument("--rgcn_layers", type=int, default=6)
    p.add_argument("--fusion_layers", type=int, default=3)
    p.add_argument("--tune_last_n_layers", type=int, default=2)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--gate_temperature", type=float, default=1.0)
    # optimisation
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[16])
    p.add_argument("--lrs", type=float, nargs="+", default=[1e-4])
    p.add_argument("--bert_lr_ratios", type=float, nargs="+", default=[0.3])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--select_on", choices=["tuned", "f1_05"], default="tuned", help="validation metric for checkpoint selection")
    p.add_argument("--subset_frac", type=float, default=1.0, help="fraction of the training set used per epoch (resampled every epoch)")
    p.add_argument("--loss", choices=["focal", "weighted_ce", "ce"], default="focal")
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42, help="seed for data shuffling at --cpg time")
    p.add_argument("--no_deterministic", action="store_true")
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--run_dir", help="(--mode test) run directory holding best.pth and args.json")
    p.add_argument("--wandb", action="store_true", help="also log to Weights & Biases (optional dependency)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    for split in ("train", "valid", "test"):
        if getattr(args, f"{split}_jsonl") is None:
            setattr(args, f"{split}_jsonl", str(Path(args.data_root) / "cleaned_data" / f"{args.dataset}_{split}.jsonl"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.cpg:
        cpg_generate(args)
    if args.embed:
        embed_generate(args)

    if args.mode == "train":
        num_relations, node_feat_dim = load_input_manifest(args)
        results = []
        for bs in args.batch_sizes:
            for lr in args.lrs:
                for ratio in args.bert_lr_ratios:
                    for seed in args.seeds:
                        r = train_run(args, device, batch_size=bs, lr=lr, ratio=ratio, seed=seed,
                                      num_relations=num_relations, node_feat_dim=node_feat_dim)
                        results.append((run_name_for(args, bs, lr, ratio, seed), r))
        print("\n===== SUMMARY (test F1 at validation-tuned threshold) =====")
        for name, r in sorted(results, key=lambda x: -x[1]["test_tuned"]["f1"]):
            print(f"{name}: test F1 {r['test_tuned']['f1']:.4f} (thr {r['test_tuned']['threshold']:.2f}) | "
                  f"F1@0.5 {r['test_05']['f1']:.4f} | best epoch {r['best_epoch']}")
        if len(args.seeds) > 1:
            by_cfg = {}
            for name, r in results:
                by_cfg.setdefault(name.rsplit("_seed", 1)[0], []).append(r["test_tuned"]["f1"])
            for cfg, f1s in by_cfg.items():
                print(f"{cfg}: mean test F1 {np.mean(f1s):.4f} +/- {np.std(f1s, ddof=1) if len(f1s) > 1 else 0:.4f} over {len(f1s)} seeds")

    if args.mode == "test":
        if not args.run_dir:
            raise SystemExit("--mode test needs --run_dir")
        run_dir = Path(args.run_dir)
        meta = json.loads((run_dir / "args.json").read_text())
        a = argparse.Namespace(**meta["args"])
        set_seed(meta["seed"], deterministic=not a.no_deterministic)
        model = build_model(a.model_type, device, num_relations=meta["num_relations"], node_feat_dim=meta["node_feat_dim"],
                            hidden_dim=a.hidden_dim, num_rgcn_layers=a.rgcn_layers, fusion_layers=a.fusion_layers,
                            tune_last_n_layers=a.tune_last_n_layers, max_length=a.max_length, ablation=a.ablation,
                            gate_temperature=a.gate_temperature)
        model.load_state_dict(torch.load(run_dir / "best.pth", map_location=device))
        input_dir = Path(args.data_root) / "input"
        val_loader, _ = make_loader(input_dir, "valid", batch_size=args.eval_batch_size, shuffle=False, seed=0)
        test_loader, _ = make_loader(input_dir, "test", batch_size=args.eval_batch_size, shuffle=False, seed=0)
        res = evaluate_run(model, run_dir, val_loader, test_loader, device)
        (run_dir / "metrics.json").write_text(json.dumps(res, indent=1))
        print(json.dumps({k: v for k, v in res.items() if k.startswith("test")}, indent=1))


if __name__ == "__main__":
    main()
