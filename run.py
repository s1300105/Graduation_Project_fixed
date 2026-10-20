#!/usr/bin/env python
"""
Corrected training / evaluation driver for the BDCA (R-GCN + CodeBERT) thesis model.

Stages
  --cpg    : jsonl splits -> source files -> Joern CPG -> graph JSON -> <split>_<slice>_cpg.pkl
  --embed  : cpg.pkl -> PyG Data with frozen-CodeBERT node features -> <split>_<slice>_input.pkl
  --mode train : train one or more (config x seed) runs, each written to runs/<run_name>/
  --mode test  : evaluate a finished run on the test split

Tasks: binary (num_classes == 2; BigVul / Devign style, P(y=1) + validation-tuned
threshold, the original code path, kept bit-identical) and multi-class
(num_classes > 2; POJ-104 / CodeNet style, argmax, macro-F1 / accuracy, no threshold).
``--graph_unit whole_file`` keeps Joern's per-file ``<global>`` graph (the whole
program) instead of the first real method.

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
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data import datamanager as dm                       # noqa: E402
from utils.process import cpg_generator as cpgproc              # noqa: E402
from utils.functions.cpg import parse_to_functions, RELATIONS, NUM_NODE_LABELS  # noqa: E402
from utils.process.embeddings import nodes_to_input, NUM_RELATIONS  # noqa: E402
from utils.functions.parse import clean_gadget                  # noqa: E402
from models.LMGNN import build_model                            # noqa: E402
from models.LMGNN import CODEBERT  # noqa: E402  default text model


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
    """``class_counts``: one training count per class (length C >= 2).

    alpha_c = (1 / C) / (count_c / total), i.e. inverse frequency normalised so
    that a balanced set gives all ones. The two-class expression is kept
    literally (0.5 / (cnt / total)) because the BigVul runs depend on it
    bit-for-bit. A class with no training sample gets alpha 1.0 (it never
    contributes to the loss) and a warning; the other classes keep their weight."""
    class_counts = tuple(class_counts)
    if len(class_counts) == 2:
        cnt0, cnt1 = class_counts
        total = max(1, cnt0 + cnt1)
        if cnt0 > 0 and cnt1 > 0:
            alpha = torch.tensor([0.5 / (cnt0 / total), 0.5 / (cnt1 / total)], dtype=torch.float32)
        else:
            alpha = torch.tensor([1.0, 1.0], dtype=torch.float32)
    elif len(class_counts) > 2:
        C = len(class_counts)
        total = max(1, sum(int(c) for c in class_counts))
        if any(int(c) <= 0 for c in class_counts):
            print(f"[WARN] {sum(1 for c in class_counts if int(c) <= 0)} of {C} classes have no training sample; "
                  f"their alpha is set to 1.0")
        alpha = torch.tensor([(1.0 / C) / (int(c) / total) if int(c) > 0 else 1.0 for c in class_counts],
                             dtype=torch.float32)
    else:
        raise ValueError(f"class_counts needs at least two classes, got {class_counts}")
    if name == "focal":
        return FocalLoss(alpha=alpha, gamma=gamma).to(device), alpha
    if name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=alpha.to(device)), alpha
    if name == "ce":
        return nn.CrossEntropyLoss(), torch.ones(len(class_counts))
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


def count_targets(files, num_classes: int = 2):
    """Per-class counts of the rows that carry a graph, as a tuple of C ints.
    Two classes: the original path (targets clipped to {0, 1}). C > 2: bincount;
    a label outside 0..C-1 raises instead of being clipped, because a silently
    clipped label would re-weight and mis-score the run without any trace."""
    if int(num_classes) == 2:
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
    C = int(num_classes)
    counts = np.zeros(C, dtype=np.int64)
    for fn in files:
        df = pd.read_pickle(fn)
        if "target" in df.columns:
            m = df["input"].notna() if "input" in df.columns else slice(None)
            vals = df.loc[m, "target"].astype("int64").to_numpy()
            if len(vals) and (vals.min() < 0 or vals.max() >= C):
                raise ValueError(f"{fn}: target outside 0..{C - 1} (min {int(vals.min())}, max {int(vals.max())}); "
                                 f"check --num_classes / input/manifest.json")
            counts += np.bincount(vals, minlength=C)
        del df
    return tuple(int(c) for c in counts)


# ============================================================================
# evaluation
# ============================================================================
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels, idxs = [], [], []
    loss_sum, n = 0.0, 0
    n_classes = None
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        y = batch.y.view(-1).long()
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        n += y.numel()
        n_classes = int(logits.size(-1))
        if n_classes == 2:
            probs.extend(F.softmax(logits, dim=-1)[:, 1].cpu().tolist())      # 1-D P(y=1)
        else:
            # [B, C] blocks: a list of Python floats would cost ~32 bytes per probability
            # (hundreds of MB per epoch for 1000 classes)
            probs.append(F.softmax(logits, dim=-1).double().cpu().numpy())
        labels.extend(y.cpu().tolist())
        bidx = getattr(batch, "idx", None)
        if bidx is None:
            idxs.extend([-1] * y.numel())
        else:
            idxs.extend([int(v) for v in (bidx.tolist() if torch.is_tensor(bidx) else bidx)])
    if n_classes is not None and n_classes != 2:
        probs = np.concatenate(probs, axis=0) if probs else np.zeros((0, n_classes))
    else:
        probs = np.asarray(probs)
    return probs, np.asarray(labels), np.asarray(idxs), loss_sum / max(1, n)


def metrics_multiclass(probs, labels):
    """Argmax metrics for ``probs`` of shape [N, C] (C > 2). Macro averages run
    over all C classes (fixed denominator, so epochs and runs are comparable
    even when a class is missing from a split; an absent class contributes 0).
    ``f1`` / ``precision`` / ``recall`` alias the macro values so that the
    selection, printing and plotting code written for the binary task works;
    ``threshold`` is None (no threshold is tuned). ``frac_pmax_over_095``
    monitors saturation of the max-class probability."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    C = int(probs.shape[1])
    preds = probs.argmax(axis=1)
    cls = list(range(C))
    per_class_f1 = f1_score(labels, preds, labels=cls, average=None, zero_division=0)
    out = {
        "threshold": None,
        "accuracy": float(accuracy_score(labels, preds)) if len(labels) else 0.0,
        "macro_f1": float(np.mean(per_class_f1)),
        "weighted_f1": float(f1_score(labels, preds, labels=cls, average="weighted", zero_division=0)) if len(labels) else 0.0,
        "macro_precision": float(precision_score(labels, preds, labels=cls, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, preds, labels=cls, average="macro", zero_division=0)),
    }
    out["f1"], out["precision"], out["recall"] = out["macro_f1"], out["macro_precision"], out["macro_recall"]
    if C > 5 and len(labels):
        top5 = np.argsort(-probs, axis=1)[:, :5]
        out["top5_accuracy"] = float((top5 == labels[:, None]).any(axis=1).mean())
    else:
        out["top5_accuracy"] = None
    out["n"] = int(len(labels))
    out["num_classes"] = C
    out["per_class_f1"] = [float(v) for v in per_class_f1]
    out["frac_pmax_over_095"] = float((probs.max(axis=1) > 0.95).mean()) if len(labels) else 0.0
    if C <= 50:
        out["confusion"] = confusion_matrix(labels, preds, labels=cls).tolist()
    return out


def metrics_at(probs, labels, t):
    """Binary: metrics at threshold ``t`` on the 1-D P(y=1). Multi-class
    (``probs.ndim == 2``): argmax metrics, ``t`` is ignored."""
    probs = np.asarray(probs)
    if probs.ndim == 2:
        return metrics_multiclass(probs, labels)
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
    """Threshold on the 0.05..0.95 grid maximising F1. Ties (e.g. saturated
    probabilities, where many thresholds give the same F1) are broken toward
    0.5; the number of tied thresholds is reported so saturation is visible.
    Multi-class probabilities ([N, C]) have no threshold: the argmax metrics are
    returned with ``threshold`` None and ``n_thresholds_tied`` 0."""
    probs = np.asarray(probs)
    if probs.ndim == 2:
        best = metrics_multiclass(probs, labels)
        best["n_thresholds_tied"] = 0
        return best
    cands = [metrics_at(probs, labels, float(t)) for t in THRESHOLDS]
    top = max(round(m["f1"], 12) for m in cands)
    tied = [m for m in cands if round(m["f1"], 12) == top]
    best = min(tied, key=lambda m: (abs(m["threshold"] - 0.5), m["threshold"]))
    best = dict(best)
    best["n_thresholds_tied"] = len(tied)
    return best


def binary_auc(probs, labels):
    """(ROC-AUC, PR-AUC) of 1-D P(y=1); (None, None) when a split holds one class only.
    Threshold-free, the standard metric for MoleculeNet-style tasks; reported next to F1."""
    probs, labels = np.asarray(probs), np.asarray(labels)
    if probs.ndim != 1 or len(np.unique(labels)) < 2:
        return None, None
    return float(roc_auc_score(labels, probs)), float(average_precision_score(labels, probs))


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
    """AdamW with separate learning rates for CodeBERT and the new modules; biases
    and the parameters of LayerNorm / Embedding modules (found by module type,
    not by name) get no weight decay."""
    no_decay_ids = set()
    for mod in model.modules():
        if isinstance(mod, (nn.LayerNorm, nn.Embedding)):
            no_decay_ids.update(id(q) for q in mod.parameters(recurse=False))
    decay, no_decay, bert_decay, bert_no_decay = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bert = "func_encoder.bert" in name
        is_nd = name.endswith(".bias") or id(p) in no_decay_ids
        (bert_no_decay if is_bert and is_nd else bert_decay if is_bert else no_decay if is_nd else decay).append(p)
    groups = []
    if decay: groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
    if no_decay: groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    if bert_decay: groups.append({"params": bert_decay, "lr": bert_lr, "weight_decay": weight_decay})
    if bert_no_decay: groups.append({"params": bert_no_decay, "lr": bert_lr, "weight_decay": 0.0})
    return torch.optim.AdamW(groups)


def run_name_for(args, batch_size, lr, ratio, seed):
    tag = args.model_type + ("" if args.ablation == "none" else f"-{args.ablation}")
    prefix = getattr(args, "run_prefix", "") or ""
    return f"{prefix}{tag}_bs{batch_size}_lr{lr:.1e}_br{ratio:.2f}_seed{seed}"


def _fmt_thr(t):
    return "argmax" if t is None else f"{t:.2f}"


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
    ep = [h.get("epoch") for h in hist]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(ep, [h.get("train_loss") for h in hist], label="train (criterion)")
    axes[0].plot(ep, [h.get("val_loss") for h in hist], label="val (CE)")
    axes[0].set_title("loss"); axes[0].legend()
    if hist[0].get("val_threshold") is None:       # multi-class: argmax metrics, no threshold
        axes[1].plot(ep, [h.get("val_macro_f1") for h in hist], label="val macro-F1 (argmax)")
        axes[1].plot(ep, [h.get("val_accuracy") for h in hist], label="val accuracy (argmax)")
        if any(h.get("val_top5") is not None for h in hist):
            axes[1].plot(ep, [h.get("val_top5") for h in hist], label="val top-5 accuracy")
        axes[1].plot(ep, [h.get("val_precision_tuned") for h in hist], label="val macro-P")
        axes[1].plot(ep, [h.get("val_recall_tuned") for h in hist], label="val macro-R")
        axes[1].set_title("validation (multi-class, argmax)"); axes[1].legend()
    else:
        axes[1].plot(ep, [h.get("val_f1_tuned") for h in hist], label="val F1 (tuned thr)")
        axes[1].plot(ep, [h.get("val_f1_05") for h in hist], label="val F1 @0.5")
        axes[1].plot(ep, [h.get("val_precision_tuned") for h in hist], label="val P (tuned)")
        axes[1].plot(ep, [h.get("val_recall_tuned") for h in hist], label="val R (tuned)")
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


def train_run(args, device, *, batch_size, lr, ratio, seed, num_relations, node_feat_dim, num_classes=2):
    set_seed(seed, deterministic=not args.no_deterministic)
    num_classes = int(num_classes)
    multiclass = num_classes != 2
    if multiclass and args.select_on == "auc":
        raise RuntimeError("--select_on auc is defined for binary runs (ROC-AUC of P(y=1)); use macro_f1 or accuracy")
    if not multiclass and args.select_on == "macro_f1":
        raise RuntimeError("--select_on macro_f1 is defined for multi-class runs; the binary task selects on "
                           "tuned / f1_05 / accuracy")
    input_dir = Path(args.data_root) / "input"
    run_dir = Path(args.runs_dir) / run_name_for(args, batch_size, lr, ratio, seed)
    if (run_dir / "args.json").exists() and not args.overwrite:
        raise RuntimeError(f"{run_dir} already holds a run; pass --overwrite to replace it")

    train_files = split_files(input_dir, "train")
    if not train_files:
        raise RuntimeError(f"no train_*_input.pkl under {input_dir}; run --cpg and --embed first")
    class_counts = count_targets(train_files, num_classes)
    total = sum(class_counts)
    max_samples = None if args.subset_frac >= 1.0 else max(1, int(total * args.subset_frac))

    train_loader, train_ds = make_loader(input_dir, "train", batch_size=batch_size, shuffle=True, seed=seed,
                                         max_samples=max_samples, buffer_size=max(256, batch_size * 16))
    val_loader, _ = make_loader(input_dir, "valid", batch_size=args.eval_batch_size, shuffle=False, seed=seed)
    test_loader, _ = make_loader(input_dir, "test", batch_size=args.eval_batch_size, shuffle=False, seed=seed)
    if val_loader is None:
        raise RuntimeError("valid split is empty; model selection needs it")
    if test_loader is None:
        raise RuntimeError("test split is empty; the run would have nothing to evaluate (need test_*_input.pkl)")
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = input_dir / "manifest.json"
    input_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    text_model, num_node_labels = resolve_text_model_and_labels(args, input_manifest)
    model = build_model(args.model_type, device, num_relations=num_relations, node_feat_dim=node_feat_dim,
                        hidden_dim=args.hidden_dim, num_rgcn_layers=args.rgcn_layers,
                        fusion_layers=args.fusion_layers, tune_last_n_layers=args.tune_last_n_layers,
                        max_length=args.max_length, ablation=args.ablation,
                        gate_temperature=args.gate_temperature, seed=seed, num_classes=num_classes,
                        model_name=text_model, num_node_labels=num_node_labels)
    n_new = sum(p.numel() for n, p in model.named_parameters() if "func_encoder" not in n)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion, alpha = build_criterion(args.loss, class_counts, args.focal_gamma, device)
    optimizer = make_optimizer(model, lr, lr * ratio, args.weight_decay)
    steps_per_epoch = math.ceil((max_samples or total) / batch_size)
    total_steps = steps_per_epoch * args.epochs
    scheduler = None
    if total_steps > 0:
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)

    meta = {
        "args": vars(args), "batch_size": batch_size, "lr": lr, "bert_lr_ratio": ratio, "seed": seed,
        "git_commit": git_commit(), "class_counts": {str(i): int(c) for i, c in enumerate(class_counts)},
        "alpha": alpha.tolist(),
        "train_samples_per_epoch": max_samples or total, "num_relations": num_relations,
        "node_feat_dim": node_feat_dim, "num_classes": num_classes,
        "text_model": text_model, "num_node_labels": num_node_labels,
        "dataset": args.dataset, "data_root": str(Path(args.data_root).resolve()),
        "graph_unit": (input_manifest or {}).get("graph_unit", getattr(args, "graph_unit", "first_method")),
        "input_manifest": input_manifest,
        "params_trainable": n_train, "params_new_non_codebert": n_new,
        "device": str(device), "torch": torch.__version__,
    }
    (run_dir / "args.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    hist_path = run_dir / "history.jsonl"
    hist_path.write_text("")
    counts_str = "+".join(str(c) for c in class_counts) if num_classes <= 10 else \
        f"{total} over {num_classes} classes (min {min(class_counts)}, max {max(class_counts)})"
    print(f"[RUN] {run_dir.name}: train {counts_str} (per epoch {max_samples or total}), "
          f"new params {n_new:,}, trainable {n_train:,}, "
          + (f"alpha={alpha.tolist()}" if num_classes <= 10 else f"alpha in [{alpha.min():.3f}, {alpha.max():.3f}]"))

    best_f1, best_epoch, wait = -1.0, -1, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(epoch)
        tr = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, args.clip_grad)
        probs, labels, _, val_loss = predict(model, val_loader, device)
        if multiclass:
            mt = best_threshold(probs, labels)          # argmax metrics, threshold None, f1 = macro-F1
            m05 = mt
            p_stat = probs.max(axis=1)                  # mean_p1 / frac_p1_mid refer to the max-class probability
        else:
            m05 = metrics_at(probs, labels, 0.5)
            mt = best_threshold(probs, labels)
            p_stat = probs
        rec = {
            "epoch": epoch, "train_loss": tr["train_loss"], "val_loss": val_loss,
            "val_f1_05": m05["f1"], "val_precision_05": m05["precision"], "val_recall_05": m05["recall"],
            "val_f1_tuned": mt["f1"], "val_precision_tuned": mt["precision"], "val_recall_tuned": mt["recall"],
            "val_threshold": mt["threshold"], "gate_code": tr["gate_code"], "gate_graph": tr["gate_graph"],
            "oom_skipped": tr["oom_skipped"], "lr": optimizer.param_groups[0]["lr"],
            "mean_p1": float(p_stat.mean()), "frac_p1_mid": float(((p_stat > 0.05) & (p_stat < 0.95)).mean()),
            "elapsed_s": time.time() - t0,
        }
        if multiclass:
            rec.update({"val_accuracy": mt["accuracy"], "val_macro_f1": mt["macro_f1"], "val_top5": mt["top5_accuracy"],
                        "p1_meaning": "max_class_probability"})
        else:               # added keys only; the recorded binary keys and values are frozen (golden test)
            rec["val_auc"], rec["val_pr_auc"] = binary_auc(probs, labels)
        with hist_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if multiclass:
            print(f"[E{epoch:03d}] loss {tr['train_loss']:.4f} | val CE {val_loss:.4f} acc {mt['accuracy']:.4f} "
                  f"macro-F1 {mt['macro_f1']:.4f} macro-P {mt['precision']:.3f} macro-R {mt['recall']:.3f}"
                  + (f" top5 {mt['top5_accuracy']:.4f}" if mt["top5_accuracy"] is not None else "")
                  + (f" | gate code {tr['gate_code']:.2f}" if tr["gate_code"] is not None else ""))
        else:
            print(f"[E{epoch:03d}] loss {tr['train_loss']:.4f} | val CE {val_loss:.4f} F1@0.5 {m05['f1']:.4f} "
                  f"F1*={mt['f1']:.4f}@{mt['threshold']:.2f} P {mt['precision']:.3f} R {mt['recall']:.3f}"
                  + (f" AUC {rec['val_auc']:.4f}" if rec.get("val_auc") is not None else "")
                  + (f" | gate code {tr['gate_code']:.2f}" if tr["gate_code"] is not None else ""))
        if args.select_on == "tuned":
            selection = mt["f1"]                        # binary: F1 at the tuned threshold; multi-class: macro-F1
        elif args.select_on == "f1_05":
            selection = m05["f1"]
        elif args.select_on == "accuracy":
            selection = mt["accuracy"]
        elif args.select_on == "macro_f1":
            selection = mt["macro_f1"]
        elif args.select_on == "auc":
            selection = rec["val_auc"] if rec["val_auc"] is not None else -1.0
        else:
            raise ValueError(f"unknown --select_on {args.select_on}")
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
    # best_val_selection_f1 keeps its (frozen) name; it holds the value of the metric named here
    sel_metric = ("roc_auc" if args.select_on == "auc" else
                  "accuracy" if args.select_on == "accuracy" else
                  "macro_f1" if multiclass or args.select_on == "macro_f1" else
                  "f1_tuned_threshold" if args.select_on == "tuned" else "f1_at_0.5")
    result.update({"best_epoch": best_epoch, "best_val_selection_f1": best_f1, "select_on": args.select_on,
                   "best_val_selection_metric": sel_metric})
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=1))
    tt = result["test_tuned"]
    if multiclass:
        print(f"[DONE] {run_dir.name}: test accuracy {tt['accuracy']:.4f} macro-F1 {tt['macro_f1']:.4f}"
              + (f" top5 {tt['top5_accuracy']:.4f}" if tt.get("top5_accuracy") is not None else "")
              + f", best epoch {best_epoch}")
    else:
        print(f"[DONE] {run_dir.name}: test F1 {tt['f1']:.4f} (thr {_fmt_thr(tt['threshold'])}), "
              f"F1@0.5 {result['test_05']['f1']:.4f}, best epoch {best_epoch}")
    return result


def evaluate_run(model, run_dir: Path, val_loader, test_loader, device):
    """Threshold chosen on validation, applied to test; both the fixed 0.5 and
    the tuned threshold are reported, and per-sample test probabilities are
    saved for paired comparisons (scripts/compare_runs.py). Multi-class runs
    (probabilities [N, C]) report argmax metrics under the same keys with
    ``threshold`` None; ``probs`` in the npz files is then 2-D."""
    vp, vl, vi, vloss = predict(model, val_loader, device)
    vt = best_threshold(vp, vl)
    out = {"val_05": metrics_at(vp, vl, 0.5), "val_tuned": vt, "val_loss": vloss,
           "num_classes": int(vp.shape[1]) if vp.ndim == 2 else 2}
    if vp.ndim == 1:
        out["val_auc"], out["val_pr_auc"] = binary_auc(vp, vl)
    np.savez(run_dir / "val_preds.npz", probs=vp, labels=vl, idx=vi)
    if test_loader is not None:
        tp, tl, ti, tloss = predict(model, test_loader, device)
        out.update({"test_05": metrics_at(tp, tl, 0.5), "test_tuned": metrics_at(tp, tl, vt["threshold"]),
                    "test_loss": tloss})
        if tp.ndim == 1:
            out["test_auc"], out["test_pr_auc"] = binary_auc(tp, tl)
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


def parse_rescue_stages(value: str) -> tuple:
    """'ABC' / 'ab' / 'A,B' -> ('A', 'B', 'C') in pipeline order; unknown letters raise (a typo must
    not silently disable the rescue pass). An empty value is refused: use --no_rescue_macros."""
    letters = [c for c in str(value).upper() if c not in ", "]
    unknown = sorted(set(letters) - set(dm.RESCUE_STAGES))
    if unknown or not letters:
        raise ValueError(f"--rescue_stages {value!r}: expected letters from {''.join(dm.RESCUE_STAGES)} "
                         f"(e.g. ABC or AB); to disable the rescue pass use --no_rescue_macros")
    return tuple(st for st in dm.RESCUE_STAGES if st in letters)


def _validate_ext_column(df: pd.DataFrame, path) -> None:
    """A prepared jsonl may carry an ``ext`` column ('.c' / '.cpp' / null). Check it
    before any Joern work (and before the old CPG directory is removed) so a mangled
    column fails fast instead of mid-way through a multi-hour run."""
    if "ext" not in df.columns:
        return
    vals = df["ext"]
    bad = vals[vals.notna() & ~vals.isin(dm._VALID_EXTS)]
    if len(bad):
        raise ValueError(f"{path}: ext column holds {len(bad)} invalid value(s), e.g. {bad.iloc[0]!r}; "
                         f"allowed: {dm._VALID_EXTS} or null")


def cpg_generate(args):
    data_root = Path(args.data_root)
    cpg_dir, joern_dir = data_root / "cpg", data_root / "joern_tmp"
    rescue_stages = parse_rescue_stages(args.rescue_stages)
    if args.normalize_graph_identifiers and args.graph_unit == "whole_file":
        raise ValueError("--normalize_graph_identifiers rewrites single functions; on whole programs it would turn "
                         "#include / #define lines and string literals into VAR_i / FUN_i tokens. Not supported "
                         "with --graph_unit whole_file")
    if not args.no_rescue_macros and args.graph_unit == "first_method":
        print(f"[INFO] rescue stages {''.join(rescue_stages) or '(none)'}"
              + (" (stage C was added after the first BigVul CPG sets; --rescue_stages AB reproduces them)"
                 if "C" in rescue_stages else ""))
    # read and validate every split before touching the previous output
    frames = []
    for split, path in (("train", args.train_jsonl), ("valid", args.valid_jsonl), ("test", args.test_jsonl)):
        df = read_jsonl(path)
        _validate_ext_column(df, path)
        frames.append((split, df))
    if cpg_dir.exists():
        shutil.rmtree(cpg_dir)
    cpg_dir.mkdir(parents=True)
    manifest = {"args": vars(args), "splits": {}}
    # --graph_unit whole_file keeps Joern's per-file <global> method (the whole program) per source file
    for split, df in frames:
        n_in = len(df)
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        if args.max_chars:
            df = df[df.func.str.len() < args.max_chars]
        if args.min_chars:
            df = df[df.func.str.len() >= args.min_chars]
        df = dm.clean(df)
        if "idx" not in df.columns:
            df["idx"] = df.index
        binary = set(int(v) for v in df.target.unique()) <= {0, 1}
        stats = {"rows_in": n_in, "rows_after_filter": len(df),
                 # a count of positives only means something for 0/1 labels (None for multi-class splits)
                 "positives_after_filter": int(df.target.sum()) if binary else None,
                 "label_counts": {str(k): int(v) for k, v in df.target.value_counts().sort_index().items()},
                 "slices": 0, "joern_failed_slices": 0, "graphs_written": 0}
        if not args.no_rescue_macros:
            for st in rescue_stages:
                stats[f"rescued_stage_{st}"] = 0            # always present, so manifests are comparable
            if args.graph_unit == "whole_file":
                stats["rescue_skipped_whole_file"] = 0      # rows lost without a rescue attempt (see below)
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
            graphs = cpgproc.json_process(str(cpg_dir), json_file, stats, graph_unit=args.graph_unit) or []
            right = pd.DataFrame(graphs, columns=["Index", "cpg"]).set_index("Index")
            left = slice_df.set_index("Index", drop=False)
            joined = left.join(right, how="inner")
            if not args.keep_intermediate:
                for f in (cpg_dir / (stem + ".bin"), Path(json_file)):
                    f.unlink(missing_ok=True)

            # --- rescue pass: functions Joern could not parse as written (macro signatures etc.) ---
            # The stages rewrite the text before the first `{` as a function signature; on a whole
            # program that text is the #include / #define / global-declaration prologue, and stage B
            # would strip its ALL-CAPS tokens (`#define MAXN 100` -> `#define 100`), so a "rescued" whole
            # file would be a graph of a corrupted program. The pass is therefore skipped for whole_file.
            if not args.no_rescue_macros and args.graph_unit == "whole_file":
                stats["rescue_skipped_whole_file"] += int((~left.index.isin(joined.index)).sum())
            elif not args.no_rescue_macros:
                lost = left.loc[~left.index.isin(joined.index)]
                for stage in rescue_stages:
                    if lost.empty:
                        break
                    variants = {ix: dict(dm.rescue_variants(code)).get(stage) for ix, code in lost["func"].items()}
                    variants = {ix: v for ix, v in variants.items() if v}
                    if not variants:
                        continue
                    rescue_df = lost.loc[list(variants)].copy()
                    rescue_df["func"] = pd.Series(variants)
                    if args.normalize_graph_identifiers:
                        rescue_df["func"] = rescue_df["func"].map(normalise_for_graph)
                    shutil.rmtree(joern_dir, ignore_errors=True)
                    dm.to_files(rescue_df, str(joern_dir))
                    rstem = f"{stem}_rescue{stage}"
                    try:
                        cpgproc.joern_parse(args.joern_cli_dir, str(joern_dir), str(cpg_dir) + "/", rstem)
                        rres = cpgproc.joern_create(args.joern_cli_dir, str(cpg_dir), str(cpg_dir), [rstem + ".bin"])
                    except Exception as e:
                        print(f"[WARN] rescue {stage} failed on {stem}: {e}")
                        continue
                    rjson = rres.get(rstem + ".bin")
                    if rjson:
                        rgraphs = cpgproc.json_process(str(cpg_dir), rjson, {}, graph_unit=args.graph_unit) or []
                        rright = pd.DataFrame(rgraphs, columns=["Index", "cpg"]).set_index("Index")
                        rjoined = lost.loc[lost.index.isin(rright.index)].join(rright, how="inner")  # keeps the ORIGINAL func text
                        if len(rjoined):
                            joined = pd.concat([joined, rjoined])
                            stats[f"rescued_stage_{stage}"] = stats.get(f"rescued_stage_{stage}", 0) + len(rjoined)
                            lost = lost.loc[~lost.index.isin(rjoined.index)]
                        if not args.keep_intermediate:
                            Path(rjson).unlink(missing_ok=True)
                    if not args.keep_intermediate:
                        (cpg_dir / (rstem + ".bin")).unlink(missing_ok=True)
                joined = joined.sort_index()

            if joined.empty:
                print(f"[{split}] slice {s}: no graphs")
                continue
            dm.write(joined, str(cpg_dir), f"{stem}.pkl")
            stats["graphs_written"] += len(joined)
        stats["rows_lost_in_joern"] = stats["rows_after_filter"] - stats["graphs_written"]
        if stats["rows_lost_in_joern"] and args.graph_unit == "whole_file":
            print(f"[WARN] {split}: {stats['rows_lost_in_joern']} of {stats['rows_after_filter']} programs contain no "
                  f"function Joern could parse (declarations only / unsupported syntax) and are excluded from ALL models.")
        elif stats["rows_lost_in_joern"]:
            print(f"[WARN] {split}: {stats['rows_lost_in_joern']} of {stats['rows_after_filter']} functions produced no "
                  f"parsable method in Joern (fragments / unsupported syntax) and are excluded from ALL models.")
        manifest["splits"][split] = stats
        print(f"[{split}] {stats}")
    shutil.rmtree(joern_dir, ignore_errors=True)
    (cpg_dir / "cpg_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))


COUNT_KEYS = ("rows", "graphs", "rows_without_graph", "rows_with_extra_graphs", "graphs_at_cap")


def _file_fingerprint(path: Path) -> dict:
    st = Path(path).stat()
    return {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def _sidecar(input_dir: Path, base: str) -> Path:
    """Per-slice record written next to <base>_input.pkl: which cpg pickle (size, mtime) and which
    settings it was built from, and its counts. --resume trusts a slice only through this record."""
    return input_dir / f"{base}_input.meta.json"


def _classes_json(args, cpg_manifest_args: dict):
    """<dataset>_classes.json next to the train jsonl the CPGs were built from (written by the
    data/scripts/make_*_json.py preparation scripts), or None."""
    train = cpg_manifest_args.get("train_jsonl") or args.train_jsonl
    dataset = cpg_manifest_args.get("dataset") or args.dataset
    if not train:
        return None
    p = Path(train).parent / f"{dataset}_classes.json"
    return p if p.exists() else None


def _legacy_record(cpg_pkl: Path, input_pkl: Path, args) -> dict:
    """Counts of a slice embedded before sidecars existed, recomputed without CodeBERT."""
    cdf = pd.read_pickle(cpg_pkl)
    idf = pd.read_pickle(input_pkl)
    extra = sum(1 for c in cdf["cpg"] if len(parse_to_functions(c, max_nodes=args.max_nodes,
                                                                 skip_global=(args.graph_unit == "first_method"))) > 1)
    graphs = [g for g in idf["input"] if g is not None]
    counts = {"rows": int(len(cdf)), "graphs": len(graphs), "rows_without_graph": int(len(cdf) - len(graphs)),
              "rows_with_extra_graphs": int(extra),
              "graphs_at_cap": sum(1 for g in graphs if int(g.num_nodes) >= int(args.max_nodes))}
    return {"counts": counts, "node_sum": sum(int(g.num_nodes) for g in graphs),
            "edge_sum": sum(int(g.edge_index.size(1)) for g in graphs)}


def embed_generate(args):
    data_root = Path(args.data_root)
    cpg_dir, input_dir = data_root / "cpg", data_root / "input"
    settings = {"reverse_edges": bool(args.reverse_edges), "max_nodes": int(args.max_nodes),
                "node_embedding": args.node_embedding, "node_max_len": int(args.node_max_len),
                "node_code_mode": args.node_code_mode, "graph_unit": args.graph_unit}
    manifest_path = input_dir / "manifest.json"

    # ---- every check that can refuse the run comes BEFORE the previous input set is touched ----
    cpg_files = sorted(cpg_dir.glob("*_cpg.pkl"))
    if not cpg_files:
        raise RuntimeError(f"no *_cpg.pkl under {cpg_dir}; run --cpg first")
    cpg_manifest = cpg_dir / "cpg_manifest.json"
    cpg_args = json.loads(cpg_manifest.read_text()).get("args", {}) if cpg_manifest.exists() else {}
    stored_unit = cpg_args.get("graph_unit", "first_method")
    if cpg_manifest.exists() and stored_unit != args.graph_unit:
        # with the wrong unit parse_to_functions returns [] for every row (whole_file CPGs hold only
        # <global>) or records a unit the graphs do not have; both are silent, so refuse
        raise RuntimeError(f"--embed --graph_unit {args.graph_unit} but {cpg_manifest} was generated with "
                           f"--graph_unit {stored_unit}; pass the same value to --cpg and --embed")
    # label range over ALL cpg rows (also the ones --resume skips)
    t_min, t_max, seen = None, None, set()
    for pkl in cpg_files:
        t = pd.read_pickle(pkl)["target"].astype("int64")
        if len(t):
            t_min = int(t.min()) if t_min is None else min(t_min, int(t.min()))
            t_max = int(t.max()) if t_max is None else max(t_max, int(t.max()))
            seen.update(int(v) for v in t.unique())
        del t
    classes_json = _classes_json(args, cpg_args)
    n_listed = len(json.loads(classes_json.read_text())) if classes_json else None
    if args.num_classes:
        num_classes = int(args.num_classes)
    elif n_listed is not None:
        num_classes = max(2, n_listed)            # the class map of the prepared dataset, not max(label)+1
    else:
        num_classes = max(2, (t_max if t_max is not None else 1) + 1)
    if t_min is not None and (t_min < 0 or t_max >= num_classes):
        raise RuntimeError(f"targets range {t_min}..{t_max} but num_classes is {num_classes} "
                           f"(labels must be 0..{num_classes - 1}; check --num_classes / the dataset's label column)")
    if n_listed is not None and n_listed != num_classes:
        print(f"[WARN] --num_classes {num_classes} but {classes_json} lists {n_listed} classes")
    if num_classes > 2 and len(seen) < num_classes:
        print(f"[WARN] {num_classes - len(seen)} of {num_classes} classes have no row in the CPG set "
              f"(lost at --cpg or absent from the prepared splits); they stay in the label space")
    prev = None
    if args.resume and manifest_path.exists():
        prev = json.loads(manifest_path.read_text())
        if prev.get("num_classes", 2) != num_classes:
            raise RuntimeError(f"--resume with a different num_classes (stored {prev.get('num_classes', 2)}, now {num_classes})")
        for k, v in settings.items():
            stored = prev.get(k, "first_method" if k == "graph_unit" else None)    # old manifests: first_method
            if stored != v:
                raise RuntimeError(f"--resume with a different setting ({k}: stored {stored!r}, now {v!r}); "
                                   f"delete {input_dir} or rerun without --resume")

    if not args.resume and input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    done = {p.name.replace("_input.pkl", "") for p in input_dir.glob("*_input.pkl")}
    records = {}
    for pkl in cpg_files:
        base = pkl.name.replace("_cpg.pkl", "")
        if args.resume and base in done:
            side = _sidecar(input_dir, base)
            inp = input_dir / f"{base}_input.pkl"
            if side.exists():
                rec = json.loads(side.read_text())
                if rec.get("settings") != settings or rec.get("num_classes") != num_classes:
                    raise RuntimeError(f"--resume: {inp} was built with other settings ({side}); "
                                       f"delete {input_dir} or rerun without --resume")
                if rec.get("cpg") == _file_fingerprint(pkl):
                    print(f"[RESUME] skip {base}")
                    records[base] = rec
                    continue
                print(f"[RESUME] {pkl.name} changed since {inp.name} was built: re-embedding {base}")
            elif inp.stat().st_mtime_ns >= pkl.stat().st_mtime_ns:
                # written before sidecars existed: trust it only if it is newer than its cpg pickle
                rec = {**_legacy_record(pkl, inp, args), "cpg": _file_fingerprint(pkl),
                       "settings": settings, "num_classes": num_classes}
                _sidecar(input_dir, base).write_text(json.dumps(rec, indent=1))
                print(f"[RESUME] skip {base} (no record; input newer than its cpg pickle, counts recomputed)")
                records[base] = rec
                continue
            else:
                print(f"[RESUME] {pkl.name} is newer than {inp.name}: re-embedding {base}")
        df = pd.read_pickle(pkl)
        counts = {k: 0 for k in COUNT_KEYS}
        node_sum = edge_sum = 0
        rows = []
        for _, row in df.iterrows():
            counts["rows"] += 1
            funcs = parse_to_functions(row.cpg, max_nodes=args.max_nodes,
                                       skip_global=(args.graph_unit == "first_method"))
            if not funcs:
                counts["rows_without_graph"] += 1
                continue
            if len(funcs) > 1:
                counts["rows_with_extra_graphs"] += 1
            nodes = funcs[0]
            g = nodes_to_input(nodes, row.target, args.max_nodes, mode=args.node_embedding,
                               reverse_edges=args.reverse_edges, max_len=args.node_max_len,
                               code_mode=args.node_code_mode, max_cache=args.node_cache_size)
            g.func = row.func
            g.idx = int(row["idx"]) if "idx" in row and pd.notna(row["idx"]) else -1
            counts["graphs"] += 1
            if int(g.num_nodes) >= int(args.max_nodes):
                counts["graphs_at_cap"] += 1
            node_sum += int(g.num_nodes)
            edge_sum += int(g.edge_index.size(1))
            rows.append({"input": g, "target": int(row.target), "func": row.func, "idx": g.idx,
                         "commit_id": row.get("commit_id", None), "project": row.get("project", None)})
        out = pd.DataFrame(rows, columns=["input", "target", "func", "idx", "commit_id", "project"])
        dm.write(out, str(input_dir), f"{base}_input.pkl")
        rec = {"counts": counts, "node_sum": node_sum, "edge_sum": edge_sum,
               "cpg": _file_fingerprint(pkl), "settings": settings, "num_classes": num_classes}
        _sidecar(input_dir, base).write_text(json.dumps(rec, indent=1))
        records[base] = rec
        del df
        gc.collect()
    # input files without a cpg pickle (a stale slice of an earlier, larger CPG set) would be trained on
    orphans = sorted(done - set(records))
    for base in orphans:
        print(f"[RESUME] removing {base}_input.pkl: no {base}_cpg.pkl in {cpg_dir}")
        (input_dir / f"{base}_input.pkl").unlink(missing_ok=True)
        _sidecar(input_dir, base).unlink(missing_ok=True)
    counts = {k: sum(int(r["counts"].get(k, 0)) for r in records.values()) for k in COUNT_KEYS}
    node_sum = sum(int(r["node_sum"]) for r in records.values())
    edge_sum = sum(int(r["edge_sum"]) for r in records.values())
    if counts["graphs"] == 0:
        raise RuntimeError(f"--embed produced no graph from {len(cpg_files)} cpg file(s) "
                           f"(rows {counts['rows']}, rows_without_graph {counts['rows_without_graph']}); "
                           f"check --graph_unit against {cpg_dir / 'cpg_manifest.json'}")
    manifest = {
        **settings,
        "num_relations": NUM_RELATIONS * (2 if args.reverse_edges else 1),
        "relations": list(RELATIONS), "node_feat_dim": 768,
        "num_classes": num_classes, "dataset": args.dataset,
        "classes_json": str(classes_json) if classes_json else None,
        "counts": counts, "counts_cover": "all", "node_sum": node_sum, "edge_sum": edge_sum,
        "nodes_mean": node_sum / counts["graphs"] if counts["graphs"] else None,
        "edges_mean": edge_sum / counts["graphs"] if counts["graphs"] else None,
        "input_files": sorted(p.name for p in input_dir.glob("*_input.pkl")),
    }
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print("[EMBED]", json.dumps(manifest))


DEFAULT_MOL_TEXT_MODEL = "DeepChem/ChemBERTa-77M-MLM"


def mol_generate(args):
    """--molgraph: jsonl splits whose ``func`` is a SMILES string -> <data_root>/input/*_input.pkl.

    The chemistry counterpart of --cpg + --embed: the molecular graph (atoms, bonds typed as
    single / double / triple / aromatic / misc, OGB one-hot atom features; utils/process/molgraph.py)
    replaces the Joern CPG with frozen-CodeBERT node features, and input/manifest.json records the
    chemical language model the text branch should use. Deterministic; no GPU needed."""
    from utils.process.molgraph import (smiles_to_data, ATOM_FEAT_DIM, NUM_ATOM_LABELS, NUM_BOND_RELATIONS,
                                        BOND_TYPES)
    data_root = Path(args.data_root)
    input_dir = data_root / "input"
    frames = [(split, read_jsonl(path)) for split, path in
              (("train", args.train_jsonl), ("valid", args.valid_jsonl), ("test", args.test_jsonl))]
    classes_json = _classes_json(args, {})
    n_listed = len(json.loads(classes_json.read_text())) if classes_json else None
    t_max = max(int(df["target"].max()) for _, df in frames if len(df))
    num_classes = int(args.num_classes or n_listed or max(2, t_max + 1))
    if t_max >= num_classes:
        raise RuntimeError(f"targets up to {t_max} but num_classes is {num_classes}")
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True)
    counts = {k: 0 for k in COUNT_KEYS}
    node_sum = edge_sum = 0
    per_split = {}
    for split, df in frames:
        rows, lost = [], []
        for rec in df.to_dict("records"):
            counts["rows"] += 1
            g = smiles_to_data(rec["func"], rec["target"])
            if g is None:
                counts["rows_without_graph"] += 1
                lost.append(rec.get("idx"))
                continue
            g.func = rec["func"]
            g.idx = int(rec["idx"])
            counts["graphs"] += 1
            node_sum += int(g.num_nodes)
            edge_sum += int(g.edge_index.size(1))
            rows.append({"input": g, "target": int(rec["target"]), "func": rec["func"], "idx": g.idx,
                         "commit_id": None, "project": rec.get("project")})
        out = pd.DataFrame(rows, columns=["input", "target", "func", "idx", "commit_id", "project"])
        dm.write(out, str(input_dir), f"{split}_0_input.pkl")
        per_split[split] = {"rows": len(df), "graphs": len(rows), "lost_idx": lost[:20],
                            "label_counts": {str(k): int(v) for k, v in out["target"].value_counts().sort_index().items()}}
    if counts["graphs"] == 0:
        raise RuntimeError("--molgraph produced no graph (is func a SMILES column?)")
    manifest = {
        "graph_unit": "molecule", "featuriser": "ogb_atom_onehot_v1",
        "reverse_edges": False, "max_nodes": None, "node_embedding": "ogb_atom_onehot",
        "num_relations": NUM_BOND_RELATIONS, "relations": list(BOND_TYPES), "node_feat_dim": ATOM_FEAT_DIM,
        "num_node_labels": NUM_ATOM_LABELS, "num_classes": num_classes, "dataset": args.dataset,
        "text_model": args.text_model or DEFAULT_MOL_TEXT_MODEL,
        "classes_json": str(classes_json) if classes_json else None,
        "counts": counts, "counts_cover": "all", "splits": per_split, "node_sum": node_sum, "edge_sum": edge_sum,
        "nodes_mean": node_sum / counts["graphs"], "edges_mean": edge_sum / counts["graphs"],
        "input_files": sorted(p.name for p in input_dir.glob("*_input.pkl")),
    }
    (input_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("[MOLGRAPH]", json.dumps({k: v for k, v in manifest.items() if k != "splits"}))


def resolve_text_model_and_labels(args, input_manifest):
    """Text encoder (HF id) and node-label vocabulary size of a run. The input builder records
    both in input/manifest.json (Joern CPGs: CodeBERT and the Joern node labels; molecules: a
    chemical LM and atom types); --text_model overrides the encoder as an explicit experiment."""
    m = input_manifest or {}
    text_model = getattr(args, "text_model", None) or m.get("text_model") or CODEBERT
    return text_model, int(m.get("num_node_labels", NUM_NODE_LABELS))


def load_input_manifest(args):
    """(num_relations, node_feat_dim, num_classes) from input/manifest.json. An
    old manifest without num_classes is binary (2). --num_classes may repeat the
    manifest's value but not contradict it; without a manifest it is the source."""
    p = Path(args.data_root) / "input" / "manifest.json"
    if p.exists():
        m = json.loads(p.read_text())
        num_classes = int(m.get("num_classes", 2))
        if args.num_classes is not None and int(args.num_classes) != num_classes:
            raise RuntimeError(f"--num_classes {args.num_classes} contradicts {p} (num_classes {num_classes}); "
                               f"the inputs were embedded for {num_classes} classes")
        return int(m.get("num_relations", NUM_RELATIONS)), int(m.get("node_feat_dim", 768)), num_classes
    return args.num_relations, 768, int(args.num_classes) if args.num_classes else 2


# ============================================================================
# main
# ============================================================================
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cpg", action="store_true", help="generate CPGs from the jsonl splits")
    p.add_argument("--embed", action="store_true", help="generate PyG inputs from the CPGs")
    p.add_argument("--molgraph", action="store_true",
                   help="chemistry: build molecular-graph inputs from jsonl splits whose func is a SMILES string "
                        "(replaces --cpg/--embed; see data/scripts/make_molnet_json.py)")
    p.add_argument("--resume", action="store_true", help="(--embed) keep already generated input files")
    p.add_argument("--mode", choices=["train", "test", "none"], default="none")
    # data
    p.add_argument("--data_root", default="data")
    p.add_argument("--dataset", default="bigvul", help="prefix of data/cleaned_data/<dataset>_<split>.jsonl")
    p.add_argument("--num_classes", type=int, default=None,
                   help="number of classes; default: inferred at --embed (max target + 1, at least 2) and read from "
                        "input/manifest.json at --mode train. 2 = binary (P(y=1) + tuned threshold), >2 = argmax")
    p.add_argument("--graph_unit", choices=["first_method", "whole_file"], default="first_method",
                   help="first_method: the first real method of each source file (BigVul-style single functions); "
                        "whole_file: Joern's per-file <global> graph, i.e. the whole program (POJ-104 / CodeNet)")
    p.add_argument("--train_jsonl"); p.add_argument("--valid_jsonl"); p.add_argument("--test_jsonl")
    p.add_argument("--max_chars", type=int, default=None, help="drop functions with >= this many chars (thesis code: 1200, undisclosed). Default: no filter")
    p.add_argument("--min_chars", type=int, default=0)
    p.add_argument("--slice_size", type=int, default=500)
    p.add_argument("--joern_cli_dir", default="joern/joern-cli/")
    p.add_argument("--normalize_graph_identifiers", action="store_true", help="apply FUN_i/VAR_j normalisation to the Joern input (thesis 3.2.2); text branch keeps raw code")
    p.add_argument("--keep_intermediate", action="store_true")
    p.add_argument("--rescue_stages", default="ABC",
                   help="rescue stages to try, as letters from ABC (validated; default ABC). Stage C (lowercase signature macros "
                        "such as av_cold / coroutine_fn, return-type macros) was added after the first BigVul runs; "
                        "pass AB to regenerate a BigVul CPG set with the earlier behaviour")
    p.add_argument("--no_rescue_macros", action="store_true",
                   help="do not retry Joern on functions whose signature it could not parse (default: retry with signature macros stripped; the text branch still sees the original code)")
    p.add_argument("--max_nodes", type=int, default=500)
    p.add_argument("--node_embedding", choices=["codebert", "fast"], default="codebert")
    p.add_argument("--node_max_len", type=int, default=128)
    p.add_argument("--node_cache_size", type=int, default=200_000, help="LRU size of the CODE-string -> CLS cache used at --embed time")
    p.add_argument("--node_code_mode", choices=["full", "first_line"], default="full",
                   help="CODE text embedded per node: full Joern CODE (thesis; container nodes carry the whole nested source) or its first line only")
    p.add_argument("--reverse_edges", action="store_true", help="add reversed edges as relations 3..5")
    p.add_argument("--num_relations", type=int, default=3, help="used only when input/manifest.json is absent")
    # model
    p.add_argument("--model_type", choices=["hybrid", "codebert", "graph_only"], default="hybrid")
    p.add_argument("--ablation", choices=["none", "no_gate", "no_cls_skip", "shuffle_edges", "no_input_norm", "no_residual"], default="none")
    p.add_argument("--text_model", default=None,
                   help="HF id of the text-branch encoder (default: input/manifest.json's text_model, else "
                        "microsoft/codebert-base). Recorded in args.json")
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
    p.add_argument("--select_on", choices=["tuned", "f1_05", "accuracy", "macro_f1", "auc"], default="tuned",
                   help="validation metric for checkpoint selection. Binary: tuned = F1 at the tuned threshold, "
                        "f1_05 = F1 at 0.5, accuracy = accuracy at the tuned threshold. Multi-class: tuned and "
                        "macro_f1 = macro-F1 (argmax), f1_05 = the same, accuracy = accuracy")
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
    p.add_argument("--run_prefix", default="", help="prefix of the run directory name, e.g. 'poj104_' (default: none)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing run directory of the same name")
    p.add_argument("--run_dir", help="(--mode test) run directory holding best.pth and args.json")
    p.add_argument("--allow_other_data_root", action="store_true",
                   help="(--mode test) evaluate on a --data_root different from the one the run was trained on")
    return p


def main(argv=None):
    # long jobs are usually run with output redirected to a file: flush per line
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except Exception:
            pass
    args = build_parser().parse_args(argv)
    for split in ("train", "valid", "test"):
        if getattr(args, f"{split}_jsonl") is None:
            setattr(args, f"{split}_jsonl", str(Path(args.data_root) / "cleaned_data" / f"{args.dataset}_{split}.jsonl"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.molgraph and (args.cpg or args.embed):
        raise SystemExit("--molgraph replaces --cpg / --embed; do not combine them")
    if args.molgraph:
        mol_generate(args)
    if args.cpg:
        cpg_generate(args)
    if args.embed:
        embed_generate(args)

    if args.mode == "train":
        num_relations, node_feat_dim, num_classes = load_input_manifest(args)
        multiclass = num_classes != 2
        results = []
        for bs in args.batch_sizes:
            for lr in args.lrs:
                for ratio in args.bert_lr_ratios:
                    for seed in args.seeds:
                        r = train_run(args, device, batch_size=bs, lr=lr, ratio=ratio, seed=seed,
                                      num_relations=num_relations, node_feat_dim=node_feat_dim, num_classes=num_classes)
                        results.append((run_name_for(args, bs, lr, ratio, seed), r))
        if multiclass:
            print(f"\n===== SUMMARY ({num_classes} classes, argmax; test macro-F1 / accuracy) =====")
        else:
            print("\n===== SUMMARY (test F1 at validation-tuned threshold) =====")
        for name, r in sorted(results, key=lambda x: -x[1]["test_tuned"]["f1"]):
            tt = r["test_tuned"]
            if multiclass:
                print(f"{name}: test macro-F1 {tt['macro_f1']:.4f} | accuracy {tt['accuracy']:.4f}"
                      + (f" | top5 {tt['top5_accuracy']:.4f}" if tt.get("top5_accuracy") is not None else "")
                      + f" | best epoch {r['best_epoch']}")
            else:
                print(f"{name}: test F1 {tt['f1']:.4f} (thr {_fmt_thr(tt['threshold'])}) | "
                      f"F1@0.5 {r['test_05']['f1']:.4f} | best epoch {r['best_epoch']}")
        if len(args.seeds) > 1:
            by_cfg = {}
            for name, r in results:
                by_cfg.setdefault(name.rsplit("_seed", 1)[0], []).append(r["test_tuned"]["f1"])
            label = "macro-F1" if multiclass else "F1"
            for cfg, f1s in by_cfg.items():
                print(f"{cfg}: mean test {label} {np.mean(f1s):.4f} +/- {np.std(f1s, ddof=1) if len(f1s) > 1 else 0:.4f} over {len(f1s)} seeds")

    if args.mode == "test":
        if not args.run_dir:
            raise SystemExit("--mode test needs --run_dir")
        run_dir = Path(args.run_dir)
        meta = json.loads((run_dir / "args.json").read_text())
        a = argparse.Namespace(**meta["args"])
        # a run belongs to one dataset: evaluating it on another data_root silently mixes datasets
        current_root = str(Path(args.data_root).resolve())
        stored_root = meta.get("data_root")
        other_root = stored_root is not None and stored_root != current_root
        if stored_root is None:
            print(f"[WARN] {run_dir / 'args.json'} predates the data_root record; evaluating on {current_root} unchecked")
        elif other_root and not args.allow_other_data_root:
            raise RuntimeError(f"--mode test on a different data_root than the run was trained on:\n"
                               f"  trained on : {stored_root}\n  now        : {current_root}\n"
                               f"pass --allow_other_data_root if this is intended")
        # whatever the root, the inputs must have the shape the model was built for (a wider head on
        # a narrower label set would evaluate silently with a macro average over absent classes)
        target_manifest = Path(args.data_root) / "input" / "manifest.json"
        if target_manifest.exists():
            tm = json.loads(target_manifest.read_text())
            for key, model_val, data_val in (("num_classes", int(meta.get("num_classes", 2)), int(tm.get("num_classes", 2))),
                                             ("num_relations", int(meta["num_relations"]), int(tm.get("num_relations", NUM_RELATIONS))),
                                             ("node_feat_dim", int(meta["node_feat_dim"]), int(tm.get("node_feat_dim", 768)))):
                if model_val != data_val:
                    raise RuntimeError(f"--mode test: the run's model has {key} {model_val} but {target_manifest} "
                                       f"says {data_val}; these inputs cannot be evaluated with this checkpoint")
        set_seed(meta["seed"], deterministic=not a.no_deterministic)
        model = build_model(a.model_type, device, num_relations=meta["num_relations"], node_feat_dim=meta["node_feat_dim"],
                            hidden_dim=a.hidden_dim, num_rgcn_layers=a.rgcn_layers, fusion_layers=a.fusion_layers,
                            tune_last_n_layers=a.tune_last_n_layers, max_length=a.max_length, ablation=a.ablation,
                            gate_temperature=a.gate_temperature, seed=meta["seed"],
                            num_classes=int(meta.get("num_classes", 2)),
                            model_name=meta.get("text_model") or CODEBERT,
                            num_node_labels=int(meta.get("num_node_labels", NUM_NODE_LABELS)))
        model.load_state_dict(torch.load(run_dir / "best.pth", map_location=device))
        input_dir = Path(args.data_root) / "input"
        val_loader, _ = make_loader(input_dir, "valid", batch_size=args.eval_batch_size, shuffle=False, seed=0)
        test_loader, _ = make_loader(input_dir, "test", batch_size=args.eval_batch_size, shuffle=False, seed=0)
        if other_root:
            # evaluation on another dataset never overwrites the run's own metrics / predictions
            out_dir = run_dir / f"reeval_{Path(current_root).name}"
            out_dir.mkdir(parents=True, exist_ok=True)
            res = evaluate_run(model, out_dir, val_loader, test_loader, device)
            res.update({"reevaluated_on": current_root, "trained_on": stored_root})
            (out_dir / "metrics.json").write_text(json.dumps(res, indent=1))
            print(f"[TEST] other data_root: results in {out_dir}")
        else:
            res = evaluate_run(model, run_dir, val_loader, test_loader, device)
            metrics_path = run_dir / "metrics.json"
            merged = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
            merged.update(res)          # keep best_epoch / best_val_selection_f1 / select_on from training
            merged["reevaluated"] = True
            metrics_path.write_text(json.dumps(merged, indent=1))
        print(json.dumps({k: v for k, v in res.items() if k.startswith("test")}, indent=1))


if __name__ == "__main__":
    main()
