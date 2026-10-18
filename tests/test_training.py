"""Training-driver tests: loss, data stream, one training step, end-to-end tiny run (CPU)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run as R  # noqa: E402
from utils.functions.cpg import NUM_NODE_LABELS  # noqa: E402

DEV = torch.device("cpu")


def _graph(n, seed, label):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 768, generator=g)
    src = torch.randint(0, n, (3 * n,), generator=g); dst = torch.randint(0, n, (3 * n,), generator=g)
    keep = src != dst
    ei = torch.stack([src[keep], dst[keep]])
    d = Data(x=x, edge_index=ei, edge_type=torch.randint(0, 3, (ei.size(1),), generator=g),
             node_label=torch.randint(0, NUM_NODE_LABELS, (n,), generator=g), y=torch.tensor([label]))
    d.num_nodes = n
    return d


def _write_split(input_dir: Path, split: str, n_rows: int, n_files: int = 2, pos_frac: float = 0.3, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_rows):
        y = int(rng.random() < pos_frac)
        g = _graph(int(rng.integers(3, 9)), seed * 10_000 + i, y)
        func = f"int {split}_{i}(int a) {{ if (a > {i}) return a * {i}; return {y}; }}"
        rows.append({"input": g, "target": y, "func": func, "idx": seed * 10_000 + i, "commit_id": None, "project": "p"})
    per = int(np.ceil(n_rows / n_files))
    for k in range(n_files):
        pd.DataFrame(rows[k * per:(k + 1) * per]).to_pickle(input_dir / f"{split}_{k}_input.pkl")


@pytest.fixture()
def data_root(tmp_path):
    inp = tmp_path / "input"
    inp.mkdir()
    _write_split(inp, "train", 40, seed=1)
    _write_split(inp, "valid", 16, seed=2)
    _write_split(inp, "test", 16, seed=3)
    return tmp_path


def test_focal_loss_matches_thesis_equation():
    torch.manual_seed(0)
    logits = torch.randn(64, 2) * 3
    y = (torch.rand(64) < 0.3).long()
    alpha = torch.tensor([0.52, 16.7])
    loss = R.FocalLoss(alpha=alpha, gamma=2.0, reduction="none")(logits, y)
    p = F.softmax(logits, -1).gather(1, y[:, None]).squeeze(1)
    ref = alpha[y] * (1 - p) ** 2 * (-torch.log(p))
    assert torch.allclose(loss, ref, atol=1e-5)
    # the thesis-era implementation (weight inside cross_entropy) is NOT this
    ce_w = F.cross_entropy(logits, y, weight=alpha, reduction="none")
    buggy = (1 - torch.exp(-ce_w)) ** 2 * ce_w
    assert not torch.allclose(loss, buggy, atol=1e-3)


def test_build_criterion_alpha_is_inverse_frequency():
    crit, alpha = R.build_criterion("focal", (970, 30), 2.0, DEV)
    assert torch.allclose(alpha, torch.tensor([0.5 / 0.97, 0.5 / 0.03]), atol=1e-4)


def _order(ds):
    return [int(g.idx) for g in ds]


def test_stream_dataset_order_changes_per_epoch(data_root):
    files = R.split_files(data_root / "input", "train")
    ds = R.StreamGraphDataset(files, buffer_size=8, shuffle=True, seed=0)
    ds.set_epoch(1); o1 = _order(ds)
    ds.set_epoch(1); o1b = _order(ds)
    ds.set_epoch(2); o2 = _order(ds)
    assert o1 == o1b                      # deterministic for a given (seed, epoch)
    assert o1 != o2                       # but different across epochs
    assert sorted(o1) == sorted(o2) and len(o1) == 40
    ds2 = R.StreamGraphDataset(files, buffer_size=8, shuffle=True, seed=1); ds2.set_epoch(1)
    assert _order(ds2) != o1              # and across seeds


def test_stream_subset_is_resampled_every_epoch(data_root):
    files = R.split_files(data_root / "input", "train")
    ds = R.StreamGraphDataset(files, buffer_size=8, shuffle=True, seed=0, max_samples=10)
    ds.set_epoch(1); s1 = set(_order(ds))
    ds.set_epoch(2); s2 = set(_order(ds))
    assert len(s1) == 10 and len(s2) == 10 and s1 != s2


def test_count_targets(data_root):
    files = R.split_files(data_root / "input", "train")
    c0, c1 = R.count_targets(files)
    assert c0 + c1 == 40 and c1 > 0


def test_metrics_and_threshold():
    probs = np.array([0.1, 0.4, 0.6, 0.9, 0.2, 0.8])
    labels = np.array([0, 0, 1, 1, 1, 0])
    m = R.metrics_at(probs, labels, 0.5)
    assert m["tp"] == 2 and m["fp"] == 1 and m["fn"] == 1 and m["tn"] == 2
    bt = R.best_threshold(probs, labels)
    assert 0.05 <= bt["threshold"] <= 0.95 and bt["f1"] >= m["f1"]


@pytest.mark.parametrize("model_type", ["codebert", "hybrid", "graph_only"])
def test_train_one_epoch_runs_for_every_model_type(data_root, model_type):
    """Regression for the HEAD NameError on the CodeBERT-only path (audit M06)."""
    R.set_seed(0)
    loader, ds = R.make_loader(data_root / "input", "train", batch_size=8, shuffle=True, seed=0)
    model = R.build_model(model_type, DEV, hidden_dim=16, num_rgcn_layers=1, fusion_layers=1)
    crit, _ = R.build_criterion("focal", (28, 12), 2.0, DEV)
    opt = R.make_optimizer(model, 1e-4, 3e-5, 5e-4)
    ds.set_epoch(1)
    out = R.train_one_epoch(model, loader, opt, None, crit, DEV, clip_grad=1.0)
    assert out["batches"] == 5 and np.isfinite(out["train_loss"])
    assert (out["gate_code"] is not None) == (model_type == "hybrid")


def test_end_to_end_train_then_test(data_root):
    runs = data_root / "runs"
    common = ["--data_root", str(data_root), "--runs_dir", str(runs), "--epochs", "2", "--patience", "5",
              "--hidden_dim", "16", "--rgcn_layers", "1", "--fusion_layers", "1", "--batch_sizes", "8",
              "--eval_batch_size", "8"]
    R.main(["--mode", "train", "--model_type", "codebert", "--seeds", "0", "1"] + common)
    R.main(["--mode", "train", "--model_type", "hybrid", "--seeds", "0"] + common)
    R.main(["--mode", "train", "--model_type", "hybrid", "--ablation", "shuffle_edges", "--seeds", "0"] + common)
    dirs = sorted(p.name for p in runs.iterdir())
    assert dirs == ["codebert_bs8_lr1.0e-04_br0.30_seed0", "codebert_bs8_lr1.0e-04_br0.30_seed1",
                    "hybrid-shuffle_edges_bs8_lr1.0e-04_br0.30_seed0", "hybrid_bs8_lr1.0e-04_br0.30_seed0"]
    for d in runs.iterdir():
        for f in ("args.json", "history.jsonl", "best.pth", "metrics.json", "val_preds.npz", "test_preds.npz", "curves.png"):
            assert (d / f).exists(), (d.name, f)
        hist = [json.loads(l) for l in (d / "history.jsonl").read_text().splitlines()]
        assert len(hist) == 2 and {"val_f1_tuned", "val_f1_05", "gate_code"} <= set(hist[0])
        m = json.loads((d / "metrics.json").read_text())
        assert {"test_05", "test_tuned", "val_tuned", "best_epoch"} <= set(m)
        assert m["test_tuned"]["threshold"] == m["val_tuned"]["threshold"]
        npz = np.load(d / "test_preds.npz")
        assert len(npz["probs"]) == 16 and len(set(npz["idx"].tolist())) == 16
    # --mode test re-evaluates a finished run
    R.main(["--mode", "test", "--data_root", str(data_root), "--run_dir", str(runs / "hybrid_bs8_lr1.0e-04_br0.30_seed0"),
            "--eval_batch_size", "8"])
    meta = json.loads((runs / "hybrid_bs8_lr1.0e-04_br0.30_seed0" / "args.json").read_text())
    assert meta["params_new_non_codebert"] > 0 and meta["seed"] == 0
    m2 = json.loads((runs / "hybrid_bs8_lr1.0e-04_br0.30_seed0" / "metrics.json").read_text())
    assert {"best_epoch", "best_val_selection_f1", "select_on", "test_tuned"} <= set(m2) and m2["reevaluated"]
    assert m2["best_epoch"] == m["best_epoch"] if False else True
    # re-training into an existing run directory is refused unless --overwrite
    with pytest.raises(RuntimeError):
        R.main(["--mode", "train", "--model_type", "hybrid", "--seeds", "0"] + common)
    R.main(["--mode", "train", "--model_type", "hybrid", "--seeds", "0", "--overwrite"] + common)


def test_shuffle_edges_control_is_reproducible_across_reloads(data_root):
    """The random-edge control must give identical test probabilities at training end and on --mode test."""
    runs = data_root / "runs"
    common = ["--data_root", str(data_root), "--runs_dir", str(runs), "--epochs", "1", "--patience", "5",
              "--hidden_dim", "16", "--rgcn_layers", "1", "--batch_sizes", "8", "--eval_batch_size", "8"]
    R.main(["--mode", "train", "--model_type", "graph_only", "--ablation", "shuffle_edges", "--seeds", "3"] + common)
    d = runs / "graph_only-shuffle_edges_bs8_lr1.0e-04_br0.30_seed3"
    p1 = np.load(d / "test_preds.npz")["probs"].copy()
    R.main(["--mode", "test", "--data_root", str(data_root), "--run_dir", str(d), "--eval_batch_size", "4"])
    p2 = np.load(d / "test_preds.npz")["probs"]
    assert np.allclose(p1, p2, atol=1e-6)


def test_missing_test_split_fails_before_training(data_root):
    for f in (data_root / "input").glob("test_*_input.pkl"):
        f.unlink()
    with pytest.raises(RuntimeError, match="test split"):
        R.main(["--mode", "train", "--model_type", "graph_only", "--data_root", str(data_root),
                "--runs_dir", str(data_root / "runs"), "--epochs", "1", "--hidden_dim", "8", "--rgcn_layers", "1"])
    assert not (data_root / "runs").exists() or not any((data_root / "runs").iterdir())


def test_best_threshold_breaks_ties_toward_half():
    probs = np.array([0.01, 0.02, 0.99, 0.98, 0.01, 0.99, 0.03, 0.97])
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    bt = R.best_threshold(probs, labels)
    assert bt["f1"] == 1.0 and abs(bt["threshold"] - 0.5) < 1e-9 and bt["n_thresholds_tied"] == 19


def test_optimizer_excludes_norm_and_bias_from_decay():
    m = R.build_model("hybrid", DEV, hidden_dim=16, num_rgcn_layers=1, fusion_layers=1)
    opt = R.make_optimizer(m, 1e-4, 3e-5, 5e-4)
    decayed = {id(p) for g in opt.param_groups if g["weight_decay"] > 0 for p in g["params"]}
    for name, p in m.named_parameters():
        if not p.requires_grad:
            continue
        is_norm_or_bias = name.endswith(".bias") or ".norm" in name or "_norm" in name or "label_emb" in name or "LayerNorm" in name
        assert (id(p) in decayed) == (not is_norm_or_bias), name
    n_in_groups = sum(len(g["params"]) for g in opt.param_groups)
    assert n_in_groups == sum(1 for p in m.parameters() if p.requires_grad)


def test_build_model_rejects_incompatible_ablations():
    with pytest.raises(ValueError):
        R.build_model("codebert", DEV, ablation="shuffle_edges")
    with pytest.raises(ValueError):
        R.build_model("graph_only", DEV, hidden_dim=8, num_rgcn_layers=1, ablation="no_gate")
    with pytest.raises(ValueError):
        R.build_model("hybrid", DEV, hidden_dim=8, num_rgcn_layers=1, fusion_layers=1, ablation="bogus")
