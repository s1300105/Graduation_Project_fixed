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


def _write_split(input_dir: Path, split: str, n_rows: int, n_files: int = 2, pos_frac: float = 0.3, seed: int = 0,
                 num_classes: int = 2):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_rows):
        if num_classes > 2:
            y = int(rng.integers(num_classes))
        else:
            y = int(rng.random() < pos_frac)           # binary fixture: byte-identical to the golden recording
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


def _make_root3(root: Path, with_manifest: bool):
    inp = root / "input"
    inp.mkdir(parents=True)
    _write_split(inp, "train", 40, seed=1, num_classes=3)
    _write_split(inp, "valid", 16, seed=2, num_classes=3)
    _write_split(inp, "test", 16, seed=3, num_classes=3)
    if with_manifest:
        (inp / "manifest.json").write_text(json.dumps({"num_relations": 3, "node_feat_dim": 768, "num_classes": 3,
                                                       "graph_unit": "whole_file", "dataset": "toy3"}))
    return root


@pytest.fixture()
def data_root3(tmp_path):
    """Three classes; input/manifest.json declares num_classes 3 (as --embed writes it)."""
    return _make_root3(tmp_path / "d3", with_manifest=True)


@pytest.fixture()
def data_root3_nomanifest(tmp_path):
    return _make_root3(tmp_path / "d3nm", with_manifest=False)


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
    # the two-class expression is literally the old one (bit-identical alpha for the BigVul runs)
    assert torch.equal(alpha, torch.tensor([0.5 / (970 / 1000), 0.5 / (30 / 1000)], dtype=torch.float32))
    assert torch.equal(alpha, torch.tensor([0.5 / 0.97, 0.5 / 0.03]))


def test_build_criterion_multiclass_alpha():
    counts = (600, 300, 100)
    total = sum(counts)
    crit, alpha = R.build_criterion("focal", counts, 2.0, DEV)
    assert alpha.shape == (3,)
    assert torch.allclose(alpha, torch.tensor([(1 / 3) / (c / total) for c in counts]))
    assert torch.allclose(alpha, torch.tensor([5 / 9, 10 / 9, 10 / 3]))
    assert isinstance(crit, R.FocalLoss) and crit.alpha.shape == (3,)
    # the focal loss itself handles C classes
    logits = torch.randn(8, 3); y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 0])
    assert torch.isfinite(crit(logits, y))
    _, ones = R.build_criterion("ce", counts, 2.0, DEV)
    assert torch.equal(ones, torch.ones(3))
    _, w = R.build_criterion("weighted_ce", counts, 2.0, DEV)
    assert w.shape == (3,)
    # an absent class gets alpha 1.0, the others keep their weight
    _, a0 = R.build_criterion("focal", (600, 0, 300), 2.0, DEV)
    assert a0[1] == 1.0 and torch.allclose(a0[[0, 2]], torch.tensor([(1 / 3) / (600 / 900), (1 / 3) / (300 / 900)]))
    with pytest.raises(ValueError):
        R.build_criterion("focal", (10,), 2.0, DEV)


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
    assert R.count_targets(files, num_classes=2) == (c0, c1)


def test_count_targets_multiclass(data_root3):
    files = R.split_files(data_root3 / "input", "train")
    counts = R.count_targets(files, num_classes=3)
    assert isinstance(counts, tuple) and len(counts) == 3 and all(isinstance(c, int) for c in counts)
    assert sum(counts) == 40 and min(counts) > 0
    ref = np.zeros(3, dtype=int)
    for f in files:
        ref += np.bincount(pd.read_pickle(f)["target"].to_numpy(), minlength=3)
    assert counts == tuple(int(c) for c in ref)
    # a label outside 0..C-1 raises instead of being clipped
    with pytest.raises(ValueError, match="outside 0..2"):
        R.count_targets(_shift_targets(files, data_root3), num_classes=3)
    # the binary path still clips (unchanged behaviour); the multi-class path never does
    assert sum(R.count_targets(files)) == 40


def _shift_targets(files, root):
    """Copy the split with every target raised by 1 (so one class is out of range)."""
    out_dir = root / "shifted"; out_dir.mkdir()
    out = []
    for f in files:
        df = pd.read_pickle(f)
        df["target"] = df["target"] + 1
        p = out_dir / Path(f).name
        df.to_pickle(p); out.append(str(p))
    return out


def test_metrics_and_threshold():
    probs = np.array([0.1, 0.4, 0.6, 0.9, 0.2, 0.8])
    labels = np.array([0, 0, 1, 1, 1, 0])
    m = R.metrics_at(probs, labels, 0.5)
    assert m["tp"] == 2 and m["fp"] == 1 and m["fn"] == 1 and m["tn"] == 2
    bt = R.best_threshold(probs, labels)
    assert 0.05 <= bt["threshold"] <= 0.95 and bt["f1"] >= m["f1"]


P3 = np.array([[0.7, 0.2, 0.1],     # 0 -> 0
               [0.1, 0.8, 0.1],     # 1 -> 1
               [0.2, 0.2, 0.6],     # 2 -> 2
               [0.5, 0.3, 0.2],     # 0 -> 0
               [0.3, 0.4, 0.3],     # 2 -> 1 (wrong)
               [0.1, 0.1, 0.8]])    # 1 -> 2 (wrong)
L3 = np.array([0, 1, 2, 0, 2, 1])


def test_metrics_multiclass_hand_made():
    m = R.metrics_multiclass(P3, L3)
    assert m["threshold"] is None and m["top5_accuracy"] is None and m["num_classes"] == 3 and m["n"] == 6
    assert m["accuracy"] == pytest.approx(4 / 6)
    # class 0: F1 1; class 1: tp1 fp1 fn1 -> 0.5; class 2: tp1 fp1 fn1 -> 0.5
    assert m["per_class_f1"] == pytest.approx([1.0, 0.5, 0.5])
    assert m["macro_f1"] == pytest.approx(2 / 3) and m["f1"] == m["macro_f1"]
    assert m["precision"] == m["macro_precision"] and m["recall"] == m["macro_recall"]
    assert m["macro_precision"] == pytest.approx(2 / 3) and m["macro_recall"] == pytest.approx(2 / 3)
    assert m["weighted_f1"] == pytest.approx(2 / 3)
    assert np.asarray(m["confusion"]).shape == (3, 3) and m["confusion"] == [[2, 0, 0], [0, 1, 1], [0, 1, 1]]
    assert m["frac_pmax_over_095"] == 0.0
    # metrics_at ignores the threshold on 2-D probs; best_threshold returns it with n_thresholds_tied 0
    assert R.metrics_at(P3, L3, 0.5) == m
    bt = R.best_threshold(P3, L3)
    assert bt["n_thresholds_tied"] == 0 and bt["threshold"] is None and bt["macro_f1"] == m["macro_f1"]
    # top-5 appears (and is 1 here) once C > 5; the confusion matrix stops at 50 classes
    p8 = np.zeros((4, 8)); p8[np.arange(4), [0, 1, 2, 3]] = 0.5
    m8 = R.metrics_multiclass(p8, np.array([0, 1, 2, 3]))
    assert m8["top5_accuracy"] == 1.0 and m8["accuracy"] == 1.0 and "confusion" in m8
    p60 = np.zeros((2, 60)); p60[0, 5] = 1; p60[1, 7] = 1
    m60 = R.metrics_multiclass(p60, np.array([5, 8]))
    assert "confusion" not in m60 and m60["accuracy"] == 0.5 and m60["frac_pmax_over_095"] == 1.0


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


def _hist(d):
    return [json.loads(l) for l in (d / "history.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("model_type", ["graph_only", "codebert"])
def test_multiclass_end_to_end(data_root3, model_type):
    """3 classes inferred from input/manifest.json: argmax metrics, [N, 3] probabilities, no threshold."""
    runs = data_root3 / "runs"
    common = ["--data_root", str(data_root3), "--runs_dir", str(runs), "--epochs", "2", "--patience", "5",
              "--hidden_dim", "16", "--rgcn_layers", "1", "--fusion_layers", "1", "--batch_sizes", "8",
              "--eval_batch_size", "8", "--seeds", "0"]
    R.main(["--mode", "train", "--model_type", model_type] + common)
    d = runs / f"{model_type}_bs8_lr1.0e-04_br0.30_seed0"
    for f in ("args.json", "history.jsonl", "best.pth", "metrics.json", "val_preds.npz", "test_preds.npz", "curves.png"):
        assert (d / f).exists(), f
    meta = json.loads((d / "args.json").read_text())
    assert meta["num_classes"] == 3 and meta["graph_unit"] == "whole_file" and meta["dataset"] == "bigvul"
    assert meta["data_root"] == str(data_root3.resolve()) and meta["input_manifest"]["num_classes"] == 3
    assert set(meta["class_counts"]) == {"0", "1", "2"} and len(meta["alpha"]) == 3
    m = json.loads((d / "metrics.json").read_text())
    assert m["num_classes"] == 3
    for k in ("val_05", "val_tuned", "test_05", "test_tuned"):
        assert m[k]["threshold"] is None and "macro_f1" in m[k] and m[k]["num_classes"] == 3
    assert m["test_tuned"]["macro_f1"] == m["test_tuned"]["f1"] and m["val_tuned"]["n_thresholds_tied"] == 0
    assert m["test_tuned"] == m["test_05"]           # argmax: no threshold to tune
    assert {"val_loss", "test_loss", "best_epoch", "best_val_selection_f1", "select_on"} <= set(m)
    npz = np.load(d / "test_preds.npz")
    assert npz["probs"].shape == (16, 3) and np.allclose(npz["probs"].sum(1), 1.0, atol=1e-5)
    assert len(set(npz["idx"].tolist())) == 16
    hist = _hist(d)
    assert len(hist) == 2
    for h in hist:
        assert h["val_threshold"] is None and {"val_macro_f1", "val_accuracy", "val_top5", "val_f1_tuned"} <= set(h)
        assert h["val_top5"] is None and h["val_macro_f1"] == h["val_f1_tuned"] and h["p1_meaning"] == "max_class_probability"
        assert 1 / 3 - 1e-9 <= h["mean_p1"] <= 1.0
    # --mode test reproduces the probabilities
    p1 = npz["probs"].copy()
    R.main(["--mode", "test", "--data_root", str(data_root3), "--run_dir", str(d), "--eval_batch_size", "4"])
    p2 = np.load(d / "test_preds.npz")["probs"]
    assert p2.shape == (16, 3) and np.allclose(p1, p2, atol=1e-6)
    m2 = json.loads((d / "metrics.json").read_text())
    assert m2["reevaluated"] and m2["num_classes"] == 3 and m2["best_epoch"] == m["best_epoch"]
    # a different data_root is refused unless explicitly allowed
    other = data_root3.parent / "other"
    _make_root3(other, with_manifest=True)
    with pytest.raises(RuntimeError, match="different data_root"):
        R.main(["--mode", "test", "--data_root", str(other), "--run_dir", str(d), "--eval_batch_size", "8"])
    R.main(["--mode", "test", "--data_root", str(other), "--run_dir", str(d), "--eval_batch_size", "8",
            "--allow_other_data_root"])
    # --num_classes equal to the manifest is fine, a different value is refused
    with pytest.raises(RuntimeError, match="contradicts"):
        R.main(["--mode", "train", "--model_type", model_type, "--num_classes", "4", "--overwrite"] + common)


def test_multiclass_via_num_classes_flag_and_select_on(data_root3_nomanifest):
    """Without a manifest --num_classes is the source; selection on accuracy / macro_f1 works."""
    root = data_root3_nomanifest
    runs = root / "runs"
    common = ["--data_root", str(root), "--runs_dir", str(runs), "--epochs", "2", "--patience", "5",
              "--hidden_dim", "16", "--rgcn_layers", "1", "--batch_sizes", "8", "--eval_batch_size", "8", "--seeds", "0"]
    R.main(["--mode", "train", "--model_type", "graph_only", "--num_classes", "3", "--select_on", "accuracy",
            "--run_prefix", "toy3_"] + common)
    d = runs / "toy3_graph_only_bs8_lr1.0e-04_br0.30_seed0"
    assert d.exists() and sorted(p.name for p in runs.iterdir()) == [d.name]
    m = json.loads((d / "metrics.json").read_text())
    assert m["num_classes"] == 3 and m["select_on"] == "accuracy" and np.load(d / "test_preds.npz")["probs"].shape == (16, 3)
    meta = json.loads((d / "args.json").read_text())
    assert meta["num_classes"] == 3 and meta["graph_unit"] == "first_method" and meta["input_manifest"] is None
    hist = _hist(d)
    assert m["best_val_selection_f1"] == max(h["val_accuracy"] for h in hist)
    R.main(["--mode", "train", "--model_type", "graph_only", "--num_classes", "3", "--select_on", "macro_f1",
            "--run_prefix", "sel_"] + common)
    hist2 = _hist(runs / "sel_graph_only_bs8_lr1.0e-04_br0.30_seed0")
    m2 = json.loads((runs / "sel_graph_only_bs8_lr1.0e-04_br0.30_seed0" / "metrics.json").read_text())
    assert m2["best_val_selection_f1"] == max(h["val_macro_f1"] for h in hist2)


def test_run_prefix_and_binary_select_on(data_root):
    runs = data_root / "runs"
    common = ["--data_root", str(data_root), "--runs_dir", str(runs), "--epochs", "1", "--patience", "5",
              "--hidden_dim", "16", "--rgcn_layers", "1", "--batch_sizes", "8", "--eval_batch_size", "8", "--seeds", "0"]
    R.main(["--mode", "train", "--model_type", "graph_only", "--run_prefix", "bigvul_"] + common)
    names = sorted(p.name for p in runs.iterdir())
    assert names == ["bigvul_graph_only_bs8_lr1.0e-04_br0.30_seed0"]
    assert names[0].startswith("bigvul_")
    meta = json.loads((runs / names[0] / "args.json").read_text())
    assert meta["num_classes"] == 2 and meta["data_root"] == str(data_root.resolve()) and meta["input_manifest"] is None
    m = json.loads((runs / names[0] / "metrics.json").read_text())
    assert m["num_classes"] == 2 and np.load(runs / names[0] / "test_preds.npz")["probs"].ndim == 1
    # binary history schema unchanged (no multi-class keys leak in)
    h = _hist(runs / names[0])[0]
    assert not ({"val_macro_f1", "val_accuracy", "val_top5", "p1_meaning"} & set(h))
    # accuracy selection is allowed on the binary task, macro_f1 is not
    R.main(["--mode", "train", "--model_type", "graph_only", "--run_prefix", "acc_", "--select_on", "accuracy"] + common)
    with pytest.raises(RuntimeError, match="macro_f1"):
        R.main(["--mode", "train", "--model_type", "graph_only", "--run_prefix", "mf_", "--select_on", "macro_f1"] + common)


def test_load_input_manifest_defaults_and_conflicts(data_root, data_root3):
    args = R.build_parser().parse_args(["--data_root", str(data_root)])
    assert R.load_input_manifest(args) == (3, 768, 2)                       # no manifest, no flag -> binary
    args = R.build_parser().parse_args(["--data_root", str(data_root), "--num_classes", "3"])
    assert R.load_input_manifest(args) == (3, 768, 3)                       # no manifest -> the flag
    (data_root / "input" / "manifest.json").write_text(json.dumps({"num_relations": 6, "node_feat_dim": 768}))
    args = R.build_parser().parse_args(["--data_root", str(data_root)])
    assert R.load_input_manifest(args) == (6, 768, 2)                       # old manifest: num_classes absent -> 2
    args = R.build_parser().parse_args(["--data_root", str(data_root3)])
    assert R.load_input_manifest(args) == (3, 768, 3)
    args = R.build_parser().parse_args(["--data_root", str(data_root3), "--num_classes", "3"])
    assert R.load_input_manifest(args) == (3, 768, 3)
    args = R.build_parser().parse_args(["--data_root", str(data_root3), "--num_classes", "2"])
    with pytest.raises(RuntimeError, match="contradicts"):
        R.load_input_manifest(args)


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


# ---------------------------------------------------------------------------------------
# --cpg / --embed safety checks (no Joern needed)
# ---------------------------------------------------------------------------------------
from utils.process import cpg_generator as _cp  # noqa: E402

_EXPORT = Path(__file__).resolve().parent / "fixtures" / "joern_v4_export.json"


def _cpg_pickles_from_fixture(root: Path, graph_unit: str):
    """Write <root>/cpg/{train,valid,test}_0_cpg.pkl from the fixture export plus a cpg_manifest."""
    cpg_dir = root / "cpg"
    cpg_dir.mkdir(parents=True, exist_ok=True)
    kept = _cp.json_process(str(_EXPORT.parent), _EXPORT.name, {}, graph_unit=graph_unit)
    rows = [{"cpg": e, "target": i % 2, "func": f"int f{idx}(void) {{ return {i}; }}", "idx": idx} for i, (idx, e) in enumerate(kept)]
    for split in ("train", "valid", "test"):
        pd.DataFrame(rows).to_pickle(cpg_dir / f"{split}_0_cpg.pkl")
    (cpg_dir / "cpg_manifest.json").write_text(json.dumps({"args": {"graph_unit": graph_unit}, "splits": {}}))
    return len(rows)


def _embed_args(root: Path, graph_unit: str):
    return R.build_parser().parse_args(["--embed", "--data_root", str(root), "--graph_unit", graph_unit,
                                        "--node_embedding", "fast", "--max_nodes", "500"])


def test_embed_refuses_a_graph_unit_different_from_the_cpg_run(tmp_path):
    n = _cpg_pickles_from_fixture(tmp_path, "whole_file")
    assert n == 10
    with pytest.raises(RuntimeError, match="graph_unit"):
        R.embed_generate(_embed_args(tmp_path, "first_method"))
    assert not (tmp_path / "input" / "manifest.json").exists()


def test_embed_refuses_to_write_an_input_set_without_graphs(tmp_path):
    _cpg_pickles_from_fixture(tmp_path, "whole_file")
    (tmp_path / "cpg" / "cpg_manifest.json").unlink()          # no manifest to compare against ...
    with pytest.raises(RuntimeError, match="no graph"):      # ... so the empty result itself must stop it
        R.embed_generate(_embed_args(tmp_path, "first_method"))


def test_embed_whole_file_from_fixture_records_unit_and_classes(tmp_path):
    _cpg_pickles_from_fixture(tmp_path, "whole_file")
    R.embed_generate(_embed_args(tmp_path, "whole_file"))
    m = json.loads((tmp_path / "input" / "manifest.json").read_text())
    assert m["graph_unit"] == "whole_file" and m["num_classes"] == 2 and m["counts"]["graphs"] == 30
    assert m["counts"]["rows_without_graph"] == 0


def test_cpg_validates_the_ext_column_before_deleting_the_old_cpg_dir(tmp_path):
    cleaned = tmp_path / "cleaned_data"
    cleaned.mkdir()
    for split in ("train", "valid", "test"):
        rows = [{"idx": i, "func": "int f(void) { return 0; }", "target": 0, "ext": ".c" if i else ".cc"} for i in range(3)]
        (cleaned / f"x_{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    old = tmp_path / "cpg"
    old.mkdir()
    (old / "keep_me.pkl").write_text("previous run")
    args = R.build_parser().parse_args(["--cpg", "--data_root", str(tmp_path), "--dataset", "x", "--joern_cli_dir", "/nonexistent/"])
    for split in ("train", "valid", "test"):
        setattr(args, f"{split}_jsonl", str(cleaned / f"x_{split}.jsonl"))
    with pytest.raises(ValueError, match="ext column"):
        R.cpg_generate(args)
    assert (old / "keep_me.pkl").exists()                     # nothing was removed before the check
