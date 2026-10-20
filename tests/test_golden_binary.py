"""Bit-identity guard for the binary (BigVul-style) pipeline.

tests/fixtures/golden_binary_v1.json holds the validation / test probabilities,
metrics.json and history.jsonl of four tiny CPU runs (graph_only, hybrid,
codebert, graph_only+shuffle_edges; seed 0, 2 epochs) recorded at commit
0da5a09 on the synthetic data of tests/test_training.py. Any later change
(multi-class support, whole-program graphs, ...) must leave these outputs
unchanged: binary callers are the existing BigVul experiments.

The comparison is exact. Floating point results depend on the torch build, so
the test is skipped when the installed torch differs from the recorded one
(regenerate with scratch script make_golden.py in that case, from a commit
that is known to be good).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run as R  # noqa: E402
from test_training import _write_split  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "golden_binary_v1.json"


@pytest.fixture(scope="module")
def golden():
    g = json.loads(GOLDEN.read_text())
    if g["torch"] != torch.__version__:
        pytest.skip(f"golden recorded with torch {g['torch']}, installed {torch.__version__}")
    return g["runs"]


@pytest.fixture(scope="module")
def golden_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("golden")
    inp = root / "input"
    inp.mkdir()
    _write_split(inp, "train", 40, seed=1)
    _write_split(inp, "valid", 16, seed=2)
    _write_split(inp, "test", 16, seed=3)
    return root


COMMON = ["--epochs", "2", "--patience", "5", "--hidden_dim", "16", "--rgcn_layers", "1", "--fusion_layers", "1",
          "--batch_sizes", "8", "--eval_batch_size", "8", "--seeds", "0"]


@pytest.mark.parametrize("model_type,ablation", [
    ("graph_only", "none"), ("graph_only", "shuffle_edges"), ("hybrid", "none"), ("codebert", "none"),
])
def test_binary_outputs_are_bit_identical(golden, golden_root, model_type, ablation):
    runs = golden_root / "runs"
    args = ["--mode", "train", "--model_type", model_type, "--data_root", str(golden_root), "--runs_dir", str(runs)] + COMMON
    if ablation != "none":
        args += ["--ablation", ablation]
    R.main(args + ["--overwrite"])
    name = R.run_name_for(R.build_parser().parse_args(args), 8, 1e-4, 0.3, 0)
    ref = golden[name]
    d = runs / name
    z = np.load(d / "test_preds.npz")
    v = np.load(d / "val_preds.npz")
    assert z["probs"].tolist() == ref["test_probs"]
    assert z["idx"].tolist() == ref["test_idx"] and z["labels"].tolist() == ref["test_labels"]
    assert v["probs"].tolist() == ref["val_probs"]
    m = json.loads((d / "metrics.json").read_text())
    for k in ("val_05", "val_tuned", "test_05", "test_tuned", "val_loss", "test_loss", "best_epoch", "best_val_selection_f1", "select_on"):
        assert m[k] == ref["metrics"][k], k
    hist = [json.loads(l) for l in (d / "history.jsonl").read_text().splitlines()]
    # every recorded key keeps its value; later changes may only ADD keys (e.g. val_auc)
    assert set(ref["history_keys"]) <= set(hist[0])
    for h, hr in zip(hist, ref["history"]):
        for k in ref["history_keys"]:
            if k == "elapsed_s":
                continue
            assert h[k] == hr[k], (k, h[k], hr[k])
