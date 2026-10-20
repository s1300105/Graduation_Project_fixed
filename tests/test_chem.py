"""Chemistry instance: SMILES -> molecular graph, MoleculeNet / OGB preparation, the edgeless
control, ROC-AUC selection / comparison and the fingerprint floor (CPU, no network)."""
import gzip
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("rdkit")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run as R  # noqa: E402
from models.LMGNN import build_model  # noqa: E402
from utils.process import molgraph as MG  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOLNET = _load("make_molnet_json", ROOT / "data" / "scripts" / "make_molnet_json.py")
CR = _load("compare_runs", ROOT / "scripts" / "compare_runs.py")
FP = _load("fingerprint_baseline", ROOT / "scripts" / "fingerprint_baseline.py")

# small real molecules from several scaffolds (benzene, pyridine, cyclohexane, acyclic, naphthalene ...)
SMILES = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C(=O)O", "Cc1ccccc1", "c1ccncc1", "Cc1ccncc1", "Oc1ccncc1",
          "C1CCCCC1O", "C1CCCCC1N", "CCO", "CCN", "CCCC", "c1ccc2ccccc2c1", "Oc1ccc2ccccc2c1", "C1CCOC1",
          "C1CCNC1", "O=C1CCCC1", "c1ccsc1", "Cc1ccsc1", "CC(=O)Oc1ccccc1C(=O)O"]


def test_aspirin_graph_has_typed_bonds_both_ways():
    d = MG.smiles_to_data("CC(=O)Oc1ccccc1C(=O)O", 1)
    assert d.num_nodes == 13 and d.x.shape == (13, MG.ATOM_FEAT_DIM) and MG.ATOM_FEAT_DIM == 174
    assert torch.all(d.x.sum(1) == 9)                                   # one hot per OGB feature
    assert d.edge_index.size(1) == 26                                   # 13 bonds, both directions
    assert d.edge_type.bincount(minlength=5).tolist() == [10, 4, 0, 12, 0]   # single, double, triple, aromatic
    pairs = set(map(tuple, d.edge_index.t().tolist()))
    assert all((j, i) in pairs for i, j in pairs)
    assert d.node_label.tolist().count(8) == 4 and d.node_label.max() < MG.NUM_ATOM_LABELS


def test_unparsable_and_bondless_molecules():
    assert MG.smiles_to_data("not a smiles", 0) is None
    ion = MG.smiles_to_data("[Na+]", 0)
    assert ion.num_nodes == 1 and ion.edge_index.tolist() == [[0], [0]] and ion.edge_type.tolist() == [0]
    assert MG.canonical_smiles("OC1=CC=CC=C1") == "Oc1ccccc1"


def test_scaffold_split_is_deterministic_and_keeps_scaffolds_together():
    from rdkit.Chem.Scaffolds import MurckoScaffold
    smiles = SMILES * 3
    a = MOLNET.scaffold_split(smiles)
    assert a == MOLNET.scaffold_split(smiles)
    by_scaffold = {}
    for s, part in zip(smiles, a):
        by_scaffold.setdefault(MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=True), set()).add(part)
    assert all(len(v) == 1 for v in by_scaffold.values())
    assert a.count("train") >= a.count("valid") and {"train", "valid", "test"} <= set(a)


def _ogb_zip(path: Path, smiles, labels, parts, label_col="p_np"):
    mol = pd.DataFrame({label_col: labels, "smiles": smiles, "mol_id": range(len(smiles))})
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("bbbp/mapping/mol.csv.gz", gzip.compress(mol.to_csv(index=False).encode()))
        for name in ("train", "valid", "test"):
            rows = [i for i, p in enumerate(parts) if p == name]
            z.writestr(f"bbbp/split/scaffold/{name}.csv.gz", gzip.compress(("\n".join(map(str, rows)) + "\n").encode()))


def test_make_molnet_json_ogb_keeps_the_shipped_split(tmp_path):
    smiles = SMILES + ["not a smiles", "c1ccccc1O"]                     # one unparsable row, one duplicate
    labels = [i % 2 for i in range(len(SMILES))] + [1, 1]
    parts = ["train"] * 14 + ["valid"] * 3 + ["test"] * 3 + ["test", "train"]
    raw = tmp_path / "raw"
    raw.mkdir()
    _ogb_zip(raw / "bbbp.zip", smiles, labels, parts)
    out = tmp_path / "out"
    stats = MOLNET.main(["--dataset", "ogb_molbbbp", "--input", str(raw), "--out_dir", str(out)])
    rows = {s: [json.loads(l) for l in (out / f"ogb_molbbbp_{s}.jsonl").read_text().splitlines()] for s in ("train", "valid", "test")}
    assert [len(rows[s]) for s in ("train", "valid", "test")] == [15, 3, 3]          # unparsable dropped, duplicate kept
    assert stats["duplicate_groups"]["groups"] == 1 and stats["duplicate_groups"]["groups_crossing_official_splits"] == 0
    assert all(r["lang"] == "smiles" and r["ext"] is None for s in rows for r in rows[s])
    assert sorted(r["idx"] for s in rows for r in rows[s]) == list(range(21))
    assert rows["valid"][0]["func"] == MG.canonical_smiles(SMILES[14])


def test_molgraph_stage_and_hybrid_no_edges_train_end_to_end(tmp_path, monkeypatch):
    """--molgraph -> graph_only and graph_only/no_edges with ROC-AUC selection; final-epoch block written."""
    cleaned = tmp_path / "cleaned_data"
    cleaned.mkdir()
    rng = np.random.default_rng(0)
    for split, n in (("train", 24), ("valid", 8), ("test", 8)):
        rows = [{"idx": i + {"train": 0, "valid": 100, "test": 200}[split], "func": SMILES[int(rng.integers(len(SMILES)))],
                 "target": int(i % 2), "lang": "smiles", "ext": None} for i in range(n)]
        (cleaned / f"toy_{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (cleaned / "toy_classes.json").write_text(json.dumps({"0": "0", "1": "1"}))
    R.main(["--molgraph", "--dataset", "toy", "--data_root", str(tmp_path)])
    m = json.loads((tmp_path / "input" / "manifest.json").read_text())
    assert m["num_relations"] == 5 and m["node_feat_dim"] == 174 and m["num_node_labels"] == MG.NUM_ATOM_LABELS
    assert m["graph_unit"] == "molecule" and m["counts"]["graphs"] == 40
    runs = tmp_path / "runs"
    common = ["--mode", "train", "--data_root", str(tmp_path), "--runs_dir", str(runs), "--epochs", "2", "--patience", "5",
              "--hidden_dim", "16", "--rgcn_layers", "2", "--batch_sizes", "8", "--eval_batch_size", "8", "--seeds", "0",
              "--loss", "ce", "--select_on", "auc"]
    R.main(common + ["--model_type", "graph_only"])
    R.main(common + ["--model_type", "graph_only", "--ablation", "no_edges"])
    for name in ("graph_only_bs8_lr1.0e-04_br0.30_seed0", "graph_only-no_edges_bs8_lr1.0e-04_br0.30_seed0"):
        mm = json.loads((runs / name / "metrics.json").read_text())
        assert mm["best_val_selection_metric"] == "roc_auc" and 0.0 <= mm["test_auc"] <= 1.0
        assert mm["final_epoch"]["epoch"] == 2 and "test_auc" in mm["final_epoch"]
        assert (runs / name / "test_preds_final.npz").exists()
        meta = json.loads((runs / name / "args.json").read_text())
        assert meta["num_node_labels"] == MG.NUM_ATOM_LABELS and meta["num_relations"] == 5
        hist = [json.loads(l) for l in (runs / name / "history.jsonl").read_text().splitlines()]
        assert all("val_auc" in h for h in hist)


def test_no_edges_ablation_ignores_the_graph_structure():
    torch.manual_seed(0)
    m = build_model("graph_only", torch.device("cpu"), num_relations=5, node_feat_dim=174, hidden_dim=16,
                    num_rgcn_layers=2, ablation="no_edges", num_node_labels=MG.NUM_ATOM_LABELS)
    m.eval()
    a = MG.smiles_to_data("CCO", 0)
    b = MG.smiles_to_data("CCO", 0)
    b.edge_index = b.edge_index.flip(1)[:, :2]                          # different edges, same atoms
    b.edge_type = b.edge_type[:2]
    from torch_geometric.data import Batch
    with torch.no_grad():
        assert torch.allclose(m(Batch.from_data_list([a])), m(Batch.from_data_list([b])))


def test_select_on_auc_is_refused_for_multiclass():
    args = R.build_parser().parse_args(["--mode", "train", "--select_on", "auc"])
    with pytest.raises(RuntimeError, match="auc"):
        R.train_run(args, torch.device("cpu"), batch_size=8, lr=1e-4, ratio=0.3, seed=0,
                    num_relations=3, node_feat_dim=768, num_classes=3)


def test_compare_runs_auc_and_two_level_bootstrap(tmp_path, capsys):
    rng = np.random.default_rng(1)
    y = np.array([0, 1] * 20)
    for g, shift in (("a", 1.0), ("b", 0.0)):
        for seed in range(3):
            d = tmp_path / f"{g}_seed{seed}"
            d.mkdir()
            p = np.clip(0.5 + shift * (y - 0.5) * 0.6 + rng.normal(0, 0.2, len(y)), 0, 1)
            for name in ("test_preds", "test_preds_final"):
                np.savez(d / f"{name}.npz", probs=p, labels=y, idx=np.arange(len(y)))
            (d / "metrics.json").write_text(json.dumps({"test_tuned": {"threshold": 0.5}}))
    from sklearn.metrics import roc_auc_score
    assert abs(CR.roc_auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) - roc_auc_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])) < 1e-12
    CR.main(["--a", str(tmp_path / "a_seed*"), "--b", str(tmp_path / "b_seed*"), "--metric", "auc",
             "--n_boot", "200", "--seed_bootstrap", "--preds", "final"])
    out = capsys.readouterr().out
    assert "ROC-AUC" in out and "two-level" in out


def test_fingerprint_baseline_writes_a_comparable_run(tmp_path):
    cleaned = tmp_path / "cleaned_data"
    cleaned.mkdir()
    for split, idx0 in (("train", 0), ("valid", 100), ("test", 200)):
        rows = [{"idx": idx0 + i, "func": s, "target": int("c" in s)} for i, s in enumerate(SMILES)]
        (cleaned / f"toy_{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    FP.main(["--data_root", str(tmp_path), "--dataset", "toy", "--runs_dir", str(tmp_path / "runs"),
             "--seeds", "0", "--n_estimators", "20", "--n_jobs", "1"])
    d = tmp_path / "runs" / "ecfp_rf_seed0"
    m = json.loads((d / "metrics.json").read_text())
    assert m["test_auc"] > 0.9 and np.load(d / "test_preds.npz")["idx"].tolist() == list(range(200, 220))
