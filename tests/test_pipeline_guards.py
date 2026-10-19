"""Guards added after the adversarial review of the task-suite change (CPU, no Joern).

--embed: refusals happen before the previous input set is removed; --resume trusts a slice only
through its per-slice record (cpg pickle size / mtime, settings) and recomputes totals from the
records, so a regenerated CPG set is never trained on through stale inputs and counts are not
double-counted. --cpg (Joern mocked): whole-file rows lost in Joern are counted without a rescue
attempt, --rescue_stages is validated, positives_after_filter is only written for 0/1 labels.
--mode test: the checkpoint's shape must match the inputs, and another data_root never
overwrites the run's own files. compare_runs: refuses pairings that are not the same test set.
"""
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ["USE_FAST_NODES_EMBEDDING"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import run as R  # noqa: E402
from test_training import _cpg_pickles_from_fixture, _embed_args, _write_split, _make_root3  # noqa: E402
from utils.data import prep  # noqa: E402

_spec = importlib.util.spec_from_file_location("compare_runs", ROOT / "scripts" / "compare_runs.py")
CR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CR)


# ------------------------------------------------------------------ --embed

def _seed_old_input(root: Path):
    inp = root / "input"
    inp.mkdir(parents=True, exist_ok=True)
    (inp / "train_9_input.pkl").write_text("previous embedding")
    (inp / "manifest.json").write_text("{}")


def test_refused_embed_keeps_the_previous_input_set(tmp_path):
    _cpg_pickles_from_fixture(tmp_path, "whole_file")
    _seed_old_input(tmp_path)
    with pytest.raises(RuntimeError, match="graph_unit"):
        R.embed_generate(_embed_args(tmp_path, "first_method"))
    assert (tmp_path / "input" / "train_9_input.pkl").read_text() == "previous embedding"
    shutil.rmtree(tmp_path / "cpg")
    with pytest.raises(RuntimeError, match="no \\*_cpg.pkl"):
        R.embed_generate(_embed_args(tmp_path, "whole_file"))
    assert (tmp_path / "input" / "train_9_input.pkl").exists()


def _manifest(root):
    return json.loads((root / "input" / "manifest.json").read_text())


def _targets(root, base):
    return pd.read_pickle(root / "input" / f"{base}_input.pkl")["target"].tolist()


def test_resume_counts_equal_a_fresh_run_after_losing_one_slice(tmp_path):
    _cpg_pickles_from_fixture(tmp_path, "whole_file")
    R.embed_generate(_embed_args(tmp_path, "whole_file"))
    fresh = _manifest(tmp_path)
    (tmp_path / "input" / "valid_0_input.pkl").unlink()
    a = _embed_args(tmp_path, "whole_file"); a.resume = True
    R.embed_generate(a)
    resumed = _manifest(tmp_path)
    for k in ("counts", "node_sum", "edge_sum", "counts_cover", "input_files"):
        assert resumed[k] == fresh[k], k


def test_resume_reembeds_a_slice_whose_cpg_pickle_changed(tmp_path):
    _cpg_pickles_from_fixture(tmp_path, "whole_file")
    R.embed_generate(_embed_args(tmp_path, "whole_file"))
    before = _targets(tmp_path, "train_0")
    time.sleep(0.01)
    pkl = tmp_path / "cpg" / "train_0_cpg.pkl"
    df = pd.read_pickle(pkl)
    df["target"] = 1 - df["target"]                           # a regenerated CPG set with other labels
    df.to_pickle(pkl)
    a = _embed_args(tmp_path, "whole_file"); a.resume = True
    R.embed_generate(a)
    assert _targets(tmp_path, "train_0") == [1 - t for t in before]
    assert _targets(tmp_path, "valid_0") == pd.read_pickle(tmp_path / "cpg" / "valid_0_cpg.pkl")["target"].tolist()


def test_resume_with_other_settings_is_refused_and_orphans_are_removed(tmp_path):
    _cpg_pickles_from_fixture(tmp_path, "whole_file")
    R.embed_generate(_embed_args(tmp_path, "whole_file"))
    a = _embed_args(tmp_path, "whole_file"); a.resume = True; a.max_nodes = 400
    with pytest.raises(RuntimeError, match="max_nodes"):
        R.embed_generate(a)
    # an input slice without a cpg pickle (left from an earlier, larger CPG set) must not be trained on
    shutil.copy(tmp_path / "input" / "train_0_input.pkl", tmp_path / "input" / "train_7_input.pkl")
    a = _embed_args(tmp_path, "whole_file"); a.resume = True
    R.embed_generate(a)
    assert not (tmp_path / "input" / "train_7_input.pkl").exists()
    assert "train_7_input.pkl" not in _manifest(tmp_path)["input_files"]


def test_resume_of_slices_without_records_recomputes_their_counts(tmp_path):
    """Input sets embedded before per-slice records existed: trusted only when newer than the cpg pickle."""
    _cpg_pickles_from_fixture(tmp_path, "whole_file")
    R.embed_generate(_embed_args(tmp_path, "whole_file"))
    fresh = _manifest(tmp_path)
    for side in (tmp_path / "input").glob("*_input.meta.json"):
        side.unlink()
    a = _embed_args(tmp_path, "whole_file"); a.resume = True
    R.embed_generate(a)
    assert _manifest(tmp_path)["counts"] == fresh["counts"]
    assert len(list((tmp_path / "input").glob("*_input.meta.json"))) == 3


def test_num_classes_comes_from_the_prepared_class_map(tmp_path):
    _cpg_pickles_from_fixture(tmp_path, "whole_file")            # targets 0/1 only
    cleaned = tmp_path / "cleaned_data"
    cleaned.mkdir()
    (cleaned / "toy_classes.json").write_text(json.dumps({str(i): f"p{i}" for i in range(4)}))
    cm = tmp_path / "cpg" / "cpg_manifest.json"
    cm.write_text(json.dumps({"args": {"graph_unit": "whole_file", "dataset": "toy",
                                       "train_jsonl": str(cleaned / "toy_train.jsonl")}, "splits": {}}))
    R.embed_generate(_embed_args(tmp_path, "whole_file"))
    m = _manifest(tmp_path)
    assert m["num_classes"] == 4 and m["classes_json"].endswith("toy_classes.json")


# ------------------------------------------------------------------ --cpg with a mocked Joern

class _FakeJoern:
    """Records the files each joern_parse call sees; a file 'parses' unless it contains `av_cold`
    (removed by rescue stage C) or the body marker `unparsable_marker` (which no stage touches).
    json_process returns one entry per parsed numbered file."""

    def __init__(self, monkeypatch):
        self.calls = []
        monkeypatch.setattr(R.cpgproc, "joern_parse", self.parse)
        monkeypatch.setattr(R.cpgproc, "joern_create", self.create)
        monkeypatch.setattr(R.cpgproc, "json_process", self.process)

    def parse(self, joern_path, input_path, output_path, file_name, *a, **k):
        files = {f.name: f.read_text() for f in Path(input_path).iterdir()}
        ok = sorted(int(n.split(".")[0]) for n, txt in files.items() if "av_cold" not in txt and "unparsable_marker" not in txt)
        Path(output_path, file_name + ".ok.json").write_text(json.dumps(ok))
        self.calls.append(file_name)

    def create(self, joern_path, in_path, out_path, cpg_files, *a, **k):
        return {f: str(Path(out_path, f.replace(".bin", "") + ".ok.json")) for f in cpg_files}

    def process(self, in_path, json_file, stats=None, graph_unit="first_method"):
        ok = json.loads(Path(json_file).read_text())
        if stats is not None:
            stats["functions_kept"] = stats.get("functions_kept", 0) + len(ok)
        return [(i, {"functions": [{"function": "f", "nodes": [], "edges": []}]}) for i in ok]


def _jsonl_root(tmp_path, rows_by_split):
    cleaned = tmp_path / "cleaned_data"
    cleaned.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        (cleaned / f"x_{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return tmp_path


def _cpg_args(root, *extra):
    a = R.build_parser().parse_args(["--cpg", "--data_root", str(root), "--dataset", "x", "--seed", "0"] + list(extra))
    for split in ("train", "valid", "test"):
        setattr(a, f"{split}_jsonl", str(Path(root) / "cleaned_data" / f"x_{split}.jsonl"))
    return a


FN_OK = "int f{i}(int a)\n{{\nreturn a + {i};\n}}\n"
FN_AVCOLD = "static av_cold int g{i}(int a)\n{{\nreturn a * {i};\n}}\n"


def _rows(n, avcold_at=(), broken_at=(), num_classes=2):
    out = []
    for i in range(n):
        func = FN_AVCOLD.format(i=i) if i in avcold_at else FN_OK.format(i=i)
        if i in broken_at:
            func = func.replace("return", "return unparsable_marker +")
        out.append({"idx": i, "func": func, "target": i % num_classes, "ext": ".c"})
    return out


def test_cpg_first_method_rescue_stages_and_counts(tmp_path, monkeypatch):
    root = _jsonl_root(tmp_path, {s: _rows(6, avcold_at=(2,), broken_at=(4,)) for s in ("train", "valid", "test")})
    fj = _FakeJoern(monkeypatch)
    R.cpg_generate(_cpg_args(root))
    st = json.loads((root / "cpg" / "cpg_manifest.json").read_text())["splits"]["train"]
    assert st["rescued_stage_C"] == 1 and st["rescued_stage_A"] == 0 and st["rescued_stage_B"] == 0
    assert st["rows_lost_in_joern"] == 1 and st["graphs_written"] == 5 and st["positives_after_filter"] == 3
    # --rescue_stages AB reproduces the earlier behaviour: no stage C, the av_cold function is lost
    assert any(c.endswith("rescueC") for c in fj.calls)
    shutil.rmtree(root / "cpg")
    n_calls = len(fj.calls)
    R.cpg_generate(_cpg_args(root, "--rescue_stages", "ab"))
    st = json.loads((root / "cpg" / "cpg_manifest.json").read_text())["splits"]["train"]
    assert "rescued_stage_C" not in st and st["rows_lost_in_joern"] == 2
    assert not any(c.endswith("rescueC") for c in fj.calls[n_calls:])


def test_cpg_whole_file_counts_losses_without_rescue_and_multiclass_positives(tmp_path, monkeypatch):
    root = _jsonl_root(tmp_path, {s: _rows(6, avcold_at=(2,), broken_at=(4,), num_classes=3) for s in ("train", "valid", "test")})
    fj = _FakeJoern(monkeypatch)
    R.cpg_generate(_cpg_args(root, "--graph_unit", "whole_file"))
    st = json.loads((root / "cpg" / "cpg_manifest.json").read_text())["splits"]["train"]
    assert st["rescue_skipped_whole_file"] == 2 == st["rows_lost_in_joern"]
    assert not any("rescue" in c for c in fj.calls)
    assert st["positives_after_filter"] is None and st["label_counts"] == {"0": 2, "1": 2, "2": 2}


@pytest.mark.parametrize("value", ["", "abd", "all", "x"])
def test_rescue_stages_typos_are_refused(tmp_path, monkeypatch, value):
    root = _jsonl_root(tmp_path, {s: _rows(2) for s in ("train", "valid", "test")})
    _FakeJoern(monkeypatch)
    with pytest.raises(ValueError, match="rescue_stages"):
        R.cpg_generate(_cpg_args(root, "--rescue_stages", value))
    assert R.parse_rescue_stages("c,a") == ("A", "C")


def test_identifier_normalisation_is_refused_on_whole_programs(tmp_path, monkeypatch):
    root = _jsonl_root(tmp_path, {s: _rows(2) for s in ("train", "valid", "test")})
    _FakeJoern(monkeypatch)
    with pytest.raises(ValueError, match="normalize_graph_identifiers"):
        R.cpg_generate(_cpg_args(root, "--graph_unit", "whole_file", "--normalize_graph_identifiers"))


# ------------------------------------------------------------------ --mode test

def _binary_root(root: Path, num_classes_in_manifest=2):
    inp = root / "input"
    inp.mkdir(parents=True)
    _write_split(inp, "train", 24, seed=1)
    _write_split(inp, "valid", 8, seed=2)
    _write_split(inp, "test", 8, seed=3)
    (inp / "manifest.json").write_text(json.dumps({"num_relations": 3, "node_feat_dim": 768,
                                                   "num_classes": num_classes_in_manifest}))
    return root


_FAST = ["--epochs", "1", "--patience", "2", "--hidden_dim", "8", "--rgcn_layers", "1", "--batch_sizes", "8",
         "--eval_batch_size", "8", "--seeds", "0"]


def test_mode_test_on_another_root_writes_a_separate_reeval_and_checks_the_shape(tmp_path):
    a = _binary_root(tmp_path / "a")
    runs = tmp_path / "runs"
    R.main(["--mode", "train", "--model_type", "graph_only", "--data_root", str(a), "--runs_dir", str(runs)] + _FAST)
    run_dir = next(runs.iterdir())
    m_before = (run_dir / "metrics.json").read_text()
    p_before = (run_dir / "test_preds.npz").read_bytes()
    m = json.loads(m_before)
    assert m["best_val_selection_metric"] == "f1_tuned_threshold"
    b = tmp_path / "b_copy"
    shutil.copytree(a, b)
    R.main(["--mode", "test", "--data_root", str(b), "--run_dir", str(run_dir), "--allow_other_data_root",
            "--eval_batch_size", "8"])
    assert (run_dir / "metrics.json").read_text() == m_before and (run_dir / "test_preds.npz").read_bytes() == p_before
    re = json.loads((run_dir / "reeval_b_copy" / "metrics.json").read_text())
    assert re["reevaluated_on"] == str(b.resolve()) and (run_dir / "reeval_b_copy" / "test_preds.npz").exists()
    # a 3-class root cannot be evaluated with a binary checkpoint, flag or not
    c = _make_root3(tmp_path / "c3", with_manifest=True)
    with pytest.raises(RuntimeError, match="num_classes"):
        R.main(["--mode", "test", "--data_root", str(c), "--run_dir", str(run_dir), "--allow_other_data_root",
                "--eval_batch_size", "8"])


# ------------------------------------------------------------------ compare_runs

def _run(d: Path, idx, labels, probs, thr=0.5):
    d.mkdir(parents=True)
    np.savez(d / "test_preds.npz", probs=np.asarray(probs, dtype=float), labels=np.asarray(labels), idx=np.asarray(idx))
    (d / "metrics.json").write_text(json.dumps({"test_tuned": {"threshold": thr}}))


def test_compare_runs_refuses_pairings_that_are_not_the_same_test_set(tmp_path, capsys):
    rng = np.random.default_rng(0)
    _run(tmp_path / "a_s0", range(10), [0, 1] * 5, rng.random(10))
    _run(tmp_path / "b_s0", range(5, 15), [0, 1] * 5, rng.random(10))        # overlapping idx 5..9, labels shifted
    with pytest.raises(SystemExit, match="disagree on the label"):
        CR.main(["--a", str(tmp_path / "a_s*"), "--b", str(tmp_path / "b_s*"), "--n_boot", "10"])
    _run(tmp_path / "c_s0", range(100, 110), [0, 1] * 5, rng.random(10))      # disjoint
    with pytest.raises(SystemExit, match="share no test idx"):
        CR.main(["--a", str(tmp_path / "a_s*"), "--b", str(tmp_path / "c_s*"), "--n_boot", "10"])
    _run(tmp_path / "d_s0", [-1] * 10, [0, 1] * 5, rng.random(10))             # inputs without idx
    with pytest.raises(SystemExit, match="duplicate or missing"):
        CR.main(["--a", str(tmp_path / "a_s*"), "--b", str(tmp_path / "d_s*"), "--n_boot", "10"])
    _run(tmp_path / "e_s0", range(10), [0, 1, 2, 0, 1, 2, 0, 1, 2, 0], rng.random((10, 3)))
    with pytest.raises(SystemExit, match="number of classes|disagree on the label"):
        CR.main(["--a", str(tmp_path / "a_s*"), "--b", str(tmp_path / "e_s*"), "--n_boot", "10", "--metric", "accuracy"])
    # partial overlap with consistent labels is allowed but reported
    _run(tmp_path / "f_s0", range(2, 10), [0, 1] * 4, rng.random(8))
    CR.main(["--a", str(tmp_path / "a_s*"), "--b", str(tmp_path / "f_s*"), "--n_boot", "10"])
    assert "only 8 test idx are shared" in capsys.readouterr().out


# ------------------------------------------------------------------ preparation inputs

def test_sharded_or_duplicated_parquet_splits_are_refused(tmp_path):
    for name in ("train-00000-of-00002.parquet", "train-00001-of-00002.parquet",
                 "validation-00000-of-00001.parquet", "test-00000-of-00001.parquet"):
        pd.DataFrame({"x": [1]}).to_parquet(tmp_path / name)
    with pytest.raises(SystemExit, match="several parquet files"):
        prep.find_hf_parquets(tmp_path)
