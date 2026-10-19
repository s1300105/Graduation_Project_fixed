"""scripts/compare_runs.py on fake run directories: binary numbers against a hand
computation, multi-class (probs [N, C]) with --metric macro_f1 / accuracy, and
the refusal of --metric f1 on multi-class runs."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run as R  # noqa: E402


def _load_module():
    spec = importlib.util.spec_from_file_location("compare_runs", ROOT / "scripts" / "compare_runs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CR = _load_module()


def _write_run(d: Path, probs, labels, idx, thr):
    d.mkdir(parents=True)
    probs = np.asarray(probs)
    np.savez(d / "test_preds.npz", probs=probs, labels=np.asarray(labels), idx=np.asarray(idx))
    tuned = {"threshold": thr}
    json.dump({"test_tuned": tuned, "val_tuned": tuned, "num_classes": 2 if probs.ndim == 1 else probs.shape[1]},
              (d / "metrics.json").open("w"))


# ------------------------------------------------------------------ binary
LABELS = np.array([1, 0, 1, 1, 0, 0, 1, 0])
IDX = np.arange(100, 108)
PA = np.array([0.9, 0.2, 0.6, 0.4, 0.1, 0.7, 0.8, 0.3])   # thr 0.5: tp 3 fp 1 fn 1 -> F1 0.75
PB = np.array([0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4])   # thr 0.5: tp 3 fp 1 fn 1 -> F1 0.75 ... same counts


@pytest.fixture()
def binary_pair(tmp_path):
    _write_run(tmp_path / "a_seed0", PA, LABELS, IDX, 0.5)
    _write_run(tmp_path / "a_seed1", PA, LABELS, IDX, 0.35)   # own tuned thr: tp 4 fp 1 fn 0 -> 8/9
    # B stores idx in another order; alignment must undo it
    perm = np.array([3, 0, 1, 2, 7, 6, 5, 4])
    _write_run(tmp_path / "b_seed0", PB[perm], LABELS[perm], IDX[perm], 0.5)
    return tmp_path


def test_binary_group_f1_matches_hand_computation(binary_pair):
    A = CR.load_group(str(binary_pair / "a_seed*"), "tuned")
    B = CR.load_group(str(binary_pair / "b_seed*"), "tuned")
    common = CR.align([A, B])
    assert common.tolist() == IDX.tolist()
    assert B[0]["labels"].tolist() == LABELS.tolist()           # re-ordered back by idx
    fa, each = CR.group_f1(A)
    assert each == pytest.approx([0.75, 8 / 9])
    assert fa == pytest.approx((0.75 + 8 / 9) / 2)
    fb, _ = CR.group_f1(B)
    assert fb == pytest.approx(0.75)
    # group_metric("f1") is the same function; the paired resample is applied to every run
    s = np.array([0, 0, 1, 2, 6, 6, 7, 5])
    assert CR.group_metric(A, "f1", s) == CR.group_f1(A, s)
    # macro-F1 / accuracy on the binary pair agree with sklearn
    from sklearn.metrics import f1_score, accuracy_score
    pred_a0 = (PA >= 0.5).astype(int)
    assert CR.group_metric(A[:1], "macro_f1")[0] == pytest.approx(f1_score(LABELS, pred_a0, average="macro"))
    assert CR.group_metric(A[:1], "accuracy")[0] == pytest.approx(accuracy_score(LABELS, pred_a0))
    # fixed threshold instead of tuned
    A05 = CR.load_group(str(binary_pair / "a_seed*"), "0.5")
    assert [r["thr"] for r in A05] == [0.5, 0.5]


def test_binary_main_runs(binary_pair, capsys):
    CR.main(["--a", str(binary_pair / "a_seed*"), "--b", str(binary_pair / "b_seed*"), "--n_boot", "50"])
    out = capsys.readouterr().out
    assert "test samples aligned by idx: 8 (positives 4)" in out
    assert "mean F1" in out and "A - B:" in out


# ------------------------------------------------------------------ multi-class
P3A = np.array([[0.7, 0.2, 0.1],     # 0 -> 0
                [0.1, 0.8, 0.1],     # 1 -> 1
                [0.2, 0.2, 0.6],     # 2 -> 2
                [0.5, 0.3, 0.2],     # 0 -> 0
                [0.3, 0.4, 0.3],     # 2 -> 1 (wrong)
                [0.1, 0.1, 0.8]])    # 1 -> 2 (wrong)
L3 = np.array([0, 1, 2, 0, 2, 1])
I3 = np.arange(6)
P3B = np.tile([0.2, 0.5, 0.3], (6, 1))      # always class 1


@pytest.fixture()
def multiclass_pair(tmp_path):
    _write_run(tmp_path / "a_seed0", P3A, L3, I3, None)
    _write_run(tmp_path / "b_seed0", P3B, L3, I3, None)
    return tmp_path


def test_multiclass_metrics_and_refusal_of_f1(multiclass_pair):
    A = CR.load_group(str(multiclass_pair / "a_seed*"), "tuned")
    B = CR.load_group(str(multiclass_pair / "b_seed*"), "tuned")
    assert A[0]["thr"] is None and A[0]["probs"].shape == (6, 3)
    CR.align([A, B])
    assert CR.predictions(A[0]).tolist() == [0, 1, 2, 0, 1, 2]
    acc, _ = CR.group_metric(A, "accuracy")
    assert acc == pytest.approx(4 / 6)
    # class 0: tp2 fp0 fn0 -> 1; class 1: tp1 fp1 fn1 -> 0.5; class 2: tp1 fp1 fn1 -> 0.5
    mf1, _ = CR.group_metric(A, "macro_f1")
    assert mf1 == pytest.approx((1 + 0.5 + 0.5) / 3)
    # B predicts class 1 for everyone: acc 2/6; F1 class1 = 2*2/(4+4) = 0.5, others 0
    assert CR.group_metric(B, "accuracy")[0] == pytest.approx(2 / 6)
    assert CR.group_metric(B, "macro_f1")[0] == pytest.approx(0.5 / 3)
    # matches run.py's metrics_multiclass
    m = R.metrics_multiclass(P3A, L3)
    assert m["macro_f1"] == pytest.approx(mf1) and m["accuracy"] == pytest.approx(acc)
    # resampled items: paired subset
    s = np.array([0, 1, 2, 3])
    assert CR.group_metric(A, "accuracy", s)[0] == pytest.approx(1.0)
    assert CR.group_metric(A, "macro_f1", s)[0] == pytest.approx(1.0)
    with pytest.raises(SystemExit, match="macro_f1 or accuracy"):
        CR.group_metric(A, "f1")


@pytest.mark.parametrize("metric", ["macro_f1", "accuracy"])
def test_multiclass_main(multiclass_pair, metric, capsys):
    CR.main(["--a", str(multiclass_pair / "a_seed*"), "--b", str(multiclass_pair / "b_seed*"),
             "--n_boot", "50", "--metric", metric])
    out = capsys.readouterr().out
    assert "3 classes, argmax predictions" in out and "A - B:" in out
    with pytest.raises(SystemExit):
        CR.main(["--a", str(multiclass_pair / "a_seed*"), "--b", str(multiclass_pair / "b_seed*"), "--n_boot", "5"])
