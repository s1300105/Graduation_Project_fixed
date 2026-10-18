import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("mkbv", Path(__file__).resolve().parents[1] / "data/scripts/make_bigvul_json.py")
mkbv = importlib.util.module_from_spec(spec); spec.loader.exec_module(mkbv)


def _df(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"idx": np.arange(n), "target": (rng.random(n) < 0.1).astype(int),
                         "project": rng.choice([f"p{i}" for i in range(30)], n), "func": [f"f{i}" for i in range(n)]})


def test_random_split_is_stratified_80_10_10():
    df = mkbv.split_dataframe(_df(), mode="random")
    sizes = df.partition.value_counts()
    assert abs(sizes["train"] - 480) <= 2 and abs(sizes["valid"] - 60) <= 2 and abs(sizes["test"] - 60) <= 2
    rates = df.groupby("partition").target.mean()
    assert rates.max() - rates.min() < 0.03


def test_project_split_keeps_projects_disjoint():
    df = mkbv.split_dataframe(_df(), mode="project", seed=1)
    parts = {p: set(df[df.partition == p].project) for p in ("train", "valid", "test")}
    assert not (parts["train"] & parts["test"]) and not (parts["train"] & parts["valid"]) and not (parts["valid"] & parts["test"])
    assert df.partition.isna().sum() == 0 and len(parts["train"]) > len(parts["test"])


def test_cleaned_code_removes_comments_and_indentation():
    out = mkbv.cleaned_code("int f() {\n  // c\n  return 1; /* x */\n}\n")
    assert "//" not in out and "/*" not in out and " c" not in out
    assert [l.strip() for l in out.splitlines() if l.strip()] == ["int f() {", "return 1;", "}"]
    assert not any(l.startswith(" ") for l in out.splitlines())
