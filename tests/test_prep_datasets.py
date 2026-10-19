"""Dataset preparation scripts (POJ-104 / CodeNet C++1000 / Devign) and their shared helpers.

Everything runs on tiny synthetic inputs built in ``tmp_path`` (no network).  The real-data
numbers live in the stats files produced under the scratch directory, not here.
"""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "prep"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prep = _load("prep_under_test", "utils/data/prep.py")
mkbv = _load("mkbv_under_test", "data/scripts/make_bigvul_json.py")
poj = _load("make_poj104_json_under_test", "data/scripts/make_poj104_json.py")
codenet = _load("make_codenet_json_under_test", "data/scripts/make_codenet_json.py")
devign = _load("make_devign_json_under_test", "data/scripts/make_devign_json.py")


def _read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def _read_all(out_dir, dataset):
    rows = {}
    for s in prep.SPLITS:
        rows[s] = _read_jsonl(Path(out_dir) / f"{dataset}_{s}.jsonl")
    classes = json.loads((Path(out_dir) / f"{dataset}_classes.json").read_text())
    stats = json.loads((Path(out_dir) / f"{dataset}_prep_stats.json").read_text())
    return rows, classes, stats


def _check_contract(rows, classes, stats):
    """The common contract: schema / idx uniqueness / dense targets / lang-ext consistency / split."""
    allrows = [r for s in prep.SPLITS for r in rows[s]]
    assert allrows
    for s in prep.SPLITS:
        for r in rows[s]:
            assert tuple(r.keys()) == prep.ROW_FIELDS
            assert r["split"] == s
            assert isinstance(r["idx"], int) and isinstance(r["target"], int) and isinstance(r["func"], str)
            assert "\r" not in r["func"]
            assert r["ext"] in (".c", ".cpp") and r["lang"] == ("cpp" if r["ext"] == ".cpp" else "c")
            assert isinstance(r["project"], str) and isinstance(r["source_id"], str)
            assert str(r["target"]) in classes
    idxs = [r["idx"] for r in allrows]
    assert len(set(idxs)) == len(idxs), "idx must be unique across splits"
    assert sorted(idxs) == list(range(len(idxs))), "idx renumbered densely from 0"
    targets = {r["target"] for r in allrows}
    assert targets == set(range(len(classes))), "targets dense 0..C-1 and every class present"
    assert {r["target"] for r in rows["train"]} == targets, "every class must appear in train"
    assert stats["rows_out_total"] == len(allrows) and stats["num_classes"] == len(classes)
    for s in prep.SPLITS:
        assert stats["splits"][s]["rows"] == len(rows[s])
        assert sum(stats["splits"][s]["label_counts"].values()) == len(rows[s])
    assert stats["steps"] and all({"step", "rows_in", "rows_out", "dropped"} <= set(st) for st in stats["steps"])
    assert "options" in stats and "sources" in stats and "ext" in stats and "knr_main" in stats


# ---------------------------------------------------------------------------------------------
# cleaning must not drift from BigVul
# ---------------------------------------------------------------------------------------------
CLEAN_CASES = [
    "int f() {\n  // c\n  return 1; /* x */\n}\n",
    "\n\nint g(int a)\n{\n\n    char *s = \"// not a comment\";\n    return a; // tail\n}\n\n",
    "/* multi\n   line */\nstatic int h(void)\n{\n\treturn '/' + '\\'' ;\n}\n",
    "#include <stdio.h>\r\nint main()\r\n{\r\n  printf(\"hi\\n\");\r\n}\r\n",
    "",
    "   \n  \n",
]


@pytest.mark.parametrize("src", CLEAN_CASES)
def test_cleaned_code_equals_make_bigvul_json(src):
    assert prep.cleaned_code(src) == mkbv.cleaned_code(src)
    assert prep.remove_empty_lines(src) == mkbv.remove_empty_lines(src)
    assert prep.remove_comments(src) == mkbv.remove_comments(src)
    assert prep.remove_space_after_newline(src) == mkbv.remove_space_after_newline(src)


def test_cleaned_code_non_string_and_prepare_code():
    assert prep.cleaned_code(None) == mkbv.cleaned_code(None) == ""
    out = prep.prepare_code("int f()\r\n{\r\n  return 1;\r\n}\r\n")
    assert "\r" not in out and out == "int f()\n{\nreturn 1;\n}\n"


def test_split_dataframe_random_matches_bigvul_for_defaults():
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({"idx": np.arange(n), "target": (rng.random(n) < 0.1).astype(int),
                       "project": rng.choice([f"p{i}" for i in range(30)], n), "func": [f"f{i}" for i in range(n)]})
    a = mkbv.split_dataframe(df, mode="random")
    b = prep.split_dataframe(df, mode="random", seed=42, fractions=(0.8, 0.1, 0.1))
    assert list(a.partition) == list(b.partition)
    c = prep.split_dataframe(df, mode="random", seed=42, fractions="0.6,0.2,0.2")
    sizes = c.partition.value_counts()
    assert abs(sizes["train"] - 360) <= 2 and abs(sizes["valid"] - 120) <= 2 and abs(sizes["test"] - 120) <= 2


def test_stratified_split_impossible_fails_loudly():
    df = pd.DataFrame({"idx": range(6), "target": [0, 0, 1, 1, 2, 2]})
    with pytest.raises(SystemExit, match="stratified split impossible"):
        prep.split_dataframe(df, mode="random", seed=1, fractions=(0.8, 0.1, 0.1))
    with pytest.raises(ValueError):
        prep.parse_fractions("0.5,0.5")


# ---------------------------------------------------------------------------------------------
# extension rule
# ---------------------------------------------------------------------------------------------
def test_knr_main_and_cpp_hints_on_fixtures():
    knr_cin = (FIX / "knr_cin.c").read_text()
    cin_noinc = (FIX / "cin_noinclude.cpp").read_text()
    plain = (FIX / "plain.c").read_text()
    modern = (FIX / "modern.cpp").read_text()
    assert prep.knr_main(knr_cin) and prep.looks_like_cpp_program(knr_cin)
    assert prep.choose_extension(knr_cin) == ".c"                 # K&R main wins over the C++ hints
    assert prep.choose_extension(knr_cin, default_ext=".cpp") == ".c"
    assert not prep.knr_main(cin_noinc) and prep.looks_like_cpp_program(cin_noinc)
    assert prep.choose_extension(cin_noinc) == ".cpp"             # include-less cin/cout program
    assert not prep.knr_main(plain) and not prep.looks_like_cpp_program(plain)
    assert prep.choose_extension(plain) == ".c"
    assert prep.choose_extension(plain, default_ext=".cpp") == ".cpp"   # CodeNet forces .cpp
    assert prep.choose_extension(modern) == ".cpp" and not prep.knr_main(modern)
    assert prep.lang_of(".c") == "c" and prep.lang_of(".cpp") == "cpp"
    with pytest.raises(ValueError):
        prep.choose_extension(plain, default_ext=".cc")


def test_knr_main_edge_cases():
    assert prep.knr_main("main()\n{\nreturn 0;\n}\n")
    assert prep.knr_main("  main ( ) {}")
    assert prep.knr_main("#include <stdio.h>\nmain(int argc, char **argv) { return 0; }")
    assert not prep.knr_main("int\nmain()\n{\n}")                       # return type on the previous line
    assert not prep.knr_main("void main()\n{\n}")                        # has a return type
    assert not prep.knr_main("int main() {\n  domain();\n}")             # not at line start / other identifier
    assert not prep.knr_main("/* main( */\nint main() { char *s = \"main(\"; return 0; }")
    assert prep.choose_extension("void main()\n{\ncin >> x;\n}") == ".cpp"
    assert prep.choose_extension("void main()\n{\nscanf(\"%d\", &x);\n}") == ".c"


def test_cpp_hints_are_conservative():
    assert not prep.looks_like_cpp_program("int set = 0; int map = 1; if (vector < n) cin = 2;")
    assert not prep.looks_like_cpp_program('const char *s = "std::string"; /* using namespace std; */')
    assert prep.looks_like_cpp_program("#include <iostream>")
    assert prep.looks_like_cpp_program("#include<bits/stdc++.h>")
    assert prep.looks_like_cpp_program("using namespace std;")
    assert prep.looks_like_cpp_program("std::vector<int> v;")
    assert prep.looks_like_cpp_program("vector<int> v;")
    assert prep.looks_like_cpp_program("template<class T> T f(T x);")
    assert prep.looks_like_cpp_program("class Foo {\n int x;\n};")
    assert prep.looks_like_cpp_program("int *p = new int[10]; delete[] p;")
    assert prep.looks_like_cpp_program("int x = Foo::bar;")
    assert prep.looks_like_cpp_program("using ll = long long;")
    assert prep.looks_like_cpp_program("vector<int>::iterator it; A::~A() {}")
    assert prep.looks_like_cpp_program("struct Foo { Foo& operator=(const Foo& o); };")
    assert prep.looks_like_cpp_program("Foo::operator == (x)")
    # false positives seen on Devign (C) and POJ-104: GCC inline-asm operand separators, a C label
    # starting with `delete`, a C variable named `operator` / `this`, `cout` as a C variable name
    assert not prep.looks_like_cpp_program('__asm__ volatile("movq %0, %%mm0" :: "r"(x));\n__asm__ __volatile__(SFENCE:::"memory");')
    assert not prep.looks_like_cpp_program('__asm__ volatile(\n"prefetch (%0)"\n"prefetch (%1)"\n::"r"(*src), "r"(*dst)\n);')
    assert not prep.looks_like_cpp_program("if (ret < 0) {\ngoto delete_and_fail;\n}\ndelete_and_fail:\nreturn ret;")
    assert not prep.looks_like_cpp_program("int deletenum;\ndeletenum = 3;")
    assert not prep.looks_like_cpp_program("opcode = (insn >> 27) & 7;\noperator = (insn >> 24) & 7;")
    assert not prep.looks_like_cpp_program("this->pid = fork();\nif (this->pid == 0) exit(0);")
    assert not prep.looks_like_cpp_program("int main()\n{\nint cout=0,len=0;\ngets(s);\ncout++;\n}")


# ---------------------------------------------------------------------------------------------
# dedup helpers
# ---------------------------------------------------------------------------------------------
def test_dedup_by_key_keeps_first_and_drops_conflicts():
    df = pd.DataFrame({"func": ["int a;", "int  a;", "int b;", "int\tb;", "int c;", "int d;"],
                       "target": [0, 0, 1, 2, 3, 3], "n": range(6)})
    df["dedup_key"] = df["func"].map(prep.ws_key)
    out, st = prep.dedup_by_key(df)
    assert list(out["n"]) == [0, 4, 5]
    assert st == {"rows_in": 6, "groups_total": 4, "conflict_groups": 1, "rows_dropped_conflict": 2,
                  "rows_dropped_duplicate": 1, "rows_out": 3}
    kept, n = prep.drop_near_empty(pd.DataFrame({"func": ["int main(void){return 0;}", "x;\n\n  ", ""]}))
    assert n == 2 and len(kept) == 1


# ---------------------------------------------------------------------------------------------
# synthetic inputs
# ---------------------------------------------------------------------------------------------
def c_prog(k: int) -> str:
    return (f"#include <stdio.h>\nint a[{k + 10}];\nint main(void)\n{{\n    int n, i, s = 0;\n"
            f"    scanf(\"%d\", &n);\n    for (i = 0; i < n; i++) s += a[i] * {k};\n"
            f"    printf(\"%d\\n\", s);\n    return 0;\n}}\n")


def cpp_prog(k: int) -> str:
    return (f"#include <iostream>\nusing namespace std;\nint main()\n{{\n    int n, s = {k};\n"
            f"    cin >> n;\n    cout << s * n << endl;\n    return 0;\n}}\n")


KNR_CIN = (FIX / "knr_cin.c").read_text()
CIN_NOINC = (FIX / "cin_noinclude.cpp").read_text()


def _write_hf_parquets(d: Path, frames: dict):
    d.mkdir(parents=True, exist_ok=True)
    for split, df in frames.items():
        df.to_parquet(d / f"{split}-00000-of-00001.parquet", index=False)


# ---------------------------------------------------------------------------------------------
# POJ-104
# ---------------------------------------------------------------------------------------------
def _poj_input(d: Path):
    """4 classes x 15 distinct programs, spread over class-disjoint HF splits like the real
    dataset, plus: class '65' duplicated verbatim across train and validation (the real 1,000-row
    overlap), a CRLF program, a whitespace-only duplicate (same label), a label conflict, a
    near-empty program, a K&R + cin program and an include-less cin program."""
    rows = {"train": [], "validation": [], "test": []}
    k = 0
    for label, split in (("1", "train"), ("2", "train"), ("65", "validation"), ("81", "test")):
        for j in range(15):
            k += 1
            code = c_prog(k) if j % 2 else cpp_prog(k)
            if label == "1" and j == 0:
                code = KNR_CIN                        # K&R + cin -> '.c'
            if label == "1" and j == 1:
                code = CIN_NOINC                      # include-less cin -> '.cpp'
            if label == "2" and j == 0:
                code = c_prog(k).replace("\n", "\r\n")  # CRLF
            rows[split].append({"id": k, "code": code, "label": label})   # ids unique within an HF split
    # class 65 duplicated across splits (identical rows)
    for r in list(rows["validation"]):
        rows["train"].append(dict(r, id=100 + r["id"]))     # HF ids are per split; keep them unique here
    # whitespace variant of an existing class-2 program: same label -> keep first only
    rows["train"].append({"id": 900, "code": rows["train"][16]["code"].replace("    ", "\t"), "label": "2"})
    # label conflict: a class-81 program also filed under class 1 -> both dropped
    rows["train"].append({"id": 901, "code": rows["test"][3]["code"], "label": "1"})
    # near-empty
    rows["test"].append({"id": 902, "code": "int x;\n", "label": "81"})
    return {s: pd.DataFrame(r, columns=["id", "code", "label"]) for s, r in rows.items()}


def test_make_poj104_json(tmp_path):
    inp = tmp_path / "poj"
    _write_hf_parquets(inp, _poj_input(inp))
    out = tmp_path / "out"
    st = poj.main(["--input", str(inp), "--out_dir", str(out), "--seed", "7"])
    rows, classes, stats = _read_all(out, "poj104")
    _check_contract(rows, classes, stats)
    assert classes == {"0": "1", "1": "2", "2": "65", "3": "81"}
    steps = {s["step"]: s for s in stats["steps"]}
    assert steps["load (concat hf splits)"]["rows_in"] == 60 + 15 + 3
    assert steps["drop exact duplicate (label, code)"]["dropped"] == 15        # the cross-split copies of class 65
    assert stats["dedup"]["conflict_groups"] == 1 and stats["dedup"]["rows_dropped_conflict"] == 2
    assert stats["dedup"]["rows_dropped_duplicate"] == 1                        # whitespace variant
    assert stats["near_empty_dropped"] == 1
    assert stats["rows_out_total"] == 60 - 1                                    # 59 programs kept (test:3 lost to the conflict)
    assert stats["knr_main"] == 1
    allrows = [r for s in prep.SPLITS for r in rows[s]]
    by_src = {r["source_id"]: r for r in allrows}
    assert by_src["train:1"]["ext"] == ".c" and by_src["train:1"]["target"] == 0        # K&R + cin
    assert by_src["train:2"]["ext"] == ".cpp"                                           # include-less cin
    assert all(r["project"] == "poj104" and r["commit_id"] is None and r["user_id"] is None for r in allrows)
    assert all(r["problem_id"] == classes[str(r["target"])] for r in allrows)
    assert {r["ext"] for r in allrows} == {".c", ".cpp"}
    assert "test:902" not in by_src and "train:901" not in by_src and "train:900" not in by_src
    assert st["rows_out_total"] == 59
    # idempotent
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    poj.main(["--input", str(inp), "--out_dir", str(out), "--seed", "7"])
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    # another seed gives another split but the same row set
    out2 = tmp_path / "out2"
    poj.main(["--input", str(inp), "--out_dir", str(out2), "--seed", "8"])
    rows2, _, _ = _read_all(out2, "poj104")
    a = {r["source_id"]: r["split"] for s in prep.SPLITS for r in rows[s]}
    b = {r["source_id"]: r["split"] for s in prep.SPLITS for r in rows2[s]}
    assert set(a) == set(b) and a != b


def test_make_poj104_json_per_class_and_drop_problems(tmp_path):
    inp = tmp_path / "poj"
    _write_hf_parquets(inp, _poj_input(inp))
    out = tmp_path / "out"
    poj.main(["--input", str(inp), "--out_dir", str(out), "--per_class", "10", "--drop_problems", "65",
              "--fractions", "0.6,0.2,0.2"])
    rows, classes, stats = _read_all(out, "poj104")
    _check_contract(rows, classes, stats)
    assert classes == {"0": "1", "1": "2", "2": "81"}
    allrows = [r for s in prep.SPLITS for r in rows[s]]
    from collections import Counter
    assert Counter(r["target"] for r in allrows) == {0: 10, 1: 10, 2: 10}
    assert stats["options"]["drop_problems"] == "65" and stats["options"]["per_class"] == 10


# ---------------------------------------------------------------------------------------------
# CodeNet
# ---------------------------------------------------------------------------------------------
def _codenet_tree(root: Path, meta_dir: Path):
    """3 problems x 8 submissions: s1 of every problem duplicates s0 (whitespace only); p00001/s7
    is a K&R program; p00002/s6 is 'Wrong Answer' in the metadata; p00003/s5 is missing from the
    metadata.  Users: one per submission except u1 (p00001/s2 and p00002/s2) and u2 (p00003/s3 and
    p00001/s3)."""
    probs = ["p00001", "p00002", "p00003"]
    meta_dir.mkdir(parents=True, exist_ok=True)
    k = 0
    for pi, prob in enumerate(probs):
        (root / prob).mkdir(parents=True, exist_ok=True)
        with (meta_dir / f"{prob}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(codenet.META_COLUMNS)
            for j in range(8):
                k += 1
                sid = f"s{pi:03d}{j:06d}"
                code = cpp_prog(k)
                if j == 1:
                    code = cpp_prog(k - 1).replace("    ", "  ")
                if prob == "p00001" and j == 7:
                    code = KNR_CIN
                (root / prob / f"{sid}.cpp").write_bytes(code.encode("utf-8"))
                if prob == "p00003" and j == 5:
                    continue                                   # missing from the metadata
                status = "Wrong Answer" if (prob == "p00002" and j == 6) else "Accepted"
                user = {("p00001", 2): "u1", ("p00002", 2): "u1", ("p00003", 3): "u2", ("p00001", 3): "u2"}.get(
                    (prob, j), f"u{prob}{j}")
                w.writerow([sid, prob, user, "1513422160", "C++", "C++14", "cpp", status, "10", "3000", str(len(code)), "4/4"])
    # an extra non-problem directory and a stray file must be ignored
    (root / "README.txt").write_text("not a problem dir")
    (root / "p00001" / "notes.md").write_text("ignored")


def test_make_codenet_json_random_with_metadata(tmp_path):
    root, meta = tmp_path / "C++1000", tmp_path / "metadata"
    _codenet_tree(root, meta)
    out = tmp_path / "out"
    codenet.main(["--input", str(root), "--out_dir", str(out), "--metadata_dir", str(meta),
                  "--per_problem", "7", "--fractions", "0.6,0.2,0.2"])
    rows, classes, stats = _read_all(out, "codenet")
    _check_contract(rows, classes, stats)
    assert classes == {"0": "p00001", "1": "p00002", "2": "p00003"}
    assert stats["metadata"]["submissions_missing_from_metadata"] == 1
    assert stats["metadata"]["status_counts"] == {"Accepted": 22, "Wrong Answer": 1}
    assert stats["dedup"]["rows_dropped_duplicate"] == 3 and stats["dedup"]["conflict_groups"] == 0
    assert stats["file_ext_in"] == {".cpp": 24}
    steps = {s["step"]: s for s in stats["steps"]}
    assert steps["read files"]["rows_in"] == 24
    assert steps["subsample --per_problem 7"]["rows_in"] == 24 - 1 - 1 - 3 and steps["subsample --per_problem 7"]["rows_out"] == 19
    assert stats["per_problem_counts"] == {"p00001": 7, "p00002": 6, "p00003": 6}
    allrows = [r for s in prep.SPLITS for r in rows[s]]
    assert len(allrows) == 19
    knr = [r for r in allrows if r["ext"] == ".c"]
    assert [r["source_id"] for r in knr] == ["s000000007"] and knr[0]["lang"] == "c" and stats["knr_main"] == 1
    assert all(r["ext"] == ".cpp" for r in allrows if r["source_id"] != "s000000007")   # default '.cpp'
    assert all(r["project"] == "codenet_cpp1000" and r["commit_id"] is None for r in allrows)
    assert all(r["problem_id"] == classes[str(r["target"])] for r in allrows)
    assert all(isinstance(r["user_id"], str) and r["user_id"].startswith("u") for r in allrows)
    assert not any(r["source_id"] in ("s001000006", "s002000005") for r in allrows)        # WA / missing
    assert not any(r["source_id"].endswith("000001") for r in allrows)                      # duplicates (s1)
    assert stats["sources"]["problem_dirs_found"] == 3 and stats["problems_selected"] == ["p00001", "p00002", "p00003"]


def test_make_codenet_json_user_split_and_problem_sampling(tmp_path):
    root, meta = tmp_path / "C++1000", tmp_path / "metadata"
    _codenet_tree(root, meta)
    out = tmp_path / "out"
    codenet.main(["--input", str(root), "--out_dir", str(out), "--metadata_dir", str(meta), "--split", "user",
                  "--per_problem", "7", "--fractions", "0.6,0.2,0.2", "--seed", "3"])
    rows, classes, stats = _read_all(out, "codenet")
    _check_contract(rows, classes, stats)
    users = {s: {r["user_id"] for r in rows[s]} for s in prep.SPLITS}
    assert not (users["train"] & users["valid"]) and not (users["train"] & users["test"]) and not (users["valid"] & users["test"])
    assert stats["users"] == {s: len(users[s]) for s in prep.SPLITS}
    assert set(stats["class_coverage"]["per_split"]) == set(prep.SPLITS)
    # --problems N: deterministic random choice of N problem dirs; targets re-densified
    out2 = tmp_path / "out2"
    codenet.main(["--input", str(root), "--out_dir", str(out2), "--problems", "2", "--per_problem", "6",
                  "--fractions", "0.6,0.2,0.2", "--seed", "3"])
    rows2, classes2, stats2 = _read_all(out2, "codenet")
    _check_contract(rows2, classes2, stats2)
    assert len(classes2) == 2 and sorted(classes2.values()) == stats2["problems_selected"]
    assert all(r["user_id"] is None for s in prep.SPLITS for r in rows2[s])          # no metadata
    assert "metadata" not in stats2
    with pytest.raises(SystemExit, match="--split user needs --metadata_dir"):
        codenet.main(["--input", str(root), "--out_dir", str(tmp_path / "x"), "--split", "user"])
    with pytest.raises(SystemExit, match="lacks metadata columns"):
        bad = tmp_path / "badmeta"; bad.mkdir()
        for p in ("p00001", "p00002", "p00003"):
            (bad / f"{p}.csv").write_text("submission_id,status\ns1,Accepted\n")
        codenet.main(["--input", str(root), "--out_dir", str(tmp_path / "y"), "--metadata_dir", str(bad)])


def test_codenet_reads_bytes_with_replacement(tmp_path):
    root = tmp_path / "C++1000"
    (root / "p00009").mkdir(parents=True)
    (root / "p00009" / "s000000001.cpp").write_bytes(b"int main(){ /* caf\xe9 */ return 0; }\n")
    rows = codenet.read_problem(root, "p00009")
    assert len(rows) == 1 and "�" in rows[0]["code"] and rows[0]["source_id"] == "s000000001"


# ---------------------------------------------------------------------------------------------
# Devign (CodeXGLUE defect detection)
# ---------------------------------------------------------------------------------------------
def _devign_input():
    def fn(k, cpp=False):
        if cpp:
            return f"static int f{k}(AVCodecContext *avctx)\n{{\n    return Foo::bar({k});\n}}\n"
        return f"static int f{k}(AVCodecContext *avctx)\n{{\n    int x = avctx->width * {k};\n    return x;\n}}\n"
    k = 0
    def rows(n, split_pos, cpp_at=None):
        nonlocal k
        out = []
        for j in range(n):
            k += 1
            out.append({"id": j, "func": fn(k, cpp=(j == cpp_at)), "target": bool(j % 2 == 0 and j < 2 * split_pos),
                        "project": "FFmpeg" if j % 3 else "qemu", "commit_id": f"c{k:04x}"})
        return out
    tr, va, te = rows(12, 3, cpp_at=4), rows(4, 1), rows(4, 1)
    # within-train duplicate (same label): keep first
    tr.append({"id": 50, "func": tr[1]["func"].replace("    ", "\t"), "target": tr[1]["target"], "project": "FFmpeg", "commit_id": "cdup"})
    # cross-split leak with the same label: train loses, test keeps
    tr.append({"id": 51, "func": te[1]["func"], "target": te[1]["target"], "project": "qemu", "commit_id": "cleak"})
    # label conflict train/valid: both dropped
    tr.append({"id": 52, "func": va[2]["func"], "target": not va[2]["target"], "project": "qemu", "commit_id": "cconf"})
    # near-empty, CRLF
    te.append({"id": 60, "func": "int x;\r\n", "target": False, "project": "qemu", "commit_id": "cne"})
    tr.append({"id": 53, "func": fn(999).replace("\n", "\r\n"), "target": True, "project": "FFmpeg", "commit_id": None})
    return {s: pd.DataFrame(r, columns=["id", "func", "target", "project", "commit_id"]) for s, r in
            (("train", tr), ("validation", va), ("test", te))}


def test_make_devign_json(tmp_path):
    inp = tmp_path / "devign"
    _write_hf_parquets(inp, _devign_input())
    out = tmp_path / "out"
    devign.main(["--input", str(inp), "--out_dir", str(out)])
    rows, classes, stats = _read_all(out, "devign")
    _check_contract(rows, classes, stats)
    assert classes == {"0": "0", "1": "1"}
    assert stats["official_split_rows"] == {"train": 16, "valid": 4, "test": 5}
    d = stats["dedup"]
    assert d["conflict_groups"] == 1 and d["rows_dropped_conflict_per_split"] == {"train": 1, "valid": 1, "test": 0}
    assert d["rows_dropped_cross_split_leak_per_split"] == {"train": 1, "valid": 0, "test": 0}
    assert d["rows_dropped_within_split_duplicate_per_split"] == {"train": 1, "valid": 0, "test": 0}
    assert stats["near_empty_dropped"] == 1
    assert [len(rows[s]) for s in prep.SPLITS] == [13, 3, 4]                 # official assignment kept
    src = {r["source_id"]: r for s in prep.SPLITS for r in rows[s]}
    assert "test:1" in src and "train:51" not in src and "train:52" not in src and "validation:2" not in src
    assert "train:50" not in src and "train:1" in src and "test:60" not in src
    assert src["train:4"]["ext"] == ".cpp" and src["train:4"]["lang"] == "cpp"
    assert sum(r["ext"] == ".c" for r in src.values()) == len(src) - 1
    assert src["train:53"]["commit_id"] is None and "\r" not in src["train:53"]["func"]
    assert src["train:0"]["commit_id"] == "c0001" and src["train:0"]["project"] == "qemu" and src["train:0"]["target"] == 1
    assert all(r["problem_id"] is None and r["user_id"] is None for r in src.values())
    # splits are written in train, valid, test order and idx follows that order
    assert [r["idx"] for r in rows["train"]] == list(range(13))
    assert [r["idx"] for r in rows["valid"]] == list(range(13, 16))
    assert [r["idx"] for r in rows["test"]] == list(range(16, 20))
    # idempotent
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    devign.main(["--input", str(inp), "--out_dir", str(out)])
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


def test_find_hf_parquets_errors_without_files(tmp_path):
    with pytest.raises(SystemExit, match="no train"):
        prep.find_hf_parquets(tmp_path)
    (tmp_path / "data").mkdir()
    for s in ("train", "validation", "test"):
        pd.DataFrame({"a": [1]}).to_parquet(tmp_path / "data" / f"{s}-00000-of-00001.parquet")
    found = prep.find_hf_parquets(tmp_path)
    assert set(found) == {"train", "validation", "test"} and all(Path(p).exists() for p in found.values())
    info = prep.source_info(found)
    assert all(len(v["sha256"]) == 64 and v["bytes"] > 0 for v in info.values())
