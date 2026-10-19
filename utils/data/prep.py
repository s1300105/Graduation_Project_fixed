"""Shared helpers for the dataset preparation scripts in ``data/scripts/make_*_json.py``.

Common contract (every ``make_<dataset>_json.py`` follows it)
=============================================================

Output
------
``<out_dir>/<dataset>_{train,valid,test}.jsonl`` -- one JSON object per line.  ``run.py --dataset
<dataset> --data_root <root>`` expects them under ``<root>/cleaned_data/``.  Next to them:

* ``<dataset>_classes.json``  -- ``{"<target int>": "<original class / problem id>"}`` (strings).
* ``<dataset>_prep_stats.json`` -- rows in/out at every step, per-split sizes and label counts,
  duplicate counts, ext distribution, K&R count, options used, source identifiers / revisions.

Row fields (in this order)
--------------------------
``idx``        int, unique across ALL splits and stable for a given seed AND the same filtering
               options (``--per_class`` / ``--drop_problems`` / ``--min_chars`` / ``--problems`` /
               ``--per_problem`` / ``--metadata_dir``; changing any of them renumbers idx).
               ``compare_runs`` aligns test items by ``idx`` and ``shuffle_edges`` seeds on it; HF
               per-split ids are NOT unique across splits, so every script renumbers from 0 in a
               deterministic order.
``func``       str, LF line endings, cleaned with the SAME ``cleaned_code`` as BigVul
               (``remove_empty_lines`` + ``remove_comments`` + ``remove_space_after_newline``;
               copied below, equality with ``data/scripts/make_bigvul_json.py`` is unit-tested).
``target``     int in ``0..C-1`` (dense).
``lang``       ``'c'`` | ``'cpp'`` (always consistent with ``ext``).
``ext``        ``'.c'`` | ``'.cpp'`` chosen by :func:`choose_extension`.
``project``    str.
``commit_id``  str or null.
``problem_id`` str or null (original problem / class id).
``source_id``  original id as str (``"<hf split>:<id>"`` for HF datasets so it is unique;
               the submission id for CodeNet).
``user_id``    str or null.
``split``      the split name (``train`` / ``valid`` / ``test``).

Extension rule (verified with Joern 4.0.400)
--------------------------------------------
``choose_extension(code)`` -> ``'.c'`` if the program has a K&R ``main`` (a line starting with
``main(`` and no return type; ``void main`` is fine as ``.cpp``, K&R is NOT: written as ``.cpp`` a
K&R program loses ``main`` entirely), else ``'.cpp'`` if C++ hints are present (C++ headers,
``using namespace``, ``std::``, ``cin``/``cout``, ``vector<``, ``template<``, ``class X {``,
``new T``, ``delete p``, ``::``, ...), else ``'.c'``.  A dataset can force a default
(CodeNet: always ``'.cpp'`` unless K&R main) with ``default_ext='.cpp'``.

Deduplication
-------------
Key = whitespace-normalised code (all whitespace runs collapsed) computed AFTER ``cleaned_code``.
Groups with ONE label keep the first row; groups with CONFLICTING labels are dropped entirely.
Near-empty programs (< 20 non-whitespace characters) are dropped.  Everything is counted.

Splitting
---------
Stratified by ``target`` with ``sklearn.model_selection.train_test_split`` (seeded), fractions
configurable (default 0.8/0.1/0.1).  The logic is a copy of ``make_bigvul_json.split_dataframe``
(generalised to arbitrary fractions / seed; identical output for the defaults).  Every class must
appear in ``train`` or the script fails loudly.

All randomness is seeded (``--seed``, default 42); the scripts are idempotent and print a
summary table at the end.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SPLITS = ("train", "valid", "test")
ROW_FIELDS = ("idx", "func", "target", "lang", "ext", "project", "commit_id", "problem_id",
              "source_id", "user_id", "split")
NEAR_EMPTY_MIN_CHARS = 20


# =====================================================================
# cleaning -- verbatim copies of data/scripts/make_bigvul_json.py (tested for equality)
# =====================================================================
def remove_empty_lines(text):
    return re.sub(r'^$\n', '', text, flags=re.MULTILINE)


def remove_comments(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)


def remove_space_after_newline(text):
    return re.sub(r'^\s+', '', text, flags=re.MULTILINE)


def cleaned_code(func_code: str) -> str:
    """Same three steps as BigVul: drop empty lines, drop comments, drop leading blanks."""
    if not isinstance(func_code, str):
        return ""

    func_code = remove_empty_lines(func_code)
    func_code = remove_comments(func_code)
    func_code = remove_space_after_newline(func_code)
    return func_code


def normalise_newlines(text: str) -> str:
    """CRLF / lone CR -> LF.  Must run BEFORE ``cleaned_code`` (its ``^$\\n`` regex needs LF)."""
    if not isinstance(text, str):
        return ""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def prepare_code(raw: str) -> str:
    """newline normalisation + BigVul cleaning: the ``func`` field of every row."""
    return cleaned_code(normalise_newlines(raw))


def ws_key(code: str) -> str:
    """Whitespace-normalised dedup key (all whitespace runs collapsed to one space)."""
    return " ".join((code or "").split())


def non_ws_chars(code: str) -> int:
    return len(re.sub(r"\s", "", code or ""))


# =====================================================================
# extension rule
# =====================================================================
_STRIP_RE = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|//[^\n]*|/\*.*?\*/', re.S)
_KNR_MAIN_LINE = re.compile(r"^[ \t]*main\s*\(", re.M)
# the previous non-blank line ends with a type / qualifier / '*' -> `int\nmain()` is not K&R
_TYPE_TAIL = re.compile(
    r"(?:\b(?:int|void|unsigned|signed|long|short|char|double|float|static|inline|extern|const|auto)\b|\*)\s*$")
_CPP_HEADERS = ("iostream", "bits/stdc++.h", "vector", "string", "algorithm", "map", "set", "queue",
                "deque", "stack", "list", "cstdio", "cstring", "cmath", "cstdlib", "climits", "cctype",
                "iomanip", "sstream", "fstream", "utility", "functional", "numeric", "bitset", "tuple",
                "unordered_map", "unordered_set", "array", "memory", "cassert", "complex", "iterator",
                "limits", "random", "chrono", "regex", "typeinfo", "exception", "stdexcept", "new")
_CPP_HINTS = re.compile(
    r"#\s*include\s*<(?:" + "|".join(re.escape(h) for h in _CPP_HEADERS) + r")\s*>"
    r"|\busing\s+namespace\b"
    r"|\busing\s+[A-Za-z_]\w*\s*="                      # using ll = long long;
    r"|\bstd\s*::"
    r"|(?<=[\w>])::(?=[A-Za-z_~])"                      # scope resolution joining two names (std::x, T<int>::y, A::~A);
                                                         # NOT the `::` operand separator of GCC inline asm (`"..." :: "r"(x)`)
    r"|\bcin\s*>>|\bcout\s*<<|\bcerr\s*<<|\bendl\b"      # (a bare `cin` is a C variable name in FFmpeg)
    r"|\b(?:vector|pair|queue|stack|deque|priority_queue|unordered_map|unordered_set|bitset)\s*<\s*[A-Za-z_][\w\s:<>,*]*>"  # vector<int>, not `vector < n` (map/set are common C identifiers)
    r"|\btemplate\s*<"
    r"|\bclass\s+[A-Za-z_]\w*\s*[:{]"
    r"|\bstruct\s+[A-Za-z_]\w*\s*\{[^}]*\b(?:public|private|operator|virtual)\b"
    r"|\bnew\s+[A-Za-z_][\w:<>]*\s*[\(\[;]"            # new Type( / new Type[ / new Type;
    r"|\bdelete\b\s*(?:\[\]\s*)?[A-Za-z_]\w*\s*;"       # delete p; / delete[] p;  (NOT the C label `delete_and_fail;`)
    r"|\bnullptr\b|\bvirtual\s+[A-Za-z_~]"
    r"|(?<=[\w&*>])\s*\boperator\s*(?:\(\)|\[\]|[-+*/%^&|~!=<>]+)\s*\("   # `T& operator=(`; NOT the C assignment `operator = (x)`
    # (`this->` is deliberately not a hint: QEMU has C functions with a variable named `this`)
    r"|\b(?:static|const|reinterpret|dynamic)_cast\s*<"
    r"|\btry\s*\{|\bcatch\s*\("
    r"|\bauto\s+[A-Za-z_]\w*\s*[=:]"                   # (bool/true/false are NOT hints: C99 stdbool)
)


def strip_literals_and_comments(code: str) -> str:
    return _STRIP_RE.sub('""', code or "")


_TYPED_MAIN = re.compile(r"\b(?:int|void|long|short|unsigned|signed|char|double|float|auto)\s+main\s*\(")


def knr_main(code: str) -> bool:
    """True when the program defines ``main`` K&R-style: ``main(`` at the start of a line with no
    return type (not even on the previous non-blank line).  Comments / literals are ignored.
    A program that defines a typed ``main`` anywhere is never K&R: after ``cleaned_code`` strips
    indentation a recursive call ``main();`` sits at column 0 and would otherwise be mistaken for
    a definition (seen in CodeNet, ~1e-5 of programs)."""
    s = strip_literals_and_comments(code)
    if _TYPED_MAIN.search(s):
        return False
    lines = s.split("\n")
    for i, line in enumerate(lines):
        if not _KNR_MAIN_LINE.match(line):
            continue
        j = i - 1
        while j >= 0 and not lines[j].strip():
            j -= 1
        prev = lines[j].strip() if j >= 0 else ""
        if prev and _TYPE_TAIL.search(prev):
            continue  # `int\nmain()` : the return type sits on the previous line
        return True
    return False


def looks_like_cpp_program(code: str) -> bool:
    """Whole-program C++ hints (headers, std, streams, templates, ...), checked on code with
    string / char literals and comments removed."""
    return bool(_CPP_HINTS.search(strip_literals_and_comments(code)))


def choose_extension(code: str, default_ext: str | None = None) -> str:
    """'.c' for K&R main; else ``default_ext`` if given ('.cpp' for CodeNet); else '.cpp' when
    C++ hints are present; else '.c'."""
    if knr_main(code):
        return ".c"
    if default_ext is not None:
        if default_ext not in (".c", ".cpp"):
            raise ValueError(default_ext)
        if default_ext == ".cpp":
            return ".cpp"
    return ".cpp" if looks_like_cpp_program(code) else ".c"


def lang_of(ext: str) -> str:
    return "cpp" if ext == ".cpp" else "c"


# =====================================================================
# dedup / filtering
# =====================================================================
def drop_near_empty(df: pd.DataFrame, code_col: str = "func", min_chars: int = NEAR_EMPTY_MIN_CHARS):
    """Drop rows with fewer than ``min_chars`` non-whitespace characters. Returns (df, n_dropped)."""
    n = df[code_col].map(non_ws_chars)
    keep = n >= min_chars
    return df[keep].copy(), int((~keep).sum())


def dedup_by_key(df: pd.DataFrame, key_col: str = "dedup_key", label_col: str = "target"):
    """Whitespace-normalised dedup (keys must already be in ``key_col``).

    * groups with one label: keep the FIRST row (row order of ``df``);
    * groups with conflicting labels: drop every row.
    Returns (df, stats)."""
    nun = df.groupby(key_col)[label_col].nunique()
    conflict_keys = set(nun[nun > 1].index)
    conflict_mask = df[key_col].isin(conflict_keys)
    kept = df[~conflict_mask]
    before = len(kept)
    kept = kept.drop_duplicates(subset=[key_col], keep="first").copy()
    stats = {
        "rows_in": int(len(df)),
        "groups_total": int(len(nun)),
        "conflict_groups": int(len(conflict_keys)),
        "rows_dropped_conflict": int(conflict_mask.sum()),
        "rows_dropped_duplicate": int(before - len(kept)),
        "rows_out": int(len(kept)),
    }
    return kept, stats


# =====================================================================
# splitting -- copy of make_bigvul_json.split_dataframe / train_val_test_split_df, generalised
# =====================================================================
def parse_fractions(spec) -> tuple[float, float, float]:
    """'0.8,0.1,0.1' or a 3-sequence -> (train, valid, test) summing to 1."""
    if isinstance(spec, str):
        parts = [float(x) for x in spec.split(",")]
    else:
        parts = [float(x) for x in spec]
    if len(parts) != 3 or any(p <= 0 for p in parts) or abs(sum(parts) - 1.0) > 1e-6:
        raise ValueError(f"fractions must be 3 positive numbers summing to 1, got {spec!r}")
    return parts[0], parts[1], parts[2]


def train_val_test_split_df(df, idcol="idx", labelcol="target", seed: int = 42,
                            fractions=(0.8, 0.1, 0.1)):
    """Add a ``partition`` column by two stratified ``train_test_split`` calls (BigVul logic)."""
    train_rat, val_rat, test_rat = fractions
    X = df[idcol]
    y = df[labelcol]
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1 - train_rat, random_state=seed, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_test,
            y_test,
            test_size=test_rat / (test_rat + val_rat),
            random_state=seed,
            stratify=y_test,
        )
    except ValueError as e:  # sklearn: "The least populated class in y has only 1 member"
        counts = Counter(y.astype(int))
        raise SystemExit(f"[prep] FATAL: stratified split impossible ({e}); every class needs >= 2 rows in "
                         f"the valid+test part. {len(counts)} classes, smallest = {min(counts.values())} rows; "
                         f"lower the number of classes, add rows or change --fractions") from e
    X_train = set(X_train)
    X_val = set(X_val)
    X_test = set(X_test)

    def path_to_label(path):
        if path in X_train:
            return "train"
        if path in X_val:
            return "valid"
        if path in X_test:
            return "test"

    df = df.copy()
    df["partition"] = df[idcol].apply(path_to_label)
    return df


def split_dataframe(df, mode: str = "random", idcol: str = "idx", labelcol: str = "target",
                    groupcol: str = "user_id", seed: int = 42, fractions=(0.8, 0.1, 0.1)):
    """Add a ``partition`` column (train/valid/test).

    mode="random" : stratified random split (make_bigvul_json.train_val_test_split_df).
    mode="group"  : every group (user) goes to exactly one split (GroupShuffleSplit, the logic of
                    make_bigvul_json's mode="project"); sizes are approximate."""
    fractions = parse_fractions(fractions)
    if mode == "random":
        return train_val_test_split_df(df, idcol=idcol, labelcol=labelcol, seed=seed, fractions=fractions)
    if mode != "group":
        raise ValueError(mode)
    from sklearn.model_selection import GroupShuffleSplit
    if groupcol not in df.columns:
        raise ValueError(f"column {groupcol!r} needed for a group split")
    train_rat, val_rat, test_rat = fractions
    groups = df[groupcol].fillna("<none>").astype(str).values
    idx = np.arange(len(df))
    gss1 = GroupShuffleSplit(n_splits=1, test_size=1 - train_rat, random_state=seed)
    tr, rest = next(gss1.split(idx, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=test_rat / (test_rat + val_rat), random_state=seed)
    va_rel, te_rel = next(gss2.split(rest, groups=groups[rest]))
    part = np.empty(len(df), dtype=object)
    part[tr] = "train"; part[rest[va_rel]] = "valid"; part[rest[te_rel]] = "test"
    df = df.copy()
    df["partition"] = part
    assert not (set(df[df.partition == "train"][groupcol]) & set(df[df.partition == "test"][groupcol]))
    assert not (set(df[df.partition == "train"][groupcol]) & set(df[df.partition == "valid"][groupcol]))
    assert not (set(df[df.partition == "valid"][groupcol]) & set(df[df.partition == "test"][groupcol]))
    return df


def class_coverage(df: pd.DataFrame, labelcol: str = "target", partcol: str = "partition") -> dict:
    """Per split: number of distinct classes and the classes missing from it."""
    classes = sorted(int(c) for c in df[labelcol].unique())
    out = {"num_classes": len(classes), "per_split": {}}
    for name in SPLITS:
        present = set(int(c) for c in df.loc[df[partcol] == name, labelcol].unique())
        missing = [c for c in classes if c not in present]
        out["per_split"][name] = {"classes_present": len(present), "classes_missing": missing}
    return out


def require_train_coverage(df: pd.DataFrame, labelcol: str = "target", partcol: str = "partition",
                           classes: dict | None = None) -> dict:
    """Fail loudly when a class is absent from train.  Returns the coverage dict."""
    cov = class_coverage(df, labelcol, partcol)
    missing = cov["per_split"]["train"]["classes_missing"]
    if missing:
        names = [classes.get(str(m), str(m)) if classes else str(m) for m in missing]
        raise SystemExit(f"[prep] FATAL: {len(missing)} class(es) have no row in train: {names[:20]}"
                         f"{' ...' if len(names) > 20 else ''} -- lower --per_* or change --fractions")
    if (df[partcol].isna()).any():
        raise SystemExit("[prep] FATAL: some rows received no partition")
    return cov


# =====================================================================
# stats / output
# =====================================================================
class StepLog:
    """Ordered record of rows in/out at every step of a script."""

    def __init__(self):
        self.steps = []

    def step(self, name: str, rows_in: int, rows_out: int, **extra):
        rec = {"step": name, "rows_in": int(rows_in), "rows_out": int(rows_out),
               "dropped": int(rows_in - rows_out)}
        rec.update(extra)
        self.steps.append(rec)
        print(f"[prep] {name:<40s} {rows_in:>8d} -> {rows_out:>8d}  (dropped {rows_in - rows_out})")
        return rec


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def label_counts(df: pd.DataFrame, col: str = "target") -> dict:
    return {str(int(k)): int(v) for k, v in sorted(Counter(df[col].astype(int)).items())}


def assign_ext(df: pd.DataFrame, default_ext: str | None = None, code_col: str = "func") -> pd.DataFrame:
    """Add ``ext``, ``lang`` and ``knr`` columns."""
    df = df.copy()
    df["knr"] = df[code_col].map(knr_main)
    df["ext"] = df[code_col].map(lambda c: choose_extension(c, default_ext))
    df["lang"] = df["ext"].map(lang_of)
    return df


def ext_stats(df: pd.DataFrame) -> dict:
    return {"ext": {str(k): int(v) for k, v in sorted(Counter(df["ext"]).items())},
            "knr_main": int(df["knr"].sum()) if "knr" in df.columns else None}


def build_rows(df: pd.DataFrame, partcol: str = "partition") -> list[dict]:
    """DataFrame -> list of row dicts with exactly ROW_FIELDS (in order)."""
    rows = []
    for rec in df.to_dict("records"):
        def opt(k):
            v = rec.get(k)
            if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
                return None
            return str(v)
        rows.append({
            "idx": int(rec["idx"]),
            "func": rec["func"],
            "target": int(rec["target"]),
            "lang": rec["lang"],
            "ext": rec["ext"],
            "project": str(rec["project"]),
            "commit_id": opt("commit_id"),
            "problem_id": opt("problem_id"),
            "source_id": str(rec["source_id"]),
            "user_id": opt("user_id"),
            "split": rec[partcol],
        })
    return rows


def write_outputs(out_dir, dataset: str, df: pd.DataFrame, classes: dict, stats: dict,
                  partcol: str = "partition") -> dict:
    """Write the three jsonl files + classes.json + prep_stats.json.  Fills ``stats['splits']``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if df["idx"].duplicated().any():
        raise SystemExit("[prep] FATAL: idx not unique across splits")
    paths = {}
    stats["splits"] = {}
    for name in SPLITS:
        sub = df[df[partcol] == name]
        rows = build_rows(sub, partcol)
        p = out_dir / f"{dataset}_{name}.jsonl"
        with p.open("w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        paths[name] = str(p)
        stats["splits"][name] = {"rows": len(rows), "label_counts": label_counts(sub),
                                 "ext": {str(k): int(v) for k, v in sorted(Counter(sub["ext"]).items())}}
    stats["rows_out_total"] = int(len(df))
    stats["num_classes"] = len(classes)
    stats["idx_range"] = [int(df["idx"].min()), int(df["idx"].max())] if len(df) else None
    cp = out_dir / f"{dataset}_classes.json"
    cp.write_text(json.dumps({str(k): str(v) for k, v in classes.items()}, indent=1, ensure_ascii=False))
    sp = out_dir / f"{dataset}_prep_stats.json"
    sp.write_text(json.dumps(stats, indent=1, ensure_ascii=False, default=str))
    paths["classes"] = str(cp)
    paths["stats"] = str(sp)
    return paths


def print_summary(dataset: str, stats: dict, paths: dict):
    print(f"\n=== {dataset}: summary ===")
    print(f"{'step':<42s}{'in':>9s}{'out':>9s}{'dropped':>9s}")
    for s in stats.get("steps", []):
        print(f"{s['step']:<42s}{s['rows_in']:>9d}{s['rows_out']:>9d}{s['dropped']:>9d}")
    print(f"{'split':<12s}{'rows':>8s}{'classes':>9s}{'.c':>8s}{'.cpp':>8s}")
    for name in SPLITS:
        s = stats["splits"][name]
        print(f"{name:<12s}{s['rows']:>8d}{len(s['label_counts']):>9d}"
              f"{s['ext'].get('.c', 0):>8d}{s['ext'].get('.cpp', 0):>8d}")
    print(f"num_classes={stats['num_classes']}  rows_out_total={stats['rows_out_total']}  "
          f"knr_main={stats.get('knr_main')}  ext={stats.get('ext')}")
    for k, v in paths.items():
        print(f"  {k:<8s} {v}")


# =====================================================================
# HF parquet helpers (POJ-104 / Devign)
# =====================================================================
HF_SPLIT_FILES = {"train": "data/train-00000-of-00001.parquet",
                  "validation": "data/validation-00000-of-00001.parquet",
                  "test": "data/test-00000-of-00001.parquet"}


def hf_download_parquets(repo_id: str, input_dir, revision: str | None = None) -> dict:
    """Download the three parquet files of an HF dataset repo into ``input_dir`` (idempotent).
    Returns source info (repo, revision sha when reachable, files)."""
    from huggingface_hub import hf_hub_download
    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, fn in HF_SPLIT_FILES.items():
        files[split] = hf_hub_download(repo_id=repo_id, filename=fn, repo_type="dataset",
                                       revision=revision, local_dir=str(input_dir))
    sha = None
    try:
        from huggingface_hub import HfApi
        sha = HfApi().dataset_info(repo_id, revision=revision).sha
    except Exception as e:  # network / auth problems must not fail the run
        print(f"[prep] could not read the HF revision sha: {type(e).__name__}: {e}")
    return {"repo_id": repo_id, "revision": sha or revision, "files": files}


def find_hf_parquets(input_dir) -> dict:
    """Locate train / validation / test parquet files under ``input_dir`` (or ``input_dir/data``)."""
    input_dir = Path(input_dir)
    found = {}
    for split in ("train", "validation", "test"):
        cands = sorted(input_dir.glob(f"{split}*.parquet")) + sorted(input_dir.glob(f"data/{split}*.parquet"))
        if not cands:
            raise SystemExit(f"[prep] FATAL: no {split}*.parquet under {input_dir} (use --hf to download)")
        if len(cands) > 1:
            # a sharded split or a second download next to the first: reading one file would drop rows silently
            raise SystemExit(f"[prep] FATAL: several parquet files match {split}*: {[str(c) for c in cands]}; "
                             f"keep exactly one file per split under {input_dir}")
        found[split] = str(cands[0])
    return found


def source_info(files: dict) -> dict:
    return {split: {"path": str(p), "sha256": file_sha256(p), "bytes": Path(p).stat().st_size}
            for split, p in files.items()}

