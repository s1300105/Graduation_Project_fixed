"""Project CodeNet C++1000 benchmark -> pipeline jsonl (N-class, whole modern C++ programs).

See the contract in utils/data/prep.py.  Input layout: ``<input>/pXXXXX/sNNNNNNNNN.cpp`` (1000 problem
dirs x 500 accepted C++ submissions).  Files are read as BYTES and decoded utf-8 with errors='replace'.

Steps:
  pick problems (--problems all | N random dirs with --seed) -> read every file of those problems
  -> optional metadata filter (--metadata_dir <Project_CodeNet/metadata> with pXXXXX.csv, official
  columns submission_id,problem_id,user_id,date,language,original_language,filename_ext,status,
  cpu_time,memory,code_size,accuracy): keep status == 'Accepted', attach user_id; submissions
  missing from the metadata are dropped and counted) -> CRLF->LF -> cleaned_code -> whitespace-
  normalised dedup (conflicting labels dropped) -> drop near-empty -> --per_problem subsample
  (AFTER dedup, deterministic) -> ext: '.cpp' unless K&R main -> target = dense index over the kept
  problems in sorted order -> split: random (stratified) | user (GroupShuffleSplit on user_id, needs
  metadata; a user's submissions never cross splits; class coverage is reported and train coverage
  enforced).

Example:
  python data/scripts/make_codenet_json.py --input /data/Project_CodeNet_C++1000 --problems 100 \
      --per_problem 100 --metadata_dir /data/Project_CodeNet/metadata --split user --out_dir data/cleaned_data
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import pandas as pd



def _load_prep():
    """Load utils/data/prep.py by path: importing the ``utils.data`` package would pull in torch
    (via datamanager), which a dataset preparation script does not need."""
    import importlib.util
    path = Path(__file__).resolve().parents[2] / "utils" / "data" / "prep.py"
    spec = importlib.util.spec_from_file_location("dataset_prep_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prep = _load_prep()

META_COLUMNS = ["submission_id", "problem_id", "user_id", "date", "language", "original_language",
                "filename_ext", "status", "cpu_time", "memory", "code_size", "accuracy"]
_PROBLEM_DIR = re.compile(r"^p\d{5}$")
_SUBMISSION = re.compile(r"^s\d+$")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="extracted Project_CodeNet_C++1000 directory")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dataset", default="codenet", help="file prefix (run.py --dataset)")
    p.add_argument("--per_problem", type=int, default=100, help="programs kept per problem after dedup (default 100)")
    p.add_argument("--problems", default="all", help="'all' or an int N: N random problem dirs (seeded)")
    p.add_argument("--metadata_dir", default=None, help="Project_CodeNet/metadata (pXXXXX.csv); enables the "
                                                        "Accepted filter, user_id and --split user")
    p.add_argument("--split", choices=["random", "user"], default="random")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fractions", default="0.8,0.1,0.1")
    p.add_argument("--min_chars", type=int, default=prep.NEAR_EMPTY_MIN_CHARS)
    return p.parse_args(argv)


def list_problems(input_dir: Path) -> list[str]:
    probs = sorted(d.name for d in input_dir.iterdir() if d.is_dir() and _PROBLEM_DIR.match(d.name))
    if not probs:
        raise SystemExit(f"[prep] FATAL: no pXXXXX directories under {input_dir}")
    return probs


def read_problem(input_dir: Path, problem: str) -> list[dict]:
    rows = []
    for f in sorted((input_dir / problem).iterdir()):
        if not f.is_file() or not _SUBMISSION.match(f.stem):
            continue
        raw = f.read_bytes().decode("utf-8", errors="replace")
        rows.append({"problem_id": problem, "source_id": f.stem, "file_ext": f.suffix, "code": raw})
    return rows


def read_metadata(metadata_dir: Path, problems: list[str]) -> pd.DataFrame:
    frames = []
    for prob in problems:
        p = metadata_dir / f"{prob}.csv"
        if not p.exists():
            raise SystemExit(f"[prep] FATAL: metadata file missing: {p}")
        m = pd.read_csv(p, dtype=str, keep_default_na=False)
        missing = [c for c in META_COLUMNS if c not in m.columns]
        if missing:
            raise SystemExit(f"[prep] FATAL: {p} lacks metadata columns {missing}; has {list(m.columns)}")
        frames.append(m[["submission_id", "problem_id", "user_id", "status", "language"]])
    return pd.concat(frames, ignore_index=True)


def main(argv=None) -> dict:
    args = parse_args(argv)
    input_dir = Path(args.input)
    log = prep.StepLog()
    stats = {"dataset": args.dataset, "options": vars(args).copy(), "steps": log.steps,
             "sources": {"input_dir": str(input_dir.resolve()), "metadata_dir": args.metadata_dir}}
    if args.split == "user" and not args.metadata_dir:
        raise SystemExit("[prep] FATAL: --split user needs --metadata_dir (user ids come from the metadata)")

    all_problems = list_problems(input_dir)
    stats["sources"]["problem_dirs_found"] = len(all_problems)
    if args.problems == "all":
        problems = all_problems
    else:
        n = int(args.problems)
        if n > len(all_problems):
            raise SystemExit(f"[prep] FATAL: --problems {n} > {len(all_problems)} dirs available")
        problems = sorted(random.Random(args.seed).sample(all_problems, n))
    stats["problems_selected"] = problems

    rows = []
    for prob in problems:
        rows.extend(read_problem(input_dir, prob))
    df = pd.DataFrame(rows, columns=["problem_id", "source_id", "file_ext", "code"])
    log.step("read files", len(df), len(df), problems=len(problems))
    stats["file_ext_in"] = {str(k): int(v) for k, v in df["file_ext"].value_counts().items()}

    if args.metadata_dir:
        meta = read_metadata(Path(args.metadata_dir), problems)
        meta = meta.drop_duplicates(subset=["submission_id"], keep="first").set_index("submission_id")
        n = len(df)
        in_meta = df["source_id"].isin(meta.index)
        stats["metadata"] = {"rows_in_metadata": int(len(meta)), "submissions_missing_from_metadata": int((~in_meta).sum())}
        df = df[in_meta].copy()
        log.step("metadata: submission present", n, len(df))
        df["status"] = meta.loc[df["source_id"], "status"].values
        df["user_id"] = meta.loc[df["source_id"], "user_id"].values
        stats["metadata"]["status_counts"] = {str(k): int(v) for k, v in df["status"].value_counts().items()}
        n = len(df)
        df = df[df["status"] == "Accepted"].copy()
        log.step("metadata: status == Accepted", n, len(df))
        df = df.drop(columns=["status"])
    else:
        df["user_id"] = None

    df["code"] = df["code"].map(prep.normalise_newlines)
    df["func"] = df["code"].map(prep.cleaned_code)
    df["dedup_key"] = df["func"].map(prep.ws_key)
    df["target_tmp"] = df["problem_id"]  # dedup conflicts are judged on the problem id
    n = len(df)
    df, dstats = prep.dedup_by_key(df, label_col="target_tmp")
    stats["dedup"] = dstats
    log.step("ws-normalised dedup (conflicts dropped)", n, len(df), **{k: v for k, v in dstats.items()
                                                                     if k not in ("rows_in", "rows_out")})
    n = len(df)
    df, n_empty = prep.drop_near_empty(df, min_chars=args.min_chars)
    stats["near_empty_dropped"] = n_empty
    log.step(f"drop near-empty (<{args.min_chars} non-ws chars)", n, len(df))

    df = df.sort_values(["problem_id", "source_id"]).reset_index(drop=True)
    n = len(df)
    keep = []
    for _, g in df.groupby("problem_id", sort=True):
        keep.extend(g.sample(n=min(len(g), args.per_problem), random_state=args.seed).index.tolist())
    df = df.loc[sorted(keep)].copy()
    log.step(f"subsample --per_problem {args.per_problem}", n, len(df))
    stats["per_problem_counts"] = {str(k): int(v) for k, v in df["problem_id"].value_counts().sort_index().items()}

    kept_problems = sorted(df["problem_id"].unique())
    classes = {i: p for i, p in enumerate(kept_problems)}
    df["target"] = df["problem_id"].map({p: i for i, p in classes.items()}).astype(int)
    df = df.drop(columns=["target_tmp"])

    df = prep.assign_ext(df, default_ext=".cpp")
    stats.update(prep.ext_stats(df))
    df = df.reset_index(drop=True)
    df["idx"] = df.index.astype(int)            # deterministic: sorted by (problem_id, submission id)
    df["project"] = "codenet_cpp1000"
    df["commit_id"] = None

    mode = "group" if args.split == "user" else "random"
    df = prep.split_dataframe(df, mode=mode, idcol="idx", labelcol="target", groupcol="user_id",
                              seed=args.seed, fractions=args.fractions)
    stats["class_coverage"] = prep.require_train_coverage(df, classes={str(k): v for k, v in classes.items()})
    if mode == "group":
        stats["users"] = {name: int(df.loc[df.partition == name, "user_id"].nunique()) for name in prep.SPLITS}
    paths = prep.write_outputs(args.out_dir, args.dataset, df, classes, stats)
    prep.print_summary(args.dataset, stats, paths)
    return stats


if __name__ == "__main__":
    main()
