"""POJ-104 (HF google/code_x_glue_cc_clone_detection_poj104) -> pipeline jsonl (104-class, whole programs).

See the contract in utils/data/prep.py.  Steps:
  concat the three HF splits (they are CLASS-disjoint: train 1..64, validation 65..80, test 81..104;
  classes 65 and 81 are duplicated across splits = 1,000 identical rows) -> CRLF->LF -> drop exact
  duplicate (label, code) rows -> target = dense index over the kept labels in numeric order
  (= int(label) - 1 when no problem is dropped; <dataset>_classes.json maps back) -> cleaned_code
  -> whitespace-normalised dedup (conflicting labels dropped) -> drop near-empty -> optional
  --per_class subsample -> choose_extension -> stratified random split (default 80/10/10).

--drop_problems: comma-separated original labels to drop before anything else (default none).
  Project CodeNet's README notes that POJ-104 problems 26 and 62 are identical problems; we keep
  them by default so the 104-class setting stays comparable with the literature.

Example:
  python data/scripts/make_poj104_json.py --input data/raw/poj104 --hf --out_dir data/cleaned_data
"""
from __future__ import annotations

import argparse
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

HF_REPO = "google/code_x_glue_cc_clone_detection_poj104"
HF_ORDER = ("train", "validation", "test")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="dir holding train/validation/test parquet (or data/ below it)")
    p.add_argument("--hf", action="store_true", help=f"download the parquet files of {HF_REPO} into --input")
    p.add_argument("--hf_repo", default=HF_REPO)
    p.add_argument("--hf_revision", default=None)
    p.add_argument("--out_dir", required=True, help="where <dataset>_{train,valid,test}.jsonl are written")
    p.add_argument("--dataset", default="poj104", help="file prefix (run.py --dataset)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fractions", default="0.8,0.1,0.1", help="train,valid,test fractions")
    p.add_argument("--per_class", type=int, default=None, help="subsample N programs per class (default all)")
    p.add_argument("--drop_problems", default="", help="comma-separated original labels to drop, e.g. '26,62'")
    p.add_argument("--min_chars", type=int, default=prep.NEAR_EMPTY_MIN_CHARS)
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    log = prep.StepLog()
    stats = {"dataset": args.dataset, "options": vars(args).copy(), "steps": log.steps}

    if args.hf:
        stats["hf"] = prep.hf_download_parquets(args.hf_repo, args.input, args.hf_revision)
    files = prep.find_hf_parquets(args.input)
    stats["sources"] = prep.source_info(files)

    parts = []
    for hf_split in HF_ORDER:
        d = pd.read_parquet(files[hf_split], columns=["id", "code", "label"])
        d["hf_split"] = hf_split
        parts.append(d)
        stats.setdefault("hf_split_rows", {})[hf_split] = int(len(d))
    df = pd.concat(parts, ignore_index=True)
    n0 = len(df)
    df["label"] = df["label"].astype(str)
    df["source_id"] = df["hf_split"] + ":" + df["id"].astype(str)
    log.step("load (concat hf splits)", n0, n0, labels=int(df["label"].nunique()))

    drop = [x.strip() for x in args.drop_problems.split(",") if x.strip()]
    if drop:
        n = len(df)
        df = df[~df["label"].isin(drop)].copy()
        log.step("drop_problems", n, len(df), dropped_labels=drop)

    df["code"] = df["code"].map(prep.normalise_newlines)
    n = len(df)
    df = df.drop_duplicates(subset=["label", "code"], keep="first").copy()
    log.step("drop exact duplicate (label, code)", n, len(df))

    labels = sorted(df["label"].unique(), key=int)
    classes = {i: lab for i, lab in enumerate(labels)}
    label_to_target = {lab: i for i, lab in classes.items()}
    df["target"] = df["label"].map(label_to_target).astype(int)
    df["problem_id"] = df["label"]

    df["func"] = df["code"].map(prep.cleaned_code)
    df["dedup_key"] = df["func"].map(prep.ws_key)
    n = len(df)
    df, dstats = prep.dedup_by_key(df)
    stats["dedup"] = dstats
    log.step("ws-normalised dedup (conflicts dropped)", n, len(df), **{k: v for k, v in dstats.items()
                                                                     if k not in ("rows_in", "rows_out")})

    n = len(df)
    df, n_empty = prep.drop_near_empty(df, min_chars=args.min_chars)
    stats["near_empty_dropped"] = n_empty
    log.step(f"drop near-empty (<{args.min_chars} non-ws chars)", n, len(df))

    if args.per_class is not None:
        n = len(df)
        keep = []
        for _, g in df.groupby("target", sort=True):
            keep.extend(g.sample(n=min(len(g), args.per_class), random_state=args.seed).index.tolist())
        df = df.loc[sorted(keep)].copy()
        log.step(f"subsample --per_class {args.per_class}", n, len(df))

    df = prep.assign_ext(df)
    stats.update(prep.ext_stats(df))
    df = df.reset_index(drop=True)
    df["idx"] = df.index.astype(int)           # deterministic: concat order of the HF files
    df["project"] = "poj104"
    df["commit_id"] = None
    df["user_id"] = None

    df = prep.split_dataframe(df, mode="random", idcol="idx", labelcol="target", seed=args.seed,
                              fractions=args.fractions)
    stats["class_coverage"] = prep.require_train_coverage(df, classes={str(k): v for k, v in classes.items()})
    paths = prep.write_outputs(args.out_dir, args.dataset, df, classes, stats)
    prep.print_summary(args.dataset, stats, paths)
    return stats


if __name__ == "__main__":
    main()
