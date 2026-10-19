"""CodeXGLUE defect detection (Devign FFmpeg + QEMU; HF google/code_x_glue_cc_defect_detection) -> jsonl.

See the contract in utils/data/prep.py.  Binary, single C functions.  The OFFICIAL split assignment
is kept (no re-split): train / validation / test parquet -> train / valid / test.  Cleaning:
  target bool -> int -> CRLF->LF + cleaned_code -> whitespace-normalised dedup:
    * groups whose labels conflict are dropped from ALL splits;
    * cross-split leaks are removed from the EARLIER split in the order train < valid < test
      (test keeps its row, train loses it);
    * within-split duplicates keep the first row;
  -> drop near-empty -> ext by choose_extension (100% '.c' on the HF data: 27,151 rows), lang consistent with ext.
idx is renumbered 0.. over train, valid, test (HF ids are per split and not unique).  --seed is only
recorded (nothing here is random).

Example:
  python data/scripts/make_devign_json.py --input data/raw/devign --hf --out_dir data/cleaned_data
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

HF_REPO = "google/code_x_glue_cc_defect_detection"
HF_TO_SPLIT = {"train": "train", "validation": "valid", "test": "test"}
SPLIT_RANK = {"train": 0, "valid": 1, "test": 2}   # later split wins a cross-split duplicate


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="dir holding train/validation/test parquet (or data/ below it)")
    p.add_argument("--hf", action="store_true", help=f"download the parquet files of {HF_REPO} into --input")
    p.add_argument("--hf_repo", default=HF_REPO)
    p.add_argument("--hf_revision", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dataset", default="devign", help="file prefix (run.py --dataset)")
    p.add_argument("--seed", type=int, default=42, help="recorded only; the official split is kept")
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
    for hf_split in ("train", "validation", "test"):
        d = pd.read_parquet(files[hf_split], columns=["id", "func", "target", "project", "commit_id"])
        d["partition"] = HF_TO_SPLIT[hf_split]
        d["source_id"] = hf_split + ":" + d["id"].astype(str)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    n0 = len(df)
    stats["official_split_rows"] = {s: int((df.partition == s).sum()) for s in prep.SPLITS}
    df["target"] = df["target"].astype(bool).astype(int)
    log.step("load (official splits)", n0, n0, **stats["official_split_rows"])

    df["func"] = df["func"].map(prep.prepare_code)
    df["dedup_key"] = df["func"].map(prep.ws_key)

    # 1) label-conflicting groups: dropped from every split
    nun = df.groupby("dedup_key")["target"].nunique()
    conflict_keys = set(nun[nun > 1].index)
    conflict_mask = df["dedup_key"].isin(conflict_keys)
    n = len(df)
    stats["dedup"] = {"groups_total": int(len(nun)), "conflict_groups": int(len(conflict_keys)),
                      "rows_dropped_conflict": int(conflict_mask.sum()),
                      "rows_dropped_conflict_per_split": {s: int((conflict_mask & (df.partition == s)).sum())
                                                          for s in prep.SPLITS}}
    df = df[~conflict_mask].copy()
    log.step("drop label-conflicting groups (all splits)", n, len(df))

    # 2) cross-split leaks: keep the row of the LATEST split (train < valid < test), drop the earlier ones
    df["rank"] = df["partition"].map(SPLIT_RANK)
    max_rank = df.groupby("dedup_key")["rank"].transform("max")
    leak_mask = df["rank"] < max_rank
    n = len(df)
    stats["dedup"]["rows_dropped_cross_split_leak_per_split"] = {s: int((leak_mask & (df.partition == s)).sum())
                                                                 for s in prep.SPLITS}
    df = df[~leak_mask].copy()
    log.step("drop cross-split leaks (earlier split loses)", n, len(df))

    # 3) within-split duplicates: keep first
    n = len(df)
    dup_mask = df.duplicated(subset=["dedup_key"], keep="first")
    stats["dedup"]["rows_dropped_within_split_duplicate_per_split"] = {s: int((dup_mask & (df.partition == s)).sum())
                                                                       for s in prep.SPLITS}
    df = df[~dup_mask].copy()
    log.step("drop within-split duplicates (keep first)", n, len(df))
    assert not df["dedup_key"].duplicated().any()
    df = df.drop(columns=["rank"])

    n = len(df)
    df, n_empty = prep.drop_near_empty(df, min_chars=args.min_chars)
    stats["near_empty_dropped"] = n_empty
    log.step(f"drop near-empty (<{args.min_chars} non-ws chars)", n, len(df))

    df = prep.assign_ext(df)
    stats.update(prep.ext_stats(df))
    df["problem_id"] = None
    df["user_id"] = None
    df["project"] = df["project"].astype(str)
    df["partition"] = pd.Categorical(df["partition"], categories=list(prep.SPLITS), ordered=True)
    df = df.sort_values("partition", kind="stable").reset_index(drop=True)   # train, valid, test; original order kept
    df["partition"] = df["partition"].astype(str)
    df["idx"] = df.index.astype(int)

    classes = {0: "0", 1: "1"}
    stats["class_coverage"] = prep.require_train_coverage(df, classes={"0": "0", "1": "1"})
    paths = prep.write_outputs(args.out_dir, args.dataset, df, classes, stats)
    prep.print_summary(args.dataset, stats, paths)
    return stats


if __name__ == "__main__":
    main()
