#!/usr/bin/env python
"""Why did functions disappear between a jsonl split and <split>_<s>_cpg.pkl?

    python scripts/diagnose_joern_losses.py --split train --slice 0 --dataset bigvul \
        [--data_root data] [--joern_cli_dir joern/joern-cli/] [--max 60] [--seed 42] [--max_chars N]

Replays the slicing done by run.py --cpg (same seed / filters), finds the
functions of that slice that have no row in the cpg pickle, summarises them
(language, project, length, signature shape) and re-runs Joern on up to --max
of them to classify each loss:

  parses_now       with the CURRENT pipeline settings Joern produces a real method and
                   json_process keeps it: the original run lost it because the code or
                   settings differed then (e.g. extension choice before the lang fix)
  rescued_A/B/C    only parses after the macro-stripping rescue pass (stage A / B / C;
                   C = lowercase signature macros such as av_cold / coroutine_fn)
  no_method        Joern produces only the <global> pseudo-method even after rescue
  pipeline_bug     Joern produced a real method but json_process dropped it -> report
  joern_failed     joern-parse / export failed for the whole batch

Writes <out>/losses.csv and prints a summary.
"""
import argparse
import collections
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.data import datamanager as dm                      # noqa: E402
from utils.process import cpg_generator as cp                 # noqa: E402

MACRO_TOKEN = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")


def first_line(code: str) -> str:
    for l in (code or "").splitlines():
        if l.strip():
            return l.strip()
    return ""


def signature_shape(code: str) -> str:
    head = code.split("{", 1)[0]
    if MACRO_TOKEN.search(first_line(code)):
        return "macro_in_signature"
    if "::" in head:
        return "cpp_scope"
    if re.match(r"\s*[A-Za-z_]\w*\s*\(", first_line(code)) and "(" in head:
        return "no_return_type"
    if "#if" in head or "#define" in head:
        return "preprocessor_in_signature"
    return "plain"


def replay_slice(jsonl_path, seed, max_chars, min_chars, slice_size, s):
    df = pd.read_json(jsonl_path, lines=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    if max_chars:
        df = df[df.func.str.len() < max_chars]
    if min_chars:
        df = df[df.func.str.len() >= min_chars]
    df = dm.clean(df)
    if "idx" not in df.columns:
        df["idx"] = df.index
    for k, part in dm.slice_frame(df, slice_size):
        if k == s:
            part = part.copy()
            part["Index"] = part.index
            return part
    raise SystemExit(f"slice {s} not found")


def _joern_pass(df: pd.DataFrame, joern_cli_dir: str, work: Path, tag: str, graph_unit: str = "first_method"):
    """Write df.func through to_files, run Joern, return ({stem: [methods]}, {stem: filename}, kept_idx)."""
    if work.exists():
        shutil.rmtree(work)
    src = work / "src"
    dm.to_files(df, str(src))
    written = {p.stem: p.name for p in src.iterdir()}
    cpg_dir = work / "cpg"
    cpg_dir.mkdir(parents=True)
    try:
        cp.joern_parse(joern_cli_dir, str(src), str(cpg_dir) + "/", f"{tag}_cpg")
        res = cp.joern_create(joern_cli_dir, str(cpg_dir), str(cpg_dir), [f"{tag}_cpg.bin"])
    except Exception as e:
        print(f"[joern FAILED in {tag}] {e}")
        return None, written, set()
    jf = res.get(f"{tag}_cpg.bin")
    if not jf:
        return None, written, set()
    raw = json.loads(re.sub(r"io\.shiftleft\.codepropertygraph\.generated\.", "", Path(jf).read_text(encoding="utf-8")))
    by_file = collections.defaultdict(list)
    for m in raw["functions"]:
        f = str(m.get("file") or "")
        if f and f != "N/A" and not f.startswith("<"):
            by_file[Path(f).stem].append(m)
    kept_idx = {idx for idx, _ in (cp.json_process(str(cpg_dir), jf, {}, graph_unit=graph_unit) or [])}
    return by_file, written, kept_idx


def rerun_joern(lost: pd.DataFrame, joern_cli_dir: str, out_dir: Path, graph_unit: str = "first_method"):
    verdict, written_all = {}, {}
    by_file, written, kept = _joern_pass(lost, joern_cli_dir, out_dir / "joern_rerun", "rerun", graph_unit)
    written_all.update(written)
    if by_file is None:
        return {k: ("joern_failed", "", "") for k in written}, written_all
    pending = {}
    for stem in written:
        methods = by_file.get(stem, [])
        real = [m for m in methods if not cp.is_global_method(m)]
        glob = [m for m in methods if cp.is_global_method(m)]
        names = ";".join(m["function"][:40] for m in real[:3])
        labels = ",".join(f"{k}:{v}" for k, v in collections.Counter(
            n.get("label") for m in glob for n in m.get("nodes", [])).most_common(4))
        if real and int(stem) in kept:
            verdict[stem] = ("parses_now", names, "")
        elif real:
            verdict[stem] = ("pipeline_bug", names, labels)
        else:
            pending[stem] = labels
    # rescue passes on what is still lost
    for stage in dm.RESCUE_STAGES:
        if not pending:
            break
        rows = lost.loc[[int(s) for s in pending]].copy()
        variants = {ix: dict(dm.rescue_variants(code)).get(stage) for ix, code in rows["func"].items()}
        variants = {ix: v for ix, v in variants.items() if v}
        if not variants:
            continue
        rows = rows.loc[list(variants)]
        rows["func"] = pd.Series(variants)
        by_file, _, kept = _joern_pass(rows, joern_cli_dir, out_dir / f"joern_rescue_{stage}", f"rescue{stage}", graph_unit)
        if by_file is None:
            continue
        for stem in list(pending):
            real = [m for m in by_file.get(stem, []) if not cp.is_global_method(m)]
            if real and int(stem) in kept:
                verdict[stem] = (f"rescued_{stage}", ";".join(m["function"][:40] for m in real[:3]), "")
                pending.pop(stem)
    for stem, labels in pending.items():
        verdict[stem] = ("no_method", "", labels)
    return verdict, written_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--slice", type=int, default=0)
    ap.add_argument("--dataset", default="bigvul")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--jsonl")
    ap.add_argument("--joern_cli_dir", default="joern/joern-cli/")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_chars", type=int, default=None)
    ap.add_argument("--min_chars", type=int, default=0)
    ap.add_argument("--slice_size", type=int, default=500)
    ap.add_argument("--max", type=int, default=60, help="how many lost functions to re-run through Joern (0 = skip)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--graph_unit", choices=["first_method", "whole_file"], default=None,
                    help="graph unit of the --cpg run being diagnosed (default: read from data_root/cpg/cpg_manifest.json, "
                         "else first_method); the verdicts are only meaningful under the same unit")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    if args.graph_unit is None:
        mf = data_root / "cpg" / "cpg_manifest.json"
        args.graph_unit = (json.loads(mf.read_text()).get("args", {}).get("graph_unit", "first_method")
                           if mf.exists() else "first_method")
    print(f"graph_unit: {args.graph_unit}")
    jsonl = args.jsonl or str(data_root / "cleaned_data" / f"{args.dataset}_{args.split}.jsonl")
    pkl = data_root / "cpg" / f"{args.split}_{args.slice}_cpg.pkl"
    out = Path(args.out or f"logs/losses_{args.split}_{args.slice}")
    out.mkdir(parents=True, exist_ok=True)

    part = replay_slice(jsonl, args.seed, args.max_chars, args.min_chars, args.slice_size, args.slice)
    kept = pd.read_pickle(pkl)
    kept_idx = set(int(v) for v in kept["idx"])
    assert kept_idx <= set(int(v) for v in part["idx"]), "replayed slice does not contain the kept rows: seed / filters differ from the --cpg run"
    lost = part[~part["idx"].astype(int).isin(kept_idx)].copy()
    print(f"slice {args.split}_{args.slice}: {len(part)} functions, kept {len(kept)}, lost {len(lost)} ({len(lost)/max(1,len(part)):.1%})")

    lost["len"] = lost.func.str.len()
    lost["shape"] = lost.func.map(signature_shape)
    lost["first_line"] = lost.func.map(first_line)
    keptf = part[part["idx"].astype(int).isin(kept_idx)]
    print("\n-- by lang (lost / total):")
    for lang, n in part.get("lang", pd.Series(["?"] * len(part))).value_counts().items():
        nl = int((lost.get("lang", pd.Series(dtype=str)) == lang).sum())
        print(f"   {lang:6s} {nl:4d} / {n:4d}  ({nl/max(1,n):.0%})")
    print("-- signature shape of lost functions:", dict(lost["shape"].value_counts()))
    print("-- signature shape of kept functions:", dict(keptf.func.map(signature_shape).value_counts()))
    print(f"-- length: lost median {lost.len.median():.0f} vs kept median {keptf.func.str.len().median():.0f}")
    if "project" in lost.columns:
        print("-- top projects among lost:", dict(lost.project.value_counts().head(8)))
    print("-- 10 example first lines of lost functions:")
    for fl in lost.first_line.head(10):
        print("   ", fl[:110])

    if args.max > 0:
        sample = lost.head(args.max)
        verdict, written = rerun_joern(sample, args.joern_cli_dir, out, graph_unit=args.graph_unit)
        sample = sample.copy()
        sample["file"] = [written.get(str(i), "") for i in sample.index]
        sample["verdict"] = [verdict.get(str(i), ("?", "", ""))[0] for i in sample.index]
        sample["joern_methods"] = [verdict.get(str(i), ("", "", ""))[1] for i in sample.index]
        sample["global_labels"] = [verdict.get(str(i), ("", "", ""))[2] for i in sample.index]
        print(f"\n-- Joern re-run on {len(sample)} lost functions:", dict(sample.verdict.value_counts()))
        print("   parses_now / rescued_* = recovered by the current pipeline settings (extension fix, macro rescue)")
        bug = sample[sample.verdict == "pipeline_bug"]
        if len(bug):
            print("   !! pipeline_bug: Joern produced a real method but json_process dropped it -> report these:")
            print(bug[["idx", "file", "joern_methods", "global_labels"]].head(10).to_string(index=False))
        nm = sample[sample.verdict == "no_method"]
        if len(nm):
            print("   no_method = only <global> even after rescue; shapes:", dict(nm["shape"].value_counts()))
            for fl in nm.first_line.head(8):
                print("      ", fl[:110])
        sample.drop(columns=["func", "cpg"], errors="ignore").to_csv(out / "losses.csv", index=False)
        print(f"\nwrote {out/'losses.csv'}")
    else:
        lost.drop(columns=["func"], errors="ignore").to_csv(out / "losses.csv", index=False)
        print(f"\nwrote {out/'losses.csv'}")


if __name__ == "__main__":
    main()
