"""MoleculeNet / OGB molecule datasets -> pipeline jsonl (binary, SMILES text + molecular graph).

The chemistry instance of the task suite: the text branch reads the SMILES string with a chemical
language model and the graph branch reads the molecular graph built from the same SMILES
(utils/process/molgraph.py; run.py --molgraph turns these jsonl files into the model inputs).

Datasets (single binary task each; the ogb_* variants carry OGB's shipped scaffold split, which is
the Hu et al. split, and keep every OGB row except SMILES the installed RDKit cannot parse):
  bbbp         MoleculeNet BBBP.csv            label p_np        (blood-brain barrier penetration)
  bace         MoleculeNet bace.csv            label Class       (BACE-1 inhibition), SMILES column mol
  clintox_tox  MoleculeNet clintox.csv.gz      label CT_TOX      (failed clinical trials for toxicity)
  hiv          MoleculeNet HIV.csv             label HIV_active  (HIV replication inhibition)
  ogb_molhiv   OGB ogbg-molhiv (hiv.zip)       label HIV_active
  ogb_molbbbp  OGB ogbg-molbbbp (bbbp.zip)     label p_np
  ogb_molbace  OGB ogbg-molbace (bace.zip)     label Class

Steps: read -> RDKit parse (unparsable rows dropped and counted) -> canonical SMILES (the text the
LM reads and the graph is built from; the original string is kept as source_id when it differs)
-> dedup on canonical SMILES (label-conflicting groups dropped, otherwise first row kept) ->
split -> <out_dir>/<dataset>_{train,valid,test}.jsonl + classes.json + prep_stats.json.

Split (default ``scaffold``): deterministic Bemis-Murcko scaffold split 80/10/10 as in Hu et al.
2020 ("Strategies for Pre-training Graph Neural Networks", chem/splitters.py): scaffolds with
chirality, scaffold sets sorted by (size, first index) in descending order as in Hu et al. (largest
first; among equal sizes the set with the larger first index first), train filled
first, then valid, then test. Molecules of one scaffold never cross splits, so the test set holds
structurally new molecules. ``random`` is a stratified random split (for contrast only);
``ogb`` uses OGB's shipped split indices (the default for every ogb_* dataset).

Row fields follow utils/data/prep.py with lang 'smiles' and ext null (no Joern).
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd


def _load_prep():
    import importlib.util
    path = Path(__file__).resolve().parents[2] / "utils" / "data" / "prep.py"
    spec = importlib.util.spec_from_file_location("dataset_prep_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prep = _load_prep()

S3 = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"
SOURCES = {
    "bbbp": {"url": S3 + "BBBP.csv", "file": "BBBP.csv", "smiles": "smiles", "label": "p_np"},
    "bace": {"url": S3 + "bace.csv", "file": "bace.csv", "smiles": "mol", "label": "Class"},
    "clintox_tox": {"url": S3 + "clintox.csv.gz", "file": "clintox.csv.gz", "smiles": "smiles", "label": "CT_TOX"},
    "hiv": {"url": S3 + "HIV.csv", "file": "HIV.csv", "smiles": "smiles", "label": "HIV_active"},
    "ogb_molhiv": {"url": "http://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/hiv.zip",
                   "file": "hiv.zip", "smiles": "smiles", "label": "HIV_active"},
    "ogb_molbbbp": {"url": "http://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/bbbp.zip",
                    "file": "bbbp.zip", "smiles": "smiles", "label": "p_np"},
    "ogb_molbace": {"url": "http://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/bace.zip",
                    "file": "bace.zip", "smiles": "smiles", "label": None},     # label column read from the file
}


def _chem():
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as e:  # pragma: no cover
        raise SystemExit("[prep] FATAL: the molecule datasets need RDKit (pip install rdkit)") from e
    RDLogger.DisableLog("rdApp.*")
    return Chem, MurckoScaffold


def download(name: str, input_dir: Path) -> Path:
    src = SOURCES[name]
    input_dir.mkdir(parents=True, exist_ok=True)
    dest = input_dir / src["file"]
    if not dest.exists():
        print(f"[prep] downloading {src['url']}")
        with urllib.request.urlopen(src["url"], timeout=300) as r, open(dest, "wb") as f:
            f.write(r.read())
    return dest


def read_source(name: str, path: Path):
    """-> (DataFrame[smiles_raw, target, source_id], official split Series or None)."""
    src = SOURCES[name]
    official = None
    if name.startswith("ogb_"):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            mol = next(n for n in names if n.endswith("mapping/mol.csv.gz"))
            df = pd.read_csv(io.BytesIO(z.read(mol)), compression="gzip")
            part = pd.Series(index=range(len(df)), dtype=object)
            for split, ours in (("train", "train"), ("valid", "valid"), ("test", "test")):
                f = next(n for n in names if n.endswith(f"split/scaffold/{split}.csv.gz"))
                rows = pd.read_csv(io.BytesIO(z.read(f)), compression="gzip", header=None)[0].astype(int)
                part.iloc[rows.values] = ours
        official = part
    else:
        df = pd.read_csv(path, compression="gzip" if str(path).endswith(".gz") else None)
    label = src["label"] or next(c for c in df.columns if c not in ("smiles", "mol_id"))
    out = pd.DataFrame({"smiles_raw": df[src["smiles"]].astype(str), "target": df[label]})
    out["source_id"] = [f"{name}:{i}" for i in range(len(out))]
    return out, official


def scaffold_split(smiles: list[str], fractions=(0.8, 0.1, 0.1)) -> list[str]:
    """Hu et al. 2020 deterministic scaffold split. Returns the split name per molecule."""
    _, MurckoScaffold = _chem()
    groups = defaultdict(list)
    for i, s in enumerate(smiles):
        groups[MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=True)].append(i)
    sets = sorted(groups.values(), key=lambda g: (len(g), g[0]), reverse=True)
    n = len(smiles)
    cut_train, cut_valid = fractions[0] * n, (fractions[0] + fractions[1]) * n
    part = [None] * n
    tr, va = [], []
    for g in sets:
        if len(tr) + len(g) > cut_train:
            if len(tr) + len(va) + len(g) > cut_valid:
                for i in g:
                    part[i] = "test"
            else:
                va.extend(g)
        else:
            tr.extend(g)
    for i in tr:
        part[i] = "train"
    for i in va:
        part[i] = "valid"
    return part


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(SOURCES))
    p.add_argument("--input", required=True, help="directory holding (or receiving with --download) the source file")
    p.add_argument("--download", action="store_true")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--split", choices=["scaffold", "random", "ogb"], default=None,
                   help="default: ogb for ogb_molhiv, scaffold otherwise")
    p.add_argument("--dedup", choices=["canonical", "none"], default=None,
                   help="canonical: drop label-conflicting duplicate groups and keep the first row of the others "
                        "(default for own splits); none: keep every row (default for ogb_*, to match OGB / Hu et al.)")
    p.add_argument("--fractions", default="0.8,0.1,0.1")
    p.add_argument("--seed", type=int, default=42, help="random split only (the scaffold split is deterministic)")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    Chem, _ = _chem()
    is_ogb = args.dataset.startswith("ogb_")
    split = args.split or ("ogb" if is_ogb else "scaffold")
    if split == "ogb" and not is_ogb:
        raise SystemExit("[prep] FATAL: --split ogb exists only for the ogb_* datasets")
    dedup = args.dedup or ("none" if is_ogb else "canonical")
    inp = Path(args.input)
    path = download(args.dataset, inp) if args.download else inp / SOURCES[args.dataset]["file"]
    if not path.exists():
        raise SystemExit(f"[prep] FATAL: {path} missing (pass --download)")
    df, official = read_source(args.dataset, path)
    stats = {"dataset": args.dataset, "source": {"path": str(path), "sha256": prep.file_sha256(path),
                                                 "url": SOURCES[args.dataset]["url"]},
             "split_mode": split, "options": vars(args), "steps": []}

    def step(name, before, after):
        stats["steps"].append({"step": name, "rows_in": int(before), "rows_out": int(after), "dropped": int(before - after)})
        print(f"[prep] {name:<44s} {before:>8d} -> {after:>8d}  (dropped {before - after})")

    if official is not None:
        df["partition"] = official.values
    n0 = len(df)
    df = df[df["target"].notna()].copy()
    df["target"] = df["target"].astype(float).astype(int)
    if not set(df["target"].unique()) <= {0, 1}:
        raise SystemExit(f"[prep] FATAL: {args.dataset} labels are not 0/1: {sorted(df['target'].unique())[:10]}")
    step("drop rows without a label", n0, len(df))
    n0 = len(df)
    df["func"] = [(lambda m: Chem.MolToSmiles(m) if m is not None and m.GetNumAtoms() > 0 else None)(Chem.MolFromSmiles(s))
                  for s in df["smiles_raw"]]
    stats["unparsable_examples"] = df.loc[df["func"].isna(), "smiles_raw"].head(10).tolist()
    df = df[df["func"].notna()].copy()
    step("drop SMILES RDKit cannot parse", n0, len(df))
    n0 = len(df)
    df["dedup_key"] = df["func"]
    nun = df.groupby("dedup_key")["target"].nunique()
    vc = df["dedup_key"].value_counts()
    stats["duplicate_groups"] = {"groups": int((vc > 1).sum()), "rows_in_groups": int(vc[vc > 1].sum()),
                                 "label_conflicting_groups": int((nun > 1).sum())}
    if official is not None:
        nparts = df.groupby("dedup_key")["partition"].nunique()
        stats["duplicate_groups"]["groups_crossing_official_splits"] = int((nparts > 1).sum())
    if dedup == "none":
        stats["dedup"] = {"mode": "none"}
    elif official is not None:
        # keep OGB's assignment; a duplicate in two official splits keeps the later split (test wins)
        rank = df["partition"].map({"train": 0, "valid": 1, "test": 2})
        df = df.assign(_r=rank).sort_values("_r", kind="mergesort", ascending=False)
        df, dstats = prep.dedup_by_key(df)
        df = df.sort_index().drop(columns="_r")
        stats["dedup"] = dstats
    else:
        df, dstats = prep.dedup_by_key(df)
        stats["dedup"] = dstats
    step("dedup canonical SMILES (conflicts dropped)" if dedup != "none" else "no dedup (duplicates kept)", n0, len(df))
    df = df.reset_index(drop=True)
    df["idx"] = range(len(df))          # order of the source file after filtering: stable for given options
    if split == "scaffold":
        df["partition"] = scaffold_split(df["func"].tolist(), prep.parse_fractions(args.fractions))
    elif split == "random":
        df = prep.split_dataframe(df, mode="random", fractions=prep.parse_fractions(args.fractions), seed=args.seed)
        df = df.sort_values("idx").reset_index(drop=True)
    df["lang"], df["ext"], df["project"] = "smiles", None, args.dataset
    df["commit_id"] = df["problem_id"] = df["user_id"] = None
    df["source_id"] = [sid if raw == can else f"{sid}|{raw}" for sid, raw, can in
                       zip(df["source_id"], df["smiles_raw"], df["func"])]
    for name in prep.SPLITS:
        if not (df["partition"] == name).any():
            raise SystemExit(f"[prep] FATAL: split {name} is empty")
    stats["positive_rate"] = {s: float(df.loc[df["partition"] == s, "target"].mean()) for s in prep.SPLITS}
    stats["atoms_per_molecule"] = df["func"].map(lambda s: Chem.MolFromSmiles(s).GetNumAtoms()).describe().round(2).to_dict()
    paths = prep.write_outputs(args.out_dir, args.dataset, df, {0: "0", 1: "1"}, stats)
    print(f"\n=== {args.dataset}: {split} split ===")
    for s in prep.SPLITS:
        print(f"{s:<6s} rows {stats['splits'][s]['rows']:>6d}  positive rate {stats['positive_rate'][s]:.3f}")
    for k, v in paths.items():
        print(f"  {k:<8s} {v}")
    return stats


if __name__ == "__main__":
    main()
