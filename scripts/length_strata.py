#!/usr/bin/env python
"""Does the text encoder miss vulnerabilities that lie beyond its window?

    python scripts/length_strata.py \\
        --group codebert "runs/codebert_bs16_lr1.0e-04_br0.30_seed*" \\
        --group hybrid   "runs/hybrid_bs16_lr1.0e-04_br0.30_seed*" \\
        [--bigvul_csv data/raw/MSR_data_cleaned.csv] [--out runs/length_strata]

The check behind the proposal "let the graph carry what the language model cannot read":
it has room only if the text encoder fails on functions whose flaw lies outside the part it
reads. Each run directory needs test_preds.npz and metrics.json (run.py). Every run is scored
at its own validation-tuned threshold (--preds final: the last epoch at 0.5), as in
compare_runs.py, and the groups are aligned on the test idx they share.

Window. The test texts are read from the run's test JSONL (args.json; --test_jsonl to
override) and tokenised as the text branch does (models/LMGNN.py: the run's text_model
tokenizer, truncation at max_length). W = max_length minus the special tokens is the number
of content tokens the encoder sees (510 for CodeBERT at 512); a function with more than W
content tokens is "truncated".

Flaw position (--bigvul_csv). The vulnerable lines of a function are the lines of
func_before that the fix removes or rewrites (difflib against func_after on
whitespace-stripped lines; blank lines never count; the LineVD convention). They are carried
through the cleaning of make_bigvul_json.py (empty lines, comments, leading whitespace) with a
character map, so their token positions are those of the text the model saw. A vulnerable
function is all_visible / some_visible / none_visible by whether its changed lines start
within the first W tokens. A fix that only adds lines (a missing check) has no changed line;
it is placed at the line in front of which the lines are inserted (the statement the check
guards) and classed insert:all_visible / insert:none_visible / ..., kept apart from the
changed-line classes. A function whose position cannot be established is counted as unknown
with the reason (no CSV row, CSV text differs from the JSONL, unchanged in the CSV, changed
lines that are blank or comments), never dropped silently.

Reading the output. Within the truncated functions, the AUC of the vulnerable ones whose flaw
is hidden (none_visible) against the non-vulnerable truncated ones is compared with the same
AUC for visible flaws (all_visible). The 95% interval of the difference resamples test items
(the runs are fixed). The same contrast with the add-only fixes added at their insertion point
is reported second. A hidden flaw needs a long function, so the hidden group can be longer
than the visible one; a model that scores by length then gets a higher AUC on the hidden
group, which pulls the comparison towards "no difference". The length-only AUC of both groups
is printed under each table to show how large that pull can be. Similar AUCs mean the model scores these functions without seeing the
flaw (length, project or style cues), so a model that reads past the window has little to gain
on this data; a clearly lower AUC for hidden flaws is the room the proposal needs. Length alone
predicts the label to some degree; its AUC is shown for every stratum.
"""
import argparse
import difflib
import glob
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODEBERT = "microsoft/codebert-base"
FLAW_CLASSES = ("all_visible", "some_visible", "none_visible")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CR = _load("compare_runs", "scripts/compare_runs.py")

# make_bigvul_json.cleaned_code is these three substitutions, in this order. Every function is
# checked against the original, so a drift between the two shows up as clean_tracker_mismatch.
_EMPTY_LINE = re.compile(r'^$\n', re.MULTILINE)
_COMMENT_OR_LITERAL = re.compile(r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"', re.DOTALL | re.MULTILINE)
_LEADING_SPACE = re.compile(r'^\s+', re.MULTILINE)


def _sub_tracked(pattern, repl, text, pos):
    """re.sub(pattern, repl, text) that carries pos (the source index of every character of text)
    along: kept characters keep theirs, a replacement takes the index of the start of its match."""
    out, out_pos, last = [], [], 0
    for m in pattern.finditer(text):
        out.append(text[last:m.start()])
        out_pos.extend(pos[last:m.start()])
        r = repl(m)
        out.append(r)
        out_pos.extend(pos[m.start():m.end()] if r == m.group(0) else [pos[m.start()]] * len(r))
        last = m.end()
    out.append(text[last:])
    out_pos.extend(pos[last:])
    return "".join(out), out_pos


def clean_tracked(code):
    """(cleaned, pos): make_bigvul_json.cleaned_code(code), and for every character of it the
    index in code it comes from."""
    text, pos = code, list(range(len(code)))
    text, pos = _sub_tracked(_EMPTY_LINE, lambda m: "", text, pos)
    text, pos = _sub_tracked(_COMMENT_OR_LITERAL, lambda m: " " if m.group(0).startswith("/") else m.group(0), text, pos)
    text, pos = _sub_tracked(_LEADING_SPACE, lambda m: "", text, pos)
    return text, pos


def changed_lines(before, after):
    """Indices of the lines of `before` that the fix removes or rewrites (difflib on
    whitespace-stripped lines, so re-indentation is not a change; blank lines never count)."""
    a, b = before.split("\n"), after.split("\n")
    sm = difflib.SequenceMatcher(None, [s.strip() for s in a], [s.strip() for s in b], autojunk=False)
    return [i for tag, i1, i2, _, _ in sm.get_opcodes() if tag in ("replace", "delete")
            for i in range(i1, i2) if a[i].strip()]


def insertion_lines(before, after):
    """For a fix that only adds lines: the line of `before` in front of which each block is
    inserted (the next non-blank line: the statement a missing check guards), or the last
    non-blank line when the block is appended at the end."""
    a, b = before.split("\n"), after.split("\n")
    sm = difflib.SequenceMatcher(None, [s.strip() for s in a], [s.strip() for s in b], autojunk=False)
    code = [i for i, s in enumerate(a) if s.strip()]
    out = set()
    for tag, i1, _, _, _ in sm.get_opcodes():
        if tag == "insert" and code:
            out.add(next((i for i in code if i >= i1), code[-1]))
    return sorted(out)


def line_spans(text):
    """(start, end) character span of every line of text, newline excluded."""
    spans, start = [], 0
    for line in text.split("\n"):
        spans.append((start, start + len(line)))
        start += len(line) + 1
    return spans


def first_tokens(pos, spans, lines, offsets, n_chars):
    """For each listed source line with a character left after cleaning, the index of the first
    token that covers one of its characters (offsets: the tokenizer's character spans)."""
    char_tok = np.full(n_chars, -1, dtype=np.int64)
    for t, (s, e) in enumerate(offsets):
        char_tok[s:e] = t
    pos = np.asarray(pos, dtype=np.int64)
    out = []
    for li in lines:
        s, e = spans[li]
        toks = char_tok[(pos >= s) & (pos < e)]
        toks = toks[toks >= 0]
        if len(toks):
            out.append(int(toks.min()))
    return out


def flaw_class(firsts, window):
    visible = [t < window for t in firsts]
    return "all_visible" if all(visible) else ("none_visible" if not any(visible) else "some_visible")


def locate(row, text, tok, window, cleaned_code):
    """(class, first token of the changed lines) of one vulnerable function: a FLAW_CLASSES
    member, 'insert:<member>' for an add-only fix, or 'unknown:<reason>' with -1."""
    if row is None:
        return "unknown:no_csv_row", -1
    before, after = row
    if not isinstance(before, str) or not isinstance(after, str):
        return "unknown:missing_func_text", -1
    clean, pos = clean_tracked(before)
    if clean != cleaned_code(before):
        return "unknown:clean_tracker_mismatch", -1
    if clean != text:
        return "unknown:csv_differs_from_jsonl", -1
    lines, prefix = changed_lines(before, after), ""
    if not lines:
        if [s.strip() for s in before.split("\n")] == [s.strip() for s in after.split("\n")]:
            return "unknown:unchanged_in_csv", -1
        lines, prefix = insertion_lines(before, after), "insert:"
    enc = tok(clean, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    firsts = first_tokens(pos, line_spans(before), lines, enc["offset_mapping"], len(clean))
    if not firsts:
        return "unknown:changed_lines_not_code", -1
    return prefix + flaw_class(firsts, window), min(firsts)


def read_bigvul(csv_path, want):
    """{idx: (func_before, func_after)} of the Big-Vul rows whose idx is in want; the idx is
    make_bigvul_json's: the 'id' column, else 'Unnamed: 0', else the row number."""
    want = {int(i) for i in want}
    header = list(pd.read_csv(csv_path, nrows=0).columns)
    missing = [c for c in ("func_before", "func_after") if c not in header]
    if missing:
        raise SystemExit(f"{csv_path}: no column {missing}")
    idcol = "id" if "id" in header else ("Unnamed: 0" if "Unnamed: 0" in header else None)
    rows = {}
    for chunk in pd.read_csv(csv_path, usecols=[c for c in (idcol, "func_before", "func_after") if c], chunksize=20000):
        ids = chunk[idcol].astype(np.int64) if idcol else pd.Series(chunk.index, index=chunk.index)
        sel = ids.isin(want)
        for i, fb, fa in zip(ids[sel], chunk.loc[sel, "func_before"], chunk.loc[sel, "func_after"]):
            rows[int(i)] = (fb, fa)
    return rows


def load_tokenizer(name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    tok.model_max_length = int(1e9)          # lengths are measured untruncated: silence the length warning
    return tok


def token_lengths(tok, texts, batch=512):
    n = []
    for i in range(0, len(texts), batch):
        n.extend(len(x) for x in tok(list(texts[i:i + batch]), add_special_tokens=False, truncation=False)["input_ids"])
    return np.asarray(n, dtype=np.int64)


def auc_pos_neg(sp, sn):
    """P(score of a positive > score of a negative), ties counted half: the ROC-AUC of the two
    sets. None when either is empty."""
    if len(sp) == 0 or len(sn) == 0:
        return None
    sn = np.sort(np.asarray(sn, dtype=np.float64))
    sp = np.asarray(sp, dtype=np.float64)
    lt = np.searchsorted(sn, sp, side="left")
    le = np.searchsorted(sn, sp, side="right")
    return float((lt + 0.5 * (le - lt)).sum() / (len(sp) * len(sn)))


def subset_scores(run, mask):
    y = run["labels"][mask].astype(bool)
    s = run["probs"][mask]
    p = s >= run["thr"]
    tp, fp, fn = int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum())
    return {"auc": auc_pos_neg(s[y], s[~y]),
            "recall": tp / (tp + fn) if tp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "f1": CR.f1_from_counts(tp, fp, fn) if tp + fp + fn else None}


def mean_sd(vals):
    """[mean, sd, n] over the runs with a value; None when no run has one."""
    v = [x for x in vals if x is not None]
    if not v:
        return None
    return [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v)]


def contrast(group, vis, hid, neg, n_boot, rng):
    c = {"n_visible": int(len(vis)), "n_hidden": int(len(hid)), "n_negative": int(len(neg)),
         "auc_visible": mean_sd([auc_pos_neg(r["probs"][vis], r["probs"][neg]) for r in group]),
         "auc_hidden": mean_sd([auc_pos_neg(r["probs"][hid], r["probs"][neg]) for r in group]),
         "recall_visible": mean_sd([float((r["probs"][vis] >= r["thr"]).mean()) if len(vis) else None for r in group]),
         "recall_hidden": mean_sd([float((r["probs"][hid] >= r["thr"]).mean()) if len(hid) else None for r in group])}
    if len(vis) and len(hid) and len(neg):
        c["visible_minus_hidden"] = visible_minus_hidden(group, vis, hid, neg, n_boot, rng)
    return c


def visible_minus_hidden(group, vis, hid, neg, n_boot, rng):
    """Seed-averaged AUC(visible vs neg) - AUC(hidden vs neg) and its 95% interval over item
    resamples (each of the three sets resampled with replacement; the runs are fixed)."""
    def stat(v, h, n):
        return float(np.mean([auc_pos_neg(r["probs"][v], r["probs"][n]) - auc_pos_neg(r["probs"][h], r["probs"][n])
                              for r in group]))
    point = stat(vis, hid, neg)
    boots = [stat(rng.choice(vis, len(vis)), rng.choice(hid, len(hid)), rng.choice(neg, len(neg))) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": point, "lo": float(lo), "hi": float(hi)}


def run_dirs(patterns, preds):
    """Finished runs only: a queue in progress leaves directories without predictions."""
    fname = "test_preds.npz" if preds == "best" else "test_preds_final.npz"
    dirs = sorted({d for p in patterns for d in glob.glob(p)
                   if (Path(d) / fname).exists() and (Path(d) / "metrics.json").exists()})
    if not dirs:
        raise SystemExit(f"no finished run matches {patterns}")
    return dirs


def _fmt(t, digits=3):
    return "-" if t is None else f"{t[0]:.{digits}f} +/- {t[1]:.{digits}f}"


def main(argv=None, tokenizer=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", nargs="+", action="append", required=True, metavar="NAME_THEN_GLOBS",
                    help="a group name and one or more run globs; repeat for every group")
    ap.add_argument("--test_jsonl", help="default: the test_jsonl recorded in the first run's args.json")
    ap.add_argument("--tokenizer", help=f"default: the first run's text_model ({CODEBERT} when not recorded)")
    ap.add_argument("--max_length", type=int, help="default: the first run's max_length (512 when not recorded)")
    ap.add_argument("--bigvul_csv", help="the Big-Vul CSV the test JSONL was made from (MSR_data_cleaned.csv): "
                                         "adds where the changed lines sit relative to the window")
    ap.add_argument("--preds", choices=("best", "final"), default="best")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="length_strata", help="directory for summary.md, summary.json and items.csv")
    args = ap.parse_args(argv)

    names, groups, all_dirs = [], [], []
    for spec in args.group:
        if len(spec) < 2:
            ap.error("--group takes a name and at least one glob")
        dirs = run_dirs(spec[1:], args.preds)
        runs = CR.load_group(dirs, "tuned", args.preds)
        if any(r["probs"].ndim != 1 for r in runs):
            raise SystemExit(f"{spec[0]}: multi-class runs; this check is for binary runs")
        names.append(spec[0])
        groups.append(runs)
        all_dirs += dirs
    common = CR.align(groups)
    labels = groups[0][0]["labels"].astype(int)

    metas = [json.loads((Path(d) / "args.json").read_text()) for d in all_dirs if (Path(d) / "args.json").exists()]
    meta = metas[0] if metas else {}
    test_jsonl = args.test_jsonl or meta.get("args", {}).get("test_jsonl")
    if not test_jsonl:
        raise SystemExit("no test_jsonl recorded in args.json: pass --test_jsonl")
    tok_name = args.tokenizer or meta.get("text_model") or CODEBERT
    max_length = int(args.max_length or meta.get("args", {}).get("max_length") or 512)
    windows = {(m.get("text_model") or CODEBERT, m.get("args", {}).get("max_length", 512)) for m in metas
               if m.get("args", {}).get("model_type") != "graph_only"}
    if len(windows) > 1:
        print(f"[WARN] the text arms differ in tokenizer / max_length {sorted(windows)}; "
              f"the window is measured with {tok_name} at {max_length}")

    df = pd.read_json(test_jsonl, lines=True)
    if df["idx"].duplicated().any():
        raise SystemExit(f"{test_jsonl}: duplicate idx")
    df = df.set_index("idx")
    absent = np.setdiff1d(common, df.index.to_numpy())
    if len(absent):
        raise SystemExit(f"{len(absent)} test idx of the runs are not in {test_jsonl}: not the data these runs used")
    df = df.loc[common]
    if not np.array_equal(df["target"].to_numpy().astype(int), labels):
        raise SystemExit(f"{test_jsonl}: its labels differ from the runs' test labels")
    texts = df["func"].astype(str).tolist()

    tok = tokenizer if tokenizer is not None else load_tokenizer(tok_name)
    window = max_length - int(tok.num_special_tokens_to_add(pair=False))
    n_tok = token_lengths(tok, texts)
    trunc = n_tok > window
    h = window // 2
    strata = [("all", np.ones(len(common), dtype=bool)),
              (f"<= {h}", n_tok <= h), (f"{h + 1}-{window}", (n_tok > h) & (n_tok <= window)),
              (f"{window + 1}-{2 * window}", (n_tok > window) & (n_tok <= 2 * window)),
              (f"> {2 * window}", n_tok > 2 * window), (f"truncated (> {window})", trunc)]

    out = {"groups": {n: [r["name"] for r in g] for n, g in zip(names, groups)}, "preds": args.preds,
           "test_jsonl": str(test_jsonl), "tokenizer": tok_name, "max_length": max_length, "window": window,
           "n_items": int(len(common)), "n_positive": int(labels.sum()), "strata": [], "flaw": None}
    for sname, m in strata:
        y = labels[m].astype(bool)
        row = {"stratum": sname, "n": int(m.sum()), "positives": int(y.sum()),
               "length_auc": auc_pos_neg(n_tok[m][y], n_tok[m][~y]), "groups": {}}
        for gname, g in zip(names, groups):
            sc = [subset_scores(r, m) for r in g]
            row["groups"][gname] = {k: mean_sd([s[k] for s in sc]) for k in ("auc", "recall", "precision", "f1")}
        out["strata"].append(row)

    flaw = np.full(len(common), "-", dtype=object)
    first_tok = np.full(len(common), -1, dtype=np.int64)
    if args.bigvul_csv:
        mb = _load("make_bigvul_json", "data/scripts/make_bigvul_json.py")
        vul = np.nonzero(labels == 1)[0]
        rows = read_bigvul(args.bigvul_csv, common[vul])
        for k in vul:
            flaw[k], first_tok[k] = locate(rows.get(int(common[k])), texts[k], tok, window, mb.cleaned_code)
        cls_all = pd.Series(flaw[vul]).value_counts().to_dict()
        cls_trunc = pd.Series(flaw[vul][trunc[vul]]).value_counts().to_dict()
        known = trunc & np.isin(flaw, list(FLAW_CLASSES))
        ft = first_tok[known]
        out["flaw"] = {"all_vulnerable": {k: int(v) for k, v in cls_all.items()},
                       "truncated_vulnerable": {k: int(v) for k, v in cls_trunc.items()},
                       "first_changed_token_of_truncated": {
                           f"[0, {window})": int((ft < window).sum()),
                           f"[{window}, {2 * window})": int(((ft >= window) & (ft < 2 * window)).sum()),
                           f">= {2 * window}": int((ft >= 2 * window).sum())},
                       "contrast": {}, "contrast_with_insertions": {}}
        neg = np.nonzero(trunc & (labels == 0))[0]
        rng = np.random.default_rng(args.seed)
        for key, extra in (("contrast", ()), ("contrast_with_insertions", ("insert:",))):
            vis = np.nonzero(trunc & np.isin(flaw, [p + "all_visible" for p in ("",) + extra]))[0]
            hid = np.nonzero(trunc & np.isin(flaw, [p + "none_visible" for p in ("",) + extra]))[0]
            out["flaw"][key + "_length_auc"] = {"visible": auc_pos_neg(n_tok[vis], n_tok[neg]),
                                                "hidden": auc_pos_neg(n_tok[hid], n_tok[neg])}
            for gname, g in zip(names, groups):
                out["flaw"][key][gname] = contrast(g, vis, hid, neg, args.n_boot, rng)

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    items = pd.DataFrame({"idx": common, "label": labels, "n_tokens": n_tok, "truncated": trunc,
                          "flaw": flaw, "first_changed_token": first_tok})
    for gname, g in zip(names, groups):
        items[f"p_{gname}"] = np.mean([r["probs"] for r in g], axis=0)
    items.to_csv(dest / "items.csv", index=False)
    (dest / "summary.json").write_text(json.dumps(out, indent=1))
    md = render(out, names)
    (dest / "summary.md").write_text(md)
    print(md)
    return out


def render(out, names):
    w = out["window"]
    md = ["# Length strata: does the text encoder miss what lies beyond its window?", "",
          f"Tokenizer {out['tokenizer']}, max_length {out['max_length']}: the encoder sees W = {w} content tokens. "
          f"Test items {out['n_items']} ({out['n_positive']} vulnerable), predictions '{out['preds']}' "
          "(each run at its validation-tuned threshold; 'final' at 0.5). Mean +/- sd over the runs of a group.", ""]
    for n in names:
        md.append(f"- {n}: {len(out['groups'][n])} run(s): {', '.join(out['groups'][n])}")
    md += ["", "## Test metrics by length (content tokens)", "",
           "| stratum | n | vulnerable | rate | length-only AUC | " + " | ".join(f"{n} AUC | {n} recall | {n} F1" for n in names) + " |",
           "|---|---|---|---|---|" + "---|---|---|" * len(names)]
    for s in out["strata"]:
        rate = s["positives"] / s["n"] if s["n"] else 0.0
        la = "-" if s["length_auc"] is None else f"{s['length_auc']:.3f}"
        cells = [f"{_fmt(s['groups'][n]['auc'])} | {_fmt(s['groups'][n]['recall'])} | {_fmt(s['groups'][n]['f1'])}" for n in names]
        md.append(f"| {s['stratum']} | {s['n']} | {s['positives']} | {rate:.3f} | {la} | " + " | ".join(cells) + " |")
    f = out["flaw"]
    if f is None:
        md += ["", "Where the vulnerable lines sit: pass --bigvul_csv (the Big-Vul CSV the test JSONL was made from)."]
        return "\n".join(md) + "\n"
    keys = list(FLAW_CLASSES) + ["insert:" + k for k in FLAW_CLASSES] \
        + sorted(k for k in f["all_vulnerable"] if k.startswith("unknown:"))
    md += ["", "## Where the changed lines sit (vulnerable functions)", "",
           "| class | all vulnerable | truncated vulnerable |", "|---|---|---|"]
    for k in keys:
        md.append(f"| {k} | {f['all_vulnerable'].get(k, 0)} | {f['truncated_vulnerable'].get(k, 0)} |")
    md += ["", "First changed token of the truncated vulnerable functions (changed-line classes): "
           + ", ".join(f"{k}: {v}" for k, v in f["first_changed_token_of_truncated"].items()), "",
           "## Hidden vs visible flaws within the truncated functions", "",
           "Positives: truncated vulnerable functions whose changed lines all start within W (visible) or none does "
           "(hidden). Negatives: the truncated non-vulnerable functions. 95% interval: resamples of the items.", ""]
    for key, title in (("contrast", "Changed lines (primary)"),
                       ("contrast_with_insertions", "Changed lines, plus add-only fixes at their insertion point")):
        md += [f"{title}:", "", "| group | AUC visible | AUC hidden | visible - hidden [95% CI] | recall visible | recall hidden |",
               "|---|---|---|---|---|---|"]
        for n in names:
            c = f[key][n]
            d = c.get("visible_minus_hidden")
            ci = "-" if d is None else f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"
            md.append(f"| {n} | {_fmt(c['auc_visible'])} (n {c['n_visible']}) | {_fmt(c['auc_hidden'])} (n {c['n_hidden']}) | "
                      f"{ci} | {_fmt(c['recall_visible'])} | {_fmt(c['recall_hidden'])} |")
        la = f[key + "_length_auc"]
        md += ["", "Length-only AUC against the same negatives: visible "
               + ("-" if la["visible"] is None else f"{la['visible']:.3f}") + ", hidden "
               + ("-" if la["hidden"] is None else f"{la['hidden']:.3f}"), ""]
    md += [f"Negatives: {f['contrast'][names[0]]['n_negative']} truncated non-vulnerable functions.", "",
           "Reading: similar AUCs for visible and hidden flaws mean the model scores these functions without seeing "
           "the flaw, so reading past the window has little to gain on this data; a clearly lower AUC for hidden "
           "flaws is the room for a model that does. Compare with the length-only AUC of the truncated stratum."]
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    main()
