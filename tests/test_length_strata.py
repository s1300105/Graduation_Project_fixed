"""scripts/length_strata.py: the tracked cleaning equals make_bigvul_json's, the changed lines and
their token positions, and the strata on fake runs with a whitespace tokenizer (no network)."""
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LS = _load("length_strata", "scripts/length_strata.py")
MB = _load("make_bigvul_json", "data/scripts/make_bigvul_json.py")


class WordTokenizer:
    """Whitespace-separated tokens with character offsets and 2 special tokens, like RoBERTa's."""

    def num_special_tokens_to_add(self, pair=False):
        return 2

    @staticmethod
    def _one(text):
        spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        return list(range(len(spans))), spans

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, truncation=False):
        assert not add_special_tokens and not truncation
        encs = [self._one(text)] if isinstance(text, str) else [self._one(t) for t in text]
        out = {"input_ids": [e[0] for e in encs]}
        if return_offsets_mapping:
            out["offset_mapping"] = [e[1] for e in encs]
        return {k: v[0] for k, v in out.items()} if isinstance(text, str) else out


TRICKY = [
    "int f(int a) {\n\n    // comment\n    int b = a; /* inline */ return b;\n}\n",
    "void g() {\n  /* multi\n     line */\n  char *s = \"// not a comment\";\n\n\n  x = '/';\n}",
    "  \n\t\nint h;\r\n  // trailing",
    "",
]


@pytest.mark.parametrize("code", TRICKY)
def test_clean_tracked_equals_make_bigvul_json(code):
    clean, pos = LS.clean_tracked(code)
    assert clean == MB.cleaned_code(code) and len(pos) == len(clean)
    # every kept character comes from the same source character; a comment becomes one space
    assert all(c == code[p] or (c == " " and code[p] == "/") for c, p in zip(clean, pos))


def test_changed_lines_ignore_reindentation_and_blank_lines():
    before = "a;\n  b;\n\nc;\nd;"
    after = "a;\nb;\nc2;\nd;"
    assert LS.changed_lines(before, after) == [3]        # "c;" rewritten; "  b;" re-indented; the blank line dropped


def test_flaw_classes_against_the_window():
    tok = WordTokenizer()
    code = "\n".join(f"s{i} x{i};" for i in range(8))       # 8 lines of 2 tokens
    clean, pos = LS.clean_tracked(code)
    offsets = tok(clean, return_offsets_mapping=True)["offset_mapping"]
    first = lambda lines: LS.first_tokens(pos, LS.line_spans(code), lines, offsets, len(clean))  # noqa: E731
    assert first([1, 5]) == [2, 10]
    window = 6                                              # lines 0-2 start within the window
    assert LS.flaw_class(first([1, 2]), window) == "all_visible"
    assert LS.flaw_class(first([2, 5]), window) == "some_visible"
    assert LS.flaw_class(first([3, 7]), window) == "none_visible"


def test_insertion_point_is_the_guarded_statement():
    assert LS.insertion_lines("a;\n\nb;\nc;", "a;\nif (x) return;\n\nb;\nc;") == [2]   # skips the blank line 1
    assert LS.insertion_lines("a;\nb;", "a;\nb;\nfree(p);") == [1]                        # appended: the last line


def test_comment_only_change_has_no_code_position():
    tok = WordTokenizer()
    before, after = "a ;\n// old note\nb ;", "a ;\n// new note\nb ;"
    assert LS.locate((before, after), MB.cleaned_code(before), tok, 6, MB.cleaned_code) == ("unknown:changed_lines_not_code", -1)


# idx: (func, target, func_after or None when the row is absent from the CSV); max_length 8 -> W = 6
FUNCS = {
    0: ("int a ;", 0, "int a ;"),
    1: ("int b ;\nbad ( ) ;", 1, "int b ;\ngood ( ) ;"),                       # 7 tokens, flaw at token 3: visible
    2: ("x x x\ny y y\nz z z\nbad2 ( ) ;", 1, "x x x\ny y y\nz z z\ngood2 ( ) ;"),  # 13 tokens, flaw at 9: hidden
    3: ("p p p\nq q q\nr r r\ns s s", 0, "p p p\nq q q\nr r r\ns s s"),       # 12 tokens, truncated negative
    4: ("ok ;", 0, "ok ;"),
    5: ("bad3 ;", 1, "good3 ;"),
    6: ("m ;", 1, None),                                                        # no CSV row
    7: ("k ;", 1, "k ;\nextra ;"),                                              # add-only, appended: at line 0
    8: ("u u u\nv v v\nw w w\ncall ( ) ;", 1,                                   # add-only, inserted before token 9
        "u u u\nv v v\nw w w\nif ( x ) return ;\ncall ( ) ;"),
}
PROBS = {"text": [[0.1, 0.8, 0.2, 0.3, 0.2, 0.9, 0.3, 0.3, 0.6],
                  [0.2, 0.7, 0.4, 0.1, 0.1, 0.6, 0.3, 0.3, 0.2]],
         "graph": [[0.5, 0.5, 0.9, 0.2, 0.1, 0.4, 0.3, 0.3, 0.3]]}


def test_strata_on_fake_runs(tmp_path):
    idx = np.array(sorted(FUNCS))
    labels = np.array([FUNCS[i][1] for i in idx])
    test_jsonl = tmp_path / "toy_test.jsonl"
    test_jsonl.write_text("".join(json.dumps({"idx": int(i), "func": FUNCS[i][0], "target": FUNCS[i][1]}) + "\n" for i in idx))
    pd.DataFrame([{"Unnamed: 0": i, "func_before": f, "func_after": a, "vul": t}
                  for i, (f, t, a) in FUNCS.items() if a is not None]).to_csv(tmp_path / "bigvul.csv", index=False)
    for g, runs in PROBS.items():
        for seed, p in enumerate(runs):
            d = tmp_path / "runs" / f"{g}_seed{seed}"
            d.mkdir(parents=True)
            np.savez(d / "test_preds.npz", probs=np.array(p), labels=labels, idx=idx)
            (d / "metrics.json").write_text(json.dumps({"test_tuned": {"threshold": 0.5}}))
            (d / "args.json").write_text(json.dumps({"text_model": "fake", "args": {
                "test_jsonl": str(test_jsonl), "max_length": 8, "model_type": "codebert" if g == "text" else "graph_only"}}))
    out = LS.main(["--group", "text", str(tmp_path / "runs" / "text_seed*"),
                   "--group", "graph", str(tmp_path / "runs" / "graph_seed*"),
                   "--bigvul_csv", str(tmp_path / "bigvul.csv"), "--n_boot", "50", "--out", str(tmp_path / "out")],
                  tokenizer=WordTokenizer())
    assert out["window"] == 6 and out["n_items"] == 9 and out["n_positive"] == 6
    strata = {s["stratum"]: s for s in out["strata"]}
    assert [strata[k]["n"] for k in ("<= 3", "4-6", "7-12", "> 12", "truncated (> 6)")] == [5, 0, 2, 2, 4]
    assert strata["all"]["groups"]["text"]["recall"][0] == pytest.approx((3 / 6 + 2 / 6) / 2)
    assert out["flaw"]["all_vulnerable"] == {"all_visible": 2, "none_visible": 1, "unknown:no_csv_row": 1,
                                             "insert:all_visible": 1, "insert:none_visible": 1}
    assert out["flaw"]["truncated_vulnerable"] == {"all_visible": 1, "none_visible": 1, "insert:none_visible": 1}
    t = out["flaw"]["contrast"]["text"]
    assert (t["n_visible"], t["n_hidden"], t["n_negative"]) == (1, 1, 1)
    assert t["auc_visible"][0] == 1.0 and t["auc_hidden"][0] == 0.5            # run 0 ranks idx 2 below idx 3
    assert t["visible_minus_hidden"]["diff"] == pytest.approx(0.5)
    assert t["recall_visible"][0] == 1.0 and t["recall_hidden"][0] == 0.0
    assert out["flaw"]["contrast"]["graph"]["auc_hidden"][0] == 1.0
    w = out["flaw"]["contrast_with_insertions"]["text"]                       # idx 8 joins the hidden flaws
    assert w["n_hidden"] == 2 and w["auc_hidden"][0] == pytest.approx(0.75)
    # tokens: visible idx 1 = 7, hidden idx 2 = 13 and idx 8 = 13, negative idx 3 = 12
    assert out["flaw"]["contrast_length_auc"] == {"visible": 0.0, "hidden": 1.0}
    items = pd.read_csv(tmp_path / "out" / "items.csv")
    assert items.set_index("idx").loc[2, "first_changed_token"] == 9 == items.set_index("idx").loc[8, "first_changed_token"]
    assert "## Hidden vs visible flaws" in (tmp_path / "out" / "summary.md").read_text()
