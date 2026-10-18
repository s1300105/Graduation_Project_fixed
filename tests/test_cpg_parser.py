"""Parser regression tests on a real-pipeline sample (data/check_data/cpg_head_check.json).

The sample holds 5 rows written by the original CPG pipeline: three real
functions and two `<global>` pseudo-methods (audit M19 / M20).
"""
import json
import os
import collections
from pathlib import Path

import pytest

os.environ["USE_FAST_NODES_EMBEDDING"] = "1"

from utils.functions.cpg import (parse_to_functions, count_edges_by_relation,
                                 is_global_function, NODE_LABEL_TO_ID)
from utils.process.embeddings import nodes_to_input, build_edges_all

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "check_data" / "cpg_head_check.json"


def _rows():
    return [json.loads(l) for l in SAMPLE.read_text(encoding="utf-8").splitlines() if l.strip()]


def _joern_truth(func_json):
    """Unique (out, in, type) triples as emitted by Joern (edge's own type field)."""
    truth = set()
    for kind in ("AST", "CFG", "PDG"):
        for nd in func_json.get(kind) or []:
            for e in nd.get("edges") or []:
                truth.add((e["out"], e["in"], e["type"]))
    return truth


def test_sample_has_global_duplicates():
    rows = _rows()
    names = [r["cpg"]["functions"][0]["function"] for r in rows]
    assert sum(is_global_function(f) for f in (r["cpg"]["functions"][0] for r in rows)) == 2
    assert names[0] == "am_get_mime_header" and names[1].endswith("<global>")


def test_global_functions_are_skipped():
    for r in _rows():
        f = r["cpg"]["functions"][0]
        funcs = parse_to_functions(r["cpg"], max_nodes=500)
        if is_global_function(f):
            assert funcs == []
        else:
            assert len(funcs) == 1


def test_edge_relations_match_joern_exactly():
    """Every produced (src, dst, relation) must be a Joern edge of that type and vice versa."""
    for r in _rows():
        f = r["cpg"]["functions"][0]
        if is_global_function(f):
            continue
        truth = _joern_truth(f)
        od = parse_to_functions(r["cpg"], max_nodes=500)[0]
        id_by_order = {n.order: nid for nid, n in od.items()}
        ei, et = build_edges_all(od)
        produced = {(id_by_order[s], id_by_order[d], ("AST", "CFG", "PDG")[t])
                    for s, d, t in zip(ei[0].tolist(), ei[1].tolist(), et.tolist())}
        # nodes dropped by the filter (UNKNOWN / COMMENT) may remove a few truth edges
        kept_ids = set(od.keys())
        truth_kept = {(u, v, t) for (u, v, t) in truth if u in kept_ids and v in kept_ids and u != v}
        assert produced == truth_kept, (f["function"], len(produced), len(truth_kept))
        by_rel = count_edges_by_relation(od)
        assert by_rel["AST"] != by_rel["CFG"] or by_rel["AST"] == 0  # AST and CFG are no longer identical sets


def test_first_function_edge_counts():
    r = _rows()[0]
    od = parse_to_functions(r["cpg"], max_nodes=500)[0]
    assert len(od) == 75
    assert count_edges_by_relation(od) == {"AST": 67, "CFG": 59, "PDG": 37}


def test_node_labels_and_data_shapes():
    r = _rows()[0]
    od = parse_to_functions(r["cpg"], max_nodes=500)[0]
    labels = collections.Counter(n.label for n in od.values())
    assert labels["IDENTIFIER"] == 26 and labels["CALL"] == 23
    assert all(n.label_id != NODE_LABEL_TO_ID["<UNK>"] for n in od.values())
    g = nodes_to_input(od, r["target"], 500, mode="fast")
    assert g.x.shape == (75, 768)
    assert g.node_label.shape == (75,)
    assert g.edge_index.shape[1] == 67 + 59 + 37
    assert set(g.edge_type.tolist()) == {0, 1, 2}
    assert g.num_nodes == 75 and int(g.y) == 0


def test_reverse_edges_double_the_edges_with_offset_relations():
    r = _rows()[0]
    od = parse_to_functions(r["cpg"], max_nodes=500)[0]
    g = nodes_to_input(od, 0, 500, mode="fast", reverse_edges=True)
    assert g.edge_index.shape[1] == 2 * (67 + 59 + 37)
    assert set(g.edge_type.tolist()) == {0, 1, 2, 3, 4, 5}


def test_flat_layout_is_accepted():
    """A function given as nodes + top-level typed edges parses identically."""
    r = _rows()[0]
    f = r["cpg"]["functions"][0]
    nodes = {}
    edges = []
    for kind in ("AST", "CFG", "PDG"):
        for nd in f[kind]:
            nodes.setdefault(nd["id"], {k: v for k, v in nd.items() if k != "edges"})
            edges.extend(nd.get("edges") or [])
    flat = {"functions": [{"function": f["function"], "id": f["id"], "nodes": list(nodes.values()), "edges": edges}]}
    od = parse_to_functions(flat, max_nodes=500)[0]
    assert len(od) == 75 and count_edges_by_relation(od) == {"AST": 67, "CFG": 59, "PDG": 37}


def test_max_nodes_cut_keeps_positioned_nodes_first():
    r = _rows()[0]
    od = parse_to_functions(r["cpg"], max_nodes=20)[0]
    assert len(od) == 20
    assert all(n.get_line_number() is not None for n in od.values())
