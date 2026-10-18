"""Regression test on a real export of joern/graph-for-funcs.sc (Joern v4.0.400) over
tests/fixtures/joern_inputs written through utils.data.datamanager.to_files
(6 C files as .c, 6 C++ files as .cpp)."""
import json
import os
from pathlib import Path

os.environ["USE_FAST_NODES_EMBEDDING"] = "1"

from utils.process import cpg_generator as cp
from utils.functions.cpg import parse_to_functions, count_edges_by_relation, NODE_LABEL_TO_ID
from utils.process.embeddings import nodes_to_input

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "tests" / "fixtures" / "joern_v4_export.json"


def test_export_layout_and_global_handling():
    raw = json.loads(EXPORT.read_text(encoding="utf-8"))
    fns = raw["functions"]
    assert len(fns) == 104
    assert sum(cp.is_global_method(f) for f in fns) == 13          # one <global> per file (+ includes)
    stats = {}
    kept = cp.json_process(str(EXPORT.parent), EXPORT.name, stats)
    assert stats["skipped_global"] == 13 and stats["functions_kept"] == 10
    assert sorted(i for i, _ in kept) == [1035, 10019, 10048, 10052, 10115, 10125, 10141, 10204, 10280, 10289]
    # 101.cpp (bare constructor fragment) and 10131.c (macro in the signature) yield no parsable method


def test_cpp_member_functions_are_real_graphs():
    kept = dict(cp.json_process(str(EXPORT.parent), EXPORT.name))
    cpp = parse_to_functions(kept[10048], max_nodes=500)[0]          # CClient::ClearServerDependentCaps
    assert len(cpp) == 80
    rel = count_edges_by_relation(cpp)
    assert rel["AST"] > 50 and rel["CFG"] > 30 and rel["PDG"] > 50
    assert kept[10048]["functions"][0]["function"].startswith("CClient.ClearServerDependentCaps")


def test_all_kept_functions_build_valid_data():
    kept = dict(cp.json_process(str(EXPORT.parent), EXPORT.name))
    for idx, entry in kept.items():
        od = parse_to_functions(entry, max_nodes=500)[0]
        assert all(n.label_id != NODE_LABEL_TO_ID["<UNK>"] for n in od.values()), idx
        rel = count_edges_by_relation(od)
        assert rel["AST"] == len(od) - 1, (idx, rel)               # AST of a method is a tree
        assert rel["CFG"] > 0 and rel["PDG"] > 0
        g = nodes_to_input(od, 1, 500, mode="fast")
        assert g.x.shape == (len(od), 768) and g.edge_index.shape[1] == sum(rel.values())
        assert set(g.edge_type.tolist()) <= {0, 1, 2}
