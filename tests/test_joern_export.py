"""Regression tests on real exports of joern/graph-for-funcs.sc (Joern v4.0.400).

tests/fixtures/joern_v4_export.json: tests/fixtures/joern_inputs written through
utils.data.datamanager.to_files (5 files written as .c and 7 as .cpp; 101.cpp is a bare
constructor-initialiser fragment that the C++ hint regex classifies as C++). One BigVul
function per file: graph_unit="first_method".

tests/fixtures/joern_v4_export_multi.json: tests/fixtures/joern_inputs_multi (see its
README.txt), five whole programs with several functions, global variables, a K&R main,
a template and a function-like macro: graph_unit="whole_file".
"""
import json
import os
import re
from pathlib import Path

import pytest

os.environ["USE_FAST_NODES_EMBEDDING"] = "1"

from utils.process import cpg_generator as cp
from utils.functions.cpg import parse_to_functions, count_edges_by_relation, NODE_LABEL_TO_ID
from utils.process.embeddings import nodes_to_input

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "tests" / "fixtures" / "joern_v4_export.json"
MULTI = ROOT / "tests" / "fixtures" / "joern_v4_export_multi.json"
MULTI_INPUTS = ROOT / "tests" / "fixtures" / "joern_inputs_multi"


def _raw(path):
    return json.loads(re.sub(r"io\.shiftleft\.codepropertygraph\.generated\.", "", path.read_text(encoding="utf-8")))


def _methods_of_file(raw, file_name):
    return [m for m in raw["functions"] if m.get("file") == file_name]


def _real_methods(methods, file_name):
    """Non-<global> methods that are not #define macro pseudo-methods (`<file>:NAME:sig`)."""
    out = [m for m in methods if not cp.is_global_method(m) and not m["function"].startswith(file_name + ":")]
    assert out == [m for m in methods if not cp.is_global_method(m) and not cp.is_macro_method(m)]
    return out


def _node_ids(m):
    return {n["id"] for n in m["nodes"]}


# ---------------------------------------------------------------------------------------
# single-function export (BigVul path, graph_unit="first_method")
# ---------------------------------------------------------------------------------------

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


def test_first_method_is_the_default_and_unchanged_by_the_keyword():
    stats_default, stats_kw = {}, {}
    a = cp.json_process(str(EXPORT.parent), EXPORT.name, stats_default)
    b = cp.json_process(str(EXPORT.parent), EXPORT.name, stats_kw, graph_unit="first_method")
    assert a == b and stats_default == stats_kw
    assert "whole_files_kept" not in stats_default


def test_unknown_graph_unit_raises():
    with pytest.raises(ValueError):
        cp.json_process(str(EXPORT.parent), EXPORT.name, {}, graph_unit="per_method")


def test_whole_file_unit_on_single_function_export_keeps_one_global_per_numbered_file():
    stats = {}
    kept = cp.json_process(str(EXPORT.parent), EXPORT.name, stats, graph_unit="whole_file")
    assert [i for i, _ in kept] == [101, 1035, 10019, 10048, 10052, 10115, 10125, 10131, 10141, 10204, 10280, 10289]
    assert all(cp.is_global_method(e["functions"][0]) for _, e in kept)
    assert stats["whole_files_kept"] == 12 and stats["functions_kept"] == 12
    assert stats["methods_folded_into_whole_file"] == 10           # the 10 real methods of first_method
    assert stats["whole_files_without_real_method"] == 2           # 101.cpp and 10131.c: Joern parsed nothing
    assert stats["skipped_unnumbered"] == 82                       # 81 call-target stubs + <includes>:<global>
    counts = {i: cp.whole_file_real_method_count(e["functions"][0]) for i, e in kept}
    assert [i for i, c in counts.items() if c == 0] == [101, 10131]
    assert all(c == 1 for i, c in counts.items() if i not in (101, 10131))


# ---------------------------------------------------------------------------------------
# graph_indexing(keep_global=...)
# ---------------------------------------------------------------------------------------

def test_graph_indexing_keep_global():
    g = {"function": "12.c:<global>", "file": "12.c", "nodes": [], "edges": []}
    assert cp.graph_indexing(g) is None
    idx, entry = cp.graph_indexing(g, keep_global=True)
    assert idx == 12 and entry == {"functions": [{"function": "12.c:<global>", "nodes": [], "edges": []}]}
    assert "file" not in entry["functions"][0]
    # still None whatever keep_global says
    for bad in ({"function": "<includes>:<global>", "file": "<includes>"},
                {"function": "<operator>.addition", "file": "<empty>"},
                {"function": "foo", "file": ""}, {"function": "foo", "file": "N/A"},
                {"function": "prog.c:<global>", "file": "prog.c"}):
        assert cp.graph_indexing(bad, keep_global=True) is None, bad
        assert cp.graph_indexing(bad) is None, bad
    assert cp.graph_indexing({"function": "f", "file": "/a/b/77.cpp"}) == (77, {"functions": [{"function": "f"}]})


def test_method_base_name():
    assert cp.method_base_name("main") == "main"
    assert cp.method_base_name("main<duplicate>0") == "main"
    assert cp.method_base_name("main:int()") == "main"
    assert cp.method_base_name("main<duplicate>0:int()") == "main"
    assert cp.method_base_name("add:T(T,T)") == "add"
    assert cp.method_base_name("204.c:SQ:ANY(1)") == "SQ"
    assert cp.method_base_name("CClient.ClearServerDependentCaps:void()") == "CClient.ClearServerDependentCaps"
    assert cp.method_base_name("Foo.bar:void(std::string,std::vector<int>)") == "Foo.bar"   # `::` inside the signature
    assert cp.method_base_name("main<duplicate>1") == "main"                 # 204.c in the multi fixture
    assert cp.method_base_name("") == ""
    assert cp.method_base_name("203.cpp:<global>") == "<global>"


# ---------------------------------------------------------------------------------------
# whole-program export (graph_unit="whole_file")
# ---------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def multi_raw():
    return _raw(MULTI)


def test_multi_fixture_sources_and_layout(multi_raw):
    assert sorted(p.name for p in MULTI_INPUTS.iterdir() if p.suffix in (".c", ".cpp")) == \
        ["201.c", "202.c", "203.cpp", "204.c", "205.cpp"]
    files = {m["file"] for m in multi_raw["functions"]}
    assert files == {"201.c", "202.c", "203.cpp", "204.c", "205.cpp", "<includes>", "<empty>"}
    assert len(multi_raw["functions"]) == 34
    assert sum(cp.is_global_method(m) for m in multi_raw["functions"]) == 6     # 5 files + <includes>


def test_first_method_keeps_the_helper_and_drops_main(multi_raw):
    """The defect whole_file fixes: Joern lists 201.c's helper (defined before main)
    first, so the BigVul-style unit keeps the helper and silently drops main."""
    names = [cp.method_base_name(m["function"]) for m in _real_methods(_methods_of_file(multi_raw, "201.c"), "201.c")]
    assert names == ["helper", "main"]
    stats = {}
    kept = dict(cp.json_process(str(MULTI.parent), MULTI.name, stats, graph_unit="first_method"))
    assert kept[201]["functions"][0]["function"] == "helper"
    assert cp.method_base_name(kept[203]["functions"][0]["function"]) == "add"        # template helper, not main
    assert cp.method_base_name(kept[205]["functions"][0]["function"]) == "dist2"
    assert stats["extra_methods_same_file"] == 4 and stats["functions_kept"] == 5 and stats["skipped_global"] == 6


def test_whole_file_one_entry_per_file_superset_of_real_methods(multi_raw):
    stats = {}
    kept = cp.json_process(str(MULTI.parent), MULTI.name, stats, graph_unit="whole_file")
    assert [i for i, _ in kept] == [201, 202, 203, 204, 205]
    assert stats["whole_files_kept"] == 5 and stats["functions_kept"] == 5
    assert stats["methods_folded_into_whole_file"] == 9            # 2+1+2+(1 real + 1 macro)+2
    assert stats["whole_files_without_real_method"] == 0
    assert stats["skipped_unnumbered"] == 20                       # 19 stubs + <includes>:<global>
    assert {i: cp.whole_file_real_method_count(e["functions"][0]) for i, e in kept} == {201: 2, 202: 1, 203: 2, 204: 1, 205: 2}
    for idx, entry in kept:
        g = entry["functions"][0]
        file_name = f"{idx}.c" if (MULTI_INPUTS / f"{idx}.c").exists() else f"{idx}.cpp"
        assert g["function"] == f"{file_name}:<global>" and "file" not in g
        gids = _node_ids(g)
        real = _real_methods(_methods_of_file(multi_raw, file_name), file_name)
        assert real, file_name
        union = set().union(*(_node_ids(m) for m in real))
        assert union < gids, file_name                             # strict superset
        # the file spine (METHOD / BLOCK / METHOD_RETURN of <global>) is in the difference
        spine = {n["id"] for n in g["nodes"] if n["label"] in ("METHOD", "BLOCK", "METHOD_RETURN")} - union
        assert len(spine) >= 3
        # all intra-method edges of the real methods are in the <global> edge list
        gedges = {(e["out"], e["in"], e["type"]) for e in g["edges"]}
        for m in real:
            assert {(e["out"], e["in"], e["type"]) for e in m["edges"]} <= gedges, m["function"]


def test_whole_file_201_contains_the_global_locals(multi_raw):
    kept = dict(cp.json_process(str(MULTI.parent), MULTI.name, graph_unit="whole_file"))
    g = kept[201]["functions"][0]
    local_codes = {n["properties"]["CODE"] for n in g["nodes"] if n["label"] == "LOCAL"}
    assert {"int table[4]", "int counter"} <= local_codes
    # ... and they are NOT in either real method
    for m in _real_methods(_methods_of_file(multi_raw, "201.c"), "201.c"):
        assert not ({"int table[4]", "int counter"} & {n["properties"]["CODE"] for n in m["nodes"] if n["label"] == "LOCAL"})
    # the initialiser literal of `table` is in the whole-file graph only
    assert any(n["label"] == "LITERAL" and n["properties"]["CODE"] == "4" and int(n["properties"]["LINE_NUMBER"]) == 1
               for n in g["nodes"])


def test_macro_pseudo_method_shares_no_node_with_global(multi_raw):
    ms = _methods_of_file(multi_raw, "204.c")
    macro = [m for m in ms if m["function"] == "204.c:SQ:ANY(1)"]
    glob = [m for m in ms if cp.is_global_method(m)]
    assert len(macro) == 1 and len(glob) == 1
    assert not (_node_ids(macro[0]) & _node_ids(glob[0]))
    assert cp.method_base_name(macro[0]["function"]) == "SQ"
    assert cp.is_macro_method(macro[0]) and not cp.is_macro_method(glob[0])
    assert not any(cp.is_macro_method(m) for m in ms if m["function"].startswith("main"))
    # the macro method is counted as folded, not kept, in whole_file mode and as an extra in first_method mode
    assert cp.graph_indexing(macro[0]) is not None


def test_kr_main_written_as_c_is_a_real_method(multi_raw):
    real = _real_methods(_methods_of_file(multi_raw, "202.c"), "202.c")
    assert [cp.method_base_name(m["function"]) for m in real] == ["main"]
    assert real[0]["function"].startswith("main<duplicate>")          # C frontend naming
    assert len(real[0]["nodes"]) > 20
    assert (MULTI_INPUTS / "202.c").read_text(encoding="utf-8").startswith("main()")


def test_modern_cpp_has_main_and_template_helper(multi_raw):
    real = _real_methods(_methods_of_file(multi_raw, "203.cpp"), "203.cpp")
    assert sorted(cp.method_base_name(m["function"]) for m in real) == ["add", "main"]
    assert {m["function"] for m in real} == {"add:T(T,T)", "main:int()"}  # C++ frontend naming
    glob = [m for m in _methods_of_file(multi_raw, "203.cpp") if cp.is_global_method(m)][0]
    assert not any(n["label"] == "UNKNOWN" for n in glob["nodes"])      # `using namespace std;` parsed cleanly as .cpp


def test_whole_file_graphs_parse_and_build_valid_data():
    kept = cp.json_process(str(MULTI.parent), MULTI.name, graph_unit="whole_file")
    for idx, entry in kept:
        assert parse_to_functions(entry, max_nodes=500) == []            # default skips <global>
        funcs = parse_to_functions(entry, max_nodes=500, skip_global=False)
        assert len(funcs) == 1
        od = funcs[0]
        labels = [n.label for n in od.values()]
        assert "UNKNOWN" not in labels and "COMMENT" not in labels
        assert all(n.label_id != NODE_LABEL_TO_ID["<UNK>"] for n in od.values()), idx
        methods = [n for n in od.values() if n.label == "METHOD"]
        assert "<global>" in {n.code for n in methods} and len(methods) >= 2   # the file spine + its functions
        blocks = [n for n in od.values() if n.label == "BLOCK"]
        src = (MULTI_INPUTS / f"{idx}.c") if (MULTI_INPUTS / f"{idx}.c").exists() else (MULTI_INPUTS / f"{idx}.cpp")
        whole = src.read_text(encoding="utf-8")
        assert any(b.code.strip() in (whole.strip(), "<empty>") or len(b.code) >= 0.8 * len(whole) for b in blocks), idx
        rel = count_edges_by_relation(od)
        assert rel["AST"] > 0 and rel["CFG"] > 0 and rel["PDG"] > 0, (idx, rel)
        assert rel["AST"] == len(od) - 1, (idx, rel)                 # <global>'s AST is still a tree
        g = nodes_to_input(od, 7, 500, mode="fast")
        assert g.x.shape == (len(od), 768) and g.num_nodes == len(od)
        assert g.edge_index.shape[1] == sum(rel.values()) and set(g.edge_type.tolist()) == {0, 1, 2}
        assert int(g.y) == 7


def test_whole_file_node_cap_cuts_later_functions_first():
    """order_nodes truncates by (line, column): with a small cap the nodes that
    survive are the earliest lines of the file, so main (last in 201.c) is cut first."""
    kept = dict(cp.json_process(str(MULTI.parent), MULTI.name, graph_unit="whole_file"))
    full = parse_to_functions(kept[201], max_nodes=500, skip_global=False)[0]
    cut = parse_to_functions(kept[201], max_nodes=25, skip_global=False)[0]
    assert len(cut) == 25 < len(full)
    kept_lines = [n.get_line_number() for n in cut.values() if n.get_line_number() is not None]
    dropped_lines = [n.get_line_number() for nid, n in full.items() if nid not in cut and n.get_line_number() is not None]
    assert max(kept_lines) <= min(dropped_lines)
    assert not any(n.code.startswith("int main()") for n in cut.values())
