# utils/process/cpg_generator.py
"""Joern driver: source files -> CPG binary -> per-function graph JSON.

Fixed relative to the thesis code (audit M20 / M26 / M39):
  * ``graph_indexing`` drops Joern's per-file ``<global>`` pseudo-method, so
    each source function yields exactly one graph row;
  * ``joern_create`` returns a {cpg_file: json_path | None} mapping so the
    caller can align slices with outputs instead of a positional zip that
    silently misaligns after any failure;
  * ``json_process`` reports how many methods were kept / skipped.

Graph units (``json_process(..., graph_unit=...)``):
  * ``"first_method"`` (default, BigVul): one row per numbered source file, the
    first real method Joern lists for that file; ``<global>`` is dropped. This is
    the thesis behaviour and stays byte-identical.
  * ``"whole_file"`` (POJ-104 / CodeNet whole programs): one row per numbered
    source file holding the file's ``<file>:<global>`` pseudo-method. Joern's
    ``<global>`` AST is a strict superset of the union of the file's real
    methods (top-level functions are AST children of ``<global>``) plus the
    global LOCALs with their initialisers, so it IS the whole-file graph. The
    file's real methods are folded into it (counted in
    ``stats["methods_folded_into_whole_file"]``). ``first_method`` would keep
    whichever function Joern lists first (a helper defined before ``main`` in
    21% of POJ-104 programs) and silently drop ``main``.

Whole-file unit (graph_unit="whole_file"): the kept graph is the file's ``<global>``
method. In Joern 4.0.400 its AST holds every top-level function and struct / class
member function of the file, the global LOCALs and their initialisers; it does NOT
hold ``#define`` function-like macro methods (``<file>:NAME:sig``), call-target
stubs (file ``<empty>``) or ``<includes>:<global>``, and a C++ lambda's body is a
separate method that this module folds back into the file graph. Its METHOD node's
CODE is ``<global>`` and its BLOCK's CODE is ``<empty>``.
"""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


def joern_parse(joern_path, input_path, output_path, file_name, timeout_sec: int = 1800):
    out_file = file_name + ".bin"
    joern_bin = Path(joern_path).expanduser().resolve() / "joern-parse"
    cmd = [str(joern_bin), str(Path(input_path).resolve()), "--output", str(Path(output_path).resolve() / out_file)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, timeout=timeout_sec,
                         cwd=str(Path(output_path).resolve()))
    if res.stdout.strip():
        print(res.stdout.strip()[-2000:])
    return out_file


def joern_create(joern_path: str, in_path: str, out_path: str, cpg_files: list, timeout_sec: int = 1800,
                 script_name: str = "graph-for-funcs.sc") -> dict:
    """Run ``graph-for-funcs.sc`` on each CPG. Returns {cpg_file: json_path or None}."""
    joern_bin = Path(joern_path).expanduser().resolve() / "joern"
    script = (Path(joern_path).expanduser().resolve() / script_name).resolve()
    if not script.exists():
        alt = Path(__file__).resolve().parents[2] / "joern" / script_name
        if alt.exists():
            script = alt
        else:
            raise FileNotFoundError(f"Joern export script not found: {script} (nor {alt})")
    in_dir = Path(in_path).expanduser().resolve()
    out_dir = Path(out_path).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for cpg_file in cpg_files:
        cpg_path = in_dir / cpg_file
        results[cpg_file] = None
        if not cpg_path.exists():
            print(f"[WARN] CPG not found: {cpg_path}")
            continue
        json_out = out_dir / (cpg_path.stem + ".json")
        cmd = [str(joern_bin), "--script", str(script),
               "--param", f"cpgFile={cpg_path}", "--param", f"outFile={json_out}"]
        print(f"[INFO] Running: {' '.join(cmd)}")
        try:
            # run inside out_dir so Joern's `workspace/` directory does not land in the repository
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_sec, cwd=str(out_dir))
        except subprocess.TimeoutExpired:
            print(f"[ERROR] Timeout after {timeout_sec}s: {cpg_file}")
            continue
        if res.returncode != 0:
            print(f"[ERROR] Joern failed on {cpg_file} (code {res.returncode})\n{res.stderr[-3000:]}")
            continue
        if json_out.exists():
            results[cpg_file] = str(json_out)
            print(f"[OK] Wrote: {json_out}")
        else:
            print(f"[ERROR] Expected output missing: {json_out}")
        # importCpg() copies the CPG into <cwd>/workspace/<name>/; drop it (it doubles disk use per slice)
        shutil.rmtree(out_dir / "workspace", ignore_errors=True)
    return results


_NUMBERED = re.compile(r"(\d+)$")


def is_global_method(graph: dict) -> bool:
    name = str(graph.get("function") or graph.get("name") or "")
    return name.endswith("<global>") or name == "<global>"


def is_lambda_method(graph: dict) -> bool:
    """A C++ lambda exported as its own method, named
    ``<file>:<global>.<enclosing>.<lambda>N:sig``. Its body is NOT part of the
    file's ``<global>`` AST (only a METHOD_REF is), so whole_file mode folds its
    nodes and edges into the file graph."""
    return "<lambda>" in str(graph.get("function") or "")


def is_macro_method(graph: dict) -> bool:
    """A ``#define`` function-like macro exported as a method: named
    ``<file>:NAME:sig`` (e.g. ``12.c:SQ:ANY(1)``) and sharing no node with ``<global>``.
    Lambdas also carry the file prefix but are real code (see is_lambda_method)."""
    if is_global_method(graph) or is_lambda_method(graph):
        return False
    f = os.path.basename(str(graph.get("file") or ""))
    return bool(f) and str(graph.get("function") or "").startswith(f + ":")


def whole_file_real_method_count(graph: dict) -> int:
    """Number of functions inside a ``<file>:<global>`` graph: its METHOD nodes
    other than the ``<global>`` spine itself. 0 means the file defines no function
    (declarations / structs / ``#define`` only, or a parse failure such as a K&R
    ``main()`` written as .cpp); such a file is not a usable graph."""
    return sum(1 for n in graph.get("nodes", []) or []
               if str(n.get("label") or "").upper() == "METHOD"
               and (n.get("properties") or {}).get("CODE") != "<global>")


_DUPLICATE_SUFFIX = re.compile(r"<duplicate>\d+")
# `12.c:` in front of a #define macro method's name (`12.c:SQ:ANY(1)`)
# the pipeline writes numeric stems (datamanager.to_files), so only `<digits>.<ext>:` is a file
# prefix; a C++ member named like an extension (`Foo.c:int()`) must keep its scope
_FILE_PREFIX = re.compile(r"\A\d+\.(?:c|cc|cpp|cxx|C|h|hh|hpp|hxx):(?=\S)")


def method_base_name(full_name: str) -> str:
    """Joern's method name without the frontend decorations: the C frontend names
    methods ``main``, ``main<duplicate>0``, ...; the C++ frontend ``main:int()``
    (the signature may itself contain ``::``, e.g. ``f:void(std::string)``); a
    ``#define`` macro method ``12.c:SQ:ANY(1)``. Returns ``main`` / ``f`` / ``SQ``."""
    name = str(full_name or "")
    if name.endswith("<global>"):
        return "<global>"
    name = _DUPLICATE_SUFFIX.sub("", name)
    name = _FILE_PREFIX.sub("", name, count=1)
    return name.split(":", 1)[0]


def graph_indexing(graph: dict, keep_global: bool = False):
    """Map one Joern method JSON to (idx, {"functions": [graph]}) using the
    numeric stem of its file name; returns None for include files
    (``<includes>:<global>``), call-target stubs (file ``<empty>``), unnumbered
    files and, unless ``keep_global`` is set, for `<global>` pseudo-methods.
    With ``keep_global=True`` a numbered file's ``<file>:<global>`` method is
    indexed like a real method (whole-file graph)."""
    f = graph.get("file", "")
    if not f or f == "N/A" or str(f).startswith("<"):
        return None
    if is_global_method(graph) and not keep_global:
        return None
    stem, _ = os.path.splitext(os.path.basename(str(f)))
    m = _NUMBERED.search(stem)
    if not m:
        return None
    idx = int(m.group(1))
    g = dict(graph)
    g.pop("file", None)
    return idx, {"functions": [g]}


GRAPH_UNITS = ("first_method", "whole_file")


def json_process(in_path, json_file, stats: dict | None = None, graph_unit: str = "first_method"):
    """Read one graph JSON and return [(idx, {"functions": [...]}) ...].

    graph_unit="first_method": methods of the same source file (idx) are merged
    into one entry so that a file with several real methods still yields one
    row per source file (the first method is used; extra methods are counted in
    ``stats["extra_methods_same_file"]``; ``<global>`` is skipped).
    graph_unit="whole_file": exactly the ``<file>:<global>`` graph of each
    numbered file is kept and the file's other methods are ignored (counted in
    ``stats["methods_folded_into_whole_file"]``); ``stats["whole_files_kept"]``
    counts the rows. A numbered file whose ``<global>`` contains NO real method
    (Joern parsed nothing: the graph would be only the 3-node file spine) is NOT
    returned, so that the caller treats the row as lost (rescue pass / loss
    accounting) instead of training on a labelled empty graph; such files are
    counted in ``stats["whole_files_without_real_method"]``.
    The return shape is the same for both units."""
    if graph_unit not in GRAPH_UNITS:
        raise ValueError(f"unknown graph_unit {graph_unit!r}; expected one of {GRAPH_UNITS}")
    path = json_file if os.path.isabs(json_file) else os.path.join(in_path, json_file)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as jf:
        cpg_string = jf.read()
    cpg_string = re.sub(r"io\.shiftleft\.codepropertygraph\.generated\.", "", cpg_string)
    cpg_json = json.loads(cpg_string)
    methods = cpg_json.get("functions", [])
    if graph_unit == "whole_file":
        return _json_process_whole_file(methods, stats)
    kept, skipped_global, skipped_other, extra = {}, 0, 0, 0
    for g in methods:
        if is_global_method(g):
            skipped_global += 1
            continue
        r = graph_indexing(g)
        if r is None:
            skipped_other += 1
            continue
        idx, entry = r
        if idx in kept:
            extra += 1          # nested / additional method in the same file
            continue
        kept[idx] = entry
    if stats is not None:
        stats["methods_total"] = stats.get("methods_total", 0) + len(methods)
        stats["skipped_global"] = stats.get("skipped_global", 0) + skipped_global
        stats["skipped_unnumbered"] = stats.get("skipped_unnumbered", 0) + skipped_other
        stats["extra_methods_same_file"] = stats.get("extra_methods_same_file", 0) + extra
        stats["functions_kept"] = stats.get("functions_kept", 0) + len(kept)
    return sorted(kept.items())


def _json_process_whole_file(methods: list, stats: dict | None):
    """graph_unit="whole_file": one entry per numbered file = its <global> method,
    plus the nodes / edges of the file's C++ lambdas (Joern exports a lambda as a
    separate method whose body is not under <global>)."""
    kept, skipped_other, folded, dup_global = {}, 0, 0, 0
    lambdas = {}                            # idx -> [lambda method graphs]
    for g in methods:
        if is_global_method(g):
            r = graph_indexing(g, keep_global=True)
            if r is None:
                skipped_other += 1          # <includes>:<global>, unnumbered files
                continue
            idx, entry = r
            if idx in kept:
                dup_global += 1             # two files with the same numeric stem (e.g. 12.c and 12.cpp)
                continue
            kept[idx] = entry
            continue
        r = graph_indexing(g)
        if r is None:
            skipped_other += 1              # call-target stubs (file <empty>), methods of include files
            continue
        folded += 1                         # real method / #define macro method of a numbered file
        if is_lambda_method(g):
            lambdas.setdefault(r[0], []).append(g)
    n_lambda = 0
    for idx, lams in lambdas.items():
        if idx not in kept:
            continue
        g = kept[idx]["functions"][0]
        have = {str(n.get("id")) for n in g.get("nodes", []) or []}
        for lam in lams:
            new_nodes = [n for n in lam.get("nodes", []) or [] if str(n.get("id")) not in have]
            have.update(str(n.get("id")) for n in new_nodes)
            g["nodes"] = list(g.get("nodes", []) or []) + new_nodes
            g["edges"] = list(g.get("edges", []) or []) + list(lam.get("edges", []) or [])
            n_lambda += 1
    # a <global> that defines no function (declarations / #define only, or a parse failure such
    # as K&R main written as .cpp) is not a graph: drop it so the row is counted as lost instead
    # of silently becoming a labelled training example made of the file spine
    empty = sorted(i for i, e in kept.items() if whole_file_real_method_count(e["functions"][0]) == 0)
    for i in empty:
        del kept[i]
    if stats is not None:
        stats["methods_total"] = stats.get("methods_total", 0) + len(methods)
        stats["skipped_unnumbered"] = stats.get("skipped_unnumbered", 0) + skipped_other
        stats["methods_folded_into_whole_file"] = stats.get("methods_folded_into_whole_file", 0) + folded
        stats["lambda_methods_folded"] = stats.get("lambda_methods_folded", 0) + n_lambda
        stats["duplicate_global_same_idx"] = stats.get("duplicate_global_same_idx", 0) + dup_global
        stats["whole_files_kept"] = stats.get("whole_files_kept", 0) + len(kept)
        stats["functions_kept"] = stats.get("functions_kept", 0) + len(kept)
        stats["whole_files_without_real_method"] = stats.get("whole_files_without_real_method", 0) + len(empty)
    return sorted(kept.items())
