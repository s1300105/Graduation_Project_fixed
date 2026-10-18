# utils/process/cpg_generator.py
"""Joern driver: source files -> CPG binary -> per-function graph JSON.

Fixed relative to the thesis code (audit M20 / M26 / M39):
  * ``graph_indexing`` drops Joern's per-file ``<global>`` pseudo-method, so
    each source function yields exactly one graph row;
  * ``joern_create`` returns a {cpg_file: json_path | None} mapping so the
    caller can align slices with outputs instead of a positional zip that
    silently misaligns after any failure;
  * ``json_process`` reports how many methods were kept / skipped.
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


def graph_indexing(graph: dict):
    """Map one Joern method JSON to (idx, {"functions": [graph]}) using the
    numeric stem of its file name; returns None for `<global>` pseudo-methods,
    include files and unnumbered files."""
    f = graph.get("file", "")
    if not f or f == "N/A" or str(f).startswith("<"):
        return None
    if is_global_method(graph):
        return None
    stem, _ = os.path.splitext(os.path.basename(str(f)))
    m = _NUMBERED.search(stem)
    if not m:
        return None
    idx = int(m.group(1))
    g = dict(graph)
    g.pop("file", None)
    return idx, {"functions": [g]}


def json_process(in_path, json_file, stats: dict | None = None):
    """Read one graph JSON and return [(idx, {"functions": [...]}) ...].
    Methods of the same source file (idx) are merged into one entry so that
    a file with several real methods still yields one row per source function
    (the first method is used; extra methods are counted in ``stats``)."""
    path = json_file if os.path.isabs(json_file) else os.path.join(in_path, json_file)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as jf:
        cpg_string = jf.read()
    cpg_string = re.sub(r"io\.shiftleft\.codepropertygraph\.generated\.", "", cpg_string)
    cpg_json = json.loads(cpg_string)
    methods = cpg_json.get("functions", [])
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
