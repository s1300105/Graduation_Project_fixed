import json
import subprocess
import os
import re
from pathlib import Path

def joern_parse(joern_path, input_path, output_path, file_name):
    out_file = file_name + ".bin"
    joern_parse_call = subprocess.run(
        ["./" + joern_path + "joern-parse", input_path, "--output", output_path + out_file],
        stdout=subprocess.PIPE, text=True, check=True)
    print(str(joern_parse_call))
    return out_file


"""
def joern_create(joern_path, in_path, out_path, cpg_files):
    
    joern_process = subprocess.Popen(["./" + joern_path + "joern"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    json_files = []
    for cpg_file in cpg_files:
        json_file_name = f"{cpg_file.split('.')[0]}.json"
        json_files.append(json_file_name)

        print(in_path+cpg_file)
        if os.path.exists(in_path+cpg_file):
            json_out = f"{os.path.abspath(out_path)}/{json_file_name}"
            import_cpg_cmd = f"importCpg(\"{os.path.abspath(in_path)}/{cpg_file}\")\r".encode()
            script_path = f"{os.path.dirname(os.path.abspath(joern_path))}/graph-for-funcs.sc"
            run_script_cmd = f"cpg.runScript(\"{script_path}\").toString() |> \"{json_out}\"\r".encode()
            joern_process.stdin.write(import_cpg_cmd)
            print(joern_process.stdout.readline().decode())
            joern_process.stdin.write(run_script_cmd)
            print(joern_process.stdout.readline().decode())
            joern_process.stdin.write("delete\r".encode())
            print(joern_process.stdout.readline().decode())

        try:
            outs, errs = joern_process.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            joern_process.kill()
            outs, errs = joern_process.communicate()
        if outs is not None:
            print(f"Outs: {outs.decode()}")
        if errs is not None:
            print(f"Errs: {errs.decode()}")
        return json_files
"""



def joern_create(joern_path: str, in_path: str, out_path: str, cpg_files: list[str], timeout_sec: int = 300):
    """
    Generates graph JSONs from CPG binaries using Joern.

    Args:
        joern_path: Path to Joern CLI directory (e.g. "joern/joern-cli/")
        in_path: Directory where CPG .bin files are stored
        out_path: Directory to write .json graph output
        cpg_files: List of CPG file names (e.g., ["0_cpg.bin", ...])
        timeout_sec: Timeout in seconds per file
    """
    joern_bin = Path(joern_path).expanduser().resolve() / "joern"
    graph_script = (Path(joern_path).expanduser().resolve() / "graph-for-funcs.sc").resolve()
    
    in_dir = Path(in_path).expanduser().resolve()
    out_dir = Path(out_path).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files: list[str] = []

    for cpg_file in cpg_files:
        cpg_path = in_dir / cpg_file
        if not cpg_path.exists():
            print(f"[WARN] CPG not found: {cpg_path}")
            continue

        json_out = out_dir / (cpg_path.stem + ".json")
        cmd = [
            str(joern_bin),
            "--script", str(graph_script),
            "--param", f"cpgFile={str(cpg_path)}",
            "--param", f"outFile={str(json_out)}"
        ]

        print(f"[INFO] Running: {' '.join(cmd)}")
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            print(f"[ERROR] Timeout after {timeout_sec}s: {cpg_file}")
            continue

        if res.stdout:
            print(res.stdout)
        if res.stderr:
            print(res.stderr)

        if res.returncode != 0:
            print(f"[ERROR] Joern failed on {cpg_file} with code {res.returncode}")
            continue

        if json_out.exists():
            json_files.append(str(json_out))
            print(f"[OK] Wrote: {json_out}")
        else:
            print(f"[ERROR] Expected output missing: {json_out}")

    return json_files





def graph_indexing(graph):
    """
    ファイル名の末尾の連番から idx を安全に抽出する。
    <includes> / N/A / 非C系拡張子はスキップ。
    例) /.../123.c -> 123, /.../456.cpp -> 456
    """
    f = graph.get("file", "")
    if not f or f == "N/A" or f.startswith("<"):
        return None  # スキップ

    base = os.path.basename(f)               # 123.c
    stem, _ = os.path.splitext(base)         # 123
    m = re.search(r"(\d+)$", stem)           # 末尾の数字を抽出
    if not m:
        return None  # 想定外のファイル名はスキップ

    idx = int(m.group(1))
    graph = dict(graph)  # defensive copy
    graph.pop("file", None)  # file は以降不要なら消す
    return idx, {"functions": [graph]}

def json_process(in_path, json_file):
    # 絶対/相対のどちらにも対応
    path = json_file if os.path.isabs(json_file) else os.path.join(in_path, json_file)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as jf:
            cpg_string = jf.read()
            cpg_string = re.sub(r"io\.shiftleft\.codepropertygraph\.generated\.", "", cpg_string)
            cpg_json = json.loads(cpg_string)
            # None を除いて返す
            container = [x for x in (graph_indexing(g) for g in cpg_json.get("functions", [])) if x is not None]
            return container
    return None
