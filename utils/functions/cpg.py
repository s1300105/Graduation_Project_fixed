# utils/functions/cpg.py

from collections import OrderedDict
import re

# === ユーティリティ ===

def _as_int(v, default=10**9):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default

def order_nodes(nodes, max_nodes):
    nodes_by_pos = sorted(
        nodes.items(),
        key=lambda it: (
            _as_int(it[1].get_line_number()),
            _as_int(it[1].get_column_number()),
            it[0],
        ),
    )
    for i, (_, node) in enumerate(nodes_by_pos):
        node.order = i
    if len(nodes_by_pos) > max_nodes:
        print(f"CPG cut - original nodes: {len(nodes_by_pos)} to max: {max_nodes}")
        nodes_by_pos = nodes_by_pos[:max_nodes]
    return OrderedDict(nodes_by_pos)

# ★ 厳格/緩和を切り替え可能に
def filter_nodes(nodes, strict=False):
    if strict:
        return {
            n_id: node
            for n_id, node in nodes.items()
            if node.has_code()
            and node.has_line_number()
            and (node.label not in ["Comment", "Unknown"])
        }
    else:
        # 緩いモード：Comment/Unknown だけ除外し、それ以外は通す
        return {
            n_id: node
            for n_id, node in nodes.items()
            if (node.label not in ["Comment", "Unknown"])
        }

# === Joern v4 再構築 ===

EDGE_KINDS = ("AST", "CFG", "PDG")

class _EdgeStub:
    __slots__ = ("type", "node_in", "node_out")
    def __init__(self, etype, node_in, node_out):
        self.type = etype
        self.node_in = node_in
        self.node_out = node_out

class _NodeStub:
    __slots__ = ("id","code","line_number","column_number","type","label","order","edges")
    def __init__(self, nid):
        self.id = nid
        self.code = ""
        self.line_number = None
        self.column_number = None
        self.type = 0
        self.label = ""
        self.order = None
        self.edges = {}  # key -> _EdgeStub

    def get_code(self): return self.code
    def get_line_number(self): return self.line_number
    def get_column_number(self): return self.column_number
    def has_code(self): return bool(self.code)
    def has_line_number(self): return self.line_number is not None

def _safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def _node_props_to_fields(nd: dict):
    props = nd.get("properties", {}) if isinstance(nd.get("properties"), dict) else {}
    code = props.get("CODE") or nd.get("code") or ""
    line = _safe_int(props.get("LINE_NUMBER") or nd.get("line"))
    col  = _safe_int(props.get("COLUMN_NUMBER") or nd.get("column"))
    label = nd.get("label") or nd.get("type") or ""
    ntype = nd.get("type") or label
    return code, line, col, ntype, label

def _ensure_node(node_map, nid, template=None):
    if nid not in node_map:
        node_map[nid] = _NodeStub(nid)
    if template:
        node = node_map[nid]
        if template.get("code"): node.code = template["code"]
        if template.get("line_number") is not None: node.line_number = template["line_number"]
        if template.get("column_number") is not None: node.column_number = template["column_number"]
        if template.get("type_name"):
            tname = template["type_name"]
            node.label = tname
            node.type = 0 if re.search(r"(Identifier|Name|Literal|Type|Decl)", tname or "", re.I) else 1
    return node_map[nid]

def _attach_edge(node_map, u, v, etype):
    ekey = f"{u}->{v}:{etype}"
    for nid, node_in, node_out in ((u,u,v),(v,u,v)):
        n = _ensure_node(node_map, nid)
        if ekey not in n.edges:
            n.edges[ekey] = _EdgeStub(etype, node_in=node_in, node_out=node_out)

def _gather_nodes_from_ast_blocks(func_json, node_map):
    """
    func_json["AST"] からノード属性を _NodeStub に写し込む。

    Joern v4 の CPG 形式では、
    - AST が「ノードのリスト」:  [ {id, properties, edges}, ... ]
    - あるいは「ブロックのリスト」: [ {nodes: [...], edges: [...]}, ... ]
    の2パターンがありうるので、両方に対応する。
    """
    ast_list = func_json.get("AST", []) or []

    for ast_item in ast_list:
        # パターン1: ast_item がブロックで、"nodes" の中にノードが入っている
        nodes_in_item = ast_item.get("nodes")
        if isinstance(nodes_in_item, list) and len(nodes_in_item) > 0:
            iterable = nodes_in_item
        else:
            # パターン2: ast_item 自体がノード
            iterable = [ast_item]

        for nd in iterable:
            nid_raw = nd.get("id")
            if nid_raw is None:
                continue

            nid = str(nid_raw)
            code, line, col, ntype, label = _node_props_to_fields(nd)

            _ensure_node(
                node_map,
                nid,
                template={
                    "code": code,
                    "line_number": line,
                    "column_number": col,
                    "type_name": ntype or label,
                },
            )


def _gather_edges_from_blocks(func_json, node_map):
    for kind in EDGE_KINDS:
        for blk in func_json.get(kind, []) or []:
            for e in blk.get("edges", []) or []:
                u = e.get("out"); v = e.get("in")
                if u is None or v is None: continue
                u = str(u); v = str(v)
                # ★ 端点からノードを必ず作る（属性は後で埋まる想定）
                _ensure_node(node_map, u)
                _ensure_node(node_map, v)
                _attach_edge(node_map, u, v, kind)

def parse_function_v4_to_nodes(func_json, max_nodes=500, strict_filter=False):
    node_map = {}
    _gather_nodes_from_ast_blocks(func_json, node_map)  # あれば属性を入れる
    _gather_edges_from_blocks(func_json, node_map)      # 端点からノードを復元
    # ★ 緩いフィルタで通す（属性が無くてもOK）
    filtered = filter_nodes(node_map, strict=strict_filter)
    if not filtered:
        return OrderedDict()
    return order_nodes(filtered, max_nodes)

def parse_to_functions(cpg, max_nodes=500):
    all_funcs = []
    for func_json in cpg.get("functions", []):
        od = parse_function_v4_to_nodes(func_json, max_nodes=max_nodes, strict_filter=False)
        if len(od) > 0:
            all_funcs.append(od)
    return all_funcs

