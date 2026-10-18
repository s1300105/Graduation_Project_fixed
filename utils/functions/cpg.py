# utils/functions/cpg.py
"""
Joern JSON -> per-function node dictionaries.

Fixed relative to the thesis code (see audit M19 / M20 / M22 / M25):
  * the relation of an edge is read from the edge's own ``type`` field
    (the original labelled every edge with the name of the list its
    source node happened to be listed in, which made AST == CFG);
  * edges are de-duplicated on (src, dst, relation) while parsing;
  * Joern's per-file ``<global>`` pseudo-method is skipped;
  * the Joern node label is kept as an integer id (``label_id``) so the
    model can embed it instead of a one-bit regex flag;
  * both the legacy layout (``AST`` / ``CFG`` / ``PDG`` node lists) and a
    flat ``nodes`` + ``edges`` layout are accepted.
"""
from collections import OrderedDict
import re

# Joern edge label -> relation name used by the model
EDGE_TYPE_MAP = {
    "AST": "AST",
    "CFG": "CFG",
    "PDG": "PDG",
    "REACHING_DEF": "PDG",
    "DDG": "PDG",
    "CDG": "PDG",
    "DATA_FLOW": "PDG",
}
RELATIONS = ("AST", "CFG", "PDG")
RELATION_TO_ID = {r: i for i, r in enumerate(RELATIONS)}

# legacy per-kind node lists written by the original graph-for-funcs.sc
EDGE_KINDS = ("AST", "CFG", "PDG")

NODE_LABELS = [
    "<PAD>", "<UNK>",
    "METHOD", "METHOD_PARAMETER_IN", "METHOD_PARAMETER_OUT", "METHOD_RETURN",
    "BLOCK", "LOCAL", "IDENTIFIER", "LITERAL", "CALL", "CONTROL_STRUCTURE",
    "RETURN", "FIELD_IDENTIFIER", "JUMP_TARGET", "TYPE_REF", "MODIFIER",
    "UNKNOWN", "MEMBER", "TYPE_DECL", "NAMESPACE_BLOCK", "COMMENT",
    "ANNOTATION", "METHOD_REF", "TYPE", "TYPE_ARGUMENT", "TYPE_PARAMETER",
    "NAMESPACE", "FILE", "IMPORT", "TEMPLATE_DOM", "ARRAY_INITIALIZER",
]
NODE_LABEL_TO_ID = {l: i for i, l in enumerate(NODE_LABELS)}
NUM_NODE_LABELS = len(NODE_LABELS)

DROP_LABELS = {"COMMENT", "UNKNOWN"}


# === utilities =============================================================

def _as_int(v, default=10**9):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


def order_nodes(nodes, max_nodes):
    """Sort by (line, column, id); nodes without a position sort last and
    are therefore the first to be cut when the graph exceeds ``max_nodes``."""
    nodes_by_pos = sorted(
        nodes.items(),
        key=lambda it: (
            _as_int(it[1].get_line_number()),
            _as_int(it[1].get_column_number()),
            it[0],
        ),
    )
    if len(nodes_by_pos) > max_nodes:
        nodes_by_pos = nodes_by_pos[:max_nodes]
    for i, (_, node) in enumerate(nodes_by_pos):
        node.order = i
    return OrderedDict(nodes_by_pos)


def filter_nodes(nodes, strict=False):
    def keep(node):
        if (node.label or "").upper() in DROP_LABELS:
            return False
        if strict:
            return node.has_code() and node.has_line_number()
        return True
    return {n_id: node for n_id, node in nodes.items() if keep(node)}


# === node / edge stubs =====================================================

class _EdgeStub:
    __slots__ = ("type", "node_in", "node_out")

    def __init__(self, etype, node_in, node_out):
        self.type = etype          # one of RELATIONS
        self.node_in = node_in     # source id (Joern "out")
        self.node_out = node_out   # destination id (Joern "in")


class _NodeStub:
    __slots__ = ("id", "code", "line_number", "column_number", "type",
                 "label", "label_id", "order", "edges")

    def __init__(self, nid):
        self.id = nid
        self.code = ""
        self.line_number = None
        self.column_number = None
        self.type = 0              # legacy one-bit flag (kept for compatibility)
        self.label = ""            # Joern label, e.g. "IDENTIFIER"
        self.label_id = NODE_LABEL_TO_ID["<UNK>"]
        self.order = None
        self.edges = {}            # key -> _EdgeStub

    def get_code(self): return self.code
    def get_line_number(self): return self.line_number
    def get_column_number(self): return self.column_number
    def has_code(self): return bool(self.code)
    def has_line_number(self): return self.line_number is not None


def _node_props_to_fields(nd: dict):
    props = nd.get("properties", {}) if isinstance(nd.get("properties"), dict) else {}
    code = props.get("CODE") or nd.get("code") or ""
    line = _safe_int(props.get("LINE_NUMBER") or nd.get("line"))
    col = _safe_int(props.get("COLUMN_NUMBER") or nd.get("column"))
    label = str(nd.get("label") or "")
    ntype = str(nd.get("type") or "")
    if not label and ntype:
        # derive the Joern label from the class name (e.g. "MethodParameterIn")
        label = re.sub(r"(?<!^)(?=[A-Z])", "_", ntype).upper()
    return code, line, col, ntype, label.upper()


def _ensure_node(node_map, nid, template=None):
    if nid not in node_map:
        node_map[nid] = _NodeStub(nid)
    node = node_map[nid]
    if template:
        if template.get("code"):
            node.code = template["code"]
        if template.get("line_number") is not None:
            node.line_number = template["line_number"]
        if template.get("column_number") is not None:
            node.column_number = template["column_number"]
        label = template.get("label") or ""
        tname = template.get("type_name") or label
        if label:
            node.label = label
            node.label_id = NODE_LABEL_TO_ID.get(label, NODE_LABEL_TO_ID["<UNK>"])
        if tname:
            node.type = 0 if re.search(r"(Identifier|Name|Literal|Type|Decl)", tname, re.I) else 1
    return node


def _attach_edge(node_map, u, v, etype):
    """Attach the directed edge u -> v (relation ``etype``) to both endpoints."""
    ekey = f"{u}->{v}:{etype}"
    for nid in (u, v):
        n = _ensure_node(node_map, nid)
        if ekey not in n.edges:
            n.edges[ekey] = _EdgeStub(etype, node_in=u, node_out=v)


# === JSON walking ==========================================================

def _iter_node_dicts(func_json):
    """Yield every node dict of a function regardless of layout."""
    for kind in EDGE_KINDS:
        for item in func_json.get(kind, []) or []:
            if not isinstance(item, dict):
                continue
            inner = item.get("nodes")
            if isinstance(inner, list) and inner:
                for nd in inner:
                    if isinstance(nd, dict):
                        yield nd
            else:
                yield item
    for nd in func_json.get("nodes", []) or []:
        if isinstance(nd, dict):
            yield nd


def _gather_nodes(func_json, node_map):
    for nd in _iter_node_dicts(func_json):
        nid_raw = nd.get("id")
        if nid_raw is None:
            continue
        code, line, col, ntype, label = _node_props_to_fields(nd)
        _ensure_node(node_map, str(nid_raw), template={
            "code": code, "line_number": line, "column_number": col,
            "type_name": ntype or label, "label": label,
        })


def _iter_edge_dicts(func_json):
    for nd in _iter_node_dicts(func_json):
        for e in nd.get("edges", []) or []:
            if isinstance(e, dict):
                yield e
    for blk in (func_json.get(k) for k in EDGE_KINDS):
        for item in blk or []:
            if isinstance(item, dict) and isinstance(item.get("edges"), list) and item.get("nodes"):
                for e in item["edges"]:
                    if isinstance(e, dict):
                        yield e
    for e in func_json.get("edges", []) or []:
        if isinstance(e, dict):
            yield e


def _gather_edges(func_json, node_map):
    """Attach every edge with the relation taken from ITS OWN ``type`` field."""
    seen = set()
    for e in _iter_edge_dicts(func_json):
        etype = EDGE_TYPE_MAP.get(str(e.get("type") or "").upper())
        if etype is None:
            continue
        u, v = e.get("out"), e.get("in")
        if u is None or v is None:
            continue
        u, v = str(u), str(v)
        if u == v:
            continue
        key = (u, v, etype)
        if key in seen:
            continue
        seen.add(key)
        _attach_edge(node_map, u, v, etype)
    return len(seen)


def is_global_function(func_json) -> bool:
    name = str(func_json.get("function") or func_json.get("name") or "")
    return name.endswith("<global>") or name == "<global>"


def parse_function_v4_to_nodes(func_json, max_nodes=500, strict_filter=False):
    node_map = {}
    _gather_nodes(func_json, node_map)
    _gather_edges(func_json, node_map)
    filtered = filter_nodes(node_map, strict=strict_filter)
    if not filtered:
        return OrderedDict()
    return order_nodes(filtered, max_nodes)


def parse_to_functions(cpg, max_nodes=500, skip_global=True, strict_filter=False):
    """Return one OrderedDict[node_id -> _NodeStub] per real function in ``cpg``."""
    all_funcs = []
    for func_json in cpg.get("functions", []):
        if skip_global and is_global_function(func_json):
            continue
        od = parse_function_v4_to_nodes(func_json, max_nodes=max_nodes, strict_filter=strict_filter)
        if len(od) > 0:
            all_funcs.append(od)
    return all_funcs


def count_edges_by_relation(nodes):
    """Debug helper: unique (src, dst, relation) triples per relation."""
    seen = set()
    for node in nodes.values():
        for e in node.edges.values():
            if e.node_in in nodes and e.node_out in nodes:
                seen.add((e.node_in, e.node_out, e.type))
    out = {r: 0 for r in RELATIONS}
    for _, _, r in seen:
        out[r] += 1
    return out
