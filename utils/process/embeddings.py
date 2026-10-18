# utils/process/embeddings.py
"""
Node features and PyG ``Data`` construction.

Fixed relative to the thesis code (audit M13 / M25; M22 only partially, see node_code_text):
  * ``x`` is the 768-d frozen-CodeBERT CLS of the node's CODE string only;
    the Joern node label travels separately as ``node_label`` (long) so the
    model can embed it (the original packed a one-bit regex flag into x);
  * optional reverse edges (relation id + 3) so messages can flow both ways;
  * every graph records ``num_nodes`` explicitly and keeps an empty-edge
    self-loop fallback.
Normalisation of the CLS vectors is done inside the model (LayerNorm on
input), not here, so cached features stay comparable across runs.
"""
import os
from collections import OrderedDict
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

try:
    from transformers import RobertaTokenizer, RobertaModel
except Exception:  # transformers is optional for the fast mode
    RobertaTokenizer = None
    RobertaModel = None

from utils.functions.cpg import RELATION_TO_ID, NODE_LABEL_TO_ID

BERT_DIM = 768
_NODES_EMBEDDERS = {}


def get_nodes_embedding(nodes_dim: int, *, mode: str = "codebert", max_len: int = 128,
                        code_mode: str = "full", max_cache: int = 200_000):
    key = (int(nodes_dim), mode, int(max_len), code_mode, int(max_cache))
    if key not in _NODES_EMBEDDERS:
        _NODES_EMBEDDERS[key] = NodesEmbedding(
            nodes_dim=int(nodes_dim), max_len=max_len, batch_size=32,
            dtype=torch.float16, mode=mode, code_mode=code_mode, max_cache=max_cache,
        )
    return _NODES_EMBEDDERS[key]


def node_code_text(node, code_mode: str = "full") -> str:
    """CODE string fed to CodeBERT for one node.
    code_mode="full": Joern's CODE property verbatim (thesis behaviour; a BLOCK /
    METHOD / CONTROL_STRUCTURE node then carries its whole enclosed source,
    audit M22). code_mode="first_line": only the first line, so container
    nodes contribute their header rather than the nested text."""
    code = node.get_code()
    code = code if isinstance(code, str) else str(code)
    if code_mode == "first_line":
        return code.split("\n", 1)[0].strip()
    return code


class NodesEmbedding:
    """Node CODE string -> 768-d vector. mode="codebert" (frozen CLS, cached by
    string) or mode="fast" (cheap deterministic features for tests)."""

    def __init__(self, nodes_dim: int, *, max_len: int = 128, batch_size: int = 32,
                 dtype=torch.float16, mode: str = "codebert", code_mode: str = "full",
                 max_cache: int = 200_000):
        self.nodes_dim = int(nodes_dim)
        self.code_mode = code_mode
        # CODE string -> owned float16 CLS vector (1.5 KB each); least-recently-used
        # entries are evicted beyond max_cache so a full-dataset --embed stays bounded
        self.max_cache = int(max_cache)
        env_fast = os.environ.get("USE_FAST_NODES_EMBEDDING", "").strip() in {"1", "true", "True"}
        self.mode = "fast" if env_fast else mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.feat_dim = BERT_DIM
        self.max_len = int(max_len)
        self.batch_size = int(batch_size)
        self._cache = OrderedDict()
        self._is_codebert = (self.mode == "codebert")
        if self._is_codebert:
            if RobertaTokenizer is None or RobertaModel is None:
                raise RuntimeError("transformers is not importable; use mode='fast' or install it.")
            self.tokenizer_bert = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
            kw = {"low_cpu_mem_usage": True}
            if self.device.type == "cuda":
                kw["dtype"] = self.dtype          # transformers >= 5 name; fall back below
            try:
                self.bert_model = RobertaModel.from_pretrained("microsoft/codebert-base", **kw)
            except TypeError:
                kw.pop("dtype", None)
                if self.device.type == "cuda":
                    kw["torch_dtype"] = self.dtype
                self.bert_model = RobertaModel.from_pretrained("microsoft/codebert-base", **kw)
            self.bert_model = self.bert_model.to(self.device)
            self.bert_model.eval()
        else:
            self.tokenizer_bert = None
            self.bert_model = None

    def __call__(self, nodes):
        emb_np = self._embed_nodes_codebert(nodes) if self._is_codebert else self._embed_nodes_fast(nodes)
        return torch.from_numpy(emb_np).float()

    def _embed_nodes_fast(self, nodes):
        node_items = list(nodes.items())
        out = np.zeros((len(node_items), self.feat_dim), dtype=np.float32)
        for i, (_, node) in enumerate(node_items):
            code = node_code_text(node, self.code_mode) or ""
            out[i, 0] = float(len(code))
            out[i, 1] = float(node.get_line_number() or 0)
            out[i, 2] = float(node.get_column_number() or 0)
            h = int.from_bytes(code.encode("utf-8")[:4].ljust(4, b"\0"), "little")
            out[i, 3] = float(h % 1_000_003)
            out[i, 4] = float(getattr(node, "label_id", 0))
        return out

    def _batch_encode(self, texts):
        return self.tokenizer_bert(texts, return_tensors="pt", padding=True,
                                   truncation=True, max_length=self.max_len)

    def _cache_put(self, key: str, vec: np.ndarray):
        self._cache[key] = np.ascontiguousarray(vec, dtype=np.float16)   # owned copy, not a view
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_cache:
            self._cache.popitem(last=False)

    def _embed_nodes_codebert(self, nodes):
        codes = [node_code_text(node, self.code_mode) for _, node in nodes.items()]
        feats = []
        with torch.inference_mode():
            for i in range(0, len(codes), self.batch_size):
                batch_codes = codes[i: i + self.batch_size]
                fresh = {}      # every vector this batch needs, immune to LRU eviction during the batch
                for c in batch_codes:
                    if c not in fresh and c in self._cache:
                        fresh[c] = self._cache[c]
                        self._cache.move_to_end(c)
                miss = [c for c in dict.fromkeys(batch_codes) if c not in fresh]
                for k in range(0, len(miss), self.batch_size):
                    chunk = miss[k: k + self.batch_size]
                    enc = self._batch_encode(chunk)
                    enc = {kk: v.to(self.device, non_blocking=True) for kk, v in enc.items()}
                    if self.device.type == "cuda":
                        with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                            hidden = self.bert_model(**enc).last_hidden_state
                    else:
                        hidden = self.bert_model(**enc).last_hidden_state
                    # slice first, then copy: never keep a view into the whole [B, L, 768] tensor
                    cls_np = hidden[:, 0].detach().to("cpu", torch.float32, copy=True).contiguous().numpy()
                    for j, c in enumerate(chunk):
                        v16 = np.ascontiguousarray(cls_np[j], dtype=np.float16)
                        fresh[c] = v16
                        self._cache_put(c, v16)
                    del enc, hidden, cls_np
                for c in batch_codes:
                    feats.append(fresh[c].astype(np.float32))   # always the fp16-rounded vector, so repeats are bit-identical
        return np.stack(feats, axis=0).astype(np.float32)


# === edges ================================================================
EDGE_MAP = dict(RELATION_TO_ID)      # {"AST": 0, "CFG": 1, "PDG": 2}
NUM_RELATIONS = len(EDGE_MAP)        # 3 forward relations (6 with reverse edges)


def build_edges_all(nodes, *, reverse_edges: bool = False):
    """Return (edge_index [2, E], edge_type [E]) over node ``order`` indices.
    Direction follows Joern: src = edge "out", dst = edge "in"."""
    src, dst, et = [], [], []
    seen = set()
    for _, node in nodes.items():
        for e in node.edges.values():
            if e.node_in not in nodes or e.node_out not in nodes:
                continue
            s, d = nodes[e.node_in].order, nodes[e.node_out].order
            if s == d:
                continue
            r = EDGE_MAP[e.type]
            if (s, d, r) in seen:
                continue
            seen.add((s, d, r))
            src.append(s); dst.append(d); et.append(r)
            if reverse_edges:
                src.append(d); dst.append(s); et.append(r + NUM_RELATIONS)
    if not src:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long), torch.tensor(et, dtype=torch.long)


def nodes_to_input(nodes, target, nodes_dim, *, mode: str = "codebert",
                   reverse_edges: bool = False, max_len: int = 128, code_mode: str = "full",
                   max_cache: int = 200_000):
    """nodes: OrderedDict[node_id -> node] (already ordered / truncated)."""
    nodes_embedding = get_nodes_embedding(nodes_dim, mode=mode, max_len=max_len, code_mode=code_mode,
                                          max_cache=max_cache)
    x = nodes_embedding(nodes)                                   # [N, 768]
    n_real = x.size(0)
    labels = torch.tensor([int(getattr(n, "label_id", NODE_LABEL_TO_ID["<UNK>"])) for n in nodes.values()],
                          dtype=torch.long)
    edge_index, edge_type = build_edges_all(nodes, reverse_edges=reverse_edges)
    if edge_index.numel() == 0 and n_real > 0:
        edge_index, _ = add_self_loops(edge_index, num_nodes=n_real)
        edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        node_label=labels,
        y=torch.tensor([int(target)], dtype=torch.long),
    )
    data.num_nodes = n_real
    return data
