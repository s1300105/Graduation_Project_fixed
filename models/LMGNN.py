# models/LMGNN.py
"""Hybrid R-GCN + CodeBERT model (BDCA) and the CodeBERT-only baseline.

Architecture follows the thesis (sections 3.3-3.5): frozen-CodeBERT node
features -> R-GCN (d_g = 200) -> projection to d_proj = 200 -> bidirectional
cross-attention blocks -> masked mean pooling -> softmax gate -> classifier.

Fixes / additions relative to the thesis code (audit ids in brackets):
  * LayerNorm on the node features before ``input_proj`` [M13];
  * Joern node label embedding instead of a one-bit regex flag [M25];
  * residual connection + dropout + LayerNorm inside the R-GCN stack
    (``rgcn_residual``) [M18];
  * ``cls_skip``: the pretrained CLS vector is concatenated to the fused
    vector so the hybrid is a strict superset of the baseline [M12];
  * ablation switches: ``graph_only``, ``use_gate=False``, ``shuffle_edges``
    (random-edge control) [M32];
  * gate temperature [M15]; no HEAD-only pre-GCN attention / 768 override [M11];
  * CodeBERT pooler is frozen (it was trainable but unused).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Batch as GeoBatch
from transformers import AutoModel, AutoTokenizer

from .layers import GraphCodeFusion, masked_mean
from utils.functions.cpg import NUM_NODE_LABELS

CODEBERT = "microsoft/codebert-base"


class CodeBERTEncoder(nn.Module):
    def __init__(self, model_name: str = CODEBERT, tune_last_n_layers: int = 2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        for p in self.bert.parameters():
            p.requires_grad = False
        if tune_last_n_layers > 0:
            for layer in self.bert.encoder.layer[-tune_last_n_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True
        # the pooler is never used (we read last_hidden_state[:, 0]); keep it frozen
        if getattr(self.bert, "pooler", None) is not None:
            for p in self.bert.pooler.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask, return_cls: bool = True):
        last = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return last[:, 0, :] if return_cls else last


def _batch_index(data, x):
    if isinstance(data, GeoBatch) and getattr(data, "ptr", None) is not None:
        counts = data.ptr[1:] - data.ptr[:-1]
        batch_idx = torch.arange(counts.size(0), device=x.device).repeat_interleave(counts)
        return batch_idx, counts.size(0), data.ptr
    batch_idx = x.new_zeros(x.size(0), dtype=torch.long)
    return batch_idx, 1, torch.tensor([0, x.size(0)], device=x.device)


def shuffle_edges_within_graph(edge_index, batch_idx, ptr, *, seed: int = 0, graph_ids=None):
    """Random-edge control: keep every edge's source and relation, replace its
    destination by a uniformly random node of the same graph.

    Deterministic per sample: the destinations of graph b are drawn from a
    generator seeded by (seed, graph id), where graph id is ``data.idx`` when
    present, so training, per-epoch validation and a later ``--mode test``
    all evaluate the same shuffled graph."""
    if edge_index.numel() == 0:
        return edge_index
    src = edge_index[0]
    g = batch_idx[src]
    dst = torch.empty_like(src)
    n_graphs = ptr.numel() - 1
    for b in range(n_graphs):
        m = g == b
        k = int(m.sum())
        if k == 0:
            continue
        n = max(1, int(ptr[b + 1] - ptr[b]))
        gid = int(graph_ids[b]) if graph_ids is not None else b
        gen = torch.Generator(device="cpu").manual_seed((seed * 1_000_003 + gid) % (2 ** 63 - 1))
        r = torch.randint(0, n, (k,), generator=gen).to(edge_index.device)
        dst[m] = ptr[b] + r
    return torch.stack([src, dst], dim=0)


class BertRGCN(nn.Module):
    def __init__(self, *, node_feat_dim: int = 768, hidden_dim: int = 200, num_rgcn_layers: int = 6,
                 num_relations: int = 3, num_node_labels: int = NUM_NODE_LABELS, label_emb_dim: int = 32,
                 fusion_layers: int = 3, num_heads: int = 4, dropout: float = 0.1, rgcn_dropout: float = 0.1,
                 rgcn_residual: bool = True, input_norm: bool = True, use_node_labels: bool = True,
                 use_gate: bool = True, gate_temperature: float = 1.0, cls_skip: bool = True,
                 graph_only: bool = False, shuffle_edges: bool = False, tune_last_n_layers: int = 2,
                 max_length: int = 512, model_name: str = CODEBERT, device=None, seed: int = 0):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.hidden_dim = hidden_dim
        self.graph_only = graph_only
        self.cls_skip = cls_skip and not graph_only
        self.shuffle_edges = shuffle_edges
        self.shuffle_seed = int(seed)
        self.max_length = max_length
        self.rgcn_residual = rgcn_residual
        self.use_node_labels = use_node_labels

        # --- text branch ---
        if not graph_only:
            self.func_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.func_encoder = CodeBERTEncoder(model_name=model_name, tune_last_n_layers=tune_last_n_layers)
            code_dim = self.func_encoder.hidden_size
        else:
            self.func_tokenizer = None
            self.func_encoder = None
            code_dim = 0

        # --- graph branch ---
        self.input_norm = nn.LayerNorm(node_feat_dim) if input_norm else nn.Identity()
        self.label_emb = nn.Embedding(num_node_labels, label_emb_dim) if use_node_labels else None
        in_dim = node_feat_dim + (label_emb_dim if use_node_labels else 0)
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.rgcn_layers = nn.ModuleList([RGCNConv(hidden_dim, hidden_dim, num_relations)
                                          for _ in range(num_rgcn_layers)])
        self.rgcn_layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_rgcn_layers)]) \
            if rgcn_residual else None
        self.rgcn_dropout = nn.Dropout(rgcn_dropout)
        self.rgcn_norm = nn.LayerNorm(hidden_dim)

        # --- fusion / head ---
        if graph_only:
            self.fusion = None
            head_in = hidden_dim
        else:
            self.fusion = GraphCodeFusion(
                code_dim=code_dim, graph_dim=hidden_dim, proj_dim=hidden_dim, num_heads=num_heads,
                ffn_hidden_dim=hidden_dim * 4, fusion_out_dim=hidden_dim, dropout=dropout,
                attn_layers=fusion_layers, use_gate=use_gate, gate_temperature=gate_temperature,
            )
            head_in = hidden_dim + (code_dim if self.cls_skip else 0)
        self.classifier = nn.Sequential(
            nn.Linear(head_in, hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, 2),
        )
        self.last_gate = None   # [B, 2] tensor from the last forward (hybrid with gate) or None

    # -- pieces --
    def encode_text(self, funcs):
        if isinstance(funcs, str):
            funcs = [funcs]
        enc = self.func_tokenizer(list(funcs), return_tensors="pt", padding=True,
                                  truncation=True, max_length=self.max_length)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        tokens = self.func_encoder(input_ids=input_ids, attention_mask=attention_mask, return_cls=False)
        return tokens, attention_mask                      # [B, L, 768], [B, L]

    def encode_graph(self, data, batch_idx, ptr):
        x = self.input_norm(data.x)
        if self.use_node_labels:
            labels = getattr(data, "node_label", None)
            if labels is None:
                labels = x.new_zeros(x.size(0), dtype=torch.long)
            x = torch.cat([x, self.label_emb(labels)], dim=-1)
        h = self.input_proj(x)
        edge_index, edge_type = data.edge_index, data.edge_type
        if self.shuffle_edges:
            gids = getattr(data, "idx", None)
            if gids is not None and not torch.is_tensor(gids):
                gids = torch.as_tensor(gids)
            if gids is not None and gids.numel() != ptr.numel() - 1:
                gids = None
            edge_index = shuffle_edges_within_graph(edge_index, batch_idx, ptr, seed=self.shuffle_seed, graph_ids=gids)
        for i, conv in enumerate(self.rgcn_layers):
            out = F.relu(conv(h, edge_index, edge_type))
            out = self.rgcn_dropout(out)
            h = self.rgcn_layer_norms[i](h + out) if self.rgcn_residual else out
        return self.rgcn_norm(h)                           # [N, hidden]

    def forward(self, data):
        batch_idx, B, ptr = _batch_index(data, data.x)
        h = self.encode_graph(data, batch_idx, ptr)
        graph_dense, graph_mask = to_dense_batch(h, batch_idx, batch_size=B)
        self.last_gate = None

        if self.graph_only:
            g_vec = masked_mean(graph_dense, graph_mask)
            return self.classifier(g_vec)

        funcs = getattr(data, "func", None)
        if funcs is None:
            raise ValueError("Batch has no `func` attribute; the hybrid model needs the source text.")
        code_emb, code_mask = self.encode_text(funcs)
        fused, gate = self.fusion(code_emb=code_emb, graph_emb=graph_dense,
                                  code_mask=code_mask, graph_mask=graph_mask)
        self.last_gate = gate
        if self.cls_skip:
            fused = torch.cat([fused, code_emb[:, 0, :]], dim=-1)
        return self.classifier(fused)


class CodeBERTOnly(nn.Module):
    """Baseline: fine-tuned CodeBERT CLS -> MLP (identical text recipe)."""

    def __init__(self, device=None, tune_last_n_layers: int = 2, max_length: int = 512,
                 hidden_dim: int = 200, model_name: str = CODEBERT):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.max_length = max_length
        self.func_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.func_encoder = CodeBERTEncoder(model_name=model_name, tune_last_n_layers=tune_last_n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(self.func_encoder.hidden_size, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2),
        )
        self.last_gate = None

    def forward(self, data):
        funcs = getattr(data, "func", None)
        if funcs is None:
            raise ValueError("Batch has no `func` attribute.")
        if isinstance(funcs, str):
            funcs = [funcs]
        enc = self.func_tokenizer(list(funcs), return_tensors="pt", padding=True,
                                  truncation=True, max_length=self.max_length)
        cls = self.func_encoder(input_ids=enc["input_ids"].to(self.device),
                                attention_mask=enc["attention_mask"].to(self.device), return_cls=True)
        return self.classifier(cls)


ABLATIONS = ("none", "no_gate", "no_cls_skip", "shuffle_edges", "no_input_norm", "no_residual")


def build_model(model_type: str, device, *, num_relations: int = 3, node_feat_dim: int = 768,
                hidden_dim: int = 200, num_rgcn_layers: int = 6, fusion_layers: int = 3,
                tune_last_n_layers: int = 2, max_length: int = 512, ablation: str = "none",
                gate_temperature: float = 1.0, rgcn_residual: bool = True, input_norm: bool = True,
                use_node_labels: bool = True, seed: int = 0):
    """model_type: hybrid | codebert | graph_only.
    ablation: none | no_gate | no_cls_skip | shuffle_edges | no_input_norm | no_residual
    (no_gate / no_cls_skip need the fusion, i.e. the hybrid; codebert takes none)."""
    if ablation not in ABLATIONS:
        raise ValueError(f"unknown ablation {ablation}")
    if model_type == "codebert":
        if ablation != "none":
            raise ValueError("ablations apply to hybrid / graph_only only; the CodeBERT baseline has no graph branch")
        return CodeBERTOnly(device=device, tune_last_n_layers=tune_last_n_layers,
                            max_length=max_length, hidden_dim=hidden_dim).to(device)
    kw = dict(node_feat_dim=node_feat_dim, hidden_dim=hidden_dim, num_rgcn_layers=num_rgcn_layers,
              num_relations=num_relations, fusion_layers=fusion_layers, tune_last_n_layers=tune_last_n_layers,
              max_length=max_length, device=device, gate_temperature=gate_temperature,
              rgcn_residual=rgcn_residual, input_norm=input_norm, use_node_labels=use_node_labels, seed=seed)
    if model_type == "graph_only":
        if ablation in ("no_gate", "no_cls_skip"):
            raise ValueError(f"ablation {ablation} needs the fusion module; graph_only has none")
        kw["graph_only"] = True
    elif model_type != "hybrid":
        raise ValueError(f"unknown model_type {model_type}")
    if ablation == "no_gate":
        kw["use_gate"] = False
    elif ablation == "no_cls_skip":
        kw["cls_skip"] = False
    elif ablation == "shuffle_edges":
        kw["shuffle_edges"] = True
    elif ablation == "no_input_norm":
        kw["input_norm"] = False
    elif ablation == "no_residual":
        kw["rgcn_residual"] = False
    return BertRGCN(**kw).to(device)
