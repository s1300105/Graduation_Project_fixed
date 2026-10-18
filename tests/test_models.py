"""Forward-pass tests for every model / ablation on a synthetic batch (CPU).
Downloads microsoft/codebert-base on first run."""
import os
import torch
import pytest
from torch_geometric.data import Data, Batch

from models.LMGNN import build_model, shuffle_edges_within_graph
from utils.functions.cpg import NUM_NODE_LABELS

DEV = torch.device("cpu")


def _graph(n, seed, label):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 768, generator=g)
    src = torch.randint(0, n, (3 * n,), generator=g)
    dst = torch.randint(0, n, (3 * n,), generator=g)
    keep = src != dst
    ei = torch.stack([src[keep], dst[keep]])
    et = torch.randint(0, 3, (ei.size(1),), generator=g)
    d = Data(x=x, edge_index=ei, edge_type=et,
             node_label=torch.randint(0, NUM_NODE_LABELS, (n,), generator=g),
             y=torch.tensor([label]))
    d.num_nodes = n
    d.func = f"int f{seed}(int a) {{ return a + {seed}; }}"
    d.idx = seed
    return d


def _batch():
    return Batch.from_data_list([_graph(5, 1, 0), _graph(9, 2, 1), _graph(3, 3, 0)])


@pytest.fixture(scope="module")
def batch():
    return _batch()


@pytest.mark.parametrize("model_type,ablation", [
    ("hybrid", "none"), ("hybrid", "no_gate"), ("hybrid", "no_cls_skip"), ("hybrid", "shuffle_edges"),
    ("hybrid", "no_input_norm"), ("hybrid", "no_residual"), ("graph_only", "none"), ("codebert", "none"),
])
def test_forward_shapes(batch, model_type, ablation):
    torch.manual_seed(0)
    m = build_model(model_type, DEV, hidden_dim=32, num_rgcn_layers=2, fusion_layers=1, ablation=ablation)
    m.train()
    logits = m(batch)
    assert logits.shape == (3, 2)
    loss = torch.nn.functional.cross_entropy(logits, batch.y.view(-1))
    loss.backward()
    grads = [p.grad for n, p in m.named_parameters() if p.requires_grad and "func_encoder" not in n]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)
    if model_type == "hybrid" and ablation != "no_gate":
        assert m.last_gate is not None and m.last_gate.shape == (3, 2)
        assert torch.allclose(m.last_gate.sum(-1), torch.ones(3), atol=1e-5)
    else:
        assert m.last_gate is None
    m.eval()
    with torch.no_grad():
        assert m(batch).shape == (3, 2)


def test_hybrid_is_superset_of_baseline_input(batch):
    """With cls_skip the classifier sees the CodeBERT CLS vector directly."""
    m = build_model("hybrid", DEV, hidden_dim=32, num_rgcn_layers=1, fusion_layers=1)
    assert m.classifier[0].in_features == 32 + 768
    m2 = build_model("hybrid", DEV, hidden_dim=32, num_rgcn_layers=1, fusion_layers=1, ablation="no_cls_skip")
    assert m2.classifier[0].in_features == 32


def test_reverse_edge_relations(batch):
    m = build_model("graph_only", DEV, hidden_dim=16, num_rgcn_layers=1, num_relations=6)
    b = Batch.from_data_list([_graph(6, 7, 1)])
    b.edge_type = torch.randint(0, 6, (b.edge_index.size(1),))
    assert m(b).shape == (1, 2)


def test_shuffle_edges_keeps_edges_inside_their_graph(batch):
    bidx = batch.batch
    ei = shuffle_edges_within_graph(batch.edge_index, bidx, batch.ptr, seed=1, graph_ids=batch.idx)
    assert ei.shape == batch.edge_index.shape
    assert torch.equal(ei[0], batch.edge_index[0])
    assert torch.equal(bidx[ei[0]], bidx[ei[1]])
    # deterministic per (seed, graph id); different seed -> different destinations
    ei2 = shuffle_edges_within_graph(batch.edge_index, bidx, batch.ptr, seed=1, graph_ids=batch.idx)
    assert torch.equal(ei, ei2)
    ei3 = shuffle_edges_within_graph(batch.edge_index, bidx, batch.ptr, seed=2, graph_ids=batch.idx)
    assert not torch.equal(ei, ei3)
    # the same graph gets the same shuffle regardless of its position in the batch
    single = Batch.from_data_list([_graph(9, 2, 1)])
    e_single = shuffle_edges_within_graph(single.edge_index, single.batch, single.ptr, seed=1, graph_ids=single.idx)
    sel = bidx[batch.edge_index[0]] == 1
    assert torch.equal(ei[1][sel] - batch.ptr[1], e_single[1])


def test_shuffle_edges_model_is_deterministic_in_eval(batch):
    m = build_model("graph_only", DEV, hidden_dim=16, num_rgcn_layers=1, ablation="shuffle_edges", seed=5)
    m.eval()
    with torch.no_grad():
        a, b = m(batch), m(batch)
    assert torch.allclose(a, b)


def test_codebert_pooler_frozen_and_last_two_layers_trainable():
    m = build_model("codebert", DEV)
    enc = m.func_encoder.bert
    assert all(not p.requires_grad for p in enc.pooler.parameters())
    assert all(p.requires_grad for p in enc.encoder.layer[-1].parameters())
    assert all(not p.requires_grad for p in enc.encoder.layer[0].parameters())
