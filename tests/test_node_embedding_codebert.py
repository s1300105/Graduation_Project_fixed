"""Runs the real CodeBERT node embedder (mode="codebert") on the 75-node sample. Slow-ish (CPU)."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from utils.functions.cpg import parse_to_functions
from utils.process import embeddings as E

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "check_data" / "cpg_head_check.json"


@pytest.fixture(scope="module")
def sample_nodes():
    row = json.loads(SAMPLE.read_text(encoding="utf-8").splitlines()[0])
    return parse_to_functions(row["cpg"], max_nodes=500)[0]


def test_codebert_node_features_shape_cache_and_code_modes(sample_nodes, monkeypatch):
    monkeypatch.delenv("USE_FAST_NODES_EMBEDDING", raising=False)
    emb = E.NodesEmbedding(nodes_dim=500, mode="codebert")
    x = emb(sample_nodes).numpy()
    assert x.shape == (75, 768) and np.isfinite(x).all()
    codes = [n.get_code() for n in sample_nodes.values()]
    assert len(emb._cache) == len(set(codes))            # one CLS per distinct CODE string
    # identical CODE strings get identical vectors regardless of position (context-free)
    dup = [c for c in set(codes) if codes.count(c) > 1][0]
    rows = [i for i, c in enumerate(codes) if c == dup]
    assert np.allclose(x[rows[0]], x[rows[1]])
    # first_line mode shortens container nodes (BLOCK = whole body in full mode)
    block = next(n for n in sample_nodes.values() if n.label == "BLOCK")
    assert "\n" in block.get_code() and "\n" not in E.node_code_text(block, "first_line")
    emb2 = E.NodesEmbedding(nodes_dim=500, mode="codebert", code_mode="first_line")
    x2 = emb2(sample_nodes).numpy()
    assert x2.shape == (75, 768)
    i_block = list(sample_nodes.values()).index(block)
    assert not np.allclose(x[i_block], x2[i_block])
