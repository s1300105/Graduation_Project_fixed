# utils/validate/visualize_graph.py
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import networkx as nx
from torch_geometric.utils import to_networkx, k_hop_subgraph, degree


def _safe_num_edges(data) -> int:
    if getattr(data, "edge_index", None) is None:
        return 0
    try:
        return int(data.edge_index.size(1))
    except Exception:
        return 0


def _pick_center_node(data, strategy: str = "max_degree", fallback: int = 0) -> int:
    """
    可視化に適した中心ノードを自動選択。
    - max_degree: 次数が最大のノード
    - first_edge: 辺があるなら最初の辺の片方
    - fallback: それでも無理なら fallback（既定0）
    """
    n = int(getattr(data, "num_nodes", 0) or 0)
    if n <= 0:
        return 0

    m = _safe_num_edges(data)
    if m == 0:
        # 無辺：とりあえず0〜n-1の範囲に収めて返す
        return min(max(fallback, 0), n - 1)

    if strategy == "max_degree":
        # 無向次数を近似：送受の合計
        d = degree(data.edge_index[0], num_nodes=n) + degree(data.edge_index[1], num_nodes=n)
        return int(torch.argmax(d).item())

    if strategy == "first_edge":
        u = int(data.edge_index[0, 0].item())
        return u

    return min(max(fallback, 0), n - 1)


def to_nx_graph(data) -> "nx.Graph":
    """
    PyG Data -> NetworkX Graph（無向）。edge_type があればエッジ属性に付与。
    ノード属性には idx と x の次元（x_dim）だけを持たせる（重くしないため）。
    """
    has_etype = hasattr(data, "edge_type") and (getattr(data, "edge_type") is not None)
    G = to_networkx(
        data,
        node_attrs=[],
        edge_attrs=["edge_type"] if has_etype else None,
        to_undirected=True,
    )

    for i in G.nodes:
        G.nodes[i]["idx"] = int(i)

    if hasattr(data, "x") and data.x is not None:
        dim = int(data.x.size(-1))
        for i in G.nodes:
            G.nodes[i]["x_dim"] = dim

    # edge_type がテンソル以外/floatのときも int に寄せる
    if has_etype and G.number_of_edges() > 0:
        for u, v, d in G.edges(data=True):
            try:
                d["edge_type"] = int(d.get("edge_type", -1))
            except Exception:
                d["edge_type"] = -1
    return G


def sample_k_hop(
    data,
    center_node: int = 0,
    num_hops: int = 2,
    max_nodes: int = 150,
):
    """
    大きいグラフを見やすくするため、中心ノードから k-hop サブグラフを抽出。
    無辺/小規模でも安全に動作。
    """
    n = int(getattr(data, "num_nodes", 0) or 0)
    m = _safe_num_edges(data)
    center_node = min(max(center_node, 0), max(n - 1, 0))

    if n == 0:
        return data  # 何もできない
    if m == 0:
        # 無辺：ノードだけ間引く
        from torch_geometric.data import Data
        sub = Data()
        if hasattr(data, "x") and data.x is not None:
            sub.x = data.x[: min(n, max_nodes)]
        sub.num_nodes = min(n, max_nodes)
        # 任意属性はそのまま
        for k in ("y", "func"):
            if hasattr(data, k):
                setattr(sub, k, getattr(data, k))
        return sub

    # 通常ケース：k-hop
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        center_node, num_hops, data.edge_index, relabel_nodes=True
    )

    from torch_geometric.data import Data
    sub = Data()
    sub.edge_index = edge_index
    sub.num_nodes = int(subset.numel())

    if hasattr(data, "x") and data.x is not None:
        sub.x = data.x[subset]
    if hasattr(data, "edge_type") and data.edge_type is not None:
        sub.edge_type = data.edge_type[edge_mask]
    if hasattr(data, "y"):
        sub.y = data.y
    if hasattr(data, "func"):
        sub.func = getattr(data, "func")

    # さらに多すぎる場合は頭から max_nodes に丸める（簡便）
    if sub.num_nodes > max_nodes:
        keep = torch.arange(max_nodes, device=edge_index.device if edge_index.is_cuda else None)
        mask = torch.isin(edge_index[0], keep) & torch.isin(edge_index[1], keep)
        sub.edge_index = edge_index[:, mask]
        if hasattr(sub, "edge_type") and sub.edge_type is not None:
            sub.edge_type = sub.edge_type[mask]
        if hasattr(sub, "x") and sub.x is not None:
            sub.x = sub.x[:max_nodes]
        sub.num_nodes = max_nodes

    return sub


def plot_pyg_graph(
    data,
    out_path: str = "data/inspect/graph.png",
    use_khop: bool = True,
    center_node: Optional[int] = None,   # None のときは自動選択
    num_hops: int = 2,
    max_nodes: int = 150,
    edge_type_labels: Optional[Dict[int, str]] = None,
    title: Optional[str] = None,
    auto_center: bool = True,            # 孤立ノード回避の自動中心選択
    show_node_ids: bool = True,
):
    """
    PyG の Data / Batch からグラフ画像を保存（matplotlib）。色は指定しない（デフォルト）。
    - 大規模グラフ: k-hop で見やすく
    - 無辺/孤立ノード: 自動で可視性の高い中心を選択
    - edge_type があれば線種・太さで区別し、凡例（任意ラベル）を表示
    """
    import matplotlib.pyplot as plt
    from torch_geometric.data import Batch

    # Batch → 先頭のグラフへ
    if isinstance(data, Batch):
        data = data.get_example(0) if hasattr(data, "get_example") else data.to_data_list()[0]

    # 中心ノードの自動選択
    if use_khop:
        if center_node is None and auto_center:
            center_node = _pick_center_node(data, strategy="max_degree", fallback=0)
        elif center_node is None:
            center_node = 0
        data = sample_k_hop(
            data,
            center_node=center_node,
            num_hops=num_hops,
            max_nodes=max_nodes,
        )

    # NetworkX へ
    G = to_nx_graph(data)

    # レイアウト（固定乱数で安定化）
    pos = nx.spring_layout(G, seed=42)

    # 単一図
    plt.figure(figsize=(8, 6))

    # ノード描画
    nx.draw_networkx_nodes(G, pos)
    if show_node_ids:
        nx.draw_networkx_labels(G, pos, labels={n: n for n in G.nodes}, font_size=7)

    # エッジ描画（edge_type ごとに線種/太さを変える。色はデフォルト）
    if G.number_of_edges() > 0:
        if any("edge_type" in G.edges[e] for e in G.edges):
            type_to_style = {}
            styles = ["solid", "dashed", "dotted", "dashdot"]
            widths = [1.0, 1.3, 1.6, 1.9]

            for (_, _, d) in G.edges(data=True):
                et = int(d.get("edge_type", -1))
                if et not in type_to_style:
                    idx = len(type_to_style) % len(styles)
                    type_to_style[et] = (styles[idx], widths[idx])

            for et, (style, width) in type_to_style.items():
                edgelist = [(u, v) for (u, v, d) in G.edges(data=True) if int(d.get("edge_type", -1)) == et]
                nx.draw_networkx_edges(G, pos, edgelist=edgelist, style=style, width=width, alpha=0.9, arrows=False)

            # 簡易凡例
            if edge_type_labels:
                from matplotlib.lines import Line2D
                handles = [
                    Line2D([0], [0], linestyle=style, linewidth=width, label=edge_type_labels.get(et, f"etype {et}"))
                    for et, (style, width) in type_to_style.items()
                ]
                if handles:
                    plt.legend(handles=handles, loc="best", fontsize=8)
        else:
            nx.draw_networkx_edges(G, pos, arrows=False)

    if title is None:
        title = f"Graph (nodes={G.number_of_nodes()}, edges={G.number_of_edges()})"
    plt.title(title)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path

