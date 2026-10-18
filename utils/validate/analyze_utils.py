# analyze_utils.py
import collections

def peek_loader(loader, max_items=3):
    """
    先頭バッチの構造・shape・dtype をざっと確認
    """
    batch = next(iter(loader))
    print("=== type(batch):", type(batch))
    if isinstance(batch, (list, tuple)):
        print(f"len(batch)={len(batch)}")
        for i, x in enumerate(batch[:max_items]):
            msg = f"[{i}] type={type(x)}"
            if hasattr(x, "shape"): msg += f" shape={tuple(x.shape)}"
            if hasattr(x, "dtype"): msg += f" dtype={getattr(x, 'dtype', None)}"
            print(msg)
    elif isinstance(batch, dict):
        for k, v in batch.items():
            msg = f"{k}: type={type(v)}"
            if hasattr(v, "shape"): msg += f" shape={tuple(v.shape)}"
            if hasattr(v, "dtype"): msg += f" dtype={getattr(v, 'dtype', None)}"
            print(msg)
    else:
        print(batch)

def describe_pyg_graph(g):
    """
    PyG Data/Batch の詳細表示（エッジタイプ分布、ノード数など）
    """
    import torch
    try:
        from torch_geometric.data import Data, Batch
    except Exception:
        print("torch_geometric が見つかりません。")
        return

    if isinstance(g, Batch):
        print(f"[Batch] num_graphs={g.num_graphs}, x={tuple(g.x.shape) if hasattr(g,'x') else None}")
        d0 = g.get_example(0) if hasattr(g, "get_example") else g.to_data_list()[0]
        print("--- first graph in batch ---")
        return describe_pyg_graph(d0)

    # Data
    if hasattr(g, "y"):
        y = g.y
        y_info = (getattr(y, "dtype", None), tuple(y.shape)) if hasattr(y, "shape") else y
        print("y:", y_info)
    if hasattr(g, "x"): print("x:", (g.x.dtype, tuple(g.x.shape)))
    if hasattr(g, "edge_index"):
        ei = g.edge_index
        print("edge_index:", (ei.dtype, tuple(ei.shape)))
        print("num_edges:", int(ei.size(1)))
    if hasattr(g, "num_nodes"): print("num_nodes:", int(g.num_nodes))

    if hasattr(g, "edge_type") and g.edge_type is not None:
        uniq, cnt = torch.unique(g.edge_type, return_counts=True)
        print("edge_type counts:", dict(zip([int(u) for u in uniq.tolist()],
                                            [int(c) for c in cnt.tolist()])))
    else:
        print("edge_type: None")

    for key in ["func", "code", "raw"]:
        if hasattr(g, key):
            val = getattr(g, key)
            if isinstance(val, str):
                preview = val[:100].replace("\n", " ")
                print(f"{key}: len={len(val)} preview={preview}...")

def summarize_graph_loader(loader, max_batches=None):
    """
    ローダ全体を走査し、平均ノード/エッジ数、edge_type分布、ラベル分布を集計
    """
    import torch
    try:
        from torch_geometric.data import Data, Batch
    except Exception:
        print("torch_geometric が見つからないため、summarize_graph_loader はスキップ。")
        return

    n_graphs = node_tot = edge_tot = 0
    etype_counter = collections.Counter()
    y_counter = collections.Counter()

    def upd_one(d: "Data"):
        nonlocal n_graphs, node_tot, edge_tot
        n_graphs += 1
        if hasattr(d, "num_nodes"): node_tot += int(d.num_nodes)
        if hasattr(d, "edge_index"): edge_tot += int(d.edge_index.size(1))
        if hasattr(d, "edge_type") and d.edge_type is not None:
            etype_counter.update(d.edge_type.detach().cpu().tolist())
        if hasattr(d, "y") and d.y is not None:
            y = d.y
            if hasattr(y, "numel") and y.numel() == 1:
                y_counter[int(y.item())] += 1
            else:
                y_counter["complex"] += 1

    for b_idx, batch in enumerate(loader, 1):
        if isinstance(batch, Batch):
            for d in batch.to_data_list(): upd_one(d)
        elif isinstance(batch, Data):
            upd_one(batch)
        elif isinstance(batch, (list, tuple)):
            for x in batch:
                if isinstance(x, Batch):
                    for d in x.to_data_list(): upd_one(d)
                elif isinstance(x, Data):
                    upd_one(x)
        if max_batches and b_idx >= max_batches:
            break

    if n_graphs == 0:
        print("グラフが検出できませんでした（PyG Data/Batch ではない可能性）。")
        return

    print("=== SUMMARY ===")
    print("graphs:", n_graphs)
    print("avg_nodes:", node_tot / n_graphs)
    print("avg_edges:", edge_tot / n_graphs)
    if etype_counter:
        print("edge_type dist (top 10):", etype_counter.most_common(10))
    if y_counter:
        print("label dist:", dict(y_counter))

def summarize_generic_loader(loader, max_batches=5):
    """
    一般テンソル/辞書/タプルのバッチ統計（shape/dtypeの出現頻度）を集計
    """
    shapes = collections.defaultdict(lambda: collections.Counter())
    dtypes = collections.defaultdict(lambda: collections.Counter())
    n_batches = 0

    def reg(name, arr):
        if hasattr(arr, "shape"): shapes[name][tuple(arr.shape)] += 1
        if hasattr(arr, "dtype"): dtypes[name][str(arr.dtype)] += 1

    for b_idx, batch in enumerate(loader, 1):
        n_batches += 1
        if isinstance(batch, (list, tuple)):
            for i, x in enumerate(batch):
                reg(f"#{i}", x)
        elif isinstance(batch, dict):
            for k, v in batch.items():
                reg(k, v)
        else:
            reg("batch", batch)
        if b_idx >= max_batches: break

    print(f"=== BATCH SUMMARY (first {n_batches} batches) ===")
    for name, cnt in shapes.items():
        print(f"{name} shapes (top3):", cnt.most_common(3))
    for name, cnt in dtypes.items():
        print(f"{name} dtypes (top3):", cnt.most_common(3))
