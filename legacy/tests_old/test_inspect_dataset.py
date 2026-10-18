# tests/test_inspect_dataset.py など
from pathlib import Path
import sys
from utils.validate.visualize_graph import plot_pyg_graph
from torch_geometric.data import Batch
from utils.validate.code_map import render_code_and_graph_html


# --- プロジェクトルートを import より先に通す（重要） ---
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))  # append ではなく insert(0) が確実

# --- ここからプロジェクト内モジュールをimport ---
from utils.data.datamanager import loads, train_val_test_split
from utils.validate.save_utils import (
    save_split_from_loaders,
    cache_loader_items,
    save_loader_sample_json,
)
from utils.validate.analyze_utils import (
    peek_loader,
    summarize_generic_loader,
    summarize_graph_loader,
    describe_pyg_graph,
)
import configs


# ルート基準の安全な出力パス
DATA_DIR = ROOT_DIR / "data"
SPLIT_JSON = DATA_DIR / "splits" / "split.json"
INSPECT_JSON = DATA_DIR / "inspect" / "train_sample.json"
CACHE_DIR = DATA_DIR / "cache"

def inspect():
    # ---- 設定とデータ読込 ----
    PATHS = configs.Paths()
    FILES = configs.Files()  # 未使用でも初期化しておく
    context = configs.Process()

    input_dataset = loads(PATHS.input)

    # ---- 分割 & ローダ作成 ----
    train_loader, val_loader, test_loader = list(
        map(
            lambda x: x.get_loader(context.batch_size, shuffle=context.shuffle),
            train_val_test_split(input_dataset, shuffle=context.shuffle),
        )
    )

    # ---- 分割インデックスの保存（再現性◎） ----
    save_split_from_loaders(
        train_loader, val_loader, test_loader,
        input_dataset_len=len(input_dataset),
        out_path=str(SPLIT_JSON),
        batch_size=context.batch_size,
        shuffle=context.shuffle,
    )

    # ---- 目視確認用に小サンプルをJSONへ ----
    save_loader_sample_json(
        train_loader,
        out_path=str(INSPECT_JSON),
        max_batches=1,
    )

    # （任意）実体キャッシュ：次回ロード高速化
    # CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # cache_loader_items(train_loader, str(CACHE_DIR / "train.pt"))
    # cache_loader_items(val_loader,   str(CACHE_DIR / "val.pt"))
    # cache_loader_items(test_loader,  str(CACHE_DIR / "test.pt"))

    # ---- クイック分析 ----
    print("\n[Peek] ---- 先頭バッチの構造 ----")
    peek_loader(train_loader)

    print("\n[Generic Summary] ---- 形状/型の頻度 ----")
    summarize_generic_loader(train_loader, max_batches=3)

    print("\n[Graph Summary] ---- グラフ統計 ----")
    try:
        summarize_graph_loader(train_loader, max_batches=50)
    except Exception as e:
        print("⚠️ summarize_graph_loader skipped:", e)

    # ---- 代表グラフの詳細 & input/func の中身 ----
    try:
        first_batch = next(iter(train_loader))

        # PyG Batchであれば詳細
        try:
            from torch_geometric.data import Batch, Data
            if isinstance(first_batch, Batch):
                print("\n[Graph Detail] ---- 代表グラフ ----")
                describe_pyg_graph(first_batch)
                # input/funcプレビュー（バッチ0番目を例に）
                d0 = first_batch.get_example(0) if hasattr(first_batch, "get_example") else first_batch.to_data_list()[0]
                if hasattr(d0, "func") and isinstance(d0.func, str):
                    preview = d0.func[:300].replace("\n", " ")
                    print(f"\n[func preview] len={len(d0.func)}: {preview}...")
                if hasattr(d0, "x"):
                    print(f"[input.x] shape={tuple(d0.x.shape)}, dtype={getattr(d0.x, 'dtype', None)}")
                if hasattr(d0, "edge_index"):
                    ei = d0.edge_index
                    print(f"[input.edge_index] shape={tuple(ei.shape)} num_edges={int(ei.size(1))}")
                if hasattr(d0, "edge_type") and d0.edge_type is not None:
                    try:
                        import torch
                        uniq, cnt = torch.unique(d0.edge_type, return_counts=True)
                        dist = dict(zip([int(u) for u in uniq.tolist()], [int(c) for c in cnt.tolist()]))
                        print(f"[input.edge_type] counts={dist}")
                    except Exception:
                        pass
            else:
                # Batchじゃない（タプル/辞書など）場合の軽い可視化
                print("\n[Info] first_batch is not a PyG Batch. describe_pyg_graph is skipped.")
        except Exception as e:
            print("⚠️ PyG not available or describe step failed:", e)

    except StopIteration:
        print("⚠️ train_loader is empty.")

    print(f"\n✅ Saved split JSON -> {SPLIT_JSON}")
    print(f"✅ Saved sample JSON -> {INSPECT_JSON}")

    first_batch = next(iter(train_loader))
    out_png = plot_pyg_graph(
        first_batch,
        out_path="data/inspect/graph_khop.png",
        use_khop=True,
        center_node=0,
        num_hops=2,
        max_nodes=150,
        edge_type_labels={0: "AST", 1: "CFG", 2: "PDG"} 
    )
    print("✅ saved:", out_png)
    save_n_graph_images(train_loader, n=10, out_dir="data/inspect/graphs")

    g = first_batch.get_example(0) if isinstance(first_batch, Batch) else first_batch

    # もし node_lines が既にあるなら渡す（例: {node_id: [line1, line2...]})
    # なければ data.node_line / node_end_line があれば自動で推測利用します。
    out = render_code_and_graph_html(
        g,
        out_html="data/inspect/graph_with_code.html",
        edge_type_labels={0:"AST", 1:"CFG", 2:"PDG"}
    )
    print("✅ wrote:", out)

    save_n_code_graph_htmls(train_loader, n=10, out_dir="data/inspect/html")



def _num_edges(g):
    if getattr(g, "edge_index", None) is None:
        return 0
    try:
        return int(g.edge_index.size(1))
    except Exception:
        return 0

def _iter_graphs(loader):
    """DataLoader -> 個々の Data を順に取り出すジェネレータ"""
    for batch in loader:
        if isinstance(batch, Batch):
            for g in batch.to_data_list():
                yield g
        else:
            yield batch

def save_n_graph_images(loader, n=10, out_dir="data/inspect/graphs"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for g in _iter_graphs(loader):
        # 辺が無いグラフはスキップ（足りなければ後で埋める）
        if _num_edges(g) == 0:
            continue
        out_png = out_dir / f"graph_{saved:02d}.png"
        plot_pyg_graph(
            g,
            out_path=str(out_png),
            use_khop=True,
            center_node=None,        # 自動で「良い中心」選択
            num_hops=2,
            max_nodes=150,
            edge_type_labels={0: "AST", 1: "CFG", 2: "PDG"},
            auto_center=True,
            show_node_ids=True,
        )
        print("✅ saved:", out_png)
        saved += 1
        if saved >= n:
            break

    # まだ枚数が足りなければ、無辺グラフも使って埋める（参考程度）
    if saved < n:
        for g in _iter_graphs(loader):
            out_png = out_dir / f"graph_{saved:02d}.png"
            plot_pyg_graph(
                g, out_path=str(out_png), use_khop=False,  # k-hop無しでそのまま
                edge_type_labels={0: "AST", 1: "CFG", 2: "PDG"},
                show_node_ids=True,
            )
            print("⚠️ no-edges graph saved:", out_png)
            saved += 1
            if saved >= n:
                break

    print(f"\n===> total saved: {saved} images to {out_dir}")


# tests/test_inspect_dataset.py の下部（save_n_graph_images の下あたり）に追加
def save_n_code_graph_htmls(loader, n=10, out_dir="data/inspect/html"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for g in _iter_graphs(loader):
        # func（元コード）が無い/空ならスキップ
        func = getattr(g, "func", None)
        if not isinstance(func, str) or len(func.strip()) == 0:
            continue

        out_html = out_dir / f"graph_with_code_{saved+1:02d}.html"
        render_code_and_graph_html(
            g,
            out_html=str(out_html),
            edge_type_labels={0: "AST", 1: "CFG", 2: "PDG"},
            title=f"Graph & Code #{saved+1}",
        )
        print("✅ saved:", out_html)
        saved += 1
        if saved >= n:
            break

    # 簡易 index.html を作成
    if saved > 0:
        index_path = out_dir / "index.html"
        links = "\n".join(
            f"<li><a href='graph_with_code_{i:02d}.html'>graph_with_code_{i:02d}.html</a></li>"
            for i in range(1, saved + 1)
        )
        index_html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Graphs</title></head>
<body><h2>Graph & Code (total={saved})</h2><ol>{links}</ol></body></html>"""
        index_path.write_text(index_html, encoding="utf-8")
        print("📄 index:", index_path)

    print(f"\n===> total saved: {saved} html files to {out_dir}")




if __name__ == "__main__":
    inspect()
