# tests/test_embed_generator.py
import sys, os, time, traceback
from pathlib import Path

# プロジェクトルートを import パスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    import run  # your run.py (configs, data, process, etc.)

    t0 = time.time()
    print("== Embed generation start ==")
    print("CPG dir        :", run.PATHS.cpg)
    print("TOKENS out dir :", run.PATHS.tokens)
    print("INPUT  out dir :", run.PATHS.input)

    # 実行
    try:
        run.Embed_generator()
    except Exception:
        print("\n[ERROR] Embed_generator failed:")
        traceback.print_exc()
        sys.exit(1)

    dt = time.time() - t0
    print(f"\n== Done in {dt:.1f}s ==")

    # ===== 出力検証 =====
    import pandas as pd
    import torch
    from torch_geometric.data import Data

    # 期待されるファイル名パターン: {file_name}_{FILES.input}
    input_dir = Path(run.PATHS.input)
    tokens_dir = Path(run.PATHS.tokens)

    # INPUTファイル列挙（サフィックス指定と .pkl の両方を拾う）
    input_suffix = getattr(run.FILES, "input", "")
    candidates = []
    if input_suffix:
        candidates.extend(input_dir.glob(f"*_{input_suffix}"))
    candidates.extend(input_dir.glob("*.pkl"))
    input_files = sorted(set(candidates))

    print(f"\n[REPORT] Found INPUT files: {len(input_files)}")
    for i, p in enumerate(input_files[:5], 1):
        print(f"  {i:>2}. {p.name}")

    # TOKENS 側も確認（存在チェックのみ）
    tokens_suffix = getattr(run.FILES, "tokens", "")
    tok_candidates = []
    if tokens_suffix:
        tok_candidates.extend(tokens_dir.glob(f"*_{tokens_suffix}"))
    tok_candidates.extend(tokens_dir.glob("*.pkl"))
    token_files = sorted(set(tok_candidates))
    print(f"\n[REPORT] Found TOKENS files: {len(token_files)}")

    if not input_files:
        print("\n[WARN] No INPUT files found. Nothing to validate.")
        return

    # ===== サンプル1件ロードしてスキーマ検査 =====
    sample_path = input_files[0]
    print(f"\n[CHECK] Load sample INPUT: {sample_path.name}")

    df = pd.read_pickle(sample_path)

    # 必須カラムの存在チェック
    required_cols = {"input", "target", "func"}
    missing = required_cols - set(df.columns)
    if missing:
        raise AssertionError(f"Missing columns in INPUT df: {missing}")

    print(f"[OK] Columns present: {list(df.columns)}")
    print(f"[OK] DataFrame shape: {df.shape}")

    # 1行取り出して Data 構造をチェック
    row0 = df.iloc[0]
    tg_data = row0["input"]
    assert isinstance(tg_data, Data), f"`input` must be torch_geometric.data.Data, got {type(tg_data)}"

    # shapes / dtypes の確認
    x = tg_data.x               # ノード特徴 [nodes_dim, feat_dim]
    edge_index = tg_data.edge_index  # [2, E]
    y = tg_data.y               # [1]

    print("\n[SCHEMA]")
    print(f"  x.shape         : {tuple(x.shape)}  dtype={x.dtype}")
    print(f"  edge_index.shape: {tuple(edge_index.shape)}  dtype={edge_index.dtype}")
    print(f"  y.shape         : {tuple(y.shape)}  dtype={y.dtype}")

    # 期待仕様に沿った軽い検証
    # nodes_dim は configs.Embed().nodes_dim
    nodes_dim = getattr(run.configs.Embed(), "nodes_dim")
    if x.shape[0] != nodes_dim:
        print(f"[WARN] x first dim != nodes_dim (x={x.shape[0]}, nodes_dim={nodes_dim}) "
              f"→ パディング仕様/保存時の整形を確認してください。")

    # edge_index の先頭次元は 2 のはず（COO）
    assert edge_index.dim() == 2 and edge_index.shape[0] == 2, \
        f"edge_index must be [2, E], got {tuple(edge_index.shape)}"

    # 型の軽いチェック
    assert x.dtype.is_floating_point, "x must be float tensor"
    assert edge_index.dtype in (torch.int64, torch.long), "edge_index must be long tensor"
    assert y.numel() == 1, "y must be scalar-like (shape [1])"

    # func テキストの存在（表示は一部）
    func_preview = str(row0["func"])[:120].replace("\n", " ")
    print(f"\n[FUNC] preview: {func_preview}...")

    # target の値域（0/1 想定）
    print(f"[TARGET] {row0['target']}")

    print("\n[OK] INPUT schema looks consistent with parse_to_nodes / nodes_to_input pipeline.")

if __name__ == "__main__":
    main()
