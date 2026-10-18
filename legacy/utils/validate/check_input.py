import pandas as pd
import torch
import json
from torch_geometric.data import Data
from pathlib import Path

in_dir = Path("/home/yudai/Project/research/Graduation_Project/data/input")
files = sorted(in_dir.glob("*_cpg_input.pkl"))

# ----- ① pkl読み込み -----
pkl = files[0]  # df0 or df1
df = pd.read_pickle(pkl)

# ----- ② ランダムに30件取得 -----
df30 = df.sample(30, random_state=42).reset_index(drop=True)

# ===== 共通ユーティリティ =====
def _infer_num_nodes(d: Data) -> int:
    n = getattr(d, "num_nodes", None)
    if isinstance(n, int) and n >= 0:
        return n
    if hasattr(d, "x") and d.x is not None and torch.is_tensor(d.x) and d.x.dim() >= 1:
        return int(d.x.size(0))
    if hasattr(d, "edge_index") and d.edge_index is not None and torch.is_tensor(d.edge_index) and d.edge_index.numel() > 0:
        return int(torch.max(d.edge_index).item() + 1)
    return 0

@torch.no_grad()
def coalesce_by_triplet_(d: Data) -> Data:
    """
    (src, dst, edge_type) が完全一致のエッジ重複を除去（in-place）。
    edge_type が無い/長さ不一致のときは (src, dst) で統合。
    """
    if not (hasattr(d, "edge_index") and torch.is_tensor(d.edge_index)):
        return d
    ei = d.edge_index
    if ei.dim() != 2 or ei.size(0) != 2:
        return d

    E = int(ei.size(1))
    et = getattr(d, "edge_type", None)
    if et is not None and torch.is_tensor(et) and et.numel() == E:
        trip = torch.stack([ei[0], ei[1], et], dim=1)  # (E, 3)
        uniq = torch.unique(trip, dim=0)
        d.edge_index = uniq[:, :2].t().contiguous()
        d.edge_type  = uniq[:, 2].contiguous()
    else:
        pairs = torch.stack([ei[0], ei[1]], dim=1)     # (E, 2)
        uniq = torch.unique(pairs, dim=0)
        d.edge_index = uniq.t().contiguous()
        # edge_type があるが長さ不一致の場合は整合を取らない（欠損扱い）
        if et is not None and torch.is_tensor(et) and et.numel() != int(d.edge_index.size(1)):
            d.edge_type = None
    return d

# ----- ③ Data → JSON 用辞書への変換（x は入れない） -----
def data_to_dict_cleaned(d: Data):
    # まず重複エッジ統合
    coalesce_by_triplet_(d)

    out = {}
    num_nodes = _infer_num_nodes(d)
    out["num_nodes"] = int(num_nodes)

    # ✅ x（埋め込み）は入れない
    out["x"] = None

    # edge_index（GPU上でも tolist できるよう CPU に移動）
    if hasattr(d, "edge_index") and d.edge_index is not None:
        out["edge_index"] = d.edge_index.detach().cpu().tolist()
    else:
        out["edge_index"] = None

    # edge_type
    if hasattr(d, "edge_type") and d.edge_type is not None:
        out["edge_type"] = d.edge_type.detach().cpu().tolist()
    else:
        out["edge_type"] = None

    # line / end_line / column / end_column
    for key in ["line", "end_line", "column", "end_column"]:
        val = getattr(d, key, None)
        if val is not None and torch.is_tensor(val):
            out[key] = val.detach().cpu().tolist()
        else:
            out[key] = None

    return out

# ----- ④ DataFrame 1行 → dict -----
def row_to_dict(row):
    d: Data = row["input"]
    return {
        "target": int(row["target"]),
        "func": row["func"],
        "input": data_to_dict_cleaned(d)
    }

# ----- ⑤ JSON 用リストへ変換 -----
json_ready = [row_to_dict(df30.iloc[i]) for i in range(len(df30))]

# ----- ⑥ JSON 保存 -----
out_path = "data/inspect/random30_no_x_coalesced.json"
Path("data/inspect").mkdir(parents=True, exist_ok=True)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(json_ready, f, indent=2, ensure_ascii=False)

print("[saved]", out_path)
