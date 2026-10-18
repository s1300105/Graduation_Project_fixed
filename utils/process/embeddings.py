import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

# transformers は codebert モードのときだけ遅延ロード
try:
    from transformers import RobertaTokenizer, RobertaModel
except Exception:
    RobertaTokenizer = None
    RobertaModel = None


_NODES_EMBEDDERS = {}

def get_nodes_embedding(nodes_dim: int, *, mode: str = "codebert"):
    """
    nodes_dim と mode ごとに NodesEmbedding を 1 回だけ生成して使い回す。
    """
    key = (int(nodes_dim), mode)
    if key not in _NODES_EMBEDDERS:
        _NODES_EMBEDDERS[key] = NodesEmbedding(
            nodes_dim=int(nodes_dim),
            max_len=128,
            batch_size=32,
            dtype=torch.float16,
            mode=mode,
        )
    return _NODES_EMBEDDERS[key]


# =========================
# Nodes → node feature
# =========================
class NodesEmbedding:
    """
    ノード特徴を作るクラス。
    - mode="codebert": CodeBERT (CLS) を使う本番モード
    - mode="fast"    : 形状を維持したダミー特徴（超高速・検証用）
    出力形状は [N, 1+768] (先頭1次元は type_id)
    """
    def __init__(
        self,
        nodes_dim: int,
        *,
        max_len: int = 128,
        batch_size: int = 32,
        dtype=torch.float16,
        mode: str = "codebert",
    ):
        self.nodes_dim = int(nodes_dim)
        assert self.nodes_dim >= 0

        # 環境変数で fast を強制（例：USE_FAST_NODES_EMBEDDING=1）
        env_fast = os.environ.get("USE_FAST_NODES_EMBEDDING", "").strip() in {"1", "true", "True"}
        if env_fast:
            mode = "fast"
        self.mode = mode

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # 出力形状: [type_id(1) + 768] = 769
        self.bert_dim = 768
        self.feat_dim = 1 + self.bert_dim

        # パディング用の固定バッファ（CPU, fp32）
        self.target = torch.zeros(self.nodes_dim, self.feat_dim, dtype=torch.float32)

        # 推論パラメータ
        self.max_len = int(max_len)
        self.batch_size = int(batch_size)

        # CodeBERT モードのみ重い初期化
        self._cache = {}
        self._is_codebert = (self.mode == "codebert")
        if self._is_codebert:
            if RobertaTokenizer is None or RobertaModel is None:
                raise RuntimeError("transformers が読み込めません。fastモードを使うか、transformers をインストールしてください。")
            self.tokenizer_bert = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
            self.bert_model = RobertaModel.from_pretrained(
                "microsoft/codebert-base",
                torch_dtype=self.dtype if self.device.type == "cuda" else None,
                low_cpu_mem_usage=True,
            ).to(self.device)
            self.bert_model.eval()
        else:
            self.tokenizer_bert = None
            self.bert_model = None

    def __call__(self, nodes):
        if self._is_codebert:
            emb_np = self._embed_nodes_codebert(nodes)   # np.array [N, feat_dim]
        else:
            emb_np = self._embed_nodes_fast(nodes)       # np.array [N, feat_dim]

        nodes_tensor = torch.from_numpy(emb_np).float()  # CPU, fp32

        

        return nodes_tensor

    # ========= fastモード =========
    def _embed_nodes_fast(self, nodes):
        """
        CodeBERTを使わず、軽量な決定的特徴で埋めた [N,769] を返す。
        [0]=type_id, [1]=len(code), [2]=line(あれば), [3]=column(あれば), [4]=hash(code[:4])、残り0
        """
        node_items = list(nodes.items())
        N = len(node_items)
        out = np.zeros((N, self.feat_dim), dtype=np.float32)
        for i, (_, node) in enumerate(node_items):
            code = node.get_code() or ""
            type_id = getattr(node, "type", 0)
            # line/column は無ければ 0
            try:
                line = node.get_line_number() or 0
            except Exception:
                line = getattr(node, "line_number", 0) or 0
            try:
                col = node.get_column_number() or 0
            except Exception:
                col = getattr(node, "column_number", 0) or 0

            out[i, 0] = float(type_id)
            out[i, 1] = float(len(code))
            out[i, 2] = float(line)
            out[i, 3] = float(col)
            h = int.from_bytes(code.encode("utf-8")[:4].ljust(4, b"\0"), "little")
            out[i, 4] = float(h % 1_000_003)
        return out

    # ========= CodeBERTモード =========
    def _batch_encode(self, texts):
        return self.tokenizer_bert(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
        )

    def _embed_nodes_codebert(self, nodes):
        node_items = list(nodes.items())
        codes, type_ids = [], []
        for _, node in node_items:
            code = node.get_code()
            codes.append(code if isinstance(code, str) else str(code))
            type_ids.append(getattr(node, "type", 0))

        feats = []
        with torch.inference_mode():
            for i in range(0, len(codes), self.batch_size):
                batch_codes = codes[i: i + self.batch_size]

                # キャッシュミスだけ推論
                miss_texts = [c for c in batch_codes if c not in self._cache]
                if miss_texts:
                    enc = self._batch_encode(miss_texts)
                    enc = {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}
                    if self.device.type == "cuda":
                        with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                            hidden = self.bert_model(**enc).last_hidden_state  # [B,L,H]
                    else:
                        hidden = self.bert_model(**enc).last_hidden_state
                    cls = hidden[:, 0]                         # [B,H]
                    cls_np = cls.detach().cpu().numpy()
                    # キャッシュへ保存（必要なら上限を設ける）
                    for j, c in enumerate(miss_texts):
                        self._cache[c] = cls_np[j]
                    # 一時テンソル開放
                    del enc, hidden, cls
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

                # バッチ全要素をキャッシュから取り出し
                for c in batch_codes:
                    feats.append(self._cache[c])

        feats = np.stack(feats, axis=0).astype(np.float32)          # [N, 768]
        type_arr = np.asarray(type_ids, dtype=np.float32).reshape(-1, 1)
        emb = np.concatenate([type_arr, feats], axis=1)             # [N, 1+768]
        return emb


# =========================
# Edges (multi-relational)
# =========================
EDGE_MAP = {
    "AST": 0, "AST_EDGE": 0,
    "CFG": 1, "CFG_EDGE": 1,
    "PDG": 2, "REACHING_DEF": 2, "DATA_FLOW": 2, "DDG": 2,
    # 必要なら追加
}

def build_edges_all(nodes):
    src, dst, et = [], [], []
    for _, node in nodes.items():
        u = node.order
        for _, e in (node.edges or {}).items():
            t = e.type
            if t not in EDGE_MAP:
                continue
            if e.node_in in nodes and nodes[e.node_in].order != u:
                src.append(nodes[e.node_in].order); dst.append(u);  et.append(EDGE_MAP[t])
            if e.node_out in nodes and nodes[e.node_out].order != u:
                src.append(u); dst.append(nodes[e.node_out].order); et.append(EDGE_MAP[t])

    if not src:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long), torch.tensor(et, dtype=torch.long)


# =========================
# Packing to PyG Data
# =========================
def nodes_to_input(nodes, target, nodes_dim, *, mode: str = "codebert"):
    """
    nodes: OrderedDict[node_id -> node]
    target: int/float
    nodes_dim: 最大ノード数（パディング幅）
    mode: "codebert" | "fast"
    """
    nodes_embedding = get_nodes_embedding(nodes_dim, mode=mode)
    full_x = nodes_embedding(nodes)                       # [nodes_dim, 1+768]; 先頭 n_real 行が実ノード
    n_real = min(len(nodes), nodes_dim)
    x = full_x[:n_real]

    edge_index, edge_type = build_edges_all(nodes)

    # エッジ0本のときは自己ループでフォールバック（ノードがある場合のみ）
    if edge_index.numel() == 0 and n_real > 0:
        edge_index, _ = add_self_loops(edge_index, num_nodes=n_real)
        edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        # 学習で CrossEntropy を使う想定なら long が安全
        y=torch.tensor([int(target)], dtype=torch.long),
    )
    data = coalesce_by_triplet_(data)
    data.num_nodes = n_real

    # ★ 位置情報(start/end line/col)は一切保持しない（メモリ節約）
    return data


@torch.no_grad()
def coalesce_by_triplet_(d: Data) -> Data:
    """
    (src, dst, edge_type) が完全一致のエッジ重複を除去（in-place）。
    """
    ei = d.edge_index
    assert ei is not None and torch.is_tensor(ei) and ei.dim() == 2 and ei.size(0) == 2

    # edge_type が無ければ (src,dst) でユニーク化
    if not (hasattr(d, "edge_type") and torch.is_tensor(d.edge_type) and d.edge_type.numel() == ei.size(1)):
        pairs = torch.stack([ei[0], ei[1]], dim=1)          # (E, 2)
        uniq = torch.unique(pairs, dim=0)                   # (E', 2)
        d.edge_index = uniq.t().contiguous()
        return d

    et = d.edge_type
    trip = torch.stack([ei[0], ei[1], et], dim=1)           # (E, 3)
    uniq = torch.unique(trip, dim=0)                        # (E', 3)
    d.edge_index = uniq[:, :2].t().contiguous()
    d.edge_type  = uniq[:, 2].contiguous()
    return d
