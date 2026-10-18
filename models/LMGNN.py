import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from .layers import GraphCodeFusion
from torch_geometric.data import Batch as GeoBatch


# ★ 置換: GGNN → R-GCN
# from torch_geometric.nn.conv import GatedGraphConv   # ← 不要
from torch_geometric.nn import RGCNConv
try:
    # PyG>=2.3 ならこちら
    from torch_geometric.utils import coalesce as pyg_coalesce
    _HAS_COALESCE = True
except Exception:
    _HAS_COALESCE = False

# === あなたの既存 Conv（そのまま利用） ==========================
# from .layers import Conv, encode_input  # encode_input は未使用でもOK
# ここでは質問に含まれている Conv 実装を前提とします
# ---------------------------------------------------------------
import torch
import torch.nn.functional as F
from torch_geometric.utils import coalesce as pyg_coalesce, add_self_loops

def _infer_num_nodes(edge_index: torch.Tensor, fallback: int | None = None) -> int:
    if edge_index is not None and edge_index.numel() > 0:
        return int(edge_index.max().item()) + 1
    return int(fallback) if fallback is not None else 0

@torch.no_grad()
def _coalesce_triplets(edge_index: torch.Tensor,
                       edge_type: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    (src, dst, edge_type) の三つ組で重複除去。
    edge_type が None の場合は (src, dst) で重複除去。
    """
    if edge_index.numel() == 0:
        return edge_index, edge_type

    # 型と連続性を保証
    edge_index = edge_index.long().contiguous()
    if edge_type is not None:
        edge_type = edge_type.long().contiguous()

    if edge_type is None or edge_type.numel() != edge_index.size(1):
        # edge_type が無い or 長さ不一致 → ペアで coalesce（type は捨てるか先頭を残す）
        ei_pairs = edge_index.t()                              # (E, 2)
        uniq, first_idx = torch.unique(ei_pairs, dim=0, return_inverse=False, return_counts=False, return_index=True)
        edge_index = uniq.t().contiguous()
        if edge_type is not None and edge_type.numel() > 0:
            edge_type = edge_type.index_select(0, first_idx.sort()[0]).contiguous()
        else:
            edge_type = None
        return edge_index, edge_type

    # 三つ組で coalesce
    trip = torch.stack([edge_index[0], edge_index[1], edge_type], dim=1)  # (E, 3)
    uniq = torch.unique(trip, dim=0)                                      # (E',3)
    edge_index = uniq[:, :2].t().contiguous()
    edge_type  = uniq[:, 2].contiguous().long()
    return edge_index, edge_type

def sanitize_for_rgcn(d, *, trim_x: bool = True, do_coalesce: bool = True):
    """
    RGCNConv に安全に渡せるよう Data/Batch を整形:
      - long 型へ正規化
      - (src,dst,etype) で重複除去
      - エッジが無い場合は自己ループ付与（etype=0）
      - num_nodes と x の行数を整合
    """
    ei = d.edge_index
    et = getattr(d, "edge_type", None)

    if ei is None:
        # エッジ自体が無いケース
        n = int(getattr(d, "num_nodes", d.x.size(0) if getattr(d, "x", None) is not None else 0))
        if n <= 0 and getattr(d, "x", None) is not None:
            n = int(d.x.size(0))
        ei, _ = add_self_loops(torch.empty(2, 0, dtype=torch.long, device=d.x.device if getattr(d,"x",None) is not None else None),
                               num_nodes=n)
        et = torch.zeros(ei.size(1), dtype=torch.long, device=ei.device)
    else:
        ei = ei.long().contiguous()
        if et is not None:
            et = et.long().contiguous()

        if do_coalesce:
            ei, et = _coalesce_triplets(ei, et)

        # 0 本になったらフォールバック
        num_nodes = int(getattr(d, "num_nodes", _infer_num_nodes(ei, getattr(d, "x", None).size(0) if getattr(d,"x",None) is not None else None)))
        if ei.numel() == 0:
            ei, _ = add_self_loops(ei, num_nodes=num_nodes)
            et = torch.zeros(ei.size(1), dtype=torch.long, device=ei.device)

    # num_nodes の整合
    num_nodes = int(getattr(d, "num_nodes", _infer_num_nodes(ei, getattr(d, "x", None).size(0) if getattr(d,"x",None) is not None else None)))
    d.num_nodes = num_nodes

    # x の行数トリム（過剰パディング対策）
    if trim_x and getattr(d, "x", None) is not None and d.x.size(0) > num_nodes:
        d.x = d.x[:num_nodes].contiguous()

    d.edge_index = ei
    if et is not None:
        d.edge_type = et
    return d


from transformers import AutoTokenizer



class BertRGCN(nn.Module):
    def __init__(self, gated_graph_conv_args, conv_args, emb_size, device, Conv=None, debug_shapes: bool = False, log_wandb: bool = False, wandb_prefix: str = 'debug'):
        super().__init__()
        self.device = device

        hidden_dim   = gated_graph_conv_args.get("out_channels", 200)
        num_relations = gated_graph_conv_args.get("num_relations", 3)
        num_layers   = gated_graph_conv_args.get("num_layers", 6)
        
        self.hidden_dim = hidden_dim
        self.code_dim   = emb_size

        # --- 1. CodeBERT Encoder ---
        self.func_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.func_encoder   = CodeBERTEncoder(
            model_name="microsoft/codebert-base",
            tune_last_n_layers=2,
        )

        # Align internal hidden dimension to CodeBERT hidden size (e.g. 768)
        # Override any previous default hidden_dim so that Graph and CodeLMs use the same dimension
        hidden_dim = self.func_encoder.hidden_size
        self.hidden_dim = hidden_dim

        # --- 2. Pre-GCN Processing (Contextualization) ★追加部分 ---
        # If CodeBERT output size equals target hidden_dim, projection can be Identity
        if self.func_encoder.hidden_size == hidden_dim:
            self.code_proj_pre = nn.Identity()
        else:
            self.code_proj_pre = nn.Linear(self.func_encoder.hidden_size, hidden_dim)
        
        # ノード(Query) が コード(Key/Value) を参照するAttention
        self.context_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.context_norm = nn.LayerNorm(hidden_dim)

        # --- 3. R-GCN Layers (Standard) ---
        self.input_proj = nn.Linear(emb_size, hidden_dim)
        self.rgcn_layers = nn.ModuleList([
            RGCNConv(hidden_dim, hidden_dim, num_relations)
            for _ in range(num_layers)
        ])
        self.rgcn_norm = nn.LayerNorm(hidden_dim)

        # --- 4. Post-GCN Fusion (Final Integration) ---
        self.fusion = GraphCodeFusion(
            code_dim=self.func_encoder.hidden_size, # 768
            graph_dim=hidden_dim,
            proj_dim=hidden_dim,
            num_heads=4,
            ffn_hidden_dim=hidden_dim * 4,
            fusion_out_dim=hidden_dim,
            dropout=0.1,
            attn_layers=3,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2)
        )
        # whether to print tensor shapes before cross-attention (useful for debugging)
        self.debug_shapes = debug_shapes
        # optional wandb logging
        self.log_wandb = log_wandb
        self.wandb_prefix = wandb_prefix
        self._wandb = None
        if self.log_wandb:
            try:
                import wandb
                self._wandb = wandb
            except Exception:
                self._wandb = None

    def forward(self, data):
        x          = data.x
        edge_index = data.edge_index
        edge_type  = data.edge_type

        # ============================================================
        # Step 1: CodeBERT Encoding (文脈情報の取得)
        # ============================================================
        funcs = getattr(data, "func", None)
        
        # バッチサイズ計算
        if isinstance(data, GeoBatch) and hasattr(data, "ptr") and data.ptr is not None:
            num_nodes_per_graph = data.ptr[1:] - data.ptr[:-1]
            batch_idx = torch.arange(num_nodes_per_graph.size(0), device=x.device) \
                            .repeat_interleave(num_nodes_per_graph)
            B = num_nodes_per_graph.size(0)
        else:
            batch_idx = x.new_zeros(x.size(0), dtype=torch.long)
            B = 1

        # CodeBERT実行
        if funcs is None:
            code_emb = x.new_zeros(B, 1, self.func_encoder.hidden_size)
            code_mask = None
        else:
            if isinstance(funcs, str):
                funcs = [funcs]
            enc = self.func_tokenizer(
                list(funcs),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            input_ids      = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            code_emb = self.func_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_cls=False,        # [B, L_code, 768]
            )
            code_mask = attention_mask   # [B, L_code]

        # ============================================================
        # Step 2: Contextualized Node Features (ノードへの文脈注入) ★追加部分
        # ============================================================
        # ノード特徴量の初期投影
        h_nodes = self.input_proj(x)  # [Total_Nodes, hidden_dim]
        
        # CodeBERT出力をGCN次元に射影: [B, L, 768] -> [B, L, 200]
        code_emb_proj = self.code_proj_pre(code_emb)

        # ノードをバッチ単位のDense形式に変換: [B, Max_Nodes, 200]
        h_nodes_dense, node_mask = to_dense_batch(h_nodes, batch_idx)

        # Attention用のマスク準備 (True=無視したい場所)
        if code_mask is not None:
            # code_maskは 1=有効, 0=padding なので反転させる
            key_padding_mask = (code_mask == 0)
        else:
            key_padding_mask = None

        # Cross-Attention: Nodes(Q) <- Code(K, V)
        # 各ノードが、コードのトークン列全体を見に行く
        # Optionally log/print shapes before attention
        if getattr(self, 'debug_shapes', False):
            try:
                kp_shape = None if key_padding_mask is None else tuple(key_padding_mask.shape)
                shape_info = {
                    f"{self.wandb_prefix}/h_nodes_dense_shape": str(tuple(h_nodes_dense.shape)),
                    f"{self.wandb_prefix}/code_emb_proj_shape": str(tuple(code_emb_proj.shape)),
                    f"{self.wandb_prefix}/key_padding_mask_shape": str(kp_shape)
                }
                # print to stdout for quick debugging
                print(f"[debug] h_nodes_dense.shape={tuple(h_nodes_dense.shape)}, code_emb_proj.shape={tuple(code_emb_proj.shape)}, key_padding_mask.shape={kp_shape}")
                # log to wandb if available
                if self._wandb is not None:
                    # wandb.log expects numeric values; log shapes as strings under step-less dict
                    try:
                        self._wandb.log(shape_info)
                    except Exception:
                        # fallback: log counts (dimensions) as numbers
                        try:
                            hd = tuple(h_nodes_dense.shape)
                            ce = tuple(code_emb_proj.shape)
                            self._wandb.log({f"{self.wandb_prefix}/h_nodes_dense_dim0": hd[0], f"{self.wandb_prefix}/h_nodes_dense_dim1": hd[1], f"{self.wandb_prefix}/code_emb_proj_dim0": ce[0], f"{self.wandb_prefix}/code_emb_proj_dim1": ce[1]})
                        except Exception:
                            pass
            except Exception:
                pass

        attn_out, _ = self.context_attn(
            query=h_nodes_dense,
            key=code_emb_proj,
            value=code_emb_proj,
            key_padding_mask=key_padding_mask
        )

        # 残差接続 & Norm (Contextualized Nodes)
        h_nodes_dense = self.context_norm(h_nodes_dense + attn_out)

        # Dense状態から元のFlatな[Total_Nodes, 200]に戻す
        h_nodes = h_nodes_dense[node_mask]

        # ============================================================
        # Step 3: R-GCN (グラフ畳み込み)
        # ============================================================
        # コンテキスト化された特徴量(h_nodes)を初期値として開始
        h = h_nodes 
        for conv in self.rgcn_layers:
            h = F.relu(conv(h, edge_index, edge_type))
        
        h = self.rgcn_norm(h)

        # ============================================================
        # Step 4: Final Fusion & Classification
        # ============================================================
        # 再度Dense化してFusion層へ
        graph_dense, graph_mask = to_dense_batch(h, batch_idx)

        # Fusion (code_embは元の768次元のものを使用 -> Fusion内で射影される)
        graph_repr = self.fusion(
            code_emb=code_emb,              # [B, L_code, 768]
            graph_emb=graph_dense,          # [B, L_graph, 200]
            code_mask=code_mask,            # [B, L_code]
            graph_mask=graph_mask,          # [B, L_graph]
        )                                   # -> [B, hidden_dim]

        logits = self.classifier(graph_repr)
        return logits
    

# models/LMGNN.py に追加

class CodeBERTOnly(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
        # CodeBERTエンコーダ (既存と同じ設定)
        self.func_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.func_encoder = CodeBERTEncoder(
            model_name="microsoft/codebert-base",
            tune_last_n_layers=2,
        )
        
        # CodeBERTの隠れ層サイズ (通常768)
        self.bert_hidden_size = self.func_encoder.hidden_size

        # 分類器
        # 元のBertRGCNのclassifier構造に似せていますが、入力次元をCodeBERTに合わせます
        self.classifier = nn.Sequential(
            nn.Linear(self.bert_hidden_size, 200), # 一度次元を落とす（元のhidden_dimに合わせる場合）
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(200, 2)
        )

    def forward(self, data):
        # 1) 関数コードのトークナイズ
        funcs = getattr(data, "func", None)
        if funcs is None:
            # 万が一funcがない場合のダミー (Batchサイズに合わせてゼロ埋め)
            # data.x からバッチサイズを推定
            B = data.x.size(0) if hasattr(data, "x") else 1
            return torch.zeros(B, 2, device=self.device)

        if isinstance(funcs, str):
            funcs = [funcs]

        enc = self.func_tokenizer(
            list(funcs),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512, 
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        # 2) CodeBERTでエンコード (return_cls=True で [CLS] トークンのみ取得)
        # [B, 768]
        cls_emb = self.func_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_cls=True 
        )

        # 3) 分類
        logits = self.classifier(cls_emb)
        return logits





# models/codebert_encoder.py
import torch
import torch.nn as nn
from transformers import AutoModel

class CodeBERTEncoder(nn.Module):
    def __init__(self, model_name: str = "microsoft/codebert-base", tune_last_n_layers: int = 2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size

        # 全層いったん凍結
        for p in self.bert.parameters():
            p.requires_grad = False

        # encoder の最後の n 層だけ解凍（ここが finetune 対象）
        encoder_layers = self.bert.encoder.layer
        for layer in encoder_layers[-tune_last_n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

        if hasattr(self.bert, "pooler"):
            for p in self.bert.pooler.parameters():
                p.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_cls: bool = True,
    ):
        """
        return_cls=True なら [B, H] (CLS)、
        False なら [B, L, H] (トークン列) を返す。
        """
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state  # [B, L, H]

        if return_cls:
            return last[:, 0, :]      # [B, H]
        else:
            return last               # [B, L, H]

