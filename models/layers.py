import os, sys, torch
PROJECT_ROOT = "/home/yudai/Project/research/Graduation_Project"
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

torch.manual_seed(2020)


def get_conv_mp_out_size(in_size, last_layer, mps):
    size = in_size

    for mp in mps:
        size = round((size - mp["kernel_size"]) / mp["stride"] + 1)

    size = size + 1 if size % 2 != 0 else size

    return int(size * last_layer["out_channels"])


def encode_input(text, tokenizer):
    max_length = 512
    input = tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
    return input.input_ids, input.attention_mask


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)


class Conv(nn.Module):
    def __init__(self, conv1d_1, conv1d_2, maxpool1d_1, maxpool1d_2, fc_1_size, fc_2_size):
        super(Conv, self).__init__()
        self.conv1d_1_args = conv1d_1
        self.conv1d_1 = nn.Conv1d(**conv1d_1)
        self.conv1d_2 = nn.Conv1d(**conv1d_2)
        self.mp_1 = nn.MaxPool1d(**maxpool1d_1)
        self.mp_2 = nn.MaxPool1d(**maxpool1d_2)
        self.drop = nn.Dropout(p=0.2)

        # --- ここから: 実ラン形状でFC in_featuresを推定 ---
        with torch.no_grad():
            C = self.conv1d_1_args["in_channels"]
            # Z 経路（hidden と x を結合した長さ）
            dummy_Z = torch.zeros(1, C, int(fc_1_size))
            Z = self.mp_1(F.relu(self.conv1d_1(dummy_Z)))
            Z = self.mp_2(self.conv1d_2(Z))
            Z_flatten_size = int(Z.shape[1] * Z.shape[-1])

            # Y 経路（hidden のみ）
            dummy_Y = torch.zeros(1, C, int(fc_2_size))
            Y = self.mp_1(F.relu(self.conv1d_1(dummy_Y)))
            Y = self.mp_2(self.conv1d_2(Y))
            Y_flatten_size = int(Y.shape[1] * Y.shape[-1])

        # Dense layers（in_features を実計測で設定）
        self.fc1 = nn.Linear(Z_flatten_size, 1)
        self.fc2 = nn.Linear(Y_flatten_size, 1)

    def forward(self, hidden, x):
        concat = torch.cat([hidden, x], 1)
        concat_size = hidden.shape[1] + x.shape[1]
        concat = concat.view(-1, self.conv1d_1_args["in_channels"], concat_size)

        Z = self.mp_1(F.relu(self.conv1d_1(concat)))
        Z = self.mp_2(self.conv1d_2(Z))

        hidden = hidden.view(-1, self.conv1d_1_args["in_channels"], hidden.shape[1])

        Y = self.mp_1(F.relu(self.conv1d_1(hidden)))
        Y = self.mp_2(self.conv1d_2(Y))

        Z_flatten_size = int(Z.shape[1] * Z.shape[-1])
        Y_flatten_size = int(Y.shape[1] * Y.shape[-1])

        Z = Z.view(-1, Z_flatten_size)
        Y = Y.view(-1, Y_flatten_size)
        res = self.fc1(Z) * self.fc2(Y)
        res = self.drop(res)
        sig = torch.sigmoid(torch.flatten(res))
        return sig


class Net(nn.Module):
    def __init__(self, gated_graph_conv_args, conv_args, emb_size, device, Conv=None):
        super().__init__()
        self.device = device

        rgcn_args = dict(gated_graph_conv_args)
        rgcn_args.setdefault("num_layers", 2)
        rgcn_args.setdefault("num_relations", 3)
        rgcn_args.setdefault("num_bases", 3)
        self.rgcn_args = rgcn_args

        hidden = rgcn_args["out_channels"]

        self.input_proj = nn.Linear(emb_size, hidden)

        self.rgcn_layers = nn.ModuleList([
            RGCNConv(
                in_channels = hidden,
                out_channels= hidden,
                num_relations= rgcn_args["num_relations"],
                num_bases   = rgcn_args["num_bases"]
            )
            for _ in range(rgcn_args["num_layers"])
        ])

        assert Conv is not None, "Conv クラス（あなたの実装）を渡してください"
        self.conv = Conv(
            **conv_args,
            fc_1_size = hidden + emb_size,
            fc_2_size = hidden
        ).to(device)

    def forward(self, data):
        from models.LMGNN import sanitize_for_rgcn
        d = sanitize_for_rgcn(data, trim_x=True, coalesce=True)
        x, edge_index, edge_type = d.x, d.edge_index, d.edge_type

        h = self.input_proj(x)
        for conv in self.rgcn_layers:
            h = F.relu(conv(h, edge_index, edge_type))

        out = self.conv(h, x)  # 既存の Conv 出力（sigmoid確率など）
        return out

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))


class PositionwiseFFN(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Module):
    """
    汎用的な Attention ブロック (Self / Cross 兼用)。
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # [B, L, D] 形式で扱う
        )

    def forward(
        self,
        query: torch.Tensor,      # [B, L_q, D]
        key: torch.Tensor,        # [B, L_k, D]
        value: torch.Tensor,      # [B, L_k, D]
        key_padding_mask=None,    # [B, L_k]
        need_weights: bool = False,
    ):
        """
        戻り値:
            out: [B, L_q, D]
        """
        out, attn_weights = self.attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
        )
        return out


class FusionBlock(nn.Module):
    """
    1層分の融合レイヤー:
    1. Self-Attention (Code & Graph)
    2. Cross-Attention (Code <-> Graph)
    3. FFN (Feed Forward Network) ★追加
    """
    def __init__(self, dim, num_heads, ffn_dim, dropout):
        super().__init__()
        # --- 1. Self-Attention ---
        #self.self_attn_code = AttentionBlock(dim, num_heads, dropout)
        #self.norm1_code = nn.LayerNorm(dim)
        
        #self.self_attn_graph = AttentionBlock(dim, num_heads, dropout)
        #self.norm1_graph = nn.LayerNorm(dim)

        # --- 2. Cross-Attention ---
        self.cross_attn_code = AttentionBlock(dim, num_heads, dropout)
        self.norm2_code = nn.LayerNorm(dim)

        self.cross_attn_graph = AttentionBlock(dim, num_heads, dropout)
        self.norm2_graph = nn.LayerNorm(dim)

        # --- 3. Position-wise FFN (各層に追加) ---
        self.ffn_code = PositionwiseFFN(dim, ffn_dim, dropout)
        self.norm3_code = nn.LayerNorm(dim)

        self.ffn_graph = PositionwiseFFN(dim, ffn_dim, dropout)
        self.norm3_graph = nn.LayerNorm(dim)

    def forward(self, code, graph, code_padding_mask, graph_padding_mask):
        # --- 1. Self-Attention ---
        # Code
        #c2 = self.self_attn_code(query=code, key=code, value=code, key_padding_mask=code_padding_mask)
        #code = self.norm1_code(code + c2)
        # Graph
        #g2 = self.self_attn_graph(query=graph, key=graph, value=graph, key_padding_mask=graph_padding_mask)
        #graph = self.norm1_graph(graph + g2)

        # --- 2. Cross-Attention ---
        # Code query, Graph key/value (CodeがGraphの情報を取り込む)
        c_cross = self.cross_attn_code(query=code, key=graph, value=graph, key_padding_mask=graph_padding_mask)
        code_next = self.norm2_code(code + c_cross)

        # Graph query, Code key/value (GraphがCodeの情報を取り込む)
        g_cross = self.cross_attn_graph(query=graph, key=code, value=code, key_padding_mask=code_padding_mask)
        graph_next = self.norm2_graph(graph + g_cross)
        
        code, graph = code_next, graph_next

        # --- 3. FFN ---
        code = self.norm3_code(code + self.ffn_code(code))
        graph = self.norm3_graph(graph + self.ffn_graph(graph))

        return code, graph


class GraphCodeFusion(nn.Module):
    """
    CodeBERT埋め込みとR-GCN埋め込みを融合するブロック。
    Standard Transformer Fusion:
      1. Self-Attention (Code & Graph independent)
      2. Cross-Attention (Code<->Graph interaction)
      3. FFN (per layer)
    """
    def __init__(
        self,
        code_dim: int,
        graph_dim: int,
        proj_dim: int = 200,
        num_heads: int = 4,
        ffn_hidden_dim: int = None,
        fusion_out_dim: int = 200,
        dropout: float = 0.1,
        attn_layers: int = 1,
    ):
        super().__init__()

        # FFNの隠れ層次元（指定がなければ投影次元の4倍が標準的）
        if ffn_hidden_dim is None:
            ffn_hidden_dim = proj_dim * 4

        self.proj_code = nn.Linear(code_dim, proj_dim)
        self.proj_graph = nn.Linear(graph_dim, proj_dim)

        # ループ構造を FusionBlock に委譲
        self.layers = nn.ModuleList([
            FusionBlock(proj_dim, num_heads, ffn_hidden_dim, dropout)
            for _ in range(attn_layers)
        ])
        
        self.gate_layer = nn.Linear(proj_dim * 2, 2)

        # Final Fusion Output
        fused_dim = proj_dim * 2
        self.ffn_final = PositionwiseFFN(fused_dim, hidden_dim=ffn_hidden_dim, dropout=dropout)
        self.norm_final = nn.LayerNorm(fused_dim)
        self.linear_out = nn.Linear(fused_dim, fusion_out_dim)

    def forward(
        self,
        code_emb: torch.Tensor,
        graph_emb: torch.Tensor,
        code_mask: torch.Tensor | None = None,   # [B, L_code] (1=有効, 0=Padding)
        graph_mask: torch.Tensor | None = None,  # [B, L_graph] (1=有効, 0=Padding)
    ):
        code_h = self.proj_code(code_emb)
        graph_h = self.proj_graph(graph_emb)

        # マスクの準備 (PyTorch MultiheadAttentionは True=無視 なので反転させる)
        code_padding_mask = (code_mask == 0) if code_mask is not None else None
        graph_padding_mask = (graph_mask == 0) if graph_mask is not None else None

        # --- Attention Layers Loop ---
        for layer in self.layers:
            code_h, graph_h = layer(
                code_h, graph_h,
                code_padding_mask, graph_padding_mask
            )

        # --- Pooling & Output ---
        
        # Code側: Masked Mean Pooling (旧 Max Pooling から変更)
        if code_mask is not None:
            # mask: [B, L], code_h: [B, L, D]
            mask_f = code_mask.float().unsqueeze(-1)  # [B, L, 1]
            code_sum = (code_h * mask_f).sum(dim=1)   # [B, D]
            denom_c = mask_f.sum(dim=1).clamp(min=1e-6)
            code_vec = code_sum / denom_c
        else:
            code_vec = code_h.mean(dim=1)

        # Graph側: Mean Pooling (変更なし)
        if graph_mask is not None:
            graph_mask_f = graph_mask.float().unsqueeze(-1)
            graph_sum = (graph_h * graph_mask_f).sum(dim=1)
            denom_g = graph_mask_f.sum(dim=1).clamp(min=1e-6)
            graph_vec = graph_sum / denom_g
        else:
            graph_vec = graph_h.mean(dim=1)



        # 1. まず両方の情報を一度結合して「状況」を把握する
        combined_features = torch.cat([code_vec, graph_vec], dim=-1)  # [B, 2*D]
        
        # 2. ゲート係数（重み）を計算する
        # Softmaxを使うことで、w_code + w_graph = 1.0 になり、バランス調整という意味合いが強まる
        gate_logits = self.gate_layer(combined_features)      # [B, 2]
        gate_weights = F.softmax(gate_logits, dim=-1)         # [B, 2] (確率分布化)

        w_code  = gate_weights[:, 0:1] # [B, 1]
        w_graph = gate_weights[:, 1:2] # [B, 1]

        # ★ ここで値を保存する (CPUに移してitem化し、計算グラフを切る)
        if self.training:
            # 訓練中のみ保存（推論時は無駄な処理を省くため）
            self.last_weights = {
                "code": w_code.mean().item(),
                "graph": w_graph.mean().item()
            }

        # 3. 重み付き結合
        fused = torch.cat([w_code * code_vec, w_graph * graph_vec], dim=-1)

        # === 変更終了 ===

        # 出力層へ
        fused = self.ffn_final(fused)
        fused = self.norm_final(fused)
        fused = self.linear_out(fused)

        return fused