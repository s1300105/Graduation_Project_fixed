import sys, os
from pathlib import Path
# Ensure project root is on sys.path (absolute)
proj_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(proj_root))

import argparse, csv, torch, numpy as np, pandas as pd
import importlib.util
import configs
# Note: avoid importing the third-party `tensor` package (Python2) which breaks execution.


from models.LMGNN import BertRGCN
from models.layers import Conv
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=50, help='max samples to analyze')
parser.add_argument('--ckpt', type=str, default=None, help='optional model checkpoint to load')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# config same as run.py
cfg = configs.BertGGNN()
gated_graph_conv_args = cfg.model["gated_graph_conv_args"]
conv_args = cfg.model["conv_args"]
emb_size = cfg.model["emb_size"]

# instantiate model to get projection layers and encoder
model = BertRGCN(gated_graph_conv_args, conv_args, emb_size, device, Conv=Conv).to(device)
model.eval()
if args.ckpt:
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd, strict=False)

OUT_CSV = Path("attn_norms.csv")
rows = [["file","idx","num_nodes","num_tokens",
         "mean_q_pre_norm","mean_k_pre_norm","mean_v_pre_norm",
         "mean_q_norm","mean_k_norm","mean_v_norm","qk_ratio", "qk_cos_pre", "qk_cos_post"]]

# find input files
input_dir = Path(configs.Paths().input)
if not input_dir.exists():
    raise FileNotFoundError(f"Input directory not found: {input_dir}")
# collect files by glob to avoid depending on utils.data helpers
files = sorted(list(input_dir.glob(f"*_{configs.Files().input}")))
count = 0
for fn in sorted(files):
    if count >= args.n: break
    df = pd.read_pickle(fn)
    for _, row in df.iterrows():
        if count >= args.n: break
        g = row.get("input", None)
        if g is None: continue
        # prepare batch (single graph)
        batch = Batch.from_data_list([g]).to(device)
        # node features -> projected query input
        x = batch.x.to(device)
        # also keep pre-projection node features for norm comparison
        # dense the raw node features so we can compute per-node norms per graph
        # create a batch index for raw x: prefer batch.batch if present
        if hasattr(batch, 'batch') and batch.batch is not None:
            batch_idx_x = batch.batch.to(x.device)
        else:
            batch_idx_x = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x_dense, _ = to_dense_batch(x, batch_idx_x)
        q_pre = x_dense[0]  # [Lg, in_dim]
        # flat node projection
        h_nodes = model.input_proj(x)                  # [Total_Nodes, hidden]
        batch_idx = torch.zeros(h_nodes.size(0), dtype=torch.long, device=h_nodes.device)
        # dense
        h_nodes_dense, node_mask = to_dense_batch(h_nodes, batch_idx)  # [B, Lg, D]
        # code embedding: use tokenizer + encoder to get token embeddings
        funcs = getattr(g, "func", None) or row.get("func", None)
        if funcs is None:
            continue
        if isinstance(funcs, str):
            funcs = [funcs]
        enc = model.func_tokenizer(funcs, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = enc["input_ids"].to(device); attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            code_emb = model.func_encoder(input_ids=input_ids, attention_mask=attention_mask, return_cls=False)  # [B, Lc, H]
            code_proj = model.code_proj_pre(code_emb)   # [B, Lc, D]
        # compute norms for first (and only) batch element
        # pre-projection code embeddings (= CodeBERT output)
        k_pre = code_emb[0]
        v_pre = code_emb[0]
        q = h_nodes_dense[0]   # [Lg, D]
        k = code_proj[0]       # [Lc, D]
        v = code_proj[0]       # [Lc, D]
        # mean-pooled vectors for cosine similarity (pre and post)
        try:
            q_pre_mean = q_pre.mean(dim=0)
            k_pre_mean = k_pre.mean(dim=0)
            q_post_mean = q.mean(dim=0)
            k_post_mean = k.mean(dim=0)
            # cosine similarity
            cos_pre = torch.nn.functional.cosine_similarity(q_pre_mean.unsqueeze(0), k_pre_mean.unsqueeze(0)).item()
            cos_post = torch.nn.functional.cosine_similarity(q_post_mean.unsqueeze(0), k_post_mean.unsqueeze(0)).item()
        except Exception:
            cos_pre = float('nan')
            cos_post = float('nan')
        # detach before converting to numpy to avoid RuntimeError when grad is enabled
        q_norms = torch.norm(q, dim=-1).detach().cpu().numpy()
        k_norms = torch.norm(k, dim=-1).detach().cpu().numpy()
        v_norms = torch.norm(v, dim=-1).detach().cpu().numpy()
        # pre-projection norms
        q_pre_norms = torch.norm(q_pre, dim=-1).detach().cpu().numpy() if q_pre is not None else np.array([])
        k_pre_norms = torch.norm(k_pre, dim=-1).detach().cpu().numpy() if k_pre is not None else np.array([])
        v_pre_norms = torch.norm(v_pre, dim=-1).detach().cpu().numpy() if v_pre is not None else np.array([])
        mean_q = float(q_norms.mean()) if q_norms.size else 0.0
        mean_k = float(k_norms.mean()) if k_norms.size else 0.0
        mean_v = float(v_norms.mean()) if v_norms.size else 0.0
        mean_q_pre = float(q_pre_norms.mean()) if q_pre_norms.size else 0.0
        mean_k_pre = float(k_pre_norms.mean()) if k_pre_norms.size else 0.0
        mean_v_pre = float(v_pre_norms.mean()) if v_pre_norms.size else 0.0
        rows.append([str(fn.name), int(getattr(g, "idx", -1)), int(q.size(0)), int(k.size(0)),
                     mean_q_pre, mean_k_pre, mean_v_pre,
                     mean_q, mean_k, mean_v, mean_q/(mean_k+1e-9), cos_pre, cos_post])
        count += 1

# write CSV
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)
print("Saved:", OUT_CSV)