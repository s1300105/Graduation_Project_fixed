#!/usr/bin/env python3
"""
Simple visualizer for attn_norms.csv produced by the analysis script.
Saves PNGs into ./plots/attn_norms/
Usage:
  python scripts/visualize_attn_norms.py --csv attn_norms.csv --out plots/attn_norms
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid")

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, default="attn_norms.csv", help="CSV file produced by compare_attn_norms.py")
parser.add_argument("--out", type=str, default="plots/attn_norms", help="output directory for plots")
parser.add_argument("--max-samples", type=int, default=None, help="max samples to plot (for large CSVs)")
args = parser.parse_args()

csv_path = Path(args.csv)
out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

if not csv_path.exists():
    print(f"CSV not found: {csv_path}")
    raise SystemExit(1)

df = pd.read_csv(csv_path)
if args.max_samples:
    df = df.head(args.max_samples)

# Only focus on pre-projection columns
# Prefer explicit 'mean_graph_pre_norm' and 'mean_codelms_pre_norm' if present, else fall back to q/k/v columns
# Map columns for backward compatibility
graph_col = None
codelms_col = None
# detect graph (Query) column
if 'mean_graph_pre_norm' in df.columns:
    graph_col = 'mean_graph_pre_norm'
elif 'mean_q_pre_norm' in df.columns:
    graph_col = 'mean_q_pre_norm'
# detect CodeLMs (Key/Value) column: prefer explicit, else average k and v if both exist
if 'mean_codelms_pre_norm' in df.columns:
    codelms_col = 'mean_codelms_pre_norm'
else:
    kcol = 'mean_k_pre_norm' if 'mean_k_pre_norm' in df.columns else None
    vcol = 'mean_v_pre_norm' if 'mean_v_pre_norm' in df.columns else None
    if kcol and vcol:
        # create an averaged CodeLMs column for convenience
        df['mean_codelms_pre_norm'] = pd.to_numeric(df[kcol], errors='coerce')
        df['mean_codelms_pre_norm'] = df['mean_codelms_pre_norm'].fillna(0.0)
        tmp_v = pd.to_numeric(df[vcol], errors='coerce').fillna(0.0)
        df['mean_codelms_pre_norm'] = (df['mean_codelms_pre_norm'] + tmp_v) / 2.0
        codelms_col = 'mean_codelms_pre_norm'
    elif kcol:
        codelms_col = kcol
    elif vcol:
        codelms_col = vcol

# If neither found, warn and exit
if graph_col is None and codelms_col is None:
    print('No recognizable pre-projection columns found (expected mean_graph_pre_norm or mean_q_pre_norm / mean_k_pre_norm/mean_v_pre_norm)')
    raise SystemExit(1)

# ensure numeric
for c in [graph_col, codelms_col]:
    if c and c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# map qk_cos_pre to a neutral name for plotting
if 'graph_codelms_cos_pre' not in df.columns:
    if 'qk_cos_pre' in df.columns:
        df['graph_codelms_cos_pre'] = pd.to_numeric(df['qk_cos_pre'], errors='coerce')

# Summary for pre columns (Graph / CodeLMs)
pre_present = [c for c in [graph_col, codelms_col] if c in df.columns]
summary = df[pre_present].describe()
print(summary)
summary.to_csv(out_dir / "summary_stats_pre_graph_codelms.csv")

# Histograms for pre norms (Graph / CodeLMs)
labels = ['Graph', 'CodeLMs']
cols_for_hist = [graph_col, codelms_col]
for col, label in zip(cols_for_hist, labels):
    if col and col in df.columns:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col].dropna(), kde=True, color="C0" if label == 'Graph' else "C1")
        plt.title(f"Mean {label} Norm (pre-projection)")
        plt.xlabel(f"Mean {label} norm")
        plt.tight_layout()
        safe_label = label.lower().replace(' ', '_')
        plt.savefig(out_dir / f"hist_{safe_label}_pre.png")
        plt.close()

# Boxplot for Graph vs CodeLMs
available_pre = [c for c in [graph_col, codelms_col] if c in df.columns]
if available_pre:
    plt.figure(figsize=(6,5))
    box_df = df[available_pre].melt(var_name="which", value_name="mean_norm")
    sns.boxplot(x="which", y="mean_norm", data=box_df, palette=sns.color_palette("Set2", n_colors=len(available_pre)), showfliers=False)
    plt.xticks(rotation=45)
    plt.title("Comparison of mean norms (pre-projection)")
    plt.tight_layout()
    plt.savefig(out_dir / "boxplot_qkv_pre.png")
    plt.close()

# Also save a Graph vs CodeLMs labeled boxplot (no fliers) for convenience
if available_pre:
    try:
        plt.figure(figsize=(6,5))
        box_df_gc = df[available_pre].melt(var_name="which", value_name="mean_norm")
        box_df_gc['which'] = box_df_gc['which'].map(lambda x: 'Graph' if x == graph_col else 'CodeLMs')
        sns.boxplot(x='which', y='mean_norm', data=box_df_gc, palette=sns.color_palette('Set2', n_colors=len(available_pre)), showfliers=False)
        plt.xticks(rotation=45)
        plt.title('Comparison of mean norms (pre-projection): Graph vs CodeLMs')
        plt.tight_layout()
        plt.savefig(out_dir / 'boxplot_graph_codelms_pre.png')
        plt.close()

        # Enhanced annotated boxplot (medians) without fliers
        plt.figure(figsize=(6,5))
        ax = sns.boxplot(x='which', y='mean_norm', data=box_df_gc, palette=sns.color_palette('Pastel1', n_colors=len(available_pre)), showfliers=False)
        medians = box_df_gc.groupby(['which'])['mean_norm'].median().values
        positions = range(len(medians))
        for pos, med in zip(positions, medians):
            ax.text(pos, med, f'{med:.3f}', horizontalalignment='center', size='small', color='black', weight='semibold')
        plt.xticks(rotation=45)
        plt.title('Boxplot of pre-projection mean norms (per-graph): Graph vs CodeLMs')
        plt.tight_layout()
        plt.savefig(out_dir / 'boxplot_graph_codelms_pre_enhanced.png')
        plt.close()
    except Exception:
        # don't fail the whole script for plotting errors
        print('Warning: failed to produce Graph vs CodeLMs pre boxplots')

# Simple overall mean bar chart (pre only)
if available_pre:
    means = [float(df[graph_col].dropna().mean()) if graph_col in df.columns else np.nan,
             float(df[codelms_col].dropna().mean()) if codelms_col in df.columns else np.nan]
    plt.figure(figsize=(6,4))
    x = np.arange(len(labels))
    plt.bar(x, means, color=["C0","C1"])
    plt.xticks(x, labels)
    plt.ylabel('Mean L2 norm (pre)')
    plt.title('Mean Norms (pre-projection): Graph vs CodeLMs')
    plt.tight_layout()
    plt.savefig(out_dir / 'bar_mean_norms_graph_codelms_pre.png')
    plt.close()
    pd.DataFrame({"label": labels, "mean_pre": means}).to_csv(out_dir / 'mean_norms_pre_graph_codelms_summary.csv', index=False)

# Cosine similarity histogram for pre (if present)
if 'graph_codelms_cos_pre' in df.columns:
    plt.figure(figsize=(6,4))
    sns.histplot(df['graph_codelms_cos_pre'].dropna(), kde=True, color="C3")
    plt.title("Distribution of Graph - CodeLMs cosine similarity (pre)")
    plt.xlabel("Cosine similarity")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_graph_codelms_cos_pre.png")
    plt.close()

# --- Enhanced distribution summaries (median/std/quantiles) ---
stat_cols = [c for c in [graph_col, codelms_col] if c in df.columns]
if stat_cols:
    # compute percentiles explicitly
    percentiles = df[stat_cols].quantile([0.25, 0.5, 0.75]).transpose()
    stats = pd.DataFrame({
        'count': df[stat_cols].count(),
        'mean': df[stat_cols].mean(),
        'median': df[stat_cols].median(),
        'std': df[stat_cols].std(),
        'min': df[stat_cols].min(),
        '25%': percentiles[0.25],
        '50%': percentiles[0.5],
        '75%': percentiles[0.75],
        'max': df[stat_cols].max()
    })
    stats.to_csv(out_dir / 'detailed_stats_pre_graph_codelms.csv')

# --- Violin plot (pre only) ---
avail = stat_cols
if avail:
    plt.figure(figsize=(8,5))
    melt_df = df[avail].melt(var_name='which', value_name='mean_norm')
    melt_df['which'] = melt_df['which'].map(lambda x: 'Graph' if x == graph_col else 'CodeLMs')
    sns.violinplot(x='which', y='mean_norm', data=melt_df, inner='quartile', palette=sns.color_palette('Set2', n_colors=len(avail)))
    # swarmplot: exclude outliers (points outside 1.5*IQR) so they are not shown as white circles
    filtered_swarm = []
    for grp in melt_df['which'].unique():
        grp_vals = melt_df.loc[melt_df['which'] == grp, 'mean_norm']
        q1 = grp_vals.quantile(0.25)
        q3 = grp_vals.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        grp_filtered = melt_df[(melt_df['which'] == grp) & (melt_df['mean_norm'] >= lower) & (melt_df['mean_norm'] <= upper)]
        filtered_swarm.append(grp_filtered)
    if filtered_swarm:
        filtered_swarm_df = pd.concat(filtered_swarm)
        sns.swarmplot(x='which', y='mean_norm', data=filtered_swarm_df, color='k', alpha=0.3, size=3)
    plt.title('Violin plot of pre-projection mean norms (per-graph): Graph vs CodeLMs')
    plt.tight_layout()
    plt.savefig(out_dir / 'violin_graph_codelms_pre.png')
    plt.close()

# --- Enhanced boxplot with medians annotated ---
if avail:
    plt.figure(figsize=(6,5))
    box_df = df[avail].melt(var_name='which', value_name='mean_norm')
    box_df['which'] = box_df['which'].map(lambda x: 'Graph' if x == graph_col else 'CodeLMs')
    ax = sns.boxplot(x='which', y='mean_norm', data=box_df, palette=sns.color_palette('Pastel1', n_colors=len(avail)), showfliers=False)
    # annotate medians
    medians = box_df.groupby(['which'])['mean_norm'].median().values
    positions = range(len(medians))
    for pos, med in zip(positions, medians):
        ax.text(pos, med, f'{med:.3f}', horizontalalignment='center', size='small', color='black', weight='semibold')
    plt.xticks(rotation=45)
    plt.title('Boxplot of pre-projection mean norms (per-graph): Graph vs CodeLMs')
    plt.tight_layout()
    plt.savefig(out_dir / 'boxplot_qkv_pre_enhanced.png')
    plt.close()

# --- Repeat above analysis for post-projection columns ---
# Prefer explicit 'mean_graph_post_norm' and 'mean_codelms_post_norm' if present, else fall back to q/k/v post columns
graph_post_col = None
codelms_post_col = None
# detect graph (Query) post column
if 'mean_graph_post_norm' in df.columns:
    graph_post_col = 'mean_graph_post_norm'
elif 'mean_q_norm' in df.columns:
    graph_post_col = 'mean_q_norm'
# detect CodeLMs (Key/Value) post column: prefer explicit, else average k and v if both exist
if 'mean_codelms_post_norm' in df.columns:
    codelms_post_col = 'mean_codelms_post_norm'
else:
    kcol_post = 'mean_k_norm' if 'mean_k_norm' in df.columns else None
    vcol_post = 'mean_v_norm' if 'mean_v_norm' in df.columns else None
    if kcol_post and vcol_post:
        # create an averaged CodeLMs post column for convenience
        df['mean_codelms_post_norm'] = pd.to_numeric(df[kcol_post], errors='coerce').fillna(0.0)
        tmp_vp = pd.to_numeric(df[vcol_post], errors='coerce').fillna(0.0)
        df['mean_codelms_post_norm'] = (df['mean_codelms_post_norm'] + tmp_vp) / 2.0
        codelms_post_col = 'mean_codelms_post_norm'
    elif kcol_post:
        codelms_post_col = kcol_post
    elif vcol_post:
        codelms_post_col = vcol_post

# If neither found, skip post analysis but continue
post_present = any([c in df.columns for c in [graph_post_col, codelms_post_col] if c])
if not post_present:
    print('No post-projection columns found; skipping post-projection plots')
else:
    # ensure numeric
    for c in [graph_post_col, codelms_post_col]:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # map qk_cos_post to neutral name
    if 'graph_codelms_cos_post' not in df.columns:
        if 'qk_cos_post' in df.columns:
            df['graph_codelms_cos_post'] = pd.to_numeric(df['qk_cos_post'], errors='coerce')

    # Summary for post columns (Graph / CodeLMs)
    post_cols = [c for c in [graph_post_col, codelms_post_col] if c in df.columns]
    if post_cols:
        summary_post = df[post_cols].describe()
        print('\nPost-projection summary:')
        print(summary_post)
        summary_post.to_csv(out_dir / "summary_stats_post_graph_codelms.csv")

    # Histograms for post norms (Graph / CodeLMs)
    labels_post = ['Graph', 'CodeLMs']
    cols_for_hist_post = [graph_post_col, codelms_post_col]
    for col, label in zip(cols_for_hist_post, labels_post):
        if col and col in df.columns:
            plt.figure(figsize=(6,4))
            sns.histplot(df[col].dropna(), kde=True, color="C0" if label == 'Graph' else "C1")
            plt.title(f"Mean {label} Norm (post-projection)")
            plt.xlabel(f"Mean {label} norm")
            plt.tight_layout()
            safe_label = label.lower().replace(' ', '_')
            plt.savefig(out_dir / f"hist_{safe_label}_post.png")
            plt.close()

    # Boxplot for Graph vs CodeLMs (post)
    available_post = [c for c in [graph_post_col, codelms_post_col] if c in df.columns]
    if available_post:
        plt.figure(figsize=(6,5))
        box_df_post = df[available_post].melt(var_name="which", value_name="mean_norm")
        sns.boxplot(x="which", y="mean_norm", data=box_df_post, palette=sns.color_palette("Set2", n_colors=len(available_post)), showfliers=False)
        plt.xticks(rotation=45)
        plt.title("Comparison of mean norms (post-projection): Graph vs CodeLMs")
        plt.tight_layout()
        plt.savefig(out_dir / "boxplot_graph_codelms_post.png")
        plt.close()

        # Simple overall mean bar chart (post only)
        if available_post:
            means_post = [float(df[graph_post_col].dropna().mean()) if graph_post_col in df.columns else np.nan,
                          float(df[codelms_post_col].dropna().mean()) if codelms_post_col in df.columns else np.nan]
            plt.figure(figsize=(6,4))
            x = np.arange(len(labels_post))
            plt.bar(x, means_post, color=["C0","C1"])
            plt.xticks(x, labels_post)
            plt.ylabel('Mean L2 norm (post)')
            plt.title('Mean Norms (post-projection): Graph vs CodeLMs')
            plt.tight_layout()
            plt.savefig(out_dir / 'bar_mean_norms_graph_codelms_post.png')
            plt.close()
            pd.DataFrame({"label": labels_post, "mean_post": means_post}).to_csv(out_dir / 'mean_norms_post_graph_codelms_summary.csv', index=False)

        # Cosine similarity histogram for post (if present)
        if 'graph_codelms_cos_post' in df.columns:
            plt.figure(figsize=(6,4))
            sns.histplot(df['graph_codelms_cos_post'].dropna(), kde=True, color="C3")
            plt.title("Distribution of Graph - CodeLMs cosine similarity (post)")
            plt.xlabel("Cosine similarity")
            plt.tight_layout()
            plt.savefig(out_dir / "hist_graph_codelms_cos_post.png")
            plt.close()

        # --- Enhanced distribution summaries for post ---
        stat_cols_post = [c for c in [graph_post_col, codelms_post_col] if c in df.columns]
        if stat_cols_post:
            percentiles_post = df[stat_cols_post].quantile([0.25, 0.5, 0.75]).transpose()
            stats_post = pd.DataFrame({
                'count': df[stat_cols_post].count(),
                'mean': df[stat_cols_post].mean(),
                'median': df[stat_cols_post].median(),
                'std': df[stat_cols_post].std(),
                'min': df[stat_cols_post].min(),
                '25%': percentiles_post[0.25],
                '50%': percentiles_post[0.5],
                '75%': percentiles_post[0.75],
                'max': df[stat_cols_post].max()
            })
            stats_post.to_csv(out_dir / 'detailed_stats_post_graph_codelms.csv')

        # --- Violin plot (post only) ---
        avail_post = stat_cols_post
        if avail_post:
            plt.figure(figsize=(8,5))
            melt_df_post = df[avail_post].melt(var_name='which', value_name='mean_norm')
            melt_df_post['which'] = melt_df_post['which'].map(lambda x: 'Graph' if x == graph_post_col else 'CodeLMs')
            sns.violinplot(x='which', y='mean_norm', data=melt_df_post, inner='quartile', palette=sns.color_palette('Set2', n_colors=len(avail_post)))
            sns.swarmplot(x='which', y='mean_norm', data=melt_df_post, color='k', alpha=0.3, size=3)
            plt.title('Violin plot of post-projection mean norms (per-graph): Graph vs CodeLMs')
            plt.tight_layout()
            plt.savefig(out_dir / 'violin_graph_codelms_post.png')
            plt.close()

        # --- Enhanced boxplot with medians annotated (post) ---
        if avail_post:
            plt.figure(figsize=(6,5))
            box_df_post = df[avail_post].melt(var_name='which', value_name='mean_norm')
            box_df_post['which'] = box_df_post['which'].map(lambda x: 'Graph' if x == graph_post_col else 'CodeLMs')
            axp = sns.boxplot(x='which', y='mean_norm', data=box_df_post, palette=sns.color_palette('Pastel1', n_colors=len(avail_post)), showfliers=False)
            medians_post = box_df_post.groupby(['which'])['mean_norm'].median().values
            positions_post = range(len(avail_post))
            for pos, med in zip(positions_post, medians_post):
                axp.text(pos, med, f'{med:.3f}', horizontalalignment='center', size='small', color='black', weight='semibold')
            plt.xticks(rotation=45)
            plt.title('Boxplot of post-projection mean norms (per-graph): Graph vs CodeLMs')
            plt.tight_layout()
            plt.savefig(out_dir / 'boxplot_graph_codelms_post_enhanced.png')
            plt.close()

print("Saved pre-projection plots to:", out_dir)
