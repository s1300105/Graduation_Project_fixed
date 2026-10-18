# check_dataset_size.py
from pathlib import Path
import pandas as pd

# === あなたのプロジェクト構成に合わせる ===
in_dir = Path("./data/input")

# === すべての *_cpg_input.pkl ファイルを探索 ===
files = sorted(in_dir.glob("*_cpg_input.pkl"))
if not files:
    raise FileNotFoundError("⚠️ data/input/ に *_cpg_input.pkl が見つかりません。")

total = 0
print("=== データ件数チェック ===")
for f in files:
    df = pd.read_pickle(f)
    n = len(df)
    total += n
    print(f"{f.name:25s}: {n:6d} samples")

print(f"\n📊 合計: {total:,} samples\n")

# オプション: 各ファイルの列名とData型も確認したい場合
sample_file = files[0]
df_sample = pd.read_pickle(sample_file)
print("例:", sample_file.name)
print("列:", list(df_sample.columns))
print("先頭行:")
print(df_sample.head(1))
