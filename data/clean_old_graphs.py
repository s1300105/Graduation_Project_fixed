import pandas as pd
from pathlib import Path
import shutil
import gc
import tempfile
import os

# === 設定 ===
IN_DIR = Path("./data/input")          # 元＆出力ディレクトリ（同じ）
BACKUP_DIR = Path("./data/input_backup")
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# 不要カラム（DataFrame列レベル）
UNUSED_COLUMNS = {
    "start_lines", "end_lines", "start_cols", "end_cols",
    "line", "end_line", "column", "end_column"
}

def clean_pyg_data_attr(d):
    """torch_geometric.data.Data を想定。不要属性を安全に削除。"""
    if hasattr(d, "__dict__"):
        for k in ("line", "end_line", "column", "end_column"):
            if hasattr(d, k):
                try:
                    delattr(d, k)
                except Exception:
                    pass
    return d

def atomic_overwrite(df: pd.DataFrame, dst_path: Path, *, protocol=5):
    """同一ディレクトリで一時ファイルに保存→置換（非圧縮）"""
    # 同一ディレクトリ配下に一時ファイル
    with tempfile.NamedTemporaryFile(dir=dst_path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.to_pickle(tmp_path, compression=None, protocol=protocol)
        os.replace(tmp_path, dst_path)  # 原子的に置換
    except Exception:
        # 失敗時は一時ファイルを削除
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise

def main():
    files = sorted(IN_DIR.glob("*_cpg_input.pkl"))
    print(f"Found {len(files)} files to clean in {IN_DIR}.\n")

    for f in files:
        print(f"→ Processing: {f.name}")
        try:
            # 元ファイルをバックアップ（未作成時のみ）
            backup_path = BACKUP_DIR / f.name
            if not backup_path.exists():
                shutil.copy2(f, backup_path)
                print(f"   📦 Backup saved → {backup_path.name}")

            # 読み込み（非圧縮前提）
            df = pd.read_pickle(f)

            # ---- DataFrameレベルの不要列削除 ----
            drop_cols = [c for c in df.columns if c in UNUSED_COLUMNS]
            if drop_cols:
                df = df.drop(columns=drop_cols)
                print(f"   - Dropped columns: {drop_cols}")

            # ---- input(Data) 内部の不要属性削除 ----
            if "input" in df.columns:
                df["input"] = [clean_pyg_data_attr(d) for d in df["input"]]
                print(f"   - Cleaned Data attributes inside 'input'")

            # ---- 非圧縮で上書き保存（原子置換）----
            atomic_overwrite(df, f, protocol=5)
            print(f"   ✔ Overwritten (no gzip) → {f.name}")

            del df
            gc.collect()

        except Exception as e:
            print(f"   ⚠ Error processing {f.name}: {e}")

    print("\n✅ Cleaning completed!")
    print(f"Output (cleaned) files stay in: {IN_DIR}\nBackups are in: {BACKUP_DIR}")

if __name__ == "__main__":
    main()
