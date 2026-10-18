import json
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import re

# =========================================
# BigVul と同じ前処理関数
# =========================================

def remove_empty_lines(text):
    return re.sub(r'^$\n', '', text, flags=re.MULTILINE)

def remove_comments(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE,
    )
    return re.sub(pattern, replacer, text)

def remove_space_after_newline(text):
    return re.sub(r'^\s+', '', text, flags=re.MULTILINE)

def cleaned_code(code):
    if not isinstance(code, str):
        return ""
    code = remove_empty_lines(code)
    code = remove_comments(code)
    code = remove_space_after_newline(code)
    return code


# =========================================
# json / jsonl ローダー
# =========================================

def load_json_or_jsonl(json_path: Path) -> pd.DataFrame:
    """
    - 先頭が '[' なら JSON 配列
    - それ以外は 1 行 1 JSON (JSONL) として読む
    """
    text = json_path.read_text(encoding="utf-8")

    if text.lstrip().startswith("["):
        data = json.loads(text)
        return pd.DataFrame(data)

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return pd.DataFrame(rows)


# =========================================
# メイン処理（primeVul 用）
# =========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_train_jsonl",
        type=str,
        default="/home/yudai/Project/research/Graduation_Project/data/raw/primevul_train.jsonl",
        help="primeVul train JSONL のパス"
    )
    parser.add_argument(
        "--input_valid_jsonl",
        type=str,
        default="/home/yudai/Project/research/Graduation_Project/data/raw/primevul_valid.jsonl",
        help="primeVul valid JSONL のパス"
    )
    parser.add_argument(
        "--input_test_jsonl",
        type=str,
        default="/home/yudai/Project/research/Graduation_Project/data/raw/primevul_test.jsonl",
        help="primeVul test JSONL のパス"
    )
    parser.add_argument(
        "--output_jsonl_base",
        type=str,
        default="data/processed/primevul_defect.jsonl",
        help="出力 JSONL のベースパス（_train/_valid/_test が付きます）"
    )
    args = parser.parse_args()

    train_path_in = Path(args.input_train_jsonl)
    valid_path_in = Path(args.input_valid_jsonl)
    test_path_in  = Path(args.input_test_jsonl)

    output_base = Path(args.output_jsonl_base)
    out_dir = output_base.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = output_base.with_suffix("").name  # "primevul_defect"

    print(f"Loading primeVul train from {train_path_in}")
    df_train = load_json_or_jsonl(train_path_in)
    df_train["partition"] = "train"

    print(f"Loading primeVul valid from {valid_path_in}")
    df_valid = load_json_or_jsonl(valid_path_in)
    df_valid["partition"] = "valid"

    print(f"Loading primeVul test from {test_path_in}")
    df_test = load_json_or_jsonl(test_path_in)
    df_test["partition"] = "test"

    # 3つを結合して一括でクレンジング & 重複処理
    df = pd.concat([df_train, df_valid, df_test], ignore_index=True)
    print("Combined primeVul shape:", df.shape)
    print(df.columns)

    # 必須カラム確認
    required_cols = {"idx", "func", "target"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"primeVul に必須カラム {required_cols} が揃っていません。欠けている: {missing}")

    # 型を揃える
    df["idx"] = df["idx"].astype(int)
    df["target"] = df["target"].astype(int)

    # クレンジング
    print("Cleaning function code...")
    df["clean_func"] = df["func"].apply(cleaned_code)

    # ===============================
    # ラベルが異なる重複関数を削除
    # ===============================
    print("Resolving duplicates with different labels...")

    groups = df.groupby("clean_func")["target"].nunique()
    conflict_funcs = groups[groups > 1].index  # ラベルが複数 → 矛盾

    print(f"Found {len(conflict_funcs)} conflicting duplicate functions.")
    before_conflict = len(df)
    df = df[~df["clean_func"].isin(conflict_funcs)].copy()
    after_conflict = len(df)
    print(f"Removed {before_conflict - after_conflict} rows because of conflicting labels.")

    # 残った重複（ラベル同じ）は 1 つだけ残す
    before_dup = len(df)
    df = df.drop_duplicates(subset=["clean_func", "target"], keep="first")
    after_dup = len(df)
    print(f"Removed {before_dup - after_dup} same-label duplicates.")

    # idx は元の値をそのまま使う（再採番しない）
    df = df.reset_index(drop=True)

    # 再度 train/valid/test に分割（もともとの partition を維持）
    df_train = df[df["partition"] == "train"].copy()
    df_valid = df[df["partition"] == "valid"].copy()
    df_test  = df[df["partition"] == "test"].copy()

    print(f"Final split sizes -> train: {len(df_train)}, valid: {len(df_valid)}, test: {len(df_test)}")

    # 出力パス
    train_out = out_dir / f"{base_name}_train.jsonl"
    valid_out = out_dir / f"{base_name}_valid.jsonl"
    test_out  = out_dir / f"{base_name}_test.jsonl"

    # ===============================
    # JSONL 出力（idx, func, target）
    # ===============================
    def write_split(df_split, path):
        print(f"Writing {len(df_split)} samples to {path}")
        with path.open("w", encoding="utf-8") as f:
            for _, row in tqdm(df_split.iterrows(), total=len(df_split)):
                obj = {
                    "idx": int(row["idx"]),          # 元の idx を使用
                    "func": row["clean_func"],       # クレンジング後コード
                    "target": int(row["target"]),
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    write_split(df_train, train_out)
    write_split(df_valid, valid_out)
    write_split(df_test,  test_out)

    print("Done!")


if __name__ == "__main__":
    main()
