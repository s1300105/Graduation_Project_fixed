import json
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import re
from sklearn.model_selection import train_test_split  # 追加

# =========================================
# データセット分割関数（ユーザー関数を流用）
# =========================================

def train_val_test_split_df(df, idcol='idx', labelcol='target'):
    """Add train/val/test column into dataframe."""
    print(df.shape, df.columns)

    X = df[idcol]
    y = df[labelcol]
    train_rat = 0.8
    val_rat = 0.1
    test_rat = 0.1

    # まず train / (val+test) に分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - train_rat, random_state=42, stratify=y
    )
    # 次に (val+test) を val / test に分割
    X_val, X_test, y_val, y_test = train_test_split(
        X_test,
        y_test,
        test_size=test_rat / (test_rat + val_rat),
        random_state=42,
        stratify=y_test,
    )

    X_train = set(X_train)
    X_val = set(X_val)
    X_test = set(X_test)
    print(f'train:{len(X_train)}, valid:{len(X_val)}, test:{len(X_test)}')

    def id_to_partition(id_):
        if id_ in X_train:
            return "train"
        if id_ in X_val:
            return "valid"
        if id_ in X_test:
            return "test"
        return None

    df["partition"] = df[idcol].apply(id_to_partition)
    return df


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
# DiverseVul ローダー（json または jsonl 自動判別）
# =========================================

def load_diversevul(json_path: Path):
    text = json_path.read_text(encoding="utf-8")

    if text.lstrip().startswith("["):
        # JSON array
        data = json.loads(text)
        return pd.DataFrame(data)

    # JSONL として1行ずつ読む
    rows = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except:
            continue
    return pd.DataFrame(rows)


# =========================================
# メイン処理
# =========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        type=str,
        default="data/raw/diversevul_20230702.json",
        help="DiverseVul JSON/JSONL のパス"
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="data/processed/diversevul_defect.jsonl",
        help="出力 JSONL のベースパス（_train/_valid/_test が付きます）"
    )
    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading DiverseVul from {input_path}")
    df = load_diversevul(input_path)
    print(df.info())

    # DiverseVul のコードカラム名は dataset により異なる
    # 一般的には "func" または "func_before"
    code_col = None
    for c in ["func", "func_before", "source_code", "code"]:
        if c in df.columns:
            code_col = c
            break
    if code_col is None:
        raise ValueError("関数コードの列 (func / func_before / code) が見つかりません")

    # ラベル列
    label_col = None
    for c in ["target", "vul", "label", "is_vulnerable"]:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise ValueError("ラベル列 (target/vul/label) が見つかりません")

    # クレンジング
    print("Cleaning function code...")
    df["clean_func"] = df[code_col].apply(cleaned_code)
    df["target"] = df[label_col].astype(int)

    # ===============================
    # ラベルが異なる重複関数を削除
    # ===============================
    print("Resolving duplicates with different labels...")

    groups = df.groupby("clean_func")["target"].nunique()
    conflict_funcs = groups[groups > 1].index  # ラベルが複数 → 矛盾

    print(f"Found {len(conflict_funcs)} conflicting duplicate functions.")

    df = df[~df["clean_func"].isin(conflict_funcs)].copy()

    # 残った重複（ラベル同じ）は 1 つだけ残す
    before = len(df)
    df = df.drop_duplicates(subset=["clean_func", "target"], keep="first")
    after = len(df)
    print(f"Removed {before - after} same-label duplicates.")

    # idx を付与
    df = df.reset_index(drop=True)
    df["idx"] = df.index

    # ===============================
    # train/valid/test に分割
    # ===============================
    print("Splitting into train/valid/test...")
    df = train_val_test_split_df(df, idcol="idx", labelcol="target")

    # 各パーティションの DataFrame
    df_train = df[df["partition"] == "train"]
    df_valid = df[df["partition"] == "valid"]
    df_test  = df[df["partition"] == "test"]

    print(f"Final split sizes -> train: {len(df_train)}, valid: {len(df_valid)}, test: {len(df_test)}")

    # 出力ファイル名（ベース名 + _train/_valid/_test）
    base = output_path.with_suffix("").name  # "diversevul_defect"
    out_dir = output_path.parent

    train_path = out_dir / f"{base}_train.jsonl"
    valid_path = out_dir / f"{base}_valid.jsonl"
    test_path  = out_dir / f"{base}_test.jsonl"

    # ===============================
    # JSONL 出力
    # ===============================
    def write_split(df_split, path):
        print(f"Writing {len(df_split)} samples to {path}")
        with path.open("w", encoding="utf-8") as f:
            for _, row in tqdm(df_split.iterrows(), total=len(df_split)):
                obj = {
                    "idx": int(row["idx"]),
                    "func": row["clean_func"],
                    "target": int(row["target"]),
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    write_split(df_train, train_path)
    write_split(df_valid, valid_path)
    write_split(df_test,  test_path)

    print("Done!")


if __name__ == "__main__":
    main()
