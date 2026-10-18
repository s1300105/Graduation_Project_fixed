import argparse
import json
from pathlib import Path
import inspect
from datetime import datetime
import subprocess
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import re
from sklearn.model_selection import train_test_split
import hashlib
import random
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    auc
)
import torch
import torch.nn.functional as F

# =====================================================================
# utils 相当の関数たち（ファイル内にそのまま含める）
# =====================================================================

# regex to remove empty lines
def remove_empty_lines(text):
    return re.sub(r'^$\n', '', text, flags=re.MULTILINE)


# regex to remove comments from a file
def remove_comments(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)


# regex to remove space before newLine character
def remove_space_before_newline(text):
    return re.sub(r'\s+$', '', text, flags=re.MULTILINE)


# regex to remove space after newLine character
def remove_space_after_newline(text):
    return re.sub(r'^\s+', '', text, flags=re.MULTILINE)


def get_dir(path) -> Path:
    """Get path, if exists. If not, create it."""
    Path(path).mkdir(exist_ok=True, parents=True)
    return path


def external_dir() -> Path:
    """Get storage external path."""
    path = storage_dir() / "external"
    Path(path).mkdir(exist_ok=True, parents=True)
    return path


def project_dir() -> Path:
    """Get project path."""
    return Path(__file__).parent.parent


def processed_dir() -> Path:
    """Get storage processed path."""
    path = storage_dir() / "processed"
    Path(path).mkdir(exist_ok=True, parents=True)
    return path


def result_dir() -> Path:
    """Get result path."""
    path = storage_dir() / "results"
    Path(path).mkdir(exist_ok=True, parents=True)
    return path


def storage_dir() -> Path:
    """Get storage path."""
    return Path(__file__).parent.parent / "storage"


def data_dir() -> Path:
    """Get dataset path."""
    path = storage_dir() / "data"
    Path(path).mkdir(exist_ok=True, parents=True)
    return path


def cache_dir() -> Path:
    """Get storage cache path."""
    path = storage_dir() / "cache"
    Path(path).mkdir(exist_ok=True, parents=True)
    return path


def debug(msg, noheader=False, sep="\t"):
    """Print to console with debug information."""
    caller = inspect.stack()[1]
    file_name = caller.filename
    ln = caller.lineno
    now = datetime.now()
    time = now.strftime("%m/%d/%Y - %H:%M:%S")
    if noheader:
        print("\t\x1b[94m{}\x1b[0m".format(msg), end="")
        return
    print(
        '\x1b[40m[{}] File "{}", line {}\x1b[0m\n\x1b[94m{}\x1b[0m'.format(
            time, file_name, ln, msg
        )
    )


def subprocess_cmd(command: str, verbose: int = 0, force_shell: bool = False):
    """Run command line process."""
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    output = process.communicate()
    if verbose > 1:
        debug(output[0].decode())
        debug(output[1].decode())
    return output


def hashstr(s):
    """Hash a string."""
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (10 ** 8)


def dfmp(df, function, columns=None, ordr=True, workers=6, cs=10, desc="Run: "):
    """Parallel apply function on dataframe."""
    if isinstance(columns, str):
        items = df[columns].tolist()
    elif isinstance(columns, list):
        items = df[columns].to_dict("records")
    elif isinstance(df, pd.DataFrame):
        items = df.to_dict("records")
    elif isinstance(df, list):
        items = df
    else:
        raise ValueError("First argument of dfmp should be pd.DataFrame or list.")

    processed = []
    desc = f"({workers} Workers) {desc}"
    with Pool(processes=workers) as p:
        map_func = getattr(p, "imap" if ordr else "imap_unordered")
        for ret in tqdm(map_func(function, items, cs), total=len(items), desc=desc):
            processed.append(ret)
    return processed


def train_val_test_split_df(df, idcol='new_id', labelcol='vul'):
    """Add train/val/test column into dataframe."""
    print(df.shape, df.columns)

    X = df[idcol]
    y = df[labelcol]
    train_rat = 0.8
    val_rat = 0.1
    test_rat = 0.1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - train_rat, random_state=42, stratify=y
    )
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

    def path_to_label(path):
        if path in X_train:
            return "train"
        if path in X_val:
            return "valid"
        if path in X_test:
            return "test"

    df["partition"] = df[idcol].apply(path_to_label)
    return df


# =====================================================================
# ここから本題：MSR_data_cleaned.csv を JSONL (idx, func, target) に変換
# =====================================================================

def cleaned_code(func_code: str) -> str:
    """
    BigVul 前処理コードと同様に、
    - 空行削除
    - コメント削除
    - 行頭の余計な空白削除
    を行う。
    """
    if not isinstance(func_code, str):
        return ""

    func_code = remove_empty_lines(func_code)
    func_code = remove_comments(func_code)
    func_code = remove_space_after_newline(func_code)
    return func_code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        default="data/raw/MSR_data_cleaned.csv",
        help="生の MSR_data_cleaned.csv へのパス",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="data/processed/bigvul_defect.jsonl",
        help="出力する JSONL ベースパス（_train/_valid/_test が付きます）",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_jsonl)
    get_dir(output_path.parent)

    print(f"Loading CSV from {input_path}")
    df = pd.read_csv(input_path)
    print(df.info())

    # 元コードと同様: "Unnamed: 0" を "id" にリネーム（ある場合）
    if "Unnamed: 0" in df.columns and "id" not in df.columns:
        df = df.rename(columns={"Unnamed: 0": "id"})

    if "func_before" not in df.columns or "vul" not in df.columns:
        raise ValueError("CSV に 'func_before' または 'vul' 列がありません。")

    # 不要な NaN を落としておく
    df = df.dropna(subset=["func_before", "vul"]).copy()

    # コードのクレンジング
    print("Cleaning func_before ...")
    df["func"] = df["func_before"].apply(cleaned_code)

    # ============================================================
    # ① 同じ func なのに vul ラベルが異なるものは「全部削除」
    # ============================================================
    print("Checking duplicate functions and label conflicts ...")
    # func ごとに vul の種類数を数える
    vul_nunique = df.groupby("func")["vul"].nunique()

    # ラベルが食い違っている func（例: vul=0 と 1 が混在）
    conflict_funcs = vul_nunique[vul_nunique > 1].index
    # ラベルが1種類だけの func（こちらは重複していても代表1件は残す）
    same_label_funcs = vul_nunique[vul_nunique == 1].index

    print(f"Total unique func: {len(vul_nunique)}")
    print(f"Funcs with label conflicts (will be removed entirely): {len(conflict_funcs)}")
    print(f"Funcs without conflict (candidate for dedup keep-one): {len(same_label_funcs)}")

    # ラベル衝突している func に属する行をすべて削除
    before_conf = len(df)
    conflict_mask = df["func"].isin(conflict_funcs)
    df = df[~conflict_mask].copy()
    after_conf = len(df)
    print(f"Removed {before_conf - after_conf} rows belonging to conflicting funcs.")

    # ============================================================
    # ② 残ったデータについては「同じ func は1行だけ残す」
    #    （この時点では func ごとに vul は1種類だけ）
    # ============================================================
    before_dup = len(df)
    df = df.drop_duplicates(subset=["func"], keep="first").copy()
    after_dup = len(df)
    print(f"Removed {before_dup - after_dup} duplicate rows (same func & same label). Remaining: {after_dup}")

    # target を int にする
    df["target"] = df["vul"].astype(int)

    # idx を改めて設定（重複除去後の DataFrame に合わせる）
    if "id" in df.columns:
        df["idx"] = df["id"].astype(int)
    else:
        df["idx"] = df.index.astype(int)

    # ============================================================
    # ③ train/valid/test に分割（stratify付き）
    # ============================================================
    print("Splitting into train/valid/test ...")
    # ここで idcol="idx", labelcol="target" を指定して再利用
    df = train_val_test_split_df(df, idcol="idx", labelcol="target")

    df_train = df[df["partition"] == "train"].copy()
    df_valid = df[df["partition"] == "valid"].copy()
    df_test  = df[df["partition"] == "test"].copy()

    print(f"Final split sizes -> train: {len(df_train)}, valid: {len(df_valid)}, test: {len(df_test)}")

    # 出力ファイル名（ベース名 + _train/_valid/_test）
    base = output_path.with_suffix("").name  # "bigvul_defect" など
    out_dir = output_path.parent

    train_path = out_dir / f"{base}_train.jsonl"
    valid_path = out_dir / f"{base}_valid.jsonl"
    test_path  = out_dir / f"{base}_test.jsonl"

    # ============================================================
    # ④ JSONL として書き出し（idx, func, target）
    # ============================================================
    def write_split(df_split, path):
        print(f"Writing {len(df_split)} samples to {path}")
        with path.open("w", encoding="utf-8") as f:
            for _, row in tqdm(df_split.iterrows(), total=len(df_split)):
                obj = {
                    "idx": int(row["idx"]),
                    "func": row["func"],
                    "target": int(row["target"]),
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    write_split(df_train, train_path)
    write_split(df_valid, valid_path)
    write_split(df_test,  test_path)

    print("Done.")
    print(f"train: {train_path}")
    print(f"valid: {valid_path}")
    print(f"test : {test_path}")


if __name__ == "__main__":
    main()
