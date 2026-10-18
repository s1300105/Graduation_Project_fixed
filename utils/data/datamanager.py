import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import glob
from ..functions import parse
from ..functions.input_dataset import InputDataset
from sklearn.model_selection import train_test_split
from os import listdir
from os.path import isfile, join


def read(path, json_file):
    fp = str(Path(path) / json_file)

    # まずは「JSON Lines 前提 + pythonエンジン」でトライ（巨大整数OK）
    try:
        return pd.read_json(fp, lines=True, engine="python")
    except ValueError:
        # JSON配列（[ {...}, {...} ]) だった場合はこちら
        try:
            return pd.read_json(fp, lines=False, engine="python")
        except ValueError:
            # それでもダメなら行ごとに標準jsonで読み込む（最強フォールバック）
            rows = []
            with open(fp, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        # 行末カンマや壊れた行をスキップ（必要に応じてログ）
                        continue
            return pd.DataFrame(rows)

def write(data_frame: pd.DataFrame, path, file_name):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)  # ディレクトリが無ければ作成

    out_path = p / file_name              # 安全なパス結合
    data_frame.to_pickle(out_path)
    print(f"[INFO] Saved: {out_path}")

def apply_filter(data_frame: pd.DataFrame, filter_func):
    return filter_func(data_frame)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    生データ用の安全なクレンジング関数。
    - 同じ行が完全に重複している場合だけ削除する。
      （project, commit_id, target, func, idx が全部同じもの）
    - 同じ func でも project / commit_id / target が違うものはそのまま残す。
    """
    df = df.copy()

    before = len(df)

    # ① func ごとにラベルが食い違っているものがないかだけチェック（削除はしない）
    if "func" in df.columns and "target" in df.columns:
        vc = df.groupby("func")["target"].nunique()
        conflict_funcs = vc[vc > 1]
        if len(conflict_funcs) > 0:
            print(f"[clean/raw] WARNING: 同じ func で target が異なるものが {len(conflict_funcs)} 個あります。"
                  " 生データでは何も削除しません。")

    # ② 完全に同じ行（全カラム一致）のみ削除
    df_clean = df.drop_duplicates(keep="first")

    after = len(df_clean)
    print(f"[clean/raw] {before} -> {after} rows "
          f"(removed {before - after} exact duplicate rows)")

    return df_clean

def drop(data_frame: pd.DataFrame, keys):
    data_frame.drop(columns=[k for k in keys if k in data_frame.columns], inplace=True)


def slice_frame(data_frame: pd.DataFrame, size:int):
    data_frame_size = len(data_frame)
    return data_frame.groupby(np.arange(data_frame_size) // size)

def to_files(data_frame: pd.DataFrame, out_path):
    os.makedirs(out_path, exist_ok=True)

    for idx, row in data_frame.iterrows():
        file_name = f"{idx}.c"
        fp = os.path.join(out_path, file_name)
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(row.func)

def create_with_index(data_frame, columns):
    """
    data_frame: pandas.DataFrame もしくは (records の) iterable
    columns   : 例 ["Index", "cpg"]
    戻り値    : Index 列を index に設定した DataFrame
    """
    # DataFrame でなければ生成
    if not isinstance(data_frame, pd.DataFrame):
        data_frame = pd.DataFrame(data_frame, columns=columns)

    # 必要な列だけに絞る（存在すれば）
    cols = [c for c in columns if c in data_frame.columns]
    if cols:
        data_frame = data_frame.loc[:, cols].copy()

    # Index 列を整数化して index に設定
    if "Index" in data_frame.columns:
        data_frame["Index"] = pd.to_numeric(data_frame["Index"], errors="coerce").astype("Int64")
        data_frame = data_frame.dropna(subset=["Index"])
        data_frame["Index"] = data_frame["Index"].astype(int)
        # drop=False にしておくと列も残る（後段の参照がある場合に安全）
        data_frame.set_index("Index", drop=False, inplace=True)

    return data_frame

def inner_join_by_index(df1, df2):
    return pd.merge(df1, df2, left_index=True, right_index=True)

def get_directory_files(directory):
    return [os.path.basename(file) for file in glob.glob(f"{directory}/*.pkl")]

def get_ratio(dataset, ratio):
    approx_size = int(len(dataset) * ratio)
    return dataset[:approx_size]

def load(path, pickle_file, ratio=1):
    """
    Pickleファイルをgzip圧縮・非圧縮の両方に対応して安全に読み込む。
    ratio < 1 の場合は一部だけサンプリングして返す。
    """
    full_path = Path(path) / pickle_file
    if not full_path.exists():
        raise FileNotFoundError(f"File not found: {full_path}")

    # gzip判定: 先頭2バイトが 0x1f 0x8b
    try:
        with open(full_path, "rb") as fh:
            head = fh.read(2)
        if head == b"\x1f\x8b":
            dataset = pd.read_pickle(full_path, compression="gzip")
        else:
            dataset = pd.read_pickle(full_path)
    except Exception as e:
        # どちらかで失敗した場合、フォールバックしてもう片方で試す
        try:
            dataset = pd.read_pickle(full_path, compression="gzip")
        except Exception:
            raise RuntimeError(f"Failed to load pickle file (gzip/non-gzip both failed): {full_path}\n{e}")

    # データ情報を出力
    dataset.info(memory_usage='deep')

    # ratio指定があれば部分サンプリング
    if ratio < 1:
        dataset = get_ratio(dataset, ratio)

    return dataset

def loads(data_sets_dir, ratio=1):
    data_sets_files = sorted([f for f in listdir(data_sets_dir) if isfile(join(data_sets_dir, f))])

    if ratio < 1:
        data_sets_files = get_ratio(data_sets_files, ratio)

    dataset = load(data_sets_dir, data_sets_files[0])
    data_sets_files.remove(data_sets_files[0])

    for ds_file in data_sets_files:
        # dataset = dataset.append(load(data_sets_dir, ds_file))
        dataset = pd.concat([dataset, load(data_sets_dir, ds_file)])

    return dataset

def tokenize(data_frame: pd.DataFrame):
    data_frame["tokens"] = data_frame["func"].apply(parse.tokenizer)
    return data_frame[["tokens", "func"]]


def train_val_test_split(data_frame: pd.DataFrame, shuffle=True):
    print("Splitting Dataset")

    false = data_frame[data_frame.target == 0]
    true = data_frame[data_frame.target == 1]

    train_false, test_false = train_test_split(false, test_size=0.2, shuffle=shuffle)
    test_false, val_false = train_test_split(test_false, test_size=0.5, shuffle=shuffle)
    train_true, test_true = train_test_split(true, test_size=0.2, shuffle=shuffle)
    test_true, val_true = train_test_split(test_true, test_size=0.5, shuffle=shuffle)

    # run = train_false.append(train_true)
    train = pd.concat([train_false, train_true])

    # val = val_false.append(val_true)
    val = pd.concat([val_false, val_true])

    # test = test_false.append(test_true)
    test = pd.concat([test_false, test_true])

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    return InputDataset(train), InputDataset(test), InputDataset(val)