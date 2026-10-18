# save_utils.py
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset

def _ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def _get_indices_from_dataset(ds):
    """torch.utils.data.Subset なら ds.indices、無ければ range(len(ds))。"""
    if hasattr(ds, "indices"):
        return [int(i) for i in ds.indices]
    return list(range(len(ds)))

def _get_dataset(obj):
    """DataLoader or Subset or Dataset から基底 dataset を取り出す."""
    if isinstance(obj, DataLoader):
        return obj.dataset
    return obj  # Subset/Dataset とみなす

def save_split_from_loaders(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    input_dataset_len: int,
    out_path: str = "artifacts/splits/split.json",
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    既に作成済みの train/val/test ローダから、分割インデックスとメタ情報をJSON保存。
    """
    train_ds = _get_dataset(train_loader)
    val_ds   = _get_dataset(val_loader)
    test_ds  = _get_dataset(test_loader)

    splits = {
        "meta": {
            "total_len": int(input_dataset_len),
            "batch_size": int(batch_size) if batch_size is not None else None,
            "shuffle": bool(shuffle) if shuffle is not None else None,
            **(extra_meta or {}),
        },
        "indices": {
            "train": _get_indices_from_dataset(train_ds),
            "val":   _get_indices_from_dataset(val_ds),
            "test":  _get_indices_from_dataset(test_ds),
        },
    }

    out_path = str(out_path)
    _ensure_dir(Path(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)
    return out_path

def load_split_indices(
    split_json_path: str,
    input_dataset,
    batch_size: int,
    shuffle_train: bool = True,
    use_dataset_get_loader: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    保存した split.json を読み込み、Subset→DataLoader を復元。
    """
    with open(split_json_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    idx = splits["indices"]

    train_subset = Subset(input_dataset, idx["train"])
    val_subset   = Subset(input_dataset, idx["val"])
    test_subset  = Subset(input_dataset, idx["test"])

    if use_dataset_get_loader and hasattr(train_subset, "get_loader"):
        train_loader = train_subset.get_loader(batch_size, shuffle=shuffle_train)
        val_loader   = val_subset.get_loader(batch_size, shuffle=False)
        test_loader  = test_subset.get_loader(batch_size, shuffle=False)
    else:
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle_train)
        val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_subset,  batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def cache_loader_items(loader: DataLoader, out_path: str) -> str:
    """
    ローダの基底データセットを1件ずつ列挙して .pt にキャッシュ（PyG DataもOK）。
    ※ローダを回さず dataset[i] を直接読むので、バッチ変換に依存しません。
    """
    ds = _get_dataset(loader)
    items = [ds[i] for i in range(len(ds))]
    out_path = str(out_path)
    _ensure_dir(Path(out_path))
    torch.save(items, out_path)
    return out_path

# ---- 小さいサンプルを JSON に吐く（可視確認用） ----

def _to_jsonable(x):
    from torch import Tensor
    pyg_Data = pyg_Batch = None
    try:
        from torch_geometric.data import Data as _D, Batch as _B
        pyg_Data, pyg_Batch = _D, _B
    except Exception:
        pass

    if isinstance(x, Tensor): return x.tolist()
    if isinstance(x, (int, float, str, bool)) or x is None: return x
    if isinstance(x, (list, tuple)): return [_to_jsonable(v) for v in x]
    if isinstance(x, dict): return {k: _to_jsonable(v) for k, v in x.items()}
    if pyg_Batch and isinstance(x, pyg_Batch):
        return [_to_jsonable(d) for d in x.to_data_list()]
    if pyg_Data and isinstance(x, pyg_Data):
        d = {}
        for k, v in x.__dict__.items():
            if k.startswith("_"): continue
            d[k] = _to_jsonable(v)
        return d
    return repr(x)

def save_loader_sample_json(
    loader: DataLoader,
    out_path: str = "artifacts/inspect/train_sample.json",
    max_batches: int = 1,
) -> str:
    """
    ローダ先頭 max_batches をJSON保存（巨大化に注意）。可視チェック用途。
    """
    it, out = iter(loader), []
    for _ in range(max_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        out.append(_to_jsonable(batch))
    out_path = str(out_path)
    _ensure_dir(Path(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out_path
