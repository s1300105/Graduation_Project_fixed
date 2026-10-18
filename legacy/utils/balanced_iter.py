# ===== utils/train/balanced_iter.py =====
from collections import deque
import random

class BalancedBatcher:
    """
    任意の (data, y) を吐くイテレータを受け取り、
    バッファして 0/1 を半々に混ぜてバッチ化して返す。
    - inner_iter: 各要素が (graph_data, label:int{0,1}) を返すイテレータ
    - batch_size: 例: 8（偶数推奨）
    - buf_cap:    ラベル別のバッファ上限
    """
    def __init__(self, inner_iter, batch_size=8, buf_cap=128):
        self.inner_iter = iter(inner_iter)
        self.batch_size = batch_size
        self.buf0 = deque(maxlen=buf_cap)
        self.buf1 = deque(maxlen=buf_cap)

    def __iter__(self):
        return self

    def __next__(self):
        # 目標は半々
        need0 = self.batch_size // 2
        need1 = self.batch_size - need0

        # バッファが足りなければ追加で読み込む
        while len(self.buf0) < need0 or len(self.buf1) < need1:
            g, y = next(self.inner_iter)  # StopIteration は上へ伝播
            if int(y) == 1:
                self.buf1.append((g, y))
            else:
                self.buf0.append((g, y))

        batch = []
        # 0から必要数
        for _ in range(need0):
            batch.append(self.buf0.popleft())
        # 1から必要数
        for _ in range(need1):
            batch.append(self.buf1.popleft())

        random.shuffle(batch)
        # DataLoader の collate に任せたいので、(gs, ys) に分解
        gs, ys = zip(*batch)
        return gs, ys
