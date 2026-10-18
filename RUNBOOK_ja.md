# GPU マシンでの実行手順（RUNBOOK）

卒論で使ったワークステーション（RTX 3090、Ubuntu）を想定。所要時間は目安で、必ず手順 6 の小規模実行で計測してから本番を回すこと。

## 0. 前提

- NVIDIA ドライバ + CUDA 12.x が動く GPU（`nvidia-smi` で確認）
- Python 3.10 または 3.11
- Java 17 以上（Joern 4 用。開発時に動作確認したのは OpenJDK 21）
- BigVul の `MSR_data_cleaned.csv`（旧リポジトリの `data/raw/` にあるもの）
- ディスク: CPG 生成の中間ファイルは既定で削除されるが、`data/input/*.pkl` は数十 GB になりうる

## 1. 取得と環境

```bash
git clone https://github.com/s1300105/Graduation_Project_fixed.git
cd Graduation_Project_fixed
python3.10 -m venv venv && . venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124   # 3090 なら cu124 でよい
pip install -r requirements.txt                                        # torch_geometric は pure-Python 版で動く（torch_scatter 不要）
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"   # True を確認
pytest -q        # 67 件。初回は CodeBERT（約 500 MB）をダウンロードする。CPU で数分
```

## 2. Joern 4

```bash
mkdir -p joern && cd joern
curl -L -o joern-cli.zip https://github.com/joernio/joern/releases/download/v4.0.400/joern-cli.zip   # 約 1.9 GB
unzip -q joern-cli.zip && rm joern-cli.zip && cd ..
joern/joern-cli/joern --version
```

`run.py` は既定で `joern/joern-cli/joern` と `joern/joern-cli/joern-parse` を呼び、書き出しスクリプトは `joern/graph-for-funcs.sc` を使う。別の場所に入れた場合は `--joern_cli_dir /path/to/joern-cli/` を付ける。

## 3. BigVul → jsonl（約 10 分）

```bash
mkdir -p data/raw data/cleaned_data
ln -s /path/to/MSR_data_cleaned.csv data/raw/MSR_data_cleaned.csv     # コピーでもよい
python data/scripts/make_bigvul_json.py \
    --input_csv data/raw/MSR_data_cleaned.csv \
    --output_jsonl data/cleaned_data/bigvul.jsonl
ls data/cleaned_data        # bigvul_train.jsonl / bigvul_valid.jsonl / bigvul_test.jsonl
```

- 旧リポジトリの `bigvul_defect_*.jsonl` を流用することもできる（`--dataset bigvul_defect`）が、`lang` / `project` 列が無いので C++ 判定が構文ヒント頼みになる。作り直しを推奨。
- プロジェクト横断分割は `--split project --output_jsonl data/cleaned_data/bigvul_xproj.jsonl`（後で `--dataset bigvul_xproj`）。

## 4. CPG 生成と埋め込み（最も時間がかかる。tmux / nohup で）

```bash
mkdir -p logs
nohup python run.py --cpg --embed --dataset bigvul > logs/cpg_embed.log 2>&1 &
tail -f logs/cpg_embed.log
```

- `--cpg`: 500 関数ずつ Joern を回す（JVM 起動 2 回/スライス）。15 万関数で約 300 スライス。1 スライス 1〜3 分なら合計 5〜15 時間。`--slice_size 1000` で JVM 回数を半減できる（メモリに余裕があれば）。
- `--embed`: 凍結 CodeBERT でノード特徴を作る。GPU なら fp16 で動く。中断したら同じフラグで `python run.py --embed --resume --dataset bigvul`（フラグが違うと拒否される）。
- 1200 字フィルタは既定で無効。論文と同じ条件に揃えたいときだけ `--max_chars 1200` を付ける。
- 終わったら必ず manifest を見る:

```bash
python -c "import json;print(json.dumps(json.load(open('data/cpg/cpg_manifest.json'))['splits'],indent=1))"
cat data/input/manifest.json
```

  - `rows_lost_in_joern`: Joern で解析できず除外された関数数（論文に書く）
  - `skipped_global`: `<global>` の除外数（0 でなければ修正が効いている）
  - `positives_after_filter`: 各分割の陽性数
  - `nodes_mean` / `edges_mean` / `num_relations`（3。`--reverse_edges` なら 6）

## 5. 任意: 変種の入力を作る

同じ `data/input` を全モデルが読むので、入力の変種ごとに `--data_root` を分ける。

```bash
# 逆辺あり
python run.py --embed --dataset bigvul --reverse_edges --data_root data_rev        # cpg は data/cpg をコピー or 再生成
# ノードの CODE を先頭行だけにする（M22 の緩和）
python run.py --embed --dataset bigvul --node_code_mode first_line --data_root data_fl
```

（`--embed` は `<data_root>/cpg/*_cpg.pkl` を読むので、`cp -r data/cpg data_rev/cpg` のように CPG を先に置く。）

## 6. 小規模で計測（30 分）

```bash
python run.py --mode train --model_type codebert --subset_frac 0.05 --epochs 2 --patience 2 --seeds 0 --runs_dir runs_smoke
python run.py --mode train --model_type hybrid   --subset_frac 0.05 --epochs 2 --patience 2 --seeds 0 --runs_dir runs_smoke
python -c "import json;[print(json.loads(l)['epoch'],round(json.loads(l)['elapsed_s'])) for l in open('runs_smoke/hybrid_bs16_lr1.0e-04_br0.30_seed0/history.jsonl')]"
```

`elapsed_s` から 1 epoch の時間を出す（5% で t 秒なら全データで約 20t 秒）。GPU メモリが足りなければ `--batch_sizes 8 --eval_batch_size 8`。

## 7. 本番（推奨順）

論文の設定は bs16 / lr 1e-4 / CodeBERT lr 比 0.3 / 200 epoch / patience 50。旧コードの曲線では検証損失の最小が epoch 30〜75、F1 は 100 epoch 以降ほぼ平坦だったので、まず `--epochs 40 --patience 10` で全腕を揃え、必要なら伸ばす。

```bash
COMMON="--epochs 40 --patience 10 --seeds 0 1 2"
# (a) グラフ枝に信号が届いているか
nohup python run.py --mode train --model_type graph_only $COMMON                    > logs/graph_only.log 2>&1
nohup python run.py --mode train --model_type hybrid --ablation shuffle_edges $COMMON > logs/hybrid_shuffle.log 2>&1
# (b) 本比較
nohup python run.py --mode train --model_type codebert $COMMON                      > logs/codebert.log 2>&1
nohup python run.py --mode train --model_type hybrid   $COMMON                      > logs/hybrid.log 2>&1
# (c) 融合の部品
nohup python run.py --mode train --model_type hybrid --ablation no_gate     $COMMON > logs/hybrid_nogate.log 2>&1
nohup python run.py --mode train --model_type hybrid --ablation no_cls_skip $COMMON > logs/hybrid_nocls.log 2>&1
```

- 1 コマンドが 3 seed を順に回す。GPU が 1 枚なら上から順に（同時に走らせない）。
- 各 run は `runs/<model>_bs16_lr1.0e-04_br0.30_seed<k>/` に `metrics.json`（閾値 0.5 と検証調整閾値の両方）、`history.jsonl`、`curves.png`、`best.pth`、`test_preds.npz` を書く。
- 同名 run があると止まる（上書きは `--overwrite`）。

## 8. 判定

```bash
# 構造は使われているか: hybrid とランダム辺対照に差があるか
python scripts/compare_runs.py --a "runs/hybrid_bs16_lr1.0e-04_br0.30_seed*" --b "runs/hybrid-shuffle_edges_bs16_lr1.0e-04_br0.30_seed*"
# 本比較
python scripts/compare_runs.py --a "runs/hybrid_bs16_lr1.0e-04_br0.30_seed*" --b "runs/codebert_bs16_lr1.0e-04_br0.30_seed*"
```

- 出力の `A - B` と 95% 区間を見る。区間が 0 をまたげば「差があるとは言えない」。
- `graph_only` が chance 付近（F1 が陽性率程度）なら、グラフ枝単体に信号が届いていない。
- `history.jsonl` の `frac_p1_mid`（確率が 0.05〜0.95 にある割合）が 0 に近づいたら出力が飽和している。`metrics.json` の `n_thresholds_tied` が大きいのも同じ兆候。

## 9. 追加実験

- 長さ別の層別: `test_preds.npz` の `idx` を jsonl の `func` 長と突き合わせて、512 トークン超とそれ以下で F1 を分ける。
- プロジェクト横断: 手順 3 の `--split project` で作った jsonl を `--dataset bigvul_xproj --data_root data_xproj` で 4〜8 をやり直す。
- ハイパーパラメータ探索は最終モデル・全データで: `--batch_sizes 8 16 32 --lrs 1e-4 5e-5 3e-5 1e-5 --bert_lr_ratios 0.1 0.2 0.3 --seeds 0 --epochs 20 --patience 5`。

## 10. よくある問題

- `CUDA out of memory`: `--batch_sizes 8 --eval_batch_size 8`。学習ループは OOM バッチを飛ばして数える（`history.jsonl` の `oom_skipped`）。0 でないなら設定を下げる。
- Joern が落ちる / 遅い: `logs/cpg_embed.log` の `[ERROR]` 行と manifest の `joern_failed_slices`。JVM のヒープは `JAVA_OPTS=-Xmx8g` などで調整。
- GPU の scatter 演算は非決定的なので同一 seed でも完全一致はしない。だから 3 seed 以上で区間を出す。
- 旧リポジトリのチェックポイントはモデル形状が違うので読めない（学習し直す）。
